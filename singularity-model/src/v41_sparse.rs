//! V4.1 sparse attention CPU building blocks, not a GPU backend.
//! Formulas follow the MIT-licensed checkpoint inference/model.py and kernel.py,
//! revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
use half::bf16;
use singularity_core::SingularityError;

fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

/// One query's indexer scores. q is [heads, dim], keys is [positions, dim],
/// and weights are the projected per-head weights including reference scaling.
pub fn index_scores(
    q: &[f32],
    keys: &[f32],
    weights: &[f32],
    dim: usize,
) -> Result<Vec<f32>, SingularityError> {
    if dim == 0
        || weights.is_empty()
        || weights.len().checked_mul(dim) != Some(q.len())
        || keys.len() % dim != 0
        || q.iter().chain(keys).chain(weights).any(|x| !x.is_finite())
    {
        return Err(error("Invalid sparse indexer input"));
    }
    Ok(keys
        .chunks_exact(dim)
        .map(|key| {
            q.chunks_exact(dim)
                .zip(weights)
                .map(|(query, &w)| {
                    let dot: f32 = query.iter().zip(key).map(|(a, b)| a * b).sum();
                    dot.max(0.0) * w
                })
                .sum()
        })
        .collect())
}

/// Candidate blocks ranked by maximum score, with the newest reachable block
/// pinned. Ties use ascending block ID; accelerator top-k tie order may differ.
pub fn candidate_blocks(
    scores: &[f32],
    reachable: usize,
    block_size: usize,
    top_blocks: usize,
) -> Result<Vec<bool>, SingularityError> {
    if block_size == 0 || reachable > scores.len() || scores.iter().any(|v| v.is_nan()) {
        return Err(error("Invalid candidate block input"));
    }
    let blocks = scores.len().div_ceil(block_size);
    let mut ranks: Vec<(usize, f32)> = (0..blocks)
        .map(|block| {
            let start = block * block_size;
            let end = (start + block_size).min(reachable);
            let score = if start < end {
                scores[start..end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            } else {
                f32::NEG_INFINITY
            };
            (block, score)
        })
        .collect();
    if reachable > 0 {
        ranks[(reachable - 1) / block_size].1 = f32::INFINITY;
    }
    ranks.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected = vec![false; scores.len()];
    for (block, score) in ranks.into_iter().take(top_blocks) {
        if score == f32::NEG_INFINITY {
            continue;
        }
        let start = block * block_size;
        for v in &mut selected[start..(start + block_size).min(scores.len())] {
            *v = true;
        }
    }
    Ok(selected)
}

/// Select reachable compressed positions, sorted into position order. `None`
/// represents padding and offset maps compressed slots into the combined KV.
pub fn compressed_topk(
    scores: &[f32],
    reachable: usize,
    candidate_mask: Option<&[bool]>,
    top_k: usize,
    offset: usize,
) -> Result<Vec<Option<usize>>, SingularityError> {
    if reachable > scores.len()
        || candidate_mask.is_some_and(|m| m.len() != scores.len())
        || scores.iter().any(|x| x.is_nan())
    {
        return Err(error("Invalid compressed top-k input"));
    }
    let width = top_k.min(scores.len());
    let mut valid: Vec<usize> = (0..reachable)
        .filter(|&i| candidate_mask.is_none_or(|m| m[i]) && scores[i] != f32::NEG_INFINITY)
        .collect();
    valid.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    valid.truncate(width);
    valid.sort_unstable();
    let mut out = Vec::with_capacity(width);
    for index in valid {
        out.push(Some(
            index
                .checked_add(offset)
                .ok_or_else(|| error("Compressed index overflow"))?,
        ));
    }
    out.resize(width, None);
    Ok(out)
}

/// Prefill window indices for each query (relative to the prefill buffer).
pub fn prefill_window_indices(
    window: usize,
    length: usize,
) -> Result<Vec<Vec<Option<usize>>>, SingularityError> {
    if window == 0 {
        return Err(error("Window size must be positive"));
    }
    let width = window.min(length);
    Ok((0..length)
        .map(|pos| {
            let start = (pos + 1).saturating_sub(window);
            (0..width)
                .map(|i| {
                    let index = start + i;
                    (index <= pos).then_some(index)
                })
                .collect()
        })
        .collect())
}

/// Decode window indices into the physical ring after writing `position`.
pub fn decode_window_indices(
    window: usize,
    position: usize,
) -> Result<Vec<Option<usize>>, SingularityError> {
    if window == 0 {
        return Err(error("Window size must be positive"));
    }
    let oldest = position % window + 1;
    Ok((oldest..window)
        .chain(0..oldest)
        .map(|slot| (slot <= position).then_some(slot))
        .collect())
}

/// Single-sequence ring cache for the uncompressed sliding-window KV.
/// Each append writes the next causal position and returns physical indices.
pub struct WindowCache {
    window: usize,
    dim: usize,
    next_position: usize,
    data: Vec<f32>,
}

impl WindowCache {
    pub fn new(window: usize, dim: usize) -> Result<Self, SingularityError> {
        let count = window
            .checked_mul(dim)
            .ok_or_else(|| error("Window cache size overflow"))?;
        if count == 0 {
            return Err(error("Invalid window cache dimensions"));
        }
        Ok(Self {
            window,
            dim,
            next_position: 0,
            data: vec![0.; count],
        })
    }

    pub fn append(&mut self, kv: &[f32]) -> Result<Vec<Option<usize>>, SingularityError> {
        if kv.len() != self.dim || kv.iter().any(|x| !x.is_finite()) {
            return Err(error("Invalid window cache value"));
        }
        let next = self
            .next_position
            .checked_add(1)
            .ok_or_else(|| error("Window position overflow"))?;
        let slot = self.next_position % self.window;
        self.data[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(kv);
        let indices = decode_window_indices(self.window, self.next_position)?;
        self.next_position = next;
        Ok(indices)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn position(&self) -> usize {
        self.next_position
    }
    pub fn clear(&mut self) {
        self.data.fill(0.0);
        self.next_position = 0;
    }
}

/// Shared latent KV attention for one query token, q [heads, dim], kv
/// [slots, dim]. Sink adds probability mass but no value. This ideal FP32
/// reference does not emulate the GPU kernel's BF16 probability rounding.
pub fn sparse_attention(
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    indices: &[Option<usize>],
    dim: usize,
    scale: f32,
    round_bf16: bool,
) -> Result<Vec<f32>, SingularityError> {
    if dim == 0
        || sink.is_empty()
        || sink.len().checked_mul(dim) != Some(q.len())
        || kv.len() % dim != 0
        || !scale.is_finite()
        || q.iter().chain(kv).chain(sink).any(|x| !x.is_finite())
    {
        return Err(error("Invalid sparse attention input"));
    }
    let slots = kv.len() / dim;
    if indices.iter().flatten().any(|&i| i >= slots) {
        return Err(error("Sparse attention index out of range"));
    }
    let mut out = vec![0.0; q.len()];
    for (head, query) in q.chunks_exact(dim).enumerate() {
        let mut scores = Vec::new();
        let mut max = sink[head];
        for &idx in indices.iter().flatten() {
            let score: f32 = query
                .iter()
                .zip(&kv[idx * dim..(idx + 1) * dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
            max = max.max(score);
            scores.push((idx, score));
        }
        let mut denom = (sink[head] - max).exp();
        for (idx, score) in scores {
            let weight = (score - max).exp();
            denom += weight;
            for j in 0..dim {
                out[head * dim + j] += weight * kv[idx * dim + j];
            }
        }
        for v in &mut out[head * dim..(head + 1) * dim] {
            *v /= denom;
            if round_bf16 {
                *v = bf16::from_f32(*v).to_f32();
            }
        }
    }
    Ok(out)
}

/// Stateful learned per-coordinate pooling of pre-projected KV and gate logits.
/// The caller performs the projection, BF16 cast, RMSNorm, RoPE and quantization.
pub struct CompressionPool {
    ratio: usize,
    dim: usize,
    kv: Vec<f32>,
    gates: Vec<f32>,
}
impl CompressionPool {
    pub fn new(ratio: usize, dim: usize) -> Result<Self, SingularityError> {
        if ratio == 0 || dim == 0 || ratio.checked_mul(dim).is_none() {
            return Err(error("Invalid compression dimensions"));
        }
        Ok(Self {
            ratio,
            dim,
            kv: Vec::with_capacity(ratio * dim),
            gates: Vec::with_capacity(ratio * dim),
        })
    }
    pub fn push(&mut self, kv: &[f32], gate: &[f32]) -> Result<Option<Vec<f32>>, SingularityError> {
        if kv.len() != self.dim
            || (self.ratio > 1 && gate.len() != self.dim)
            || kv.iter().chain(gate).any(|v| !v.is_finite())
        {
            return Err(error("Invalid compression input"));
        }
        if self.ratio == 1 {
            return Ok(Some(kv.to_vec()));
        }
        self.kv.extend_from_slice(kv);
        self.gates.extend_from_slice(gate);
        if self.kv.len() < self.ratio * self.dim {
            return Ok(None);
        }
        let mut result = vec![0.0; self.dim];
        for (j, out) in result.iter_mut().enumerate() {
            let max = (0..self.ratio)
                .map(|i| self.gates[i * self.dim + j])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0;
            for i in 0..self.ratio {
                let w = (self.gates[i * self.dim + j] - max).exp();
                denom += w;
                *out += w * self.kv[i * self.dim + j];
            }
            *out /= denom;
        }
        self.kv.clear();
        self.gates.clear();
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Vectors {
        index: Vec<f32>,
        attention: Vec<f32>,
        pool: Vec<Vec<f32>>,
    }
    fn vectors() -> Vectors {
        serde_json::from_str(include_str!("../tests/fixtures/v41_sparse_vectors.json")).unwrap()
    }
    #[test]
    fn indexer_and_attention_match_torch_reference() {
        let q: Vec<f32> = (0..3)
            .flat_map(|i| (0..4).map(move |j| (i * 3 + j - 5) as f32 / 8.0))
            .collect();
        let kv: Vec<f32> = (0..8)
            .flat_map(|i| (0..4).map(move |j| (i * 2 - j - 4) as f32 / 8.0))
            .collect();
        let v = vectors();
        let scores = index_scores(&q, &kv, &[0.25, -0.5, 0.75], 4).unwrap();
        for (a, b) in scores.iter().zip(v.index) {
            assert!((a - b).abs() < 1e-6);
        }
        let out = sparse_attention(
            &q,
            &kv,
            &[-0.5, 0.25, 1.0],
            &[Some(0), Some(3), None, Some(5), Some(7)],
            4,
            0.5,
            true,
        )
        .unwrap();
        for (a, b) in out.iter().zip(v.attention) {
            assert!((a - b).abs() <= 0.004, "{a} != {b}");
        }
        assert_eq!(
            sparse_attention(&q, &kv, &[-0.5, 0.25, 1.0], &[None, None], 4, 0.5, false).unwrap(),
            vec![0.; 12]
        );
        assert!(sparse_attention(&q, &kv, &[-0.5, 0.25, 1.0], &[Some(8)], 4, 0.5, false).is_err());
    }
    #[test]
    fn causal_windows_and_two_stage_retrieval() {
        let prefill = prefill_window_indices(3, 5).unwrap();
        assert_eq!(prefill[0], vec![Some(0), None, None]);
        assert_eq!(prefill[3], vec![Some(1), Some(2), Some(3)]);
        assert_eq!(
            decode_window_indices(3, 1).unwrap(),
            vec![None, Some(0), Some(1)]
        );
        assert_eq!(
            decode_window_indices(3, 4).unwrap(),
            vec![Some(2), Some(0), Some(1)]
        );
        let scores = [9., 8., 7., 6., 5., 4., 3., 2., 1., 0.];
        let mask = candidate_blocks(&scores, 9, 3, 2).unwrap();
        assert_eq!(
            mask,
            [true, true, true, false, false, false, true, true, true, false]
        );
        assert_eq!(
            compressed_topk(&scores, 9, Some(&mask), 4, 10).unwrap(),
            [Some(10), Some(11), Some(12), Some(16)]
        );
        assert_eq!(
            compressed_topk(&scores, 2, None, 4, 10).unwrap(),
            [Some(10), Some(11), None, None]
        );
        assert!(candidate_blocks(&scores, 11, 3, 2).is_err());
    }

    #[test]
    fn ring_cache_matches_prefill_attention_across_wraparound() {
        let full: Vec<f32> = (0..18).map(|i| (i as f32 - 9.) / 8.).collect();
        let mut cache = WindowCache::new(3, 2).unwrap();
        let prefill = prefill_window_indices(3, 9).unwrap();
        for pos in 0..9 {
            let indices = cache.append(&full[pos * 2..pos * 2 + 2]).unwrap();
            let expected =
                sparse_attention(&[0.5, -0.25], &full, &[0.7], &prefill[pos], 2, 0.7, false)
                    .unwrap();
            let actual =
                sparse_attention(&[0.5, -0.25], cache.data(), &[0.7], &indices, 2, 0.7, false)
                    .unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6);
            }
        }
        assert_eq!(cache.position(), 9);
        assert!(cache.append(&[1.]).is_err());
        assert_eq!(cache.position(), 9);
        cache.clear();
        assert_eq!(cache.position(), 0);
        assert!(cache.data().iter().all(|v| *v == 0.));
    }
    #[test]
    fn pooled_groups_match_torch_and_carry_partial_state() {
        let mut pool = CompressionPool::new(2, 4).unwrap();
        let expected = vectors().pool;
        for i in 0..5 {
            let kv: Vec<f32> = (0..4).map(|j| (i * 3 + j - 5) as f32 / 8.).collect();
            let gate: Vec<f32> = (0..4).map(|j| (i - j * 2) as f32 / 4.).collect();
            let result = pool.push(&kv, &gate).unwrap();
            if i % 2 == 0 {
                assert!(result.is_none());
            } else {
                for (a, b) in result.unwrap().iter().zip(&expected[i as usize / 2]) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
        }
        let result = pool.push(&[0.; 4], &[0.; 4]).unwrap();
        assert!(result.is_some());
        assert_eq!(
            CompressionPool::new(1, 2)
                .unwrap()
                .push(&[1., 2.], &[])
                .unwrap(),
            Some(vec![1., 2.])
        );
    }
}
