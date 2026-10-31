//! CPU reference for V4.1 Engram table layout and n-gram addressing.
//!
//! Based on the MIT-licensed checkpoint's `inference/engram.py` at revision
//! 1bae0e1088993b9b82de08f59de5be5a34e42170. This module handles vocabulary
//! compression and hashing; lookup and gating are in `v41_engram`.
use singularity_core::SingularityError;
use std::collections::HashSet;

use crate::model_error as error;

#[derive(Debug, Clone)]
pub struct EngramLayerLayout {
    pub layer_id: usize,
    /// In n-gram order (2..=max), then head order.
    pub moduli: Vec<u64>,
    pub offsets: Vec<u64>,
    pub rows: u64,
}

#[derive(Debug, Clone)]
pub struct EngramLayout {
    pub max_ngram: usize,
    pub heads: usize,
    pub layers: Vec<EngramLayerLayout>,
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d <= n / d {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

impl EngramLayout {
    /// Build from the checkpoint and verify the prime-derived table sizes.
    pub fn from_config(text: &crate::v41_config::TextConfig) -> Result<Self, SingularityError> {
        let layout = Self::new(
            &text.engram_layer_ids,
            text.engram_max_ngram_size,
            text.engram_n_heads,
            text.engram_vocab_size,
        )?;
        if text.engram_head_dim == 0
            || text.engram_compressed_vocab_size == 0
            || layout
                .layers
                .iter()
                .map(|layer| layer.rows)
                .ne(text.engram_num_embeddings.iter().copied())
        {
            return Err(error(
                "Engram table dimensions do not match checkpoint configuration",
            ));
        }
        Ok(layout)
    }

    pub fn new(
        layer_ids: &[usize],
        max_ngram: usize,
        heads: usize,
        vocab_size: u64,
    ) -> Result<Self, SingularityError> {
        if layer_ids.is_empty() || max_ngram < 2 || heads == 0 || vocab_size < 2 {
            return Err(error("Invalid Engram layout parameters"));
        }
        let columns = (max_ngram - 1)
            .checked_mul(heads)
            .ok_or_else(|| error("Engram column count overflow"))?;
        let mut seen = HashSet::new();
        let mut layers = Vec::with_capacity(layer_ids.len());
        for &layer_id in layer_ids {
            let mut moduli = Vec::with_capacity(columns);
            let mut offsets = Vec::with_capacity(columns);
            let mut rows = 0u64;
            for _ in 1..max_ngram {
                let mut candidate = vocab_size - 1;
                for _ in 0..heads {
                    loop {
                        candidate = candidate
                            .checked_add(1)
                            .ok_or_else(|| error("Engram prime overflow"))?;
                        if is_prime(candidate) && seen.insert(candidate) {
                            break;
                        }
                    }
                    offsets.push(rows);
                    rows = rows
                        .checked_add(candidate)
                        .ok_or_else(|| error("Engram row count overflow"))?;
                    moduli.push(candidate);
                }
            }
            layers.push(EngramLayerLayout {
                layer_id,
                moduli,
                offsets,
                rows,
            });
        }
        Ok(Self {
            max_ngram,
            heads,
            layers,
        })
    }

    /// Hash one position, using compressed IDs in sequence order. `None` denotes
    /// a dead/image token and blocks all older lookback. Multipliers must be the
    /// checkpoint's per-layer NumPy PCG64-derived values, in layer order.
    pub fn hash_position(
        &self,
        sequence: &[Option<u32>],
        position: usize,
        pad: u32,
        multipliers: &[Vec<u64>],
    ) -> Result<Vec<Vec<u64>>, SingularityError> {
        if position >= sequence.len()
            || multipliers.len() != self.layers.len()
            || multipliers.iter().any(|row| row.len() != self.max_ngram)
        {
            return Err(error("Invalid Engram hash input dimensions"));
        }
        let mut blocked = false;
        let tokens: Vec<u32> = (0..self.max_ngram)
            .map(|shift| {
                let value = position.checked_sub(shift).and_then(|i| sequence[i]);
                if value.is_none() {
                    blocked = true;
                }
                if blocked {
                    pad
                } else {
                    value.unwrap()
                }
            })
            .collect();
        self.layers
            .iter()
            .zip(multipliers)
            .map(|(layer, factors)| {
                let product = |token: u32, factor: u64| -> Result<u64, SingularityError> {
                    let value = (token as u64)
                        .checked_mul(factor)
                        .ok_or_else(|| error("Engram hash product overflow"))?;
                    if value > i64::MAX as u64 {
                        return Err(error("Engram hash product exceeds signed reference range"));
                    }
                    Ok(value)
                };
                let mut rolling = product(tokens[0], factors[0])?;
                let mut hashes = Vec::with_capacity(layer.moduli.len());
                for n in 1..self.max_ngram {
                    rolling ^= product(tokens[n], factors[n])?;
                    for h in 0..self.heads {
                        let i = (n - 1) * self.heads + h;
                        hashes.push(layer.offsets[i] + rolling % layer.moduli[i]);
                    }
                }
                Ok(hashes)
            })
            .collect()
    }
}

/// NumPy PCG64 seed 10007 * layer, compressed vocabulary size 99092,
/// max n-gram size 4, for the pinned Flash-1017 checkpoint only.
pub const FLASH_1017_MULTIPLIERS: [[u64; 4]; 2] = [
    [
        76632096046245,
        4839876093313,
        35959672319349,
        73987337458391,
    ],
    [
        67716810739261,
        51510806800915,
        30921347202721,
        82619226485591,
    ],
];

/// Per-sequence incremental hash state for the pinned Flash-1017 tokenizer.
/// Stores only the n-gram lookback, and resets independently between requests.
#[derive(Clone)]
pub struct Flash1017HashState {
    layout: EngramLayout,
    token_map: std::sync::Arc<[u32]>,
    pad: u32,
    history: std::collections::VecDeque<Option<u32>>,
}

impl Flash1017HashState {
    pub fn new(
        config: &crate::v41_config::TextConfig,
        token_map: std::sync::Arc<[u32]>,
        pad_token_id: u32,
    ) -> Result<Self, SingularityError> {
        if config.engram_layer_ids != [1, 14]
            || config.engram_max_ngram_size != 4
            || config.engram_compressed_vocab_size != 99092
            || token_map.is_empty()
            || token_map
                .iter()
                .any(|&id| id as usize >= config.engram_compressed_vocab_size)
        {
            return Err(error("Incompatible Flash-1017 Engram hash configuration"));
        }
        let pad = *token_map
            .get(pad_token_id as usize)
            .ok_or_else(|| error("Engram pad token outside tokenizer"))?;
        Ok(Self {
            layout: EngramLayout::from_config(config)?,
            token_map,
            pad,
            history: std::collections::VecDeque::new(),
        })
    }

    /// `None` marks an image span and prevents hashing across that boundary.
    /// Returns hash IDs in configured layer order for the appended token.
    pub fn push(&mut self, token: Option<u32>) -> Result<Vec<Vec<u64>>, SingularityError> {
        let compressed = match token {
            Some(id) => Some(
                *self
                    .token_map
                    .get(id as usize)
                    .ok_or_else(|| error("Engram token outside tokenizer"))?,
            ),
            None => None,
        };
        let mut next = self.history.clone();
        next.push_back(compressed);
        if next.len() > self.layout.max_ngram {
            next.pop_front();
        }
        let tokens: Vec<_> = next.iter().copied().collect();
        let multipliers: Vec<_> = FLASH_1017_MULTIPLIERS
            .iter()
            .map(|row| row.to_vec())
            .collect();
        let result =
            self.layout
                .hash_position(&tokens, tokens.len() - 1, self.pad, &multipliers)?;
        self.history = next;
        Ok(result)
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Build the Engram-specific compressed vocabulary from the checkpoint tokenizer.
/// IDs are assigned in original token-ID order; malformed or incompatible
/// tokenizer assets fail instead of silently producing a different hash space.
pub fn compressed_vocabulary(
    tokenizer: &tokenizers::Tokenizer,
    expected_size: usize,
) -> Result<Vec<u32>, SingularityError> {
    use std::collections::HashMap;
    use tokenizers::normalizers::replace::ReplacePattern;
    use tokenizers::normalizers::{Lowercase, Replace, Strip, StripAccents, NFD, NFKC};
    use tokenizers::{NormalizedString, Normalizer};
    let convert_error = |e: Box<dyn std::error::Error + Send + Sync>| {
        error(&format!("Engram tokenizer normalization: {e}"))
    };
    let whitespace =
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ").map_err(convert_error)?;
    let single_space =
        Replace::new(ReplacePattern::Regex("^ $".into()), "\u{e000}").map_err(convert_error)?;
    let restore = Replace::new("\u{e000}", " ").map_err(convert_error)?;
    let strip = Strip::new(true, true);
    let normalizers: [&dyn Normalizer; 8] = [
        &NFKC,
        &NFD,
        &StripAccents,
        &Lowercase,
        &whitespace,
        &single_space,
        &strip,
        &restore,
    ];
    let count = tokenizer.get_vocab_size(true);
    let mut keys: HashMap<String, u32> = HashMap::new();
    let mut mapping = Vec::with_capacity(count);
    for id in 0..count {
        let id = u32::try_from(id).map_err(|_| error("Tokenizer vocabulary exceeds u32"))?;
        let text = tokenizer.decode(&[id], false).map_err(convert_error)?;
        let key = if text.contains('\u{fffd}') {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| error("Missing tokenizer ID"))?
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            for normalizer in normalizers {
                normalizer
                    .normalize(&mut normalized)
                    .map_err(convert_error)?;
            }
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next =
            u32::try_from(keys.len()).map_err(|_| error("Compressed vocabulary exceeds u32"))?;
        mapping.push(*keys.entry(key).or_insert(next));
    }
    if keys.len() != expected_size {
        return Err(error(&format!(
            "Compressed vocabulary size mismatch: expected {expected_size}, got {}",
            keys.len()
        )));
    }
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_hashes_match_full_history_and_reset() {
        let config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let map: std::sync::Arc<[u32]> = (0..99092u32).collect::<Vec<_>>().into();
        let mut state = Flash1017HashState::new(&config.text_config, map, 2).unwrap();
        let layout = EngramLayout::from_config(&config.text_config).unwrap();
        let multipliers: Vec<_> = FLASH_1017_MULTIPLIERS.iter().map(|x| x.to_vec()).collect();
        let tokens = [
            Some(7),
            Some(11),
            None,
            Some(19),
            Some(23),
            Some(31),
            Some(41),
        ];
        for (i, &token) in tokens.iter().enumerate() {
            assert_eq!(
                state.push(token).unwrap(),
                layout.hash_position(&tokens, i, 2, &multipliers).unwrap()
            );
        }
        assert!(state.push(Some(99999)).is_err());
        let expected = layout
            .hash_position(&[Some(23), Some(31), Some(41), Some(5)], 3, 2, &multipliers)
            .unwrap();
        assert_eq!(state.push(Some(5)).unwrap(), expected);
        state.clear();
        assert_eq!(
            state.push(Some(7)).unwrap(),
            layout
                .hash_position(&[Some(7)], 0, 2, &multipliers)
                .unwrap()
        );
    }
    #[test]
    fn flash_layout_and_masked_history_match_reference() {
        let config: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let layout = EngramLayout::from_config(&config.text_config).unwrap();
        assert_eq!(
            layout.layers.iter().map(|l| l.rows).collect::<Vec<_>>(),
            [384006168, 384016682]
        );
        assert_eq!(
            (layout.layers[0].moduli[0], layout.layers[1].moduli[23]),
            (16000057, 16000889)
        );
        let factors = FLASH_1017_MULTIPLIERS
            .iter()
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        let tokens = [Some(7), Some(11), None, Some(19), Some(23)];
        let hashes = layout.hash_position(&tokens, 4, 2, &factors).unwrap();
        assert_eq!(
            hashes[0],
            [
                4154010, 24833484, 35441074, 56303406, 71603239, 82038269, 103256237, 112496759,
                130003622, 144901934, 164213493, 182995918, 202882774, 212829214, 227752636,
                255363185, 256560522, 274001417, 301847113, 315770783, 330387000, 344311580,
                363178097, 373654792
            ]
        );
        assert_eq!(
            hashes[1],
            [
                11621315, 21417254, 42774701, 57014231, 64134364, 86524925, 102649781, 126720270,
                140282775, 157352483, 170185631, 188326022, 195816841, 221452925, 237522650,
                242882215, 262672742, 280110661, 297232370, 313714413, 331484054, 350393947,
                366556970, 383369419
            ]
        );
        assert_eq!(
            layout.hash_position(&tokens, 3, 2, &factors).unwrap(),
            layout.hash_position(&[Some(19)], 0, 2, &factors).unwrap()
        );
        assert!(layout.hash_position(&tokens, 9, 2, &factors).is_err());
    }

    #[test]
    #[ignore = "requires SINGULARITY_V41_TOKENIZER pointing to the pinned checkpoint tokenizer.json"]
    fn checkpoint_compressed_vocabulary_matches_reference() {
        use sha2::{Digest, Sha256};
        let path = std::env::var("SINGULARITY_V41_TOKENIZER").expect("tokenizer path");
        let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
        let mapping = compressed_vocabulary(&tokenizer, 99092).unwrap();
        assert_eq!(mapping.len(), 129280);
        let mut hash = Sha256::new();
        for id in mapping {
            hash.update(id.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
        );
    }
}
