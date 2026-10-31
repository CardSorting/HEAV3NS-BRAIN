//! CPU reference for V4.1 image grid planning, token layout and RGB patch order.
//! The caller must decode and resize/pad the image before `patchify_rgb`.
use crate::{
    v41_compressor::Bf16Matrix,
    v41_config::{TextConfig, VisionConfig},
    SafetensorsMmapLoader,
};
use half::bf16;
use singularity_core::SingularityError;
fn error(reason: &str) -> SingularityError {
    SingularityError::ModelError {
        reason: reason.into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub pixel_height: usize,
    pub pixel_width: usize,
    pub vit_height: usize,
    pub vit_width: usize,
    pub llm_height: usize,
    pub llm_width: usize,
}
impl ImageGrid {
    pub fn token_count(&self) -> Result<usize, SingularityError> {
        self.llm_width
            .checked_add(1)
            .and_then(|w| self.llm_height.checked_mul(w))
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| error("Image token count overflow"))
    }
}
fn grid(height: usize, width: usize, p: usize, down: usize) -> Result<ImageGrid, SingularityError> {
    if p == 0 || down == 0 || height == 0 || width == 0 || height % p != 0 || width % p != 0 {
        return Err(error("Invalid image grid dimensions"));
    }
    let vh = height / p;
    let vw = width / p;
    Ok(ImageGrid {
        pixel_height: height,
        pixel_width: width,
        vit_height: vh,
        vit_width: vw,
        llm_height: vh.div_ceil(down),
        llm_width: vw.div_ceil(down),
    })
}
/// Match checkpoint `plan_image_grid` for positive image dimensions.
pub fn plan_image_grid(
    width: usize,
    height: usize,
    c: &VisionConfig,
) -> Result<ImageGrid, SingularityError> {
    let p = c.patch_size;
    let down = c.downsample_ratio;
    if width == 0
        || height == 0
        || p == 0
        || down == 0
        || c.max_image_tokens < 4
        || c.max_wh_ratio == Some(0)
    {
        return Err(error("Invalid image resize configuration"));
    }
    let mut w = width;
    let mut h = height;
    if let Some(ratio) = c.max_wh_ratio {
        w = w.min(
            h.checked_mul(ratio)
                .ok_or_else(|| error("Image size overflow"))?,
        );
    }
    let area = w
        .checked_mul(h)
        .ok_or_else(|| error("Image size overflow"))?;
    if area < c.min_pixels {
        let scale = (c.min_pixels as f64 / area as f64).sqrt();
        w = (w as f64 * scale) as usize;
        h = (h as f64 * scale) as usize;
    }
    let rounded = |n: usize| {
        n.checked_add(p - 1)
            .map(|v| v / p)
            .and_then(|v| v.checked_mul(p))
            .ok_or_else(|| error("Image size overflow"))
    };
    let mut plan = grid(rounded(h)?, rounded(w)?, p, down)?;
    if plan.token_count()? > c.max_image_tokens {
        let ratio = h as f64 / w as f64;
        let max_w = (((c.max_image_tokens - 2) as f64 / ratio + 0.25).sqrt()) - 0.5;
        let max_h = max_w * ratio;
        let cell = p
            .checked_mul(down)
            .ok_or_else(|| error("Image size overflow"))?;
        let (new_h, new_w) = if max_w < 1.0 {
            (
                ((c.max_image_tokens - 2) / 2)
                    .checked_mul(cell)
                    .ok_or_else(|| error("Image size overflow"))?,
                cell,
            )
        } else if max_h < 1.0 {
            (
                cell,
                (c.max_image_tokens - 3)
                    .checked_mul(cell)
                    .ok_or_else(|| error("Image size overflow"))?,
            )
        } else {
            let beta = (max_w.floor() * cell as f64 / w as f64)
                .min(max_h.floor() * cell as f64 / h as f64);
            (
                ((h as f64 * beta / p as f64).floor() as usize)
                    .checked_mul(p)
                    .ok_or_else(|| error("Image size overflow"))?,
                ((w as f64 * beta / p as f64).floor() as usize)
                    .checked_mul(p)
                    .ok_or_else(|| error("Image size overflow"))?,
            )
        };
        plan = grid(new_h, new_w, p, down)?;
        if plan.token_count()? > c.max_image_tokens {
            return Err(error("Image resize exceeded token budget"));
        }
    }
    Ok(plan)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Text,
    ImageStart,
    Image,
    ImageNewline,
    ImageEnd,
}
pub fn image_types(grid: ImageGrid) -> Result<Vec<TokenType>, SingularityError> {
    if grid.llm_height == 0 || grid.llm_width == 0 {
        return Err(error("Invalid image token grid"));
    }
    let mut types = Vec::with_capacity(grid.token_count()?);
    types.push(TokenType::ImageStart);
    for _ in 0..grid.llm_height {
        types.extend(std::iter::repeat_n(TokenType::Image, grid.llm_width));
        types.push(TokenType::ImageNewline);
    }
    types.push(TokenType::ImageEnd);
    Ok(types)
}
#[derive(Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub grid: ImageGrid,
}
pub struct ExpandedTokens {
    pub tokens: Vec<u32>,
    pub types: Vec<TokenType>,
    pub images: Vec<ImageSpan>,
}
pub fn expand_image_tokens(
    tokens: &[u32],
    image_token: u32,
    grids: &[ImageGrid],
) -> Result<ExpandedTokens, SingularityError> {
    if tokens.iter().filter(|&&t| t == image_token).count() != grids.len() {
        return Err(error("Image placeholder count does not match images"));
    }
    let mut out = ExpandedTokens {
        tokens: Vec::new(),
        types: Vec::new(),
        images: Vec::new(),
    };
    let mut grids = grids.iter();
    for &token in tokens {
        if token == image_token {
            let grid = *grids.next().unwrap();
            let types = image_types(grid)?;
            out.images.push(ImageSpan {
                start: out.tokens.len(),
                grid,
            });
            out.tokens
                .extend(std::iter::repeat_n(image_token, types.len()));
            out.types.extend(types);
        } else {
            out.tokens.push(token);
            out.types.push(TokenType::Text);
        }
    }
    Ok(out)
}
/// Convert already resized RGB8 pixels (row-major HWC) to patch-major CHW BF16 values.
pub fn patchify_rgb(
    rgb: &[u8],
    grid: ImageGrid,
    patch: usize,
) -> Result<Vec<f32>, SingularityError> {
    if patch == 0
        || grid.vit_height.checked_mul(patch) != Some(grid.pixel_height)
        || grid.vit_width.checked_mul(patch) != Some(grid.pixel_width)
    {
        return Err(error("Invalid patch dimensions"));
    }
    let count = grid
        .pixel_height
        .checked_mul(grid.pixel_width)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| error("Image size overflow"))?;
    if rgb.len() != count {
        return Err(error("Invalid RGB payload size"));
    }
    let mut out = Vec::with_capacity(count);
    for y in 0..grid.vit_height {
        for x in 0..grid.vit_width {
            for channel in 0..3 {
                for py in 0..patch {
                    for px in 0..patch {
                        let index = (((y * patch + py) * grid.pixel_width) + (x * patch + px)) * 3
                            + channel;
                        out.push(
                            bf16::from_f32(((rgb[index] as f32 / 255.0) - 0.5) / 0.5).to_f32(),
                        );
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Deterministically resize a row-major HWC RGB8 image to `(dst_width, dst_height)`
/// using pixel-center bilinear interpolation.
pub fn resize_rgb_bilinear(
    rgb: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> Result<Vec<u8>, SingularityError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(error("Invalid RGB resize dimensions"));
    }
    let expected = src_width
        .checked_mul(src_height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| error("Image size overflow"))?;
    if rgb.len() != expected {
        return Err(error("Invalid RGB payload size"));
    }
    if src_width == dst_width && src_height == dst_height {
        return Ok(rgb.to_vec());
    }
    let out_len = dst_width
        .checked_mul(dst_height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| error("Image size overflow"))?;
    let mut out = Vec::with_capacity(out_len);
    for dy in 0..dst_height {
        let sy = ((dy as f64 + 0.5) * (src_height as f64 / dst_height as f64) - 0.5)
            .clamp(0.0, (src_height - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_height - 1);
        let wy = sy - y0 as f64;
        for dx in 0..dst_width {
            let sx = ((dx as f64 + 0.5) * (src_width as f64 / dst_width as f64) - 0.5)
                .clamp(0.0, (src_width - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_width - 1);
            let wx = sx - x0 as f64;
            for c in 0..3 {
                let p00 = rgb[(y0 * src_width + x0) * 3 + c] as f64;
                let p01 = rgb[(y0 * src_width + x1) * 3 + c] as f64;
                let p10 = rgb[(y1 * src_width + x0) * 3 + c] as f64;
                let p11 = rgb[(y1 * src_width + x1) * 3 + c] as f64;
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                let value = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
                out.push(value);
            }
        }
    }
    Ok(out)
}

/// Plan the V4.1 image grid for an arbitrary `(width, height)` RGB8 image,
/// resize it to `(grid.pixel_width, grid.pixel_height)`, and convert to patch-major BF16 values.
pub fn prepare_image_patches(
    rgb: &[u8],
    width: usize,
    height: usize,
    config: &VisionConfig,
) -> Result<(ImageGrid, Vec<f32>), SingularityError> {
    let grid = plan_image_grid(width, height, config)?;
    let resized = resize_rgb_bilinear(rgb, width, height, grid.pixel_width, grid.pixel_height)?;
    let patches = patchify_rgb(&resized, grid, config.patch_size)?;
    Ok((grid, patches))
}

/// Match the aligner's padded `unfold`: output rows are grid-major, and each
/// row is ordered [channel, block_y, block_x].
pub fn aligner_unfold(
    features: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    ratio: usize,
) -> Result<Vec<f32>, SingularityError> {
    if height == 0
        || width == 0
        || channels == 0
        || ratio == 0
        || height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(channels))
            != Some(features.len())
    {
        return Err(error("Invalid aligner feature dimensions"));
    }
    let count = height
        .div_ceil(ratio)
        .checked_mul(width.div_ceil(ratio))
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul(ratio))
        .and_then(|n| n.checked_mul(ratio))
        .ok_or_else(|| error("Aligner size overflow"))?;
    let mut out = Vec::with_capacity(count);
    for gy in 0..height.div_ceil(ratio) {
        for gx in 0..width.div_ceil(ratio) {
            for c in 0..channels {
                for dy in 0..ratio {
                    for dx in 0..ratio {
                        let y = gy * ratio + dy;
                        let x = gx * ratio + dx;
                        out.push(if y < height && x < width {
                            features[(y * width + x) * channels + c]
                        } else {
                            0.0
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Assemble the image span embeddings from aligner rows and learned markers.
pub fn image_span_embeddings(
    grid: ImageGrid,
    hidden: usize,
    features: &[f32],
    start: &[f32],
    newline: &[f32],
    end: &[f32],
) -> Result<Vec<f32>, SingularityError> {
    let rows = grid
        .llm_height
        .checked_mul(grid.llm_width)
        .ok_or_else(|| error("Image embedding size overflow"))?;
    if hidden == 0
        || rows.checked_mul(hidden) != Some(features.len())
        || [start, newline, end].iter().any(|v| v.len() != hidden)
    {
        return Err(error("Invalid image embedding dimensions"));
    }
    let size = grid
        .token_count()?
        .checked_mul(hidden)
        .ok_or_else(|| error("Image embedding size overflow"))?;
    let mut output = Vec::with_capacity(size);
    output.extend_from_slice(start);
    for row in 0..grid.llm_height {
        let begin = row * grid.llm_width * hidden;
        output.extend_from_slice(&features[begin..begin + grid.llm_width * hidden]);
        output.extend_from_slice(newline);
    }
    output.extend_from_slice(end);
    Ok(output)
}

/// Checkpoint aligner: padded vision features -> BF16 linear -> GELU -> BF16 linear.
pub struct Aligner<'a> {
    first: Bf16Matrix<'a>,
    second: Bf16Matrix<'a>,
    bias1: Vec<f32>,
    bias2: Vec<f32>,
    start: Vec<f32>,
    newline: Vec<f32>,
    end: Vec<f32>,
    vision_dim: usize,
    hidden: usize,
    ratio: usize,
}
fn gelu(x: f32) -> f32 {
    // Absolute error of the erf approximation is below 1.5e-7 for finite x.
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly =
        (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    let erf = (1.0 - poly * (-a * a).exp()).copysign(z);
    0.5 * x * (1.0 + erf)
}
impl<'a> Aligner<'a> {
    pub fn load(
        loader: &'a SafetensorsMmapLoader,
        vision: &VisionConfig,
        text: &TextConfig,
    ) -> Result<Self, SingularityError> {
        let ratio = vision.downsample_ratio;
        let hidden = text.hidden_size;
        let dim = vision.hidden_size;
        let input = dim
            .checked_mul(ratio)
            .and_then(|x| x.checked_mul(ratio))
            .ok_or_else(|| error("Aligner dimensions overflow"))?;
        if ratio == 0 || hidden == 0 || input == 0 {
            return Err(error("Invalid aligner dimensions"));
        }
        let vector = |name: &str| -> Result<Vec<f32>, SingularityError> {
            let info = loader
                .get_tensor_info(name)
                .ok_or_else(|| error(&format!("Missing tensor {name}")))?;
            if info.dtype != "BF16" || info.shape != [hidden] {
                return Err(error(&format!("Invalid tensor {name}")));
            }
            loader.get_tensor_f32(name)
        };
        Ok(Self {
            first: Bf16Matrix::load(loader, "aligner.w1.weight", hidden, input)?,
            second: Bf16Matrix::load(loader, "aligner.w2.weight", hidden, hidden)?,
            bias1: vector("aligner.w1.bias")?,
            bias2: vector("aligner.w2.bias")?,
            start: vector("image_start")?,
            newline: vector("image_newline")?,
            end: vector("image_end")?,
            vision_dim: dim,
            hidden,
            ratio,
        })
    }
    /// Return the learned image-token rows in LLM grid order.
    pub fn forward(
        &self,
        features: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, SingularityError> {
        let unfolded = aligner_unfold(features, height, width, self.vision_dim, self.ratio)?;
        let input_dim = self.vision_dim * self.ratio * self.ratio;
        let output_size = height
            .div_ceil(self.ratio)
            .checked_mul(width.div_ceil(self.ratio))
            .and_then(|n| n.checked_mul(self.hidden))
            .ok_or_else(|| error("Aligner output size overflow"))?;
        let mut result = Vec::with_capacity(output_size);
        for row in unfolded.chunks_exact(input_dim) {
            let input: Vec<f32> = row.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
            let first = self.first.project(&input, false)?;
            let activated: Vec<f32> = first
                .iter()
                .zip(&self.bias1)
                .map(|(x, b)| bf16::from_f32(gelu(bf16::from_f32(x + b).to_f32())).to_f32())
                .collect();
            let second = self.second.project(&activated, false)?;
            result.extend(
                second
                    .iter()
                    .zip(&self.bias2)
                    .map(|(x, b)| bf16::from_f32(x + b).to_f32()),
            );
        }
        Ok(result)
    }
    pub fn span(&self, grid: ImageGrid, features: &[f32]) -> Result<Vec<f32>, SingularityError> {
        if grid.vit_height.div_ceil(self.ratio) != grid.llm_height
            || grid.vit_width.div_ceil(self.ratio) != grid.llm_width
        {
            return Err(error("Image grid does not match aligner"));
        }
        let rows = self.forward(features, grid.vit_height, grid.vit_width)?;
        image_span_embeddings(
            grid,
            self.hidden,
            &rows,
            &self.start,
            &self.newline,
            &self.end,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct GridVector {
        width: usize,
        height: usize,
        result: [usize; 4],
    }
    #[derive(Deserialize)]
    struct Vectors {
        grids: Vec<GridVector>,
        rgb: Vec<u8>,
        patches: Vec<f32>,
        types: Vec<i64>,
        vit: [usize; 2],
        unfold: Vec<f32>,
        aligned: Vec<f32>,
    }
    #[test]
    #[ignore = "requires SINGULARITY_V41_ALIGNER pinned shard-1 range"]
    fn real_checkpoint_aligner_matches_pytorch() {
        use sha2::{Digest, Sha256};
        use std::io::Write;
        #[derive(Deserialize)]
        struct Expected {
            sha256: String,
            output: Vec<f32>,
        }
        let expected: Expected = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_aligner_real_vectors.json"
        ))
        .unwrap();
        let weights = std::fs::read(std::env::var("SINGULARITY_V41_ALIGNER").unwrap()).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&weights)), expected.sha256);
        let header = serde_json::json!({
            "aligner.w1.bias":{"dtype":"BF16","shape":[5120],"data_offsets":[0,10240]},
            "aligner.w1.weight":{"dtype":"BF16","shape":[5120,9216],"data_offsets":[10240,94382080]},
            "aligner.w2.bias":{"dtype":"BF16","shape":[5120],"data_offsets":[94382080,94392320]},
            "aligner.w2.weight":{"dtype":"BF16","shape":[5120,5120],"data_offsets":[94392320,146821120]},
            "image_start":{"dtype":"BF16","shape":[5120],"data_offsets":[146821120,146831360]},
            "image_newline":{"dtype":"BF16","shape":[5120],"data_offsets":[146831360,146841600]},
            "image_end":{"dtype":"BF16","shape":[5120],"data_offsets":[146841600,146851840]}
        });
        let header = serde_json::to_vec(&header).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&weights).unwrap();
        file.write_all(&vec![0; 30720]).unwrap();
        drop(file);
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        let aligner = Aligner::load(&loader, &cfg.vision_config, &cfg.text_config).unwrap();
        let features: Vec<f32> = (0..9216)
            .map(|i| (((i * 17) % 97) as i32 - 48) as f32 / 64.)
            .collect();
        let actual = aligner.forward(&features, 3, 3).unwrap();
        assert_eq!(actual.len(), expected.output.len());
        let error: f64 = actual
            .iter()
            .zip(&expected.output)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum();
        let norm: f64 = expected.output.iter().map(|x| (*x as f64).powi(2)).sum();
        let rms = (error / norm).sqrt();
        eprintln!("aligner relative RMS: {rms:.8}");
        assert!(rms < 0.01);
    }
    #[test]
    fn grids_patches_and_image_tokens_match_checkpoint() {
        let vectors: Vectors =
            serde_json::from_str(include_str!("../tests/fixtures/v41_vision_vectors.json"))
                .unwrap();
        let cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        for v in vectors.grids {
            let g = plan_image_grid(v.width, v.height, &cfg.vision_config).unwrap();
            assert_eq!(
                [g.llm_height, g.llm_width, g.pixel_height, g.pixel_width],
                v.result,
                "{}x{}",
                v.width,
                v.height
            );
            assert!(g.token_count().unwrap() <= cfg.vision_config.max_image_tokens);
        }
        let g = grid(4, 4, 2, 2).unwrap();
        assert_eq!([g.vit_height, g.vit_width], vectors.vit);
        assert_eq!(patchify_rgb(&vectors.rgb, g, 2).unwrap(), vectors.patches);
        let types = image_types(g).unwrap();
        let as_int: Vec<i64> = types
            .iter()
            .map(|t| match t {
                TokenType::ImageStart => 0,
                TokenType::Image => 1,
                TokenType::ImageNewline => 2,
                TokenType::ImageEnd => 3,
                TokenType::Text => -1,
            })
            .collect();
        assert_eq!(as_int, vectors.types);
        let expanded = expand_image_tokens(&[4, 99, 5, 99], 99, &[g, g]).unwrap();
        assert_eq!(expanded.images[0].start, 1);
        assert_eq!(expanded.images[1].start, 2 + types.len());
        assert_eq!(expanded.tokens.len(), 2 + 2 * types.len());
        assert!(expand_image_tokens(&[99], 99, &[]).is_err());
        assert!(patchify_rgb(&vectors.rgb[..3], g, 2).is_err());
        assert!(plan_image_grid(0, 1, &cfg.vision_config).is_err());
        let features: Vec<f32> = (0..24).map(|i| i as f32).collect();
        assert_eq!(
            aligner_unfold(&features, 3, 4, 2, 3).unwrap(),
            vectors.unfold
        );
        let two = grid(6, 9, 1, 3).unwrap();
        let rows: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let embedded =
            image_span_embeddings(two, 2, &rows, &[100., 101.], &[200., 201.], &[300., 301.])
                .unwrap();
        assert_eq!(
            embedded,
            vec![
                100., 101., 0., 1., 2., 3., 4., 5., 200., 201., 6., 7., 8., 9., 10., 11., 200.,
                201., 300., 301.
            ]
        );
        assert!(image_span_embeddings(two, 2, &rows[..10], &[0.; 2], &[0.; 2], &[0.; 2]).is_err());
    }

    #[test]
    fn checkpoint_aligner_matches_pytorch() {
        let vectors: Vectors =
            serde_json::from_str(include_str!("../tests/fixtures/v41_vision_vectors.json"))
                .unwrap();
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        cfg.vision_config.hidden_size = 2;
        cfg.vision_config.downsample_ratio = 3;
        cfg.text_config.hidden_size = 4;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let w1: Vec<f32> = (0..72)
            .map(|i| ((i * 7 % 17) as i32 - 8) as f32 / 16.)
            .collect();
        let w2: Vec<f32> = (0..16)
            .map(|i| ((i * 5 % 11) as i32 - 5) as f32 / 8.)
            .collect();
        for (name, shape, values) in [
            ("aligner.w1.weight", vec![4, 18], w1),
            ("aligner.w1.bias", vec![4], vec![0.125, -0.25, 0.375, -0.5]),
            ("aligner.w2.weight", vec![4, 4], w2),
            ("aligner.w2.bias", vec![4], vec![0.25, 0.125, -0.125, -0.25]),
            ("image_start", vec![4], vec![1.; 4]),
            ("image_newline", vec![4], vec![2.; 4]),
            ("image_end", vec![4], vec![3.; 4]),
        ] {
            let start = payload.len();
            for value in values {
                payload.extend(bf16::from_f32(value).to_le_bytes());
            }
            header.insert(name.into(),serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(h);
        file.extend(payload);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, file).unwrap();
        let loader = SafetensorsMmapLoader::open(path).unwrap();
        let aligner = Aligner::load(&loader, &cfg.vision_config, &cfg.text_config).unwrap();
        let features: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let actual = aligner.forward(&features, 3, 4).unwrap();
        assert_eq!(actual.len(), vectors.aligned.len());
        for (a, b) in actual.iter().zip(&vectors.aligned) {
            assert!((a - b).abs() <= 0.01 * b.abs().max(0.01), "{a} != {b}");
        }
        let grid = grid(3, 4, 1, 3).unwrap();
        let span = aligner.span(grid, &features).unwrap();
        assert_eq!(&span[..4], &[1.; 4]);
        assert_eq!(&span[span.len() - 4..], &[3.; 4]);
    }

    #[test]
    fn resize_and_prepare_image_patches_work() {
        let mut cfg: crate::v41_config::V41Config = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek-v41-flash-1017-config.json"
        ))
        .unwrap();
        cfg.vision_config.patch_size = 2;
        cfg.vision_config.downsample_ratio = 2;
        cfg.vision_config.min_pixels = 16;
        cfg.vision_config.max_image_tokens = 64;

        let src = vec![0u8, 128, 255, 64, 128, 192, 32, 64, 96, 255, 0, 128];
        assert_eq!(resize_rgb_bilinear(&src, 2, 2, 2, 2).unwrap(), src);
        assert!(resize_rgb_bilinear(&src, 0, 2, 2, 2).is_err());
        assert!(resize_rgb_bilinear(&src[..3], 2, 2, 2, 2).is_err());

        let (planned, patches) = prepare_image_patches(&src, 2, 2, &cfg.vision_config).unwrap();
        assert_eq!((planned.pixel_width, planned.pixel_height), (4, 4));
        assert_eq!(patches.len(), 4 * 4 * 3);
        assert!(patches
            .iter()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    }
}
