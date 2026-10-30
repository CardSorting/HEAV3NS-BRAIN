//! Checked byte layouts for native V4.1 quantized matrix-vector execution.
use singularity_core::SingularityError;

#[derive(Debug, PartialEq, Eq)]
pub struct V41MatvecLayout {
    pub weight_bytes: usize,
    pub scale_bytes: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub activation_blocks: usize,
}

impl V41MatvecLayout {
    pub fn new(rows: usize, cols: usize, expert: bool) -> Result<Self, SingularityError> {
        let bad = || SingularityError::HardwareError {
            detail: "Invalid or overflowing V4.1 quantized matrix dimensions".into(),
        };
        if rows == 0
            || cols == 0
            || cols % 32 != 0
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
        {
            return Err(bad());
        }
        let blocks = cols / 32;
        Ok(Self {
            weight_bytes: rows
                .checked_mul(if expert { cols / 2 } else { cols })
                .ok_or_else(bad)?,
            scale_bytes: (if expert { rows } else { rows.div_ceil(32) })
                .checked_mul(blocks)
                .ok_or_else(bad)?,
            input_bytes: cols.checked_mul(2).ok_or_else(bad)?,
            output_bytes: rows.checked_mul(2).ok_or_else(bad)?,
            activation_blocks: blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_checkpoint_layouts_and_integer_boundaries() {
        let dense = V41MatvecLayout::new(512, 5120, false).unwrap();
        assert_eq!(
            (
                dense.weight_bytes,
                dense.scale_bytes,
                dense.input_bytes,
                dense.output_bytes
            ),
            (2621440, 2560, 10240, 1024)
        );
        let expert = V41MatvecLayout::new(2304, 5120, true).unwrap();
        assert_eq!((expert.weight_bytes, expert.scale_bytes), (5898240, 368640));
        assert!(V41MatvecLayout::new(0, 32, false).is_err());
        assert!(V41MatvecLayout::new(32, 31, true).is_err());
        assert!(V41MatvecLayout::new(usize::MAX, 32, true).is_err());
    }
}
