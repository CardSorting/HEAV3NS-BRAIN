// Independent correctness-first V4.1 CUDA reference kernels.
// Quantization: BF16 input -> E4M3FN + per-32 UE8M0 scale.
// Matvec: E4M3FN activations times either E4M3FN dense or packed E2M1FN
// experts, with the checkpoint's block scale layouts. No tensor cores.
// Compiled by NVRTC at runtime; this is not an optimized B200 kernel.

__device__ __forceinline__ float v41_e4(unsigned char x) {
    unsigned int exponent = (x >> 3) & 15u;
    unsigned int mantissa = x & 7u;
    float value;
    if (exponent == 15u && mantissa == 7u) value = __uint_as_float(0x7fc00000u);
    else if (exponent == 0u) value = (float)mantissa * 0.001953125f;
    else value = (1.0f + (float)mantissa * 0.125f) * __uint_as_float((exponent + 120u) << 23);
    return (x & 128u) ? -value : value;
}

__device__ __forceinline__ float v41_scale(unsigned char x) {
    if (x == 0) return __uint_as_float(0x00400000u);
    if (x == 255) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(((unsigned int)x) << 23);
}

__device__ __forceinline__ float v41_fp4(unsigned char x) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value = table[x & 7u];
    return (x & 8u) ? -value : value;
}

__device__ __forceinline__ unsigned char v41_encode_e4(float x) {
    unsigned int sign = __float_as_uint(x) >> 31;
    float v = __uint_as_float(__float_as_uint(x) & 0x7fffffffu);
    if (v > 448.0f) v = 448.0f;
    unsigned int low = 0, high = 126;
    while (low < high) {
        unsigned int mid = low + (high - low) / 2;
        if (v41_e4((unsigned char)mid) < v) low = mid + 1;
        else high = mid;
    }
    unsigned int selected = low;
    if (low != 0) {
        float dl = v - v41_e4((unsigned char)(low - 1));
        float du = v41_e4((unsigned char)low) - v;
        if (dl < du || (dl == du && ((low - 1) & 1u) == 0)) selected = low - 1;
    }
    return (unsigned char)(selected | (sign << 7));
}

extern "C" __global__ void v41_quantize(const unsigned short* input, unsigned char* quantized,
                                          unsigned char* scales, unsigned int blocks, unsigned int* status) {
    unsigned int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    float maxv = 0.0001f;
    for (unsigned int j = 0; j < 32; ++j) {
        unsigned int bits = ((unsigned int)input[block * 32u + j]) << 16;
        if ((bits & 0x7f800000u) == 0x7f800000u) { atomicOr(status, 1u); bits = 0; }
        float value = __uint_as_float(bits & 0x7fffffffu);
        if (value > maxv) maxv = value;
    }
    unsigned int bits = __float_as_uint(maxv * (1.0f / 448.0f));
    unsigned int exponent = ((bits >> 23) & 255u) + ((bits & 0x007fffffu) != 0u);
    if (exponent == 0 || exponent >= 255) { atomicOr(status, 2u); exponent = 127; }
    float scale = v41_scale((unsigned char)exponent);
    scales[block] = (unsigned char)exponent;
    for (unsigned int j = 0; j < 32; ++j) {
        unsigned int bits_x = ((unsigned int)input[block * 32u + j]) << 16;
        float x = __uint_as_float(bits_x);
        if ((bits_x & 0x7f800000u) == 0x7f800000u) x = 0.0f;
        quantized[block * 32u + j] = v41_encode_e4(x / scale);
    }
}

extern "C" __global__ void v41_matvec(const unsigned char* weights, const unsigned char* weight_scales,
                                        const unsigned char* x, const unsigned char* x_scales,
                                        unsigned short* output, unsigned int rows, unsigned int cols,
                                        unsigned int expert) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int blocks = cols / 32u;
    float result = 0.0f;
    for (unsigned int block = 0; block < blocks; ++block) {
        float accum = 0.0f;
        for (unsigned int j = 0; j < 32; ++j) {
            unsigned int col = block * 32u + j;
            unsigned long long idx = (unsigned long long)row * (expert ? cols / 2u : cols) + (expert ? col / 2u : col);
            unsigned char packed = weights[idx];
            float w = expert ? v41_fp4((col & 1u) ? packed >> 4 : packed & 15u) : v41_e4(packed);
            accum += v41_e4(x[col]) * w;
        }
        unsigned long long scale_idx = (unsigned long long)(expert ? row : row / 32u) * blocks + block;
        result += accum * v41_scale(x_scales[block]) * v41_scale(weight_scales[scale_idx]);
    }
    unsigned int bits = __float_as_uint(result);
    // Round to nearest even. Preserve NaNs as NaNs instead of overflowing their mantissa.
    if ((bits & 0x7fffffffu) > 0x7f800000u) output[row] = (unsigned short)((bits >> 16) | 0x0040u);
    else output[row] = (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
