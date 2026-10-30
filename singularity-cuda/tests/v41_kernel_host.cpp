// Host-only syntax and smoke check for CUDA scalar kernel math.
// This does not compile PTX or validate execution on an accelerator.
#include <cassert>
#include <cstdint>
#include <cstring>
#define __device__
#define __global__
#define __forceinline__ inline
struct Dim { unsigned int x; };
Dim blockIdx, blockDim, threadIdx;
inline float __uint_as_float(unsigned int n) { float x; std::memcpy(&x, &n, sizeof(x)); return x; }
inline unsigned int __float_as_uint(float x) { unsigned int n; std::memcpy(&n, &x, sizeof(n)); return n; }
inline unsigned int atomicOr(unsigned int* p, unsigned int v) { unsigned int old=*p; *p |= v; return old; }
#include "../src/kernels/v41_quant.cu"

int main() {
    for (unsigned int i=0;i<=126;++i) {
        assert(v41_encode_e4(v41_e4((unsigned char)i))==i);
        assert(v41_encode_e4(v41_e4((unsigned char)(i|0x80)))==(i|0x80));
        if (i<126) {
            float midpoint=(v41_e4((unsigned char)i)+v41_e4((unsigned char)(i+1)))*0.5f;
            assert(v41_encode_e4(midpoint)==(i%2==0 ? i : i+1));
        }
    }
    unsigned short x[32]; for (auto& v : x) v=0x3f80;
    unsigned char q[32]={}, qs[1]={}, weights[64], scales[2]={127,127};
    unsigned int status=0; unsigned short out[2]={};
    blockIdx.x=0; blockDim.x=1; threadIdx.x=0;
    v41_quantize(x,q,qs,1,&status);
    assert(status==0 && qs[0]==119);
    for (auto v:q) assert(v41_e4(v)==256.0f);
    std::memset(weights,0x38,sizeof(weights));
    v41_matvec(weights,scales,q,qs,out,2,32,0);
    assert(out[0]==0x4200);
    threadIdx.x=1; v41_matvec(weights,scales,q,qs,out,2,32,0); assert(out[1]==0x4200);
    std::memset(weights,0x21,16); threadIdx.x=0;
    v41_matvec(weights,scales,q,qs,out,1,32,1); assert(out[0]==0x41c0);
    x[0]=0x7f80; v41_quantize(x,q,qs,1,&status); assert(status==1);
}
