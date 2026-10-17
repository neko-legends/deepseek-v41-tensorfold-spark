// Vision towers' glue (cuda/vit.zig; LFM2-VL's SigLIP2, RADIO): LayerNorm, Linear epilogues, activations, unshuffle.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

namespace {

constexpr int NT = 256;    // threads a row block
constexpr int MAXE = 8;    // elements a thread holds: LayerNorm rows up to NT * MAXE = 2048 wide

// A block's values summed: warp butterflies, then the warps in order; every thread gets the sum.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = red[0];
    for (int w = 1; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

// torch's GELU kernels in fp32: the tanh approximation, and the exact form through erf.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kappa = 0.044715f;
    const float cube = __fmul_rn(__fmul_rn(x, x), x);
    const float inner = __fmul_rn(beta, __fadd_rn(x, __fmul_rn(kappa, cube)));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float alpha = M_SQRT1_2;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, alpha))));
}

}  // namespace

// Block a row: y = bf16((x - mean) * rstd * w + b), mean and variance in two fixed-order passes.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                       const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                       float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        if (threadIdx.x + k * NT >= D) continue;
        x[k] = __fsub_rn(x[k], mean);
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(x[k], rstd), bf(W[i])), bf(B[i])));
    }
}

// Block (row, column block): o = bf16(acc + bias), for a Linear whose cuBLASLt algorithm cannot fuse its bias.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_bias(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 __nv_bfloat16* __restrict__ O, int N) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(__fadd_rn(ACC[at], bf(BIAS[c])));
}

// A Linear's bf16 output: bf16(act(x)) (1 tanh GELU, 2 erf GELU, 3 relu squared), then bf16(r + x) when R.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_pointwise(const __nv_bfloat16* X, const __nv_bfloat16* R, __nv_bfloat16* O,
                                                                      int64_t count, int act) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at >= count) return;
    float v = bf(X[at]);
    if (act == 1) v = rbf(gelu_tanh(v));
    if (act == 2) v = rbf(gelu_erf(v));
    if (act == 3) v = rbf(v > 0.0f ? __fmul_rn(v, v) : 0.0f);
    O[at] = __float2bfloat16_rn(R ? __fadd_rn(bf(R[at]), v) : v);
}

// Block an output row (i, j) of an h x w grid: rows (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) side by side.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_unshuffle(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int w,
                                                                      int C) {
    const int o = blockIdx.x, i = o / (w / 2), j = o % (w / 2);
    for (int e = threadIdx.x; e < 4 * C; e += NT) {
        const int q = e / C;
        const int64_t src = static_cast<int64_t>(2 * i + q / 2) * w + 2 * j + q % 2;
        Y[static_cast<int64_t>(o) * 4 * C + e] = X[src * C + e % C];
    }
}

// Element i: bf16(x), round to nearest even (a picture's float patches for the tower's first Linear).
extern "C" __global__ void __launch_bounds__(NT) tf_vit_cast(const float* __restrict__ X, __nv_bfloat16* __restrict__ Y, int64_t count) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at < count) Y[at] = __float2bfloat16_rn(X[at]);
}

namespace {

// An output index's two bilinear sources over `in` samples resized to `out` (half-pixel centres, clamped) and weights.
struct Tap {
    int lo, hi;
    float l0, l1;
};

__device__ __forceinline__ Tap tap(int in, int out, int i) {
    if (in == out) return {i, i, 1.0f, 0.0f};
    const float scale = __fdiv_rn(static_cast<float>(in), static_cast<float>(out));
    const float src = fmaxf(__fsub_rn(__fmul_rn(scale, __fadd_rn(static_cast<float>(i), 0.5f)), 0.5f), 0.0f);
    const int lo = min(static_cast<int>(floorf(src)), in - 1);
    const float l1 = fminf(fmaxf(__fsub_rn(src, static_cast<float>(lo)), 0.0f), 1.0f);
    return {lo, lo + (lo < in - 1 ? 1 : 0), __fsub_rn(1.0f, l1), l1};
}

}  // namespace

// Block a patch (y, x) of a grid gw wide: o = bf16(x + bf16(the side^2 table resized to m^2, at (y, x))), RADIO's CPE.
extern "C" __global__ void __launch_bounds__(NT) tf_vit_position_add(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ T,
                                                                       __nv_bfloat16* __restrict__ O, int gw, int m, int side, int C) {
    const int r = blockIdx.x, y = r / gw, x = r % gw;
    const Tap ty = tap(side, m, y), tx = tap(side, m, x);
    const float w00 = __fmul_rn(ty.l0, tx.l0), w01 = __fmul_rn(ty.l0, tx.l1);
    const float w10 = __fmul_rn(ty.l1, tx.l0), w11 = __fmul_rn(ty.l1, tx.l1);
    const __nv_bfloat16* t00 = T + (static_cast<int64_t>(ty.lo) * side + tx.lo) * C;
    const __nv_bfloat16* t01 = T + (static_cast<int64_t>(ty.lo) * side + tx.hi) * C;
    const __nv_bfloat16* t10 = T + (static_cast<int64_t>(ty.hi) * side + tx.lo) * C;
    const __nv_bfloat16* t11 = T + (static_cast<int64_t>(ty.hi) * side + tx.hi) * C;
    const int64_t at = static_cast<int64_t>(r) * C;
    for (int c = threadIdx.x; c < C; c += NT) {
        const float far = __fadd_rn(__fmul_rn(bf(t10[c]), w10), __fmul_rn(bf(t11[c]), w11));
        const float p = __fadd_rn(__fmul_rn(bf(t00[c]), w00), __fadd_rn(__fmul_rn(bf(t01[c]), w01), far));
        O[at + c] = __float2bfloat16_rn(__fadd_rn(rbf(p), bf(X[at + c])));
    }
}

// ---- ours: GLM-5.3-Flash's tower (families/glm/cuda_vision.zig), HF Glm5NextVisionModel's bf16 rounding points ----

namespace {

constexpr int GLMV_MAXE = 16;  // the merger's LayerNorm: rows up to NT * 16 = 4096 wide

// fp32 sum of squares of a row's elements this thread holds, across the block (fixed order).
__device__ __forceinline__ float glmv_rstd(float ss, int D, float eps, float* red) {
    const float var = __fdiv_rn(block_sum(ss, red), static_cast<float>(D));
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
}

}  // namespace

// Block a row: Glm5NextRMSNorm in bf16, y = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), rows up to 2048 wide.
extern "C" __global__ void __launch_bounds__(NT) tf_glmv_rms(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                              __nv_bfloat16* __restrict__ Y, int D, float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[MAXE];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        ss = __fmaf_rn(x[k], x[k], ss);
    }
    const float rstd = glmv_rstd(ss, D, eps, red);
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fmul_rn(bf(W[i]), rbf(__fmul_rn(x[k], rstd))));
    }
}

// Block a row of qkv [rows, 3 * 16 * 64] (a patch, block-major over 2x2 windows of a grid gw patches wide): its q and
// k heads RMS-normed over the 64 dims (q_norm / k_norm, rounded as tf_glmv_rms), then the 2-D rope in fp32 with the
// patch's (h, w): angle j < 16 = h * INV[j], 16 <= j < 32 = w * INV[j - 16], dims j and j + 32 alike; rotate_half over
// all 64; bf16 back in place. A warp a head (32 heads: 16 q, then 16 k), two dims a lane.
extern "C" __global__ void __launch_bounds__(NT) tf_glmv_qk_rope(__nv_bfloat16* __restrict__ QKV, const __nv_bfloat16* __restrict__ QW,
                                                                  const __nv_bfloat16* __restrict__ KW, const float* __restrict__ INV, int gw,
                                                                  float eps) {
    constexpr int H = 16, Dh = 64;
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int win = r >> 2, wx_n = gw >> 1;
    const float ph = static_cast<float>((win / wx_n) * 2 + ((r >> 1) & 1));
    const float pw = static_cast<float>((win % wx_n) * 2 + (r & 1));
    for (int head = warp; head < 2 * H; head += NT / 32) {
        __nv_bfloat16* x = QKV + static_cast<int64_t>(r) * 3 * H * Dh + head * Dh;  // q heads 0..15, k heads 16..31
        const __nv_bfloat16* w = head < H ? QW : KW;
        const float a = bf(x[lane]), b = bf(x[lane + 32]);
        const float ss = warp_sum(__fmaf_rn(b, b, __fmul_rn(a, a)));
        const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(Dh)), eps)));
        const float na = rbf(__fmul_rn(bf(w[lane]), rbf(__fmul_rn(a, rstd))));
        const float nb = rbf(__fmul_rn(bf(w[lane + 32]), rbf(__fmul_rn(b, rstd))));
        const float angle = __fmul_rn(lane < 16 ? ph : pw, INV[lane & 15]);
        float s, c;
        sincosf(angle, &s, &c);
        // dims lane and lane + 32 share the angle: out_lo = lo * c - hi * s, out_hi = hi * c + lo * s
        x[lane] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(na, c), __fmul_rn(-nb, s)));
        x[lane + 32] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(nb, c), __fmul_rn(na, s)));
    }
}

// Element i of [rows, N] from GU [rows, 2N] (a row's gate then up): bf16(bf16(silu(min(g, L))) * clamp(u, -L, L)).
extern "C" __global__ void __launch_bounds__(NT) tf_glmv_swiglu(const __nv_bfloat16* __restrict__ GU, __nv_bfloat16* __restrict__ O, int64_t rows,
                                                                 int N, float limit) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at >= rows * N) return;
    const int64_t row = at / N, c = at % N;
    const float g = fminf(bf(GU[row * 2 * N + c]), limit);
    const float u = fminf(fmaxf(bf(GU[row * 2 * N + N + c]), -limit), limit);
    O[at] = __float2bfloat16_rn(__fmul_rn(rbf(tf_silu(g)), u));
}

// tf_vit_layer_norm for rows up to 4096 wide (the merger's post_projection_norm).
extern "C" __global__ void __launch_bounds__(NT) tf_glmv_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                     const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, int D,
                                                                     float eps) {
    __shared__ float red[NT / 32];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * D;
    float x[GLMV_MAXE];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < GLMV_MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        x[k] = bf(X[at + i]);
        acc = __fadd_rn(acc, x[k]);
    }
    const float mean = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    acc = 0.0f;
#pragma unroll
    for (int k = 0; k < GLMV_MAXE; ++k) {
        if (threadIdx.x + k * NT >= D) continue;
        x[k] = __fsub_rn(x[k], mean);
        acc = __fadd_rn(acc, __fmul_rn(x[k], x[k]));
    }
    const float var = __fdiv_rn(block_sum(acc, red), static_cast<float>(D));
    const float rstd = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, eps)));
#pragma unroll
    for (int k = 0; k < GLMV_MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i < D) Y[at + i] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(x[k], rstd), bf(W[i])), bf(B[i])));
    }
}

// Block (row, column block) of `rows` image rows [rows, D] into the stream rows [rows][C][D] (every hc copy).
extern "C" __global__ void __launch_bounds__(NT) tf_glmv_splice(const __nv_bfloat16* __restrict__ SRC, __nv_bfloat16* __restrict__ DST, int D,
                                                                 int C) {
    const int r = blockIdx.x, i = blockIdx.y * NT + threadIdx.x;
    if (i >= D) return;
    const __nv_bfloat16 v = SRC[static_cast<int64_t>(r) * D + i];
    for (int c = 0; c < C; ++c) DST[(static_cast<int64_t>(r) * C + c) * D + i] = v;
}
