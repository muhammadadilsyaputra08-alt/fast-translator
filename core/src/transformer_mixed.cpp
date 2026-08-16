#include "transformer_mixed.h"
#include "thread_pool.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FASTAI_HAVE_NEON 1
#endif

namespace transformer_mixed {

// Real INT8 multiply-accumulate (not sign-only like ternary) — standard
// PTQ, fp32 activations. Declared at namespace scope (not anonymous) so
// neon_selftest.cpp can call both variants directly to cross-check them
// on-device, since this sandbox has no ARM cross-compiler to test the
// NEON path itself (see notes/NEON_OPTIMIZATION_FINDINGS.md).
void matmul_int8_scalar(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                         int o_begin, int o_end) {
    if (o_end < 0) o_end = d_out;
    for (int o = o_begin; o < o_end; ++o) {
        float acc = 0.0f;
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) acc += static_cast<float>(row[i]) * x[i];
        out[o] = acc * scale;
    }
}

#ifdef FASTAI_HAVE_NEON
// Processes 8 int8 weights per iteration: widen int8 -> int16 -> int32 ->
// float32 (NEON has no direct int8->float conversion, so this widening
// chain is the standard idiom), then fused-multiply-add against 2x
// float32x4 activation lanes. This trades BitNet's "no multiply" scalar
// trick for real SIMD throughput -- on real hardware FMA is fast enough
// that this is expected to win despite doing genuine multiplication.
void matmul_int8_neon(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out,
                       int o_begin, int o_end) {
    if (o_end < 0) o_end = d_out;
    for (int o = o_begin; o < o_end; ++o) {
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= d_in; i += 8) {
            int8x8_t w8 = vld1_s8(row + i);
            int16x8_t w16 = vmovl_s8(w8);
            int32x4_t w32_lo = vmovl_s16(vget_low_s16(w16));
            int32x4_t w32_hi = vmovl_s16(vget_high_s16(w16));
            float32x4_t wf_lo = vcvtq_f32_s32(w32_lo);
            float32x4_t wf_hi = vcvtq_f32_s32(w32_hi);
            float32x4_t x_lo = vld1q_f32(x + i);
            float32x4_t x_hi = vld1q_f32(x + i + 4);
            acc0 = vmlaq_f32(acc0, wf_lo, x_lo);
            acc1 = vmlaq_f32(acc1, wf_hi, x_hi);
        }
        float32x4_t acc = vaddq_f32(acc0, acc1);
        float sum = vaddvq_f32(acc); // AArch64-only horizontal add; fine, we only target arm64-v8a
        for (; i < d_in; ++i) sum += static_cast<float>(row[i]) * x[i]; // remainder, d_in % 8 != 0
        out[o] = sum * scale;
    }
}

// Parallelized across output neurons via the process-wide thread pool
// (core/include/thread_pool.h). Each thread computes a fully disjoint
// range of `out[o_begin..o_end)` using the existing single-threaded NEON
// kernel above -- no partial sums are combined across threads, so this is
// bit-identical to running the whole [0, d_out) range on one thread, just
// faster on multi-core devices (e.g. Dimensity 6300: 2xA76 + 6xA55).
// Below a small size threshold we skip the pool entirely: for small
// matmuls (e.g. GQA's narrower wk/wv projections), thread hand-off/sync
// overhead can exceed the compute time saved.
void matmul_int8(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out) {
    constexpr int kMinParallelWork = 64; // d_out below this: not worth spreading across threads
    if (d_out < kMinParallelWork) {
        matmul_int8_neon(out, x, w, scale, d_in, d_out);
        return;
    }
    threadpool::global_pool().parallel_for(d_out, [&](int begin, int end) {
        matmul_int8_neon(out, x, w, scale, d_in, d_out, begin, end);
    });
}
#else
void matmul_int8(float* out, const float* x, const int8_t* w, float scale, int d_in, int d_out) {
    constexpr int kMinParallelWork = 64;
    if (d_out < kMinParallelWork) {
        matmul_int8_scalar(out, x, w, scale, d_in, d_out);
        return;
    }
    threadpool::global_pool().parallel_for(d_out, [&](int begin, int end) {
        matmul_int8_scalar(out, x, w, scale, d_in, d_out, begin, end);
    });
}
#endif

namespace {

void rmsnorm(float* out, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) ss += x[i] * x[i];
    ss = 1.0f / std::sqrt(ss / size + 1e-5f);
    for (int i = 0; i < size; ++i) out[i] = weight[i] * (x[i] * ss);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) max_val = std::max(max_val, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

void matmul_ternary(float* out, const float* x, const int8_t* w, float alpha, int d_in, int d_out) {
    for (int o = 0; o < d_out; ++o) {
        float acc = 0.0f;
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) {
            int8_t wv = row[i];
            if (wv == 1) acc += x[i];
            else if (wv == -1) acc -= x[i];
        }
        out[o] = acc * alpha;
    }
}

void matmul_fp32(float* out, const float* x, const float* w, int d_in, int d_out) {
    for (int o = 0; o < d_out; ++o) {
        float sum = 0.0f;
        const float* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) sum += row[i] * x[i];
        out[o] = sum;
    }
}

// Batched INT8 matmul: for each output row (weight row, loaded once),
// compute the dot product against ALL n_tokens input vectors before
// moving to the next row. This is the cache-locality win batching buys —
// each weight row is read from memory once and reused n_tokens times,
// instead of once per token as in the sequential single-token path.
void matmul_int8_batch(float* out, const float* x, const int8_t* w, float scale,
                        int d_in, int d_out, int n_tokens) {
    for (int o = 0; o < d_out; ++o) {
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        for (int t = 0; t < n_tokens; ++t) {
            const float* xt = x + static_cast<size_t>(t) * d_in;
            float acc = 0.0f;
            for (int i = 0; i < d_in; ++i) acc += static_cast<float>(row[i]) * xt[i];
            out[static_cast<size_t>(t) * d_out + o] = acc * scale;
        }
    }
}

void matmul_ternary_batch(float* out, const float* x, const int8_t* w, float alpha,
                           int d_in, int d_out, int n_tokens) {
    for (int o = 0; o < d_out; ++o) {
        const int8_t* row = w + static_cast<size_t>(o) * d_in;
        for (int t = 0; t < n_tokens; ++t) {
            const float* xt = x + static_cast<size_t>(t) * d_in;
            float acc = 0.0f;
            for (int i = 0; i < d_in; ++i) {
                int8_t wv = row[i];
                if (wv == 1) acc += xt[i];
                else if (wv == -1) acc -= xt[i];
            }
            out[static_cast<size_t>(t) * d_out + o] = acc * alpha;
        }
    }
}

} // namespace

void RunState::init(const fllm::TransformerConfig& c) {
    int head_size = c.dim / c.n_heads;
    int kv_dim = c.n_kv_heads * head_size;
    x.resize(c.dim); xb.resize(c.dim); xb2.resize(c.dim);
    hb.resize(c.hidden_dim); hb2.resize(c.hidden_dim);
    q.resize(c.dim); k.resize(kv_dim); v.resize(kv_dim);
    att.resize(static_cast<size_t>(c.n_heads) * c.seq_len);
    logits.resize(c.vocab_size);
    key_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim);
    value_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim);
}

MixedPrecisionTransformer::MixedPrecisionTransformer(const fllm::FllmModel& model, FfnPrecision ffn_precision)
    : config_(model.transformer_config), w_(model.transformer_weights), ffn_precision_(ffn_precision) {
    state_.init(config_);
}

float* MixedPrecisionTransformer::forward(int token, int pos) {
    const auto& p = config_;
    const auto& w = w_;
    RunState& s = state_;
    int dim = p.dim;
    int head_size = dim / p.n_heads;
    int kv_dim = p.n_kv_heads * head_size;
    int group_size = p.n_heads / p.n_kv_heads; // GQA: query heads sharing each KV head
    int hidden_dim = p.hidden_dim;

    const float* content_row = w.token_embedding_table.data() + static_cast<size_t>(token) * dim;
    std::memcpy(s.x.data(), content_row, dim * sizeof(float));

    const float* freq_real_row = w.freq_cis_real.data() + static_cast<size_t>(pos) * (head_size / 2);
    const float* freq_imag_row = w.freq_cis_imag.data() + static_cast<size_t>(pos) * (head_size / 2);

    for (int l = 0; l < p.n_layers; ++l) {
        rmsnorm(s.xb.data(), s.x.data(), w.rms_att_weight.data() + static_cast<size_t>(l) * dim, dim);

        // Attention: INT8 (real multiply), higher fidelity than ternary.
        matmul_int8(s.q.data(), s.xb.data(), w.wq.data() + static_cast<size_t>(l) * dim * dim, w.wq_alpha[l], dim, dim);
        matmul_int8(s.k.data(), s.xb.data(), w.wk.data() + static_cast<size_t>(l) * kv_dim * dim, w.wk_alpha[l], dim, kv_dim);
        matmul_int8(s.v.data(), s.xb.data(), w.wv.data() + static_cast<size_t>(l) * kv_dim * dim, w.wv_alpha[l], dim, kv_dim);

        for (int h = 0; h < p.n_heads; ++h) {
            float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float q0 = q_head[2 * i], q1 = q_head[2 * i + 1];
                q_head[2 * i] = q0 * fcr - q1 * fci;
                q_head[2 * i + 1] = q0 * fci + q1 * fcr;
            }
        }
        for (int h = 0; h < p.n_kv_heads; ++h) {
            float* k_head = s.k.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float k0 = k_head[2 * i], k1 = k_head[2 * i + 1];
                k_head[2 * i] = k0 * fcr - k1 * fci;
                k_head[2 * i + 1] = k0 * fci + k1 * fcr;
            }
        }

        size_t layer_cache_off = static_cast<size_t>(l) * p.seq_len * kv_dim;
        std::memcpy(s.key_cache.data() + layer_cache_off + static_cast<size_t>(pos) * kv_dim, s.k.data(), kv_dim * sizeof(float));
        std::memcpy(s.value_cache.data() + layer_cache_off + static_cast<size_t>(pos) * kv_dim, s.v.data(), kv_dim * sizeof(float));

        for (int h = 0; h < p.n_heads; ++h) {
            int kv_h = h / group_size;
            const float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            float* att_row = s.att.data() + static_cast<size_t>(h) * p.seq_len;
            for (int t = 0; t <= pos; ++t) {
                const float* k_t = s.key_cache.data() + layer_cache_off + static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kv_h) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += q_head[i] * k_t[i];
                score /= std::sqrt(static_cast<float>(head_size));
                att_row[t] = score;
            }
            softmax(att_row, pos + 1);

            float* out_head = s.xb.data() + static_cast<size_t>(h) * head_size;
            std::fill(out_head, out_head + head_size, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* v_t = s.value_cache.data() + layer_cache_off + static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kv_h) * head_size;
                float a = att_row[t];
                for (int i = 0; i < head_size; ++i) out_head[i] += a * v_t[i];
            }
        }

        matmul_int8(s.xb2.data(), s.xb.data(), w.wo.data() + static_cast<size_t>(l) * dim * dim, w.wo_alpha[l], dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        // FFN: ternary or INT8, depending on this instance's configuration.
        rmsnorm(s.xb.data(), s.x.data(), w.rms_ffn_weight.data() + static_cast<size_t>(l) * dim, dim);
        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim);
            matmul_ternary(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim);
        } else {
            matmul_int8(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim);
            matmul_int8(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim);
        }
        for (int i = 0; i < hidden_dim; ++i) {
            float v = s.hb[i];
            float silu = v / (1.0f + std::exp(-v));
            s.hb[i] = silu * s.hb2[i];
        }
        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim);
        } else {
            matmul_int8(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim);
        }
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight.data(), dim);
    matmul_fp32(s.logits.data(), s.x.data(), (w.wcls.empty() ? w.token_embedding_table : w.wcls).data(), dim, p.vocab_size);

    return s.logits.data();
}

std::vector<std::vector<float>> MixedPrecisionTransformer::forward_batch(
    const std::vector<int>& tokens, int start_pos) {

    const auto& p = config_;
    const auto& w = w_;
    int n = static_cast<int>(tokens.size());
    int dim = p.dim, hidden_dim = p.hidden_dim, head_size = dim / p.n_heads;

    // Batch-local scratch buffers, laid out [n_tokens x dim] etc.
    std::vector<float> x(static_cast<size_t>(n) * dim);
    std::vector<float> xb(static_cast<size_t>(n) * dim), xb2(static_cast<size_t>(n) * dim);
    std::vector<float> hb(static_cast<size_t>(n) * hidden_dim), hb2(static_cast<size_t>(n) * hidden_dim);
    std::vector<float> q(static_cast<size_t>(n) * dim), k(static_cast<size_t>(n) * dim), v(static_cast<size_t>(n) * dim);

    for (int t = 0; t < n; ++t) {
        const float* emb = w.token_embedding_table.data() + static_cast<size_t>(tokens[t]) * dim;
        std::copy(emb, emb + dim, x.begin() + static_cast<size_t>(t) * dim);
    }

    for (int l = 0; l < p.n_layers; ++l) {
        for (int t = 0; t < n; ++t) {
            rmsnorm(&xb[static_cast<size_t>(t) * dim], &x[static_cast<size_t>(t) * dim],
                    w.rms_att_weight.data() + static_cast<size_t>(l) * dim, dim);
        }

        matmul_int8_batch(q.data(), xb.data(), w.wq.data() + static_cast<size_t>(l) * dim * dim, w.wq_alpha[l], dim, dim, n);
        matmul_int8_batch(k.data(), xb.data(), w.wk.data() + static_cast<size_t>(l) * dim * dim, w.wk_alpha[l], dim, dim, n);
        matmul_int8_batch(v.data(), xb.data(), w.wv.data() + static_cast<size_t>(l) * dim * dim, w.wv_alpha[l], dim, dim, n);

        // RoPE per batch position (each token has its own absolute pos).
        for (int t = 0; t < n; ++t) {
            int pos = start_pos + t;
            const float* fr = w.freq_cis_real.data() + static_cast<size_t>(pos) * (head_size / 2);
            const float* fi = w.freq_cis_imag.data() + static_cast<size_t>(pos) * (head_size / 2);
            for (int h = 0; h < p.n_heads; ++h) {
                float* qh = &q[static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size];
                float* kh = &k[static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size];
                for (int i = 0; i < head_size / 2; ++i) {
                    float fcr = fr[i], fci = fi[i];
                    float q0 = qh[2*i], q1 = qh[2*i+1];
                    qh[2*i] = q0*fcr - q1*fci; qh[2*i+1] = q0*fci + q1*fcr;
                    float k0 = kh[2*i], k1 = kh[2*i+1];
                    kh[2*i] = k0*fcr - k1*fci; kh[2*i+1] = k0*fci + k1*fcr;
                }
            }
        }

        // Write this batch's K,V into the persistent KV cache before doing
        // attention, so later positions in the SAME batch can see earlier
        // ones (causal, including intra-batch dependencies).
        size_t layer_cache_off = static_cast<size_t>(l) * p.seq_len * dim;
        for (int t = 0; t < n; ++t) {
            int pos = start_pos + t;
            std::copy(&k[static_cast<size_t>(t) * dim], &k[static_cast<size_t>(t) * dim] + dim,
                      state_.key_cache.begin() + layer_cache_off + static_cast<size_t>(pos) * dim);
            std::copy(&v[static_cast<size_t>(t) * dim], &v[static_cast<size_t>(t) * dim] + dim,
                      state_.value_cache.begin() + layer_cache_off + static_cast<size_t>(pos) * dim);
        }

        for (int t = 0; t < n; ++t) {
            int pos = start_pos + t;
            for (int h = 0; h < p.n_heads; ++h) {
                const float* q_head = &q[static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size];
                std::vector<float> att_row(pos + 1);
                for (int tt = 0; tt <= pos; ++tt) {
                    const float* k_t = state_.key_cache.data() + layer_cache_off + static_cast<size_t>(tt) * dim + static_cast<size_t>(h) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; ++i) score += q_head[i] * k_t[i];
                    att_row[tt] = score / std::sqrt(static_cast<float>(head_size));
                }
                softmax(att_row.data(), pos + 1);

                float* out_head = &xb[static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size];
                std::fill(out_head, out_head + head_size, 0.0f);
                for (int tt = 0; tt <= pos; ++tt) {
                    const float* v_t = state_.value_cache.data() + layer_cache_off + static_cast<size_t>(tt) * dim + static_cast<size_t>(h) * head_size;
                    float a = att_row[tt];
                    for (int i = 0; i < head_size; ++i) out_head[i] += a * v_t[i];
                }
            }
        }

        matmul_int8_batch(xb2.data(), xb.data(), w.wo.data() + static_cast<size_t>(l) * dim * dim, w.wo_alpha[l], dim, dim, n);
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < dim; ++i) x[static_cast<size_t>(t) * dim + i] += xb2[static_cast<size_t>(t) * dim + i];

        for (int t = 0; t < n; ++t) {
            rmsnorm(&xb[static_cast<size_t>(t) * dim], &x[static_cast<size_t>(t) * dim],
                    w.rms_ffn_weight.data() + static_cast<size_t>(l) * dim, dim);
        }

        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary_batch(hb.data(), xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim, n);
            matmul_ternary_batch(hb2.data(), xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim, n);
        } else {
            matmul_int8_batch(hb.data(), xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim, n);
            matmul_int8_batch(hb2.data(), xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim, n);
        }
        for (size_t i = 0; i < hb.size(); ++i) {
            float vv = hb[i];
            float silu = vv / (1.0f + std::exp(-vv));
            hb[i] = silu * hb2[i];
        }
        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary_batch(xb.data(), hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim, n);
        } else {
            matmul_int8_batch(xb.data(), hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim, n);
        }
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < dim; ++i) x[static_cast<size_t>(t) * dim + i] += xb[static_cast<size_t>(t) * dim + i];
    }

    std::vector<std::vector<float>> results(n, std::vector<float>(p.vocab_size));
    for (int t = 0; t < n; ++t) {
        std::vector<float> xt(x.begin() + static_cast<size_t>(t) * dim, x.begin() + static_cast<size_t>(t + 1) * dim);
        rmsnorm(xt.data(), xt.data(), w.rms_final_weight.data(), dim);
        matmul_fp32(results[t].data(), xt.data(), (w.wcls.empty() ? w.token_embedding_table : w.wcls).data(), dim, p.vocab_size);
    }
    return results;
}

size_t MixedPrecisionTransformer::kv_cache_bytes() const {
    size_t n_active = active_positions_.size();
    size_t per_layer_bytes = n_active * config_.dim * 2 * sizeof(float); // key + value
    return per_layer_bytes * config_.n_layers;
}

float* MixedPrecisionTransformer::forward_evictable(int token, int true_pos, const kvcache::RocketKVEvictor& evictor) {
    const auto& p = config_;
    const auto& w = w_;
    RunState& s = state_;
    int dim = p.dim, head_size = dim / p.n_heads, hidden_dim = p.hidden_dim;
    int kv_dim = p.n_kv_heads * head_size;
    int group_size = p.n_heads / p.n_kv_heads;

    if (evict_key_cache_.empty()) {
        evict_key_cache_.assign(p.n_layers, {});
        evict_value_cache_.assign(p.n_layers, {});
    }

    const float* content_row = w.token_embedding_table.data() + static_cast<size_t>(token) * dim;
    std::memcpy(s.x.data(), content_row, dim * sizeof(float));

    const float* freq_real_row = w.freq_cis_real.data() + static_cast<size_t>(true_pos) * (head_size / 2);
    const float* freq_imag_row = w.freq_cis_imag.data() + static_cast<size_t>(true_pos) * (head_size / 2);

    for (int l = 0; l < p.n_layers; ++l) {
        rmsnorm(s.xb.data(), s.x.data(), w.rms_att_weight.data() + static_cast<size_t>(l) * dim, dim);

        matmul_int8(s.q.data(), s.xb.data(), w.wq.data() + static_cast<size_t>(l) * dim * dim, w.wq_alpha[l], dim, dim);
        matmul_int8(s.k.data(), s.xb.data(), w.wk.data() + static_cast<size_t>(l) * kv_dim * dim, w.wk_alpha[l], dim, kv_dim);
        matmul_int8(s.v.data(), s.xb.data(), w.wv.data() + static_cast<size_t>(l) * kv_dim * dim, w.wv_alpha[l], dim, kv_dim);

        for (int h = 0; h < p.n_heads; ++h) {
            float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float q0 = q_head[2*i], q1 = q_head[2*i+1];
                q_head[2*i] = q0*fcr - q1*fci; q_head[2*i+1] = q0*fci + q1*fcr;
            }
        }
        for (int h = 0; h < p.n_kv_heads; ++h) {
            float* k_head = s.k.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float k0 = k_head[2*i], k1 = k_head[2*i+1];
                k_head[2*i] = k0*fcr - k1*fci; k_head[2*i+1] = k0*fci + k1*fcr;
            }
        }

        // Append this position's K,V to the active (post-eviction) cache.
        evict_key_cache_[l].insert(evict_key_cache_[l].end(), s.k.begin(), s.k.end());
        evict_value_cache_[l].insert(evict_value_cache_[l].end(), s.v.begin(), s.v.end());

        // Attention over ALL currently-active positions for this layer
        // (not 0..pos -- the active set may have holes from prior eviction).
        size_t n_active_before = evict_key_cache_[l].size() / kv_dim; // includes the just-appended entry
        for (int h = 0; h < p.n_heads; ++h) {
            int kv_h = h / group_size;
            const float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            std::vector<float> att_row(n_active_before);
            for (size_t t = 0; t < n_active_before; ++t) {
                const float* k_t = evict_key_cache_[l].data() + t * kv_dim + static_cast<size_t>(kv_h) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += q_head[i] * k_t[i];
                att_row[t] = score / std::sqrt(static_cast<float>(head_size));
            }
            softmax(att_row.data(), static_cast<int>(n_active_before));

            float* out_head = s.xb.data() + static_cast<size_t>(h) * head_size;
            std::fill(out_head, out_head + head_size, 0.0f);
            for (size_t t = 0; t < n_active_before; ++t) {
                const float* v_t = evict_value_cache_[l].data() + t * kv_dim + static_cast<size_t>(kv_h) * head_size;
                float a = att_row[t];
                for (int i = 0; i < head_size; ++i) out_head[i] += a * v_t[i];
            }
        }

        matmul_int8(s.xb2.data(), s.xb.data(), w.wo.data() + static_cast<size_t>(l) * dim * dim, w.wo_alpha[l], dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        rmsnorm(s.xb.data(), s.x.data(), w.rms_ffn_weight.data() + static_cast<size_t>(l) * dim, dim);
        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim);
            matmul_ternary(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim);
        } else {
            matmul_int8(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim);
            matmul_int8(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim);
        }
        for (int i = 0; i < hidden_dim; ++i) {
            float v = s.hb[i];
            float silu = v / (1.0f + std::exp(-v));
            s.hb[i] = silu * s.hb2[i];
        }
        if (ffn_precision_ == FfnPrecision::Ternary) {
            matmul_ternary(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim);
        } else {
            matmul_int8(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim);
        }
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    active_positions_.push_back(true_pos);

    // Eviction check (after processing this position through all layers,
    // so the just-added entry participates in scoring like everything else).
    if (evictor.should_evict_check(active_positions_.size())) {
        // NOTE: evict_key_cache_/evict_value_cache_ entries are stored with
        // stride kv_dim (n_kv_heads * head_size), NOT dim -- they hold s.k/s.v,
        // which under GQA (n_kv_heads < n_heads, e.g. TinyLlama: 4 vs 32) are
        // narrower than the full embedding dim. Using `dim` here (as this code
        // previously did) only happened to work for MHA models like stories15M
        // where kv_dim == dim; for TinyLlama it indexes far past the actual
        // buffer size once eviction triggers, causing a SIGSEGV inside
        // std::vector::assign (confirmed via on-device tombstone).
        std::vector<kvcache::KVEntry> scoring_entries(active_positions_.size());
        for (size_t t = 0; t < active_positions_.size(); ++t) {
            scoring_entries[t].key.assign(
                evict_key_cache_[0].begin() + t * kv_dim, evict_key_cache_[0].begin() + (t + 1) * kv_dim);
        }
        auto kept = evictor.select_kept_indices(scoring_entries);

        std::vector<int> new_positions;
        std::vector<std::vector<float>> new_keys(p.n_layers), new_values(p.n_layers);
        for (int l = 0; l < p.n_layers; ++l) { new_keys[l].reserve(kept.size()*kv_dim); new_values[l].reserve(kept.size()*kv_dim); }

        for (size_t idx : kept) {
            new_positions.push_back(active_positions_[idx]);
            for (int l = 0; l < p.n_layers; ++l) {
                new_keys[l].insert(new_keys[l].end(),
                    evict_key_cache_[l].begin() + idx*kv_dim, evict_key_cache_[l].begin() + (idx+1)*kv_dim);
                new_values[l].insert(new_values[l].end(),
                    evict_value_cache_[l].begin() + idx*kv_dim, evict_value_cache_[l].begin() + (idx+1)*kv_dim);
            }
        }
        active_positions_ = std::move(new_positions);
        evict_key_cache_ = std::move(new_keys);
        evict_value_cache_ = std::move(new_values);
    }

    rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight.data(), dim);
    matmul_fp32(s.logits.data(), s.x.data(), (w.wcls.empty() ? w.token_embedding_table : w.wcls).data(), dim, p.vocab_size);
    return s.logits.data();
}

} // namespace transformer_mixed

namespace transformer_mixed {
void MixedPrecisionTransformer::reset_evictable_cache() {
    active_positions_.clear();
    evict_key_cache_.clear();
    evict_value_cache_.clear();
}
} // namespace transformer_mixed
