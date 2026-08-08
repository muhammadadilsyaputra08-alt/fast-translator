#include "transformer_bitnet.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace transformer_bitnet {

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

// Ternary matmul with fp32 activations: out[o] = alpha * sum_i sign(W[o,i]) * x[i],
// where sign(W) in {-1,0,+1}. No real multiplication against the weight —
// only add/sub/skip — which is the core BitNet speed argument. Activations
// stay fp32 here (a full W1.58A8 kernel would also quantize x to int8;
// core/include/bitnet_kernel.h implements that int8 variant and is used
// where an int8 activation pipeline is wired up).
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

// Regular fp32 matmul, used for the (unquantized) output classifier, which
// shares the token embedding table — matches stories15M's shared_weights.
void matmul_fp32(float* out, const float* x, const float* w, int d_in, int d_out) {
    for (int o = 0; o < d_out; ++o) {
        float sum = 0.0f;
        const float* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) sum += row[i] * x[i];
        out[o] = sum;
    }
}

} // namespace

void RunState::init(const fllm::TransformerConfig& c) {
    x.resize(c.dim);
    xb.resize(c.dim);
    xb2.resize(c.dim);
    hb.resize(c.hidden_dim);
    hb2.resize(c.hidden_dim);
    q.resize(c.dim);
    k.resize(c.dim);
    v.resize(c.dim);
    att.resize(static_cast<size_t>(c.n_heads) * c.seq_len);
    logits.resize(c.vocab_size);
    key_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * c.dim);
    value_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * c.dim);
}

BitnetTransformer::BitnetTransformer(const fllm::FllmModel& model)
    : config_(model.transformer_config), w_(model.transformer_weights) {
    state_.init(config_);
}

float* BitnetTransformer::forward(int token, int pos) {
    const auto& p = config_;
    const auto& w = w_;
    RunState& s = state_;
    int dim = p.dim;
    int head_size = dim / p.n_heads;
    int hidden_dim = p.hidden_dim;

    const float* content_row = w.token_embedding_table.data() + static_cast<size_t>(token) * dim;
    std::memcpy(s.x.data(), content_row, dim * sizeof(float));

    const float* freq_real_row = w.freq_cis_real.data() + static_cast<size_t>(pos) * (head_size / 2);
    const float* freq_imag_row = w.freq_cis_imag.data() + static_cast<size_t>(pos) * (head_size / 2);

    for (int l = 0; l < p.n_layers; ++l) {
        rmsnorm(s.xb.data(), s.x.data(), w.rms_att_weight.data() + static_cast<size_t>(l) * dim, dim);

        matmul_ternary(s.q.data(), s.xb.data(), w.wq.data() + static_cast<size_t>(l) * dim * dim, w.wq_alpha[l], dim, dim);
        matmul_ternary(s.k.data(), s.xb.data(), w.wk.data() + static_cast<size_t>(l) * dim * dim, w.wk_alpha[l], dim, dim);
        matmul_ternary(s.v.data(), s.xb.data(), w.wv.data() + static_cast<size_t>(l) * dim * dim, w.wv_alpha[l], dim, dim);

        for (int h = 0; h < p.n_heads; ++h) {
            float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            float* k_head = s.k.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float q0 = q_head[2 * i], q1 = q_head[2 * i + 1];
                q_head[2 * i] = q0 * fcr - q1 * fci;
                q_head[2 * i + 1] = q0 * fci + q1 * fcr;
                float k0 = k_head[2 * i], k1 = k_head[2 * i + 1];
                k_head[2 * i] = k0 * fcr - k1 * fci;
                k_head[2 * i + 1] = k0 * fci + k1 * fcr;
            }
        }

        size_t layer_cache_off = static_cast<size_t>(l) * p.seq_len * dim;
        std::memcpy(s.key_cache.data() + layer_cache_off + static_cast<size_t>(pos) * dim, s.k.data(), dim * sizeof(float));
        std::memcpy(s.value_cache.data() + layer_cache_off + static_cast<size_t>(pos) * dim, s.v.data(), dim * sizeof(float));

        for (int h = 0; h < p.n_heads; ++h) {
            const float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            float* att_row = s.att.data() + static_cast<size_t>(h) * p.seq_len;
            for (int t = 0; t <= pos; ++t) {
                const float* k_t = s.key_cache.data() + layer_cache_off + static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += q_head[i] * k_t[i];
                score /= std::sqrt(static_cast<float>(head_size));
                att_row[t] = score;
            }
            softmax(att_row, pos + 1);

            float* out_head = s.xb.data() + static_cast<size_t>(h) * head_size;
            std::fill(out_head, out_head + head_size, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* v_t = s.value_cache.data() + layer_cache_off + static_cast<size_t>(t) * dim + static_cast<size_t>(h) * head_size;
                float a = att_row[t];
                for (int i = 0; i < head_size; ++i) out_head[i] += a * v_t[i];
            }
        }

        matmul_ternary(s.xb2.data(), s.xb.data(), w.wo.data() + static_cast<size_t>(l) * dim * dim, w.wo_alpha[l], dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        rmsnorm(s.xb.data(), s.x.data(), w.rms_ffn_weight.data() + static_cast<size_t>(l) * dim, dim);
        matmul_ternary(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w1_alpha[l], dim, hidden_dim);
        matmul_ternary(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, w.w3_alpha[l], dim, hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            float v = s.hb[i];
            float silu = v / (1.0f + std::exp(-v));
            s.hb[i] = silu * s.hb2[i];
        }
        matmul_ternary(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, w.w2_alpha[l], hidden_dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight.data(), dim);
    matmul_fp32(s.logits.data(), s.x.data(), w.token_embedding_table.data(), dim, p.vocab_size);

    return s.logits.data();
}

} // namespace transformer_bitnet
