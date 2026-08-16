#include "transformer.h"
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace transformer {

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
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

// out[d] = sum_i W[d,i] * x[i], W is row-major [d_out x d_in].
void matmul(float* out, const float* x, const float* w, int d_in, int d_out) {
    for (int o = 0; o < d_out; ++o) {
        float sum = 0.0f;
        const float* row = w + static_cast<size_t>(o) * d_in;
        for (int i = 0; i < d_in; ++i) sum += row[i] * x[i];
        out[o] = sum;
    }
}

// Reads a Weights struct from an open, already-positioned ifstream, given
// the checkpoint's fixed section ordering. Supports GQA: wk/wv use kv_dim
// (n_kv_heads * head_size) instead of dim when n_kv_heads < n_heads;
// n_kv_heads == n_heads (stories15M) is the kv_dim == dim special case.
void load_weights(std::ifstream& f, const Config& c, Weights& w, bool shared_classifier) {
    int head_size = c.dim / c.n_heads;
    int kv_dim = c.n_kv_heads * head_size;

    auto read_vec = [&](std::vector<float>& v, size_t count) {
        v.resize(count);
        f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    };

    read_vec(w.token_embedding_table, static_cast<size_t>(c.vocab_size) * c.dim);
    read_vec(w.rms_att_weight, static_cast<size_t>(c.n_layers) * c.dim);
    read_vec(w.wq, static_cast<size_t>(c.n_layers) * c.dim * c.dim);
    read_vec(w.wk, static_cast<size_t>(c.n_layers) * kv_dim * c.dim);
    read_vec(w.wv, static_cast<size_t>(c.n_layers) * kv_dim * c.dim);
    read_vec(w.wo, static_cast<size_t>(c.n_layers) * c.dim * c.dim);
    read_vec(w.rms_ffn_weight, static_cast<size_t>(c.n_layers) * c.dim);
    read_vec(w.w1, static_cast<size_t>(c.n_layers) * c.hidden_dim * c.dim);
    read_vec(w.w2, static_cast<size_t>(c.n_layers) * c.dim * c.hidden_dim);
    read_vec(w.w3, static_cast<size_t>(c.n_layers) * c.hidden_dim * c.dim);
    read_vec(w.rms_final_weight, static_cast<size_t>(c.dim));
    read_vec(w.freq_cis_real, static_cast<size_t>(c.seq_len) * (head_size / 2));
    read_vec(w.freq_cis_imag, static_cast<size_t>(c.seq_len) * (head_size / 2));

    // Unshared classifier (e.g. TinyLlama): one more tensor follows,
    // the actual lm_head weights, separate from token_embedding_table.
    if (!shared_classifier) {
        read_vec(w.wcls, static_cast<size_t>(c.vocab_size) * c.dim);
    }
}

} // namespace

void RunState::init(const Config& c) {
    int head_size = c.dim / c.n_heads;
    int kv_dim = c.n_kv_heads * head_size;

    x.resize(c.dim);
    xb.resize(c.dim);
    xb2.resize(c.dim);
    hb.resize(c.hidden_dim);
    hb2.resize(c.hidden_dim);
    q.resize(c.dim);
    k.resize(kv_dim);
    v.resize(kv_dim);
    att.resize(static_cast<size_t>(c.n_heads) * c.seq_len);
    logits.resize(c.vocab_size);
    key_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim);
    value_cache.resize(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim);
}

std::unique_ptr<Transformer> Transformer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[transformer] cannot open " << path << "\n";
        return nullptr;
    }

    Config c{};
    f.read(reinterpret_cast<char*>(&c), sizeof(Config));
    if (!f) {
        std::cerr << "[transformer] failed to read config header\n";
        return nullptr;
    }
    bool shared_classifier = c.vocab_size > 0;
    if (!shared_classifier) c.vocab_size = -c.vocab_size;

    auto t = std::unique_ptr<Transformer>(new Transformer());
    t->config_ = c;
    t->shared_classifier_ = shared_classifier;
    load_weights(f, c, t->w_, shared_classifier);
    if (!f && !f.eof()) {
        std::cerr << "[transformer] failed to read weight tensors (truncated file?)\n";
        return nullptr;
    }
    t->state_.init(c);
    return t;
}

float* Transformer::forward(int token, int pos) {
    const Config& p = config_;
    Weights& w = w_;
    RunState& s = state_;
    int dim = p.dim;
    int head_size = dim / p.n_heads;
    int kv_dim = p.n_kv_heads * head_size;
    int hidden_dim = p.hidden_dim;
    // Under GQA, `group_size` query heads share each KV head; when
    // n_kv_heads == n_heads (MHA, e.g. stories15M) group_size == 1, i.e.
    // every query head has its own KV head, same as before this change.
    int group_size = p.n_heads / p.n_kv_heads;

    const float* content_row = w.token_embedding_table.data() + static_cast<size_t>(token) * dim;
    std::memcpy(s.x.data(), content_row, dim * sizeof(float));

    const float* freq_real_row = w.freq_cis_real.data() + static_cast<size_t>(pos) * (head_size / 2);
    const float* freq_imag_row = w.freq_cis_imag.data() + static_cast<size_t>(pos) * (head_size / 2);

    for (int l = 0; l < p.n_layers; ++l) {
        rmsnorm(s.xb.data(), s.x.data(), w.rms_att_weight.data() + static_cast<size_t>(l) * dim, dim);

        matmul(s.q.data(), s.xb.data(), w.wq.data() + static_cast<size_t>(l) * dim * dim, dim, dim);
        matmul(s.k.data(), s.xb.data(), w.wk.data() + static_cast<size_t>(l) * kv_dim * dim, dim, kv_dim);
        matmul(s.v.data(), s.xb.data(), w.wv.data() + static_cast<size_t>(l) * kv_dim * dim, dim, kv_dim);

        // RoPE: Q rotates over all n_heads; K rotates over only n_kv_heads
        // (it has fewer heads under GQA — rotating "extra" heads that don't
        // exist would read/write out of bounds).
        for (int h = 0; h < p.n_heads; ++h) {
            float* q_head = s.q.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float q0 = q_head[2 * i], q1 = q_head[2 * i + 1];
                q_head[2 * i]     = q0 * fcr - q1 * fci;
                q_head[2 * i + 1] = q0 * fci + q1 * fcr;
            }
        }
        for (int h = 0; h < p.n_kv_heads; ++h) {
            float* k_head = s.k.data() + static_cast<size_t>(h) * head_size;
            for (int i = 0; i < head_size / 2; ++i) {
                float fcr = freq_real_row[i], fci = freq_imag_row[i];
                float k0 = k_head[2 * i], k1 = k_head[2 * i + 1];
                k_head[2 * i]     = k0 * fcr - k1 * fci;
                k_head[2 * i + 1] = k0 * fci + k1 * fcr;
            }
        }

        size_t layer_cache_off = static_cast<size_t>(l) * p.seq_len * kv_dim;
        std::memcpy(s.key_cache.data() + layer_cache_off + static_cast<size_t>(pos) * kv_dim, s.k.data(), kv_dim * sizeof(float));
        std::memcpy(s.value_cache.data() + layer_cache_off + static_cast<size_t>(pos) * kv_dim, s.v.data(), kv_dim * sizeof(float));

        // Multi-head causal attention: query head h reads from KV head
        // (h / group_size) — under MHA (group_size==1) this is just h,
        // identical to the pre-GQA behavior.
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

        matmul(s.xb2.data(), s.xb.data(), w.wo.data() + static_cast<size_t>(l) * dim * dim, dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        rmsnorm(s.xb.data(), s.x.data(), w.rms_ffn_weight.data() + static_cast<size_t>(l) * dim, dim);
        matmul(s.hb.data(), s.xb.data(), w.w1.data() + static_cast<size_t>(l) * hidden_dim * dim, dim, hidden_dim);
        matmul(s.hb2.data(), s.xb.data(), w.w3.data() + static_cast<size_t>(l) * hidden_dim * dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            float v = s.hb[i];
            float silu = v / (1.0f + std::exp(-v));
            s.hb[i] = silu * s.hb2[i];
        }
        matmul(s.xb.data(), s.hb.data(), w.w2.data() + static_cast<size_t>(l) * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight.data(), dim);
    matmul(s.logits.data(), s.x.data(), (shared_classifier_ ? w.token_embedding_table : w.wcls).data(), dim, p.vocab_size);

    return s.logits.data();
}

} // namespace transformer
