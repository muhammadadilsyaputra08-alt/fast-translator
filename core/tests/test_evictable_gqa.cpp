#include "fllm_parser.h"
#include "transformer_mixed.h"
#include <iostream>
#include <random>

int main() {
    // Model kecil TAPI dengan GQA nyata (n_kv_heads < n_heads), persis kondisi
    // TinyLlama (dim=2048/n_heads=32/n_kv_heads=4 -> kv_dim=256 != dim). Kalau
    // kv_dim == dim (MHA biasa, kasus stories15M), bug stride ini tidak akan
    // pernah ketahuan -- makanya test lama tidak menangkapnya.
    fllm::FllmModel model;
    model.has_transformer = true;
    auto& tc = model.transformer_config;
    tc.dim = 64; tc.hidden_dim = 128; tc.n_layers = 2;
    tc.n_heads = 8; tc.n_kv_heads = 2; // GQA: kv_dim = 2 * (64/8) = 16, jauh dari dim=64
    tc.vocab_size = 50; tc.seq_len = 64;

    int head_size = tc.dim / tc.n_heads;
    int kv_dim = tc.n_kv_heads * head_size;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> disti(-100, 100);

    auto randf = [&](size_t n){ std::vector<float> v(n); for (auto& x : v) x = distf(rng); return v; };
    auto randi = [&](size_t n){ std::vector<int8_t> v(n); for (auto& x : v) x = static_cast<int8_t>(disti(rng)); return v; };

    auto& w = model.transformer_weights;
    w.token_embedding_table = randf((size_t)tc.vocab_size * tc.dim);
    w.rms_att_weight = randf((size_t)tc.n_layers * tc.dim);
    w.rms_ffn_weight = randf((size_t)tc.n_layers * tc.dim);
    w.rms_final_weight = randf(tc.dim);
    w.freq_cis_real = randf((size_t)tc.seq_len * head_size / 2);
    w.freq_cis_imag = randf((size_t)tc.seq_len * head_size / 2);

    w.wq = randi((size_t)tc.n_layers * tc.dim * tc.dim);       w.wq_alpha = randf(tc.n_layers);
    w.wk = randi((size_t)tc.n_layers * kv_dim * tc.dim);       w.wk_alpha = randf(tc.n_layers);
    w.wv = randi((size_t)tc.n_layers * kv_dim * tc.dim);       w.wv_alpha = randf(tc.n_layers);
    w.wo = randi((size_t)tc.n_layers * tc.dim * tc.dim);       w.wo_alpha = randf(tc.n_layers);
    w.w1 = randi((size_t)tc.n_layers * tc.hidden_dim * tc.dim); w.w1_alpha = randf(tc.n_layers);
    w.w2 = randi((size_t)tc.n_layers * tc.dim * tc.hidden_dim); w.w2_alpha = randf(tc.n_layers);
    w.w3 = randi((size_t)tc.n_layers * tc.hidden_dim * tc.dim); w.w3_alpha = randf(tc.n_layers);
    // wcls kosong -> shared classifier

    transformer_mixed::MixedPrecisionTransformer xf(model, transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8);

    // Threshold rendah (8) supaya eviction PASTI ke-trigger jauh sebelum
    // seq_len (64) habis -- ini yang memicu jalur buggy sebelumnya.
    kvcache::RocketKVEvictor evictor(/*evict_threshold=*/8, /*keep_top_n=*/4);

    std::cout << "Menjalankan 30 posisi dengan eviction aktif (threshold=8)...\n";
    for (int pos = 0; pos < 30; ++pos) {
        int token = pos % tc.vocab_size;
        float* logits = xf.forward_evictable(token, pos, evictor);
        if (!logits) { std::cout << "[FAIL] logits null di pos " << pos << "\n"; return 1; }
    }
    std::cout << "[PASS] 30 posisi selesai tanpa crash (eviction path GQA aman)\n";
    return 0;
}
