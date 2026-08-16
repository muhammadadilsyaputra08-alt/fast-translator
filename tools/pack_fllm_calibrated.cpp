// Calibrated .fllm packer using GPTQ-style ternary quantization.
//
// Unlike pack_fllm.cpp (naive per-weight absmean threshold), this collects
// real activation statistics by running the unquantized fp32 model over a
// handful of calibration prompts, then quantizes each weight matrix using
// sequential error-compensated rounding (gptq_ternary.h) so that rounding
// error is absorbed by not-yet-quantized weights instead of every weight
// independently contributing uncorrected error.
#include "fllm_format.h"
#include "fllm_parser.h"
#include "tokenizer.h"
#include "bitnet_kernel.h"
#include "gptq_ternary.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace {

struct RawWeights {
    std::vector<float> token_embedding_table, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, freq_r, freq_i;
    int32_t dim, hidden, n_layers, n_heads, vocab, seq;
};

RawWeights load_raw(const std::string& path) {
    std::ifstream cf(path, std::ios::binary);
    RawWeights r{};
    int32_t n_kv;
    cf.read((char*)&r.dim,4); cf.read((char*)&r.hidden,4); cf.read((char*)&r.n_layers,4);
    cf.read((char*)&r.n_heads,4); cf.read((char*)&n_kv,4); cf.read((char*)&r.vocab,4); cf.read((char*)&r.seq,4);
    if (r.vocab < 0) r.vocab = -r.vocab;
    int head_size = r.dim / r.n_heads;
    auto rd = [&](size_t n){ std::vector<float> v(n); cf.read((char*)v.data(), n*4); return v; };
    r.token_embedding_table = rd((size_t)r.vocab * r.dim);
    r.rms_att = rd((size_t)r.n_layers * r.dim);
    r.wq = rd((size_t)r.n_layers * r.dim * r.dim);
    r.wk = rd((size_t)r.n_layers * r.dim * r.dim);
    r.wv = rd((size_t)r.n_layers * r.dim * r.dim);
    r.wo = rd((size_t)r.n_layers * r.dim * r.dim);
    r.rms_ffn = rd((size_t)r.n_layers * r.dim);
    r.w1 = rd((size_t)r.n_layers * r.dim * r.hidden);
    r.w2 = rd((size_t)r.n_layers * r.dim * r.hidden);
    r.w3 = rd((size_t)r.n_layers * r.dim * r.hidden);
    r.rms_final = rd(r.dim);
    r.freq_r = rd((size_t)r.seq * (head_size/2));
    r.freq_i = rd((size_t)r.seq * (head_size/2));
    return r;
}

void rmsnorm(float* out, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) ss += x[i]*x[i];
    ss = 1.0f/std::sqrt(ss/size + 1e-5f);
    for (int i = 0; i < size; ++i) out[i] = weight[i]*(x[i]*ss);
}
void softmax(float* x, int n) {
    float m = x[0]; for (int i=1;i<n;++i) m=std::max(m,x[i]);
    float s=0; for(int i=0;i<n;++i){x[i]=std::exp(x[i]-m); s+=x[i];}
    for(int i=0;i<n;++i) x[i]/=s;
}
void matmul_fp32(float* out, const float* x, const float* w, int din, int dout) {
    for (int o=0;o<dout;++o){float s=0; const float* row=w+(size_t)o*din; for(int i=0;i<din;++i) s+=row[i]*x[i]; out[o]=s;}
}

struct ActivationSink {
    std::vector<std::vector<std::vector<float>>> qkv_input, wo_input, ffn_input, w2_input;
    void init(int n_layers) {
        qkv_input.resize(n_layers); wo_input.resize(n_layers);
        ffn_input.resize(n_layers); w2_input.resize(n_layers);
    }
};

void run_calibration_forward(const RawWeights& r, const tokenizer::Tokenizer& tok,
                              const std::string& prompt, ActivationSink& sink) {
    int dim=r.dim, hidden=r.hidden, n_layers=r.n_layers, n_heads=r.n_heads, seq_len=r.seq;
    int head_size = dim/n_heads;

    std::vector<float> x(dim), xb(dim), xb2(dim), hb(hidden), hb2(hidden), q(dim), k(dim), v(dim);
    std::vector<float> att((size_t)n_heads*seq_len);
    std::vector<float> key_cache((size_t)n_layers*seq_len*dim), value_cache((size_t)n_layers*seq_len*dim);

    auto tokens = tok.encode(prompt, true);
    for (int pos = 0; pos < (int)tokens.size() && pos < seq_len; ++pos) {
        int token = tokens[pos];
        const float* emb = r.token_embedding_table.data() + (size_t)token*dim;
        std::copy(emb, emb+dim, x.data());
        const float* fr = r.freq_r.data() + (size_t)pos*(head_size/2);
        const float* fi = r.freq_i.data() + (size_t)pos*(head_size/2);

        for (int l = 0; l < n_layers; ++l) {
            rmsnorm(xb.data(), x.data(), r.rms_att.data()+(size_t)l*dim, dim);
            sink.qkv_input[l].emplace_back(xb.begin(), xb.end());

            int64_t dim2 = (int64_t)dim*dim;
            matmul_fp32(q.data(), xb.data(), r.wq.data()+l*dim2, dim, dim);
            matmul_fp32(k.data(), xb.data(), r.wk.data()+l*dim2, dim, dim);
            matmul_fp32(v.data(), xb.data(), r.wv.data()+l*dim2, dim, dim);

            for (int h=0;h<n_heads;++h){
                float* qh=q.data()+h*head_size; float* kh=k.data()+h*head_size;
                for (int i=0;i<head_size/2;++i){
                    float fcr=fr[i], fci=fi[i];
                    float q0=qh[2*i],q1=qh[2*i+1]; qh[2*i]=q0*fcr-q1*fci; qh[2*i+1]=q0*fci+q1*fcr;
                    float k0=kh[2*i],k1=kh[2*i+1]; kh[2*i]=k0*fcr-k1*fci; kh[2*i+1]=k0*fci+k1*fcr;
                }
            }
            size_t off=(size_t)l*seq_len*dim;
            std::copy(k.begin(),k.end(), key_cache.begin()+off+(size_t)pos*dim);
            std::copy(v.begin(),v.end(), value_cache.begin()+off+(size_t)pos*dim);
            for (int h=0;h<n_heads;++h){
                float* qh=q.data()+h*head_size; float* ar=att.data()+h*seq_len;
                for (int t=0;t<=pos;++t){
                    const float* kt=key_cache.data()+off+(size_t)t*dim+h*head_size;
                    float sc=0; for(int i=0;i<head_size;++i) sc+=qh[i]*kt[i];
                    ar[t]=sc/std::sqrt((float)head_size);
                }
                softmax(ar,pos+1);
                float* oh=xb.data()+h*head_size; std::fill(oh,oh+head_size,0.0f);
                for (int t=0;t<=pos;++t){
                    const float* vt=value_cache.data()+off+(size_t)t*dim+h*head_size;
                    float a=ar[t]; for(int i=0;i<head_size;++i) oh[i]+=a*vt[i];
                }
            }
            sink.wo_input[l].emplace_back(xb.begin(), xb.end());
            matmul_fp32(xb2.data(), xb.data(), r.wo.data()+l*dim2, dim, dim);
            for (int i=0;i<dim;++i) x[i]+=xb2[i];

            rmsnorm(xb.data(), x.data(), r.rms_ffn.data()+(size_t)l*dim, dim);
            sink.ffn_input[l].emplace_back(xb.begin(), xb.end());
            int64_t dh=(int64_t)dim*hidden;
            matmul_fp32(hb.data(), xb.data(), r.w1.data()+l*dh, dim, hidden);
            matmul_fp32(hb2.data(), xb.data(), r.w3.data()+l*dh, dim, hidden);
            for (int i=0;i<hidden;++i){ float vv=hb[i]; float silu=vv/(1.0f+std::exp(-vv)); hb[i]=silu*hb2[i]; }
            sink.w2_input[l].emplace_back(hb.begin(), hb.end());
            matmul_fp32(xb.data(), hb.data(), r.w2.data()+l*dh, hidden, dim);
            for (int i=0;i<dim;++i) x[i]+=xb[i];
        }
    }
}

// Quantizes one [n_layers, d_out, d_in] tensor, layer by layer, using
// per-layer GPTQ compensation against that layer's calibration Hessian.
void quantize_tensor_gptq(
    const std::vector<float>& raw, int n_layers, int64_t per_layer_count, int d_out, int d_in,
    const std::vector<std::vector<std::vector<float>>>& activations_per_layer,
    std::vector<int8_t>& out_codes, std::vector<float>& out_alpha) {

    out_codes.resize(raw.size());
    out_alpha.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        std::vector<float> layer_w(raw.begin() + l * per_layer_count, raw.begin() + (l + 1) * per_layer_count);

        float alpha = 0.0f;
        bitnet::quantize_ternary(layer_w, &alpha); // reuse absmean rule just to derive the scale
        out_alpha[l] = alpha;

        auto hess = gptq::Hessian::from_activations(activations_per_layer[l], d_in, 0.01);

        std::vector<float> scratch = layer_w; // mutated in place by GPTQ compensation
        std::vector<int8_t> codes;
        gptq::quantize_ternary_gptq(scratch, d_out, d_in, alpha, hess, codes);

        std::copy(codes.begin(), codes.end(), out_codes.begin() + l * per_layer_count);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <checkpoint.bin> <tokenizer.bin> <output.fllm>\n";
        return 1;
    }
    std::string checkpoint_path = argv[1], tokenizer_path = argv[2], output_path = argv[3];

    auto r = load_raw(checkpoint_path);
    std::cout << "checkpoint: dim=" << r.dim << " hidden=" << r.hidden << " layers=" << r.n_layers
              << " heads=" << r.n_heads << " vocab=" << r.vocab << " seq_len=" << r.seq << "\n";

    auto tok = tokenizer::Tokenizer::load(tokenizer_path, r.vocab);
    if (!tok) { std::cerr << "failed to load tokenizer\n"; return 1; }

    std::vector<std::string> calib_prompts = {
        "Once upon a time, there was a little girl named",
        "The little dog ran across the",
        "One day, a boy found a big red",
        "She was very happy because",
        "The sun was shining and the birds were",
        "Once upon a time, in a small village, there lived",
        "He looked at the sky and saw a",
        "The cat sat on the mat and watched the",
    };

    std::cout << "running calibration forward pass over " << calib_prompts.size() << " prompts...\n";
    ActivationSink sink;
    sink.init(r.n_layers);
    for (const auto& p : calib_prompts) run_calibration_forward(r, *tok, p, sink);

    int total_samples = 0;
    for (auto& v : sink.qkv_input) total_samples += (int)v.size();
    std::cout << "collected " << total_samples << " calibration samples per layer-group\n";

    fllm::FllmModel model;
    model.header.model_type = 1;
    model.has_transformer = true;
    model.transformer_config = {r.dim, r.hidden, r.n_layers, r.n_heads, r.n_heads, r.vocab, r.seq};

    auto& w = model.transformer_weights;
    w.token_embedding_table = r.token_embedding_table;
    w.rms_att_weight = r.rms_att;
    w.rms_ffn_weight = r.rms_ffn;
    w.rms_final_weight = r.rms_final;
    w.freq_cis_real = r.freq_r;
    w.freq_cis_imag = r.freq_i;

    std::cout << "quantizing wq (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.wq, r.n_layers, (int64_t)r.dim*r.dim, r.dim, r.dim, sink.qkv_input, w.wq, w.wq_alpha);
    std::cout << "quantizing wk (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.wk, r.n_layers, (int64_t)r.dim*r.dim, r.dim, r.dim, sink.qkv_input, w.wk, w.wk_alpha);
    std::cout << "quantizing wv (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.wv, r.n_layers, (int64_t)r.dim*r.dim, r.dim, r.dim, sink.qkv_input, w.wv, w.wv_alpha);
    std::cout << "quantizing wo (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.wo, r.n_layers, (int64_t)r.dim*r.dim, r.dim, r.dim, sink.wo_input, w.wo, w.wo_alpha);
    std::cout << "quantizing w1 (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.w1, r.n_layers, (int64_t)r.dim*r.hidden, r.hidden, r.dim, sink.ffn_input, w.w1, w.w1_alpha);
    std::cout << "quantizing w3 (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.w3, r.n_layers, (int64_t)r.dim*r.hidden, r.hidden, r.dim, sink.ffn_input, w.w3, w.w3_alpha);
    std::cout << "quantizing w2 (GPTQ-calibrated)...\n";
    quantize_tensor_gptq(r.w2, r.n_layers, (int64_t)r.dim*r.hidden, r.dim, r.hidden, sink.w2_input, w.w2, w.w2_alpha);

    std::ifstream tf(tokenizer_path, std::ios::binary);
    model.tokenizer_json.assign((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());

    model.kv_config = {true, false, 16, 5.0f, 512};
    model.medusa_config = {4, static_cast<uint32_t>(r.dim), 0.8f, 0.5f};
    model.grammar_gbnf = "";
    model.weights = {};

    if (!fllm::write_fllm(output_path, model)) {
        std::cerr << "failed to write " << output_path << "\n";
        return 1;
    }
    std::cout << "wrote " << output_path << "\n";
    return 0;
}
