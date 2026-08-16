// Diagnostic-only experiment: measures whether attention or FFN weights are
// more sensitive to post-training ternary quantization, by running forward
// passes with only one group quantized at a time.
#include "fllm_parser.h"
#include "tokenizer.h"
#include "bitnet_kernel.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

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
void matmul_ternary(float* out, const float* x, const int8_t* w, float alpha, int din, int dout) {
    for (int o=0;o<dout;++o){
        float acc=0; const int8_t* row=w+(size_t)o*din;
        for (int i=0;i<din;++i){int8_t wv=row[i]; if(wv==1) acc+=x[i]; else if(wv==-1) acc-=x[i];}
        out[o]=acc*alpha;
    }
}

// mode: 0=all fp32 (baseline), 1=quantize attention only, 2=quantize FFN only, 3=quantize both
std::string run_experiment(const RawWeights& r, const tokenizer::Tokenizer& tok, int mode, const std::string& prompt, int max_tokens) {
    int dim=r.dim, hidden=r.hidden, n_layers=r.n_layers, n_heads=r.n_heads, seq_len=r.seq, vocab=r.vocab;
    int head_size = dim/n_heads;
    int64_t dim2 = (int64_t)dim*dim, dh=(int64_t)dim*hidden;

    // Quantize per-layer as needed
    std::vector<int8_t> q_wq,q_wk,q_wv,q_wo,q_w1,q_w2,q_w3;
    std::vector<float> a_wq,a_wk,a_wv,a_wo,a_w1,a_w2,a_w3;
    bool quant_att = (mode==1||mode==3);
    bool quant_ffn = (mode==2||mode==3);

    auto quant_all = [&](const std::vector<float>& src, int64_t per_layer, std::vector<int8_t>& out, std::vector<float>& alpha){
        out.resize(src.size()); alpha.resize(n_layers);
        for (int l=0;l<n_layers;++l){
            std::vector<float> slice(src.begin()+l*per_layer, src.begin()+(l+1)*per_layer);
            float a; auto t = bitnet::quantize_ternary(slice,&a);
            std::copy(t.begin(),t.end(), out.begin()+l*per_layer);
            alpha[l]=a;
        }
    };
    if (quant_att) { quant_all(r.wq,dim2,q_wq,a_wq); quant_all(r.wk,dim2,q_wk,a_wk); quant_all(r.wv,dim2,q_wv,a_wv); quant_all(r.wo,dim2,q_wo,a_wo); }
    if (quant_ffn) { quant_all(r.w1,dh,q_w1,a_w1); quant_all(r.w2,dh,q_w2,a_w2); quant_all(r.w3,dh,q_w3,a_w3); }

    std::vector<float> x(dim), xb(dim), xb2(dim), hb(hidden), hb2(hidden), q(dim), k(dim), v(dim);
    std::vector<float> att((size_t)n_heads*seq_len), logits(vocab);
    std::vector<float> key_cache((size_t)n_layers*seq_len*dim), value_cache((size_t)n_layers*seq_len*dim);

    auto prompt_tokens = tok.encode(prompt, true);
    std::string output;
    int pos=0, token=prompt_tokens[0], prev_token=token;

    while (pos < max_tokens + (int)prompt_tokens.size()) {
        const float* emb = r.token_embedding_table.data() + (size_t)token*dim;
        std::copy(emb, emb+dim, x.data());
        const float* fr = r.freq_r.data() + (size_t)pos*(head_size/2);
        const float* fi = r.freq_i.data() + (size_t)pos*(head_size/2);

        for (int l=0;l<n_layers;++l) {
            rmsnorm(xb.data(), x.data(), r.rms_att.data()+(size_t)l*dim, dim);
            if (quant_att) {
                matmul_ternary(q.data(), xb.data(), q_wq.data()+l*dim2, a_wq[l], dim, dim);
                matmul_ternary(k.data(), xb.data(), q_wk.data()+l*dim2, a_wk[l], dim, dim);
                matmul_ternary(v.data(), xb.data(), q_wv.data()+l*dim2, a_wv[l], dim, dim);
            } else {
                matmul_fp32(q.data(), xb.data(), r.wq.data()+l*dim2, dim, dim);
                matmul_fp32(k.data(), xb.data(), r.wk.data()+l*dim2, dim, dim);
                matmul_fp32(v.data(), xb.data(), r.wv.data()+l*dim2, dim, dim);
            }
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
            if (quant_att) matmul_ternary(xb2.data(), xb.data(), q_wo.data()+l*dim2, a_wo[l], dim, dim);
            else matmul_fp32(xb2.data(), xb.data(), r.wo.data()+l*dim2, dim, dim);
            for (int i=0;i<dim;++i) x[i]+=xb2[i];

            rmsnorm(xb.data(), x.data(), r.rms_ffn.data()+(size_t)l*dim, dim);
            if (quant_ffn) {
                matmul_ternary(hb.data(), xb.data(), q_w1.data()+l*dh, a_w1[l], dim, hidden);
                matmul_ternary(hb2.data(), xb.data(), q_w3.data()+l*dh, a_w3[l], dim, hidden);
            } else {
                matmul_fp32(hb.data(), xb.data(), r.w1.data()+l*dh, dim, hidden);
                matmul_fp32(hb2.data(), xb.data(), r.w3.data()+l*dh, dim, hidden);
            }
            for (int i=0;i<hidden;++i){ float vv=hb[i]; float silu=vv/(1.0f+std::exp(-vv)); hb[i]=silu*hb2[i]; }
            if (quant_ffn) matmul_ternary(xb.data(), hb.data(), q_w2.data()+l*dh, a_w2[l], hidden, dim);
            else matmul_fp32(xb.data(), hb.data(), r.w2.data()+l*dh, hidden, dim);
            for (int i=0;i<dim;++i) x[i]+=xb[i];
        }
        rmsnorm(x.data(), x.data(), r.rms_final.data(), dim);
        matmul_fp32(logits.data(), x.data(), r.token_embedding_table.data(), dim, vocab);

        int next;
        if (pos+1 < (int)prompt_tokens.size()) next = prompt_tokens[pos+1];
        else { int best=0; float bv=logits[0]; for(int i=1;i<vocab;++i) if(logits[i]>bv){bv=logits[i];best=i;} next=best; }

        if (pos >= (int)prompt_tokens.size()-1) output += tok.decode(prev_token, token);
        if (next == tok.eos_id()) break;
        prev_token=token; token=next; ++pos;
    }
    output += tok.decode(prev_token, token);
    return output;
}

int main() {
    auto r = load_raw("/mnt/user-data/uploads/stories15M.bin");
    auto tok = tokenizer::Tokenizer::load("/mnt/user-data/uploads/tokenizer.bin", r.vocab);

    std::cout << "=== mode 0: baseline fp32 ===\n" << run_experiment(r, *tok, 0, "Once upon a time", 40) << "\n\n";
    std::cout << "=== mode 1: attention quantized, FFN fp32 ===\n" << run_experiment(r, *tok, 1, "Once upon a time", 40) << "\n\n";
    std::cout << "=== mode 2: FFN quantized, attention fp32 ===\n" << run_experiment(r, *tok, 2, "Once upon a time", 40) << "\n\n";
    std::cout << "=== mode 3: both quantized (current pack_fllm behavior) ===\n" << run_experiment(r, *tok, 3, "Once upon a time", 40) << "\n\n";
    return 0;
}
