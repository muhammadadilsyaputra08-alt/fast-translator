#include "fllm_parser.h"
#include "sha256.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace fllm {

namespace {

// RAII wrapper around mmap() -- used by read_fllm() below so the ~1.5GB
// .fllm file is mapped by the OS instead of malloc'd + fully read into a
// std::vector<uint8_t>. Mapped pages are file-backed and clean (never
// written to), so the OS can reclaim them under memory pressure without
// ever touching swap -- unlike a plain heap buffer, which must stay fully
// resident until freed. This was the main contributor to the ~2.7GB peak
// RSS that triggered a LOW_MEMORY kill on-device (confirmed via logcat:
// "reason=3 (LOW_MEMORY) ... pss=2,7GB").
struct MappedFile {
    void* data = nullptr;
    size_t size = 0;

    bool open_file(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < 0) {
            ::close(fd);
            return false;
        }
        size = static_cast<size_t>(st.st_size);
        if (size == 0) {
            ::close(fd);
            return false;
        }

        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd); // safe to close once mmap'd -- the mapping stays valid
        if (data == MAP_FAILED) {
            data = nullptr;
            return false;
        }
        return true;
    }

    ~MappedFile() {
        if (data) ::munmap(data, size);
    }

    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back((v >> (i * 8)) & 0xff);
}
void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back((v >> (i * 8)) & 0xff);
}
void write_i32(std::vector<uint8_t>& buf, int32_t v) { write_u32(buf, static_cast<uint32_t>(v)); }
void write_f32(std::vector<uint8_t>& buf, float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    write_u32(buf, bits);
}
void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}
void write_bytes(std::vector<uint8_t>& buf, const std::vector<int8_t>& s) {
    write_u64(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}
void write_floats(std::vector<uint8_t>& buf, const std::vector<float>& v) {
    write_u64(buf, v.size());
    for (float f : v) write_f32(buf, f);
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
uint32_t read_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
int32_t read_i32(const uint8_t* p) { return static_cast<int32_t>(read_u32(p)); }
float read_f32(const uint8_t* p) {
    uint32_t bits = read_u32(p);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Reads a length-prefixed float vector, advancing `p` past it.
std::vector<float> read_floats_adv(const uint8_t*& p) {
    uint64_t count = read_u64(p); p += 8;
    std::vector<float> v(count);
    for (uint64_t i = 0; i < count; ++i) { v[i] = read_f32(p); p += 4; }
    return v;
}
// Reads a length-prefixed int8 vector, advancing `p` past it.
std::vector<int8_t> read_bytes_adv(const uint8_t*& p) {
    uint64_t count = read_u64(p); p += 8;
    std::vector<int8_t> v(reinterpret_cast<const int8_t*>(p), reinterpret_cast<const int8_t*>(p) + count);
    p += count;
    return v;
}

void append(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

} // namespace

bool write_fllm(const std::string& path, const FllmModel& model) {
    std::vector<uint8_t> tokenizer_buf;
    write_str(tokenizer_buf, model.tokenizer_json);

    std::vector<uint8_t> embeddings_buf;
    write_floats(embeddings_buf, model.embeddings);

    std::vector<uint8_t> grammar_buf;
    write_str(grammar_buf, model.grammar_gbnf);

    std::vector<uint8_t> kv_buf;
    kv_buf.push_back(model.kv_config.enable_minicache ? 1 : 0);
    kv_buf.push_back(model.kv_config.enable_rocketkv ? 1 : 0);
    write_u32(kv_buf, model.kv_config.compress_every);
    write_f32(kv_buf, model.kv_config.minicache_merge_ratio);
    write_u32(kv_buf, model.kv_config.rocketkv_evict_threshold);

    std::vector<uint8_t> medusa_buf;
    write_u32(medusa_buf, model.medusa_config.num_heads);
    write_u32(medusa_buf, model.medusa_config.hidden_dim);
    write_f32(medusa_buf, model.medusa_config.epsilon);
    write_f32(medusa_buf, model.medusa_config.delta);

    std::vector<uint8_t> weights_buf;
    write_bytes(weights_buf, model.weights);

    std::vector<uint8_t> tconfig_buf;
    std::vector<uint8_t> tweights_buf;
    if (model.has_transformer) {
        const auto& tc = model.transformer_config;
        write_i32(tconfig_buf, tc.dim);
        write_i32(tconfig_buf, tc.hidden_dim);
        write_i32(tconfig_buf, tc.n_layers);
        write_i32(tconfig_buf, tc.n_heads);
        write_i32(tconfig_buf, tc.n_kv_heads);
        write_i32(tconfig_buf, tc.vocab_size);
        write_i32(tconfig_buf, tc.seq_len);

        const auto& w = model.transformer_weights;
        write_floats(tweights_buf, w.token_embedding_table);
        write_floats(tweights_buf, w.rms_att_weight);
        write_floats(tweights_buf, w.rms_ffn_weight);
        write_floats(tweights_buf, w.rms_final_weight);
        write_floats(tweights_buf, w.freq_cis_real);
        write_floats(tweights_buf, w.freq_cis_imag);

        write_bytes(tweights_buf, w.wq); write_floats(tweights_buf, w.wq_alpha);
        write_bytes(tweights_buf, w.wk); write_floats(tweights_buf, w.wk_alpha);
        write_bytes(tweights_buf, w.wv); write_floats(tweights_buf, w.wv_alpha);
        write_bytes(tweights_buf, w.wo); write_floats(tweights_buf, w.wo_alpha);
        write_bytes(tweights_buf, w.w1); write_floats(tweights_buf, w.w1_alpha);
        write_bytes(tweights_buf, w.w2); write_floats(tweights_buf, w.w2_alpha);
        write_bytes(tweights_buf, w.w3); write_floats(tweights_buf, w.w3_alpha);
        write_floats(tweights_buf, w.wcls); // empty vector if shared classifier
    }

    FllmHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.model_type = model.header.model_type;

    uint64_t offset = sizeof(FllmHeader);
    hdr.weights_offset = offset;               offset += weights_buf.size();
    hdr.tokenizer_offset = offset;             offset += tokenizer_buf.size();
    hdr.embeddings_offset = offset;            offset += embeddings_buf.size();
    hdr.grammar_offset = offset;               offset += grammar_buf.size();
    hdr.kv_cache_offset = offset;              offset += kv_buf.size();
    hdr.medusa_offset = offset;                offset += medusa_buf.size();
    hdr.transformer_config_offset = model.has_transformer ? offset : 0;
    offset += tconfig_buf.size();
    hdr.transformer_weights_offset = model.has_transformer ? offset : 0;
    offset += tweights_buf.size();

    std::vector<uint8_t> payload;
    append(payload, weights_buf);
    append(payload, tokenizer_buf);
    append(payload, embeddings_buf);
    append(payload, grammar_buf);
    append(payload, kv_buf);
    append(payload, medusa_buf);
    append(payload, tconfig_buf);
    append(payload, tweights_buf);
    hdr.checksum = sha256::hash_first8(payload.data(), payload.size());

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    return out.good();
}

bool validate_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    FllmHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in) return false;
    return hdr.magic == kMagic && hdr.version == kVersion;
}

std::optional<FllmModel> read_fllm(const std::string& path) {
    MappedFile mapped;
    if (!mapped.open_file(path)) return std::nullopt;
    if (mapped.size < sizeof(FllmHeader)) return std::nullopt;

    const uint8_t* buf = reinterpret_cast<const uint8_t*>(mapped.data);

    FllmHeader hdr{};
    std::memcpy(&hdr, buf, sizeof(hdr));

    if (hdr.magic != kMagic) {
        std::cerr << "[fllm] invalid magic number\n";
        return std::nullopt;
    }
    if (hdr.version != kVersion) {
        std::cerr << "[fllm] unsupported version (file is v" << std::hex << hdr.version
                   << ", reader expects v" << kVersion << std::dec << ")\n";
        return std::nullopt;
    }

    uint64_t computed = sha256::hash_first8(buf + sizeof(hdr), mapped.size - sizeof(hdr));
    if (computed != hdr.checksum) {
        std::cerr << "[fllm] checksum mismatch: file is corrupt\n";
        return std::nullopt;
    }

    FllmModel model;
    model.header = hdr;

    const uint8_t* base = buf;

    // NOTE: for the TinyLlama INT8 pack path (tools/pack_fllm.cpp) this field is
    // always empty (`model.weights = {}`), so parsing it costs ~nothing for our
    // actual model.fllm. It's kept here because test_fase1.cpp asserts round-trip
    // equality on it as part of the .fllm format contract, even though no runtime
    // consumer (engine.cpp/transformer_mixed.cpp/transformer_bitnet.cpp) reads it.
    {
        const uint8_t* p = base + hdr.weights_offset;
        uint64_t len = read_u64(p);
        p += 8;
        model.weights.assign(reinterpret_cast<const int8_t*>(p), reinterpret_cast<const int8_t*>(p) + len);
    }
    {
        const uint8_t* p = base + hdr.tokenizer_offset;
        uint64_t len = read_u64(p);
        p += 8;
        model.tokenizer_json.assign(reinterpret_cast<const char*>(p), len);
    }
    {
        const uint8_t* p = base + hdr.embeddings_offset;
        uint64_t count = read_u64(p);
        p += 8;
        model.embeddings.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            model.embeddings[i] = read_f32(p);
            p += 4;
        }
    }
    {
        const uint8_t* p = base + hdr.grammar_offset;
        uint64_t len = read_u64(p);
        p += 8;
        model.grammar_gbnf.assign(reinterpret_cast<const char*>(p), len);
    }
    {
        const uint8_t* p = base + hdr.kv_cache_offset;
        model.kv_config.enable_minicache = p[0] != 0;
        model.kv_config.enable_rocketkv = p[1] != 0;
        p += 2;
        model.kv_config.compress_every = read_u32(p); p += 4;
        model.kv_config.minicache_merge_ratio = read_f32(p); p += 4;
        model.kv_config.rocketkv_evict_threshold = read_u32(p);
    }
    {
        const uint8_t* p = base + hdr.medusa_offset;
        model.medusa_config.num_heads = read_u32(p); p += 4;
        model.medusa_config.hidden_dim = read_u32(p); p += 4;
        model.medusa_config.epsilon = read_f32(p); p += 4;
        model.medusa_config.delta = read_f32(p);
    }

    if (hdr.transformer_config_offset != 0 && hdr.transformer_weights_offset != 0) {
        model.has_transformer = true;

        const uint8_t* p = base + hdr.transformer_config_offset;
        auto& tc = model.transformer_config;
        tc.dim = read_i32(p);         p += 4;
        tc.hidden_dim = read_i32(p);  p += 4;
        tc.n_layers = read_i32(p);    p += 4;
        tc.n_heads = read_i32(p);     p += 4;
        tc.n_kv_heads = read_i32(p);  p += 4;
        tc.vocab_size = read_i32(p);  p += 4;
        tc.seq_len = read_i32(p);

        const uint8_t* wp = base + hdr.transformer_weights_offset;
        auto& w = model.transformer_weights;
        w.token_embedding_table = read_floats_adv(wp);
        w.rms_att_weight = read_floats_adv(wp);
        w.rms_ffn_weight = read_floats_adv(wp);
        w.rms_final_weight = read_floats_adv(wp);
        w.freq_cis_real = read_floats_adv(wp);
        w.freq_cis_imag = read_floats_adv(wp);

        w.wq = read_bytes_adv(wp); w.wq_alpha = read_floats_adv(wp);
        w.wk = read_bytes_adv(wp); w.wk_alpha = read_floats_adv(wp);
        w.wv = read_bytes_adv(wp); w.wv_alpha = read_floats_adv(wp);
        w.wo = read_bytes_adv(wp); w.wo_alpha = read_floats_adv(wp);
        w.w1 = read_bytes_adv(wp); w.w1_alpha = read_floats_adv(wp);
        w.w2 = read_bytes_adv(wp); w.w2_alpha = read_floats_adv(wp);
        w.w3 = read_bytes_adv(wp); w.w3_alpha = read_floats_adv(wp);
        w.wcls = read_floats_adv(wp); // empty if this model uses a shared classifier
    }

    return model;
}

} // namespace fllm
