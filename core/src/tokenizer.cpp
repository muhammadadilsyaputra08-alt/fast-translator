#include "tokenizer.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <iterator>

namespace tokenizer {

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& path, int vocab_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[tokenizer] cannot open " << path << "\n";
        return nullptr;
    }
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_from_bytes(bytes, vocab_size);
}

std::unique_ptr<Tokenizer> Tokenizer::load_from_bytes(const std::string& bytes, int vocab_size) {
    if (bytes.size() < 4) return nullptr;
    size_t pos = 0;
    auto read_i32 = [&]() -> int32_t {
        int32_t v;
        std::memcpy(&v, bytes.data() + pos, 4);
        pos += 4;
        return v;
    };
    auto read_f32 = [&]() -> float {
        float v;
        std::memcpy(&v, bytes.data() + pos, 4);
        pos += 4;
        return v;
    };

    int32_t max_token_length = read_i32();
    (void)max_token_length;

    auto tok = std::unique_ptr<Tokenizer>(new Tokenizer());
    tok->vocab_.resize(vocab_size);
    tok->vocab_scores_.resize(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        if (pos + 8 > bytes.size()) {
            std::cerr << "[tokenizer] truncated buffer at vocab entry " << i << "\n";
            return nullptr;
        }
        float score = read_f32();
        int32_t len = read_i32();
        if (pos + static_cast<size_t>(len) > bytes.size()) {
            std::cerr << "[tokenizer] truncated buffer reading string at entry " << i << "\n";
            return nullptr;
        }
        std::string s = bytes.substr(pos, len);
        pos += len;
        tok->vocab_[i] = s;
        tok->vocab_scores_[i] = score;
        tok->str_to_id_[s] = i;
    }

    return tok;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;
    if (add_bos) tokens.push_back(bos_id());

    // Step 1: seed with one token per UTF-8 codepoint (falling back to raw
    // byte tokens "<0xXX>" for anything not directly in the vocab).
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int char_len = 1;
        if ((c & 0x80) == 0x00) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        char_len = std::min<int>(char_len, static_cast<int>(text.size() - i));

        std::string piece = text.substr(i, char_len);
        auto it = str_to_id_.find(piece);
        if (it != str_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            // Byte fallback: encode each raw byte as "<0xXX>".
            for (int b = 0; b < char_len; ++b) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(text[i + b]));
                auto bit = str_to_id_.find(buf);
                if (bit != str_to_id_.end()) tokens.push_back(bit->second);
            }
        }
        i += char_len;
    }

    // Step 2: greedy BPE — repeatedly merge the single best-scoring
    // adjacent pair (by the merged string's vocab score) until no more
    // merges are found in the vocab.
    while (tokens.size() >= 2) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_id = -1;

        for (size_t t = 0; t + 1 < tokens.size(); ++t) {
            std::string merged = vocab_[tokens[t]] + vocab_[tokens[t + 1]];
            auto it = str_to_id_.find(merged);
            if (it != str_to_id_.end() && vocab_scores_[it->second] > best_score) {
                best_score = vocab_scores_[it->second];
                best_idx = static_cast<int>(t);
                best_id = it->second;
            }
        }

        if (best_idx == -1) break; // no more valid merges

        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    std::string piece = vocab_[token];
    // SentencePiece convention: strip a single leading space right after BOS.
    if (prev_token == bos_id() && !piece.empty() && piece[0] == ' ') {
        piece = piece.substr(1);
    }
    // Raw byte fallback tokens look like "<0xXX>" — render as the actual byte.
    if (piece.size() == 6 && piece.rfind("<0x", 0) == 0 && piece.back() == '>') {
        unsigned int byte_val = 0;
        std::sscanf(piece.c_str(), "<0x%02X>", &byte_val);
        return std::string(1, static_cast<char>(byte_val));
    }
    return piece;
}

} // namespace tokenizer
