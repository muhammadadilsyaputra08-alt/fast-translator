#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tokenizer {

class Tokenizer {
public:
    // Loads a tokenizer.bin in the llama2.c format: int32 max_token_length,
    // then vocab_size entries of (float32 score, int32 len, char[len] text).
    static std::unique_ptr<Tokenizer> load(const std::string& path, int vocab_size);

    // Same as load(), but reads from an in-memory buffer (used when the
    // tokenizer bytes are embedded inside a .fllm file rather than a
    // standalone tokenizer.bin on disk).
    static std::unique_ptr<Tokenizer> load_from_bytes(const std::string& bytes, int vocab_size);

    // Encodes `text` into token ids using greedy BPE merging (highest-score
    // adjacent pair wins each round, same algorithm SentencePiece/llama2.c
    // use). If `add_bos` is true, prepends token id 1.
    std::vector<int> encode(const std::string& text, bool add_bos) const;

    // Converts a single token id to its display string. `prev_token` is
    // used only to strip the SentencePiece leading-space artifact right
    // after BOS, matching llama2.c's decode() convention.
    std::string decode(int prev_token, int token) const;

    int bos_id() const { return 1; }
    int eos_id() const { return 2; }

private:
    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::unordered_map<std::string, int> str_to_id_;
};

} // namespace tokenizer
