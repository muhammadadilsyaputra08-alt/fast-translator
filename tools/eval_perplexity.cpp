// Milestone 2: objective perplexity evaluator.
//
// Computes cross-entropy / perplexity on held-out text (NOT the calibration
// prompts used for packing) for either the fp32 baseline or a quantized
// .fllm model, so quantization quality can be compared with a real metric
// instead of eyeballing generated text.
//
// Build: g++ -std=c++17 -O2 -I ../core/include \
//   ../core/src/transformer.cpp ../core/src/transformer_bitnet.cpp \
//   ../core/src/tokenizer.cpp ../core/src/fllm_parser.cpp eval_perplexity.cpp -o eval_perplexity
// Run:   ./eval_perplexity fp32   stories15M.bin tokenizer.bin
//        ./eval_perplexity fllm  model.fllm
#include "transformer.h"
#include "transformer_bitnet.h"
#include "transformer_mixed.h"
#include "tokenizer.h"
#include "fllm_parser.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

// Held-out evaluation prompts — deliberately DIFFERENT from the calibration
// prompts used in pack_fllm_calibrated.cpp, so this measures generalization,
// not just how well the model reconstructs the exact calibration set.
static const std::vector<std::string> kEvalPrompts = {
    "The little rabbit hopped through the garden and found a",
    "Mom said it was time for bed, so Tim",
    "The children played happily in the park until the",
    "A brave knight walked into the dark forest to find the",
    "Every morning, the old man fed the birds in his",
};

double softmax_log_prob(const float* logits, int vocab_size, int target_token) {
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) max_val = std::max(max_val, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) sum += std::exp(logits[i] - max_val);
    double log_sum = std::log(sum) + max_val;
    return logits[target_token] - log_sum; // log p(target)
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage:\n  " << argv[0] << " fp32 <checkpoint.bin> <tokenizer.bin>\n"
                  << "  " << argv[0] << " fllm <model.fllm>\n";
        return 1;
    }
    std::string mode = argv[1];

    double total_log_prob = 0.0;
    int total_tokens = 0;

    if (mode == "fp32") {
        auto model = transformer::Transformer::load(argv[2]);
        auto tok = tokenizer::Tokenizer::load(argv[3], model->config().vocab_size);
        if (!model || !tok) { std::cerr << "failed to load\n"; return 1; }

        for (const auto& prompt : kEvalPrompts) {
            auto tokens = tok->encode(prompt, true);
            for (int pos = 0; pos + 1 < (int)tokens.size(); ++pos) {
                float* logits = model->forward(tokens[pos], pos);
                total_log_prob += softmax_log_prob(logits, model->config().vocab_size, tokens[pos + 1]);
                total_tokens++;
            }
        }
    } else if (mode == "fllm") {
        auto loaded = fllm::read_fllm(argv[2]);
        if (!loaded || !loaded->has_transformer) { std::cerr << "failed to load .fllm or no transformer section\n"; return 1; }
        auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);
        transformer_bitnet::BitnetTransformer model(*loaded);

        for (const auto& prompt : kEvalPrompts) {
            auto tokens = tok->encode(prompt, true);
            for (int pos = 0; pos + 1 < (int)tokens.size(); ++pos) {
                float* logits = model.forward(tokens[pos], pos);
                total_log_prob += softmax_log_prob(logits, loaded->transformer_config.vocab_size, tokens[pos + 1]);
                total_tokens++;
            }
        }
    } else if (mode == "mixed" || mode == "int8all") {
        auto loaded = fllm::read_fllm(argv[2]);
        if (!loaded || !loaded->has_transformer) { std::cerr << "failed to load .fllm or no transformer section\n"; return 1; }
        auto tok = tokenizer::Tokenizer::load_from_bytes(loaded->tokenizer_json, loaded->transformer_config.vocab_size);
        auto ffn_prec = (mode == "int8all") ? transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Int8
                                             : transformer_mixed::MixedPrecisionTransformer::FfnPrecision::Ternary;
        transformer_mixed::MixedPrecisionTransformer model(*loaded, ffn_prec);

        for (const auto& prompt : kEvalPrompts) {
            auto tokens = tok->encode(prompt, true);
            for (int pos = 0; pos + 1 < (int)tokens.size(); ++pos) {
                float* logits = model.forward(tokens[pos], pos);
                total_log_prob += softmax_log_prob(logits, loaded->transformer_config.vocab_size, tokens[pos + 1]);
                total_tokens++;
            }
        }
    } else {
        std::cerr << "unknown mode: " << mode << "\n";
        return 1;
    }

    double avg_neg_log_prob = -total_log_prob / total_tokens;
    double perplexity = std::exp(avg_neg_log_prob);

    std::cout << "tokens evaluated: " << total_tokens << "\n";
    std::cout << "average cross-entropy (nats/token): " << avg_neg_log_prob << "\n";
    std::cout << "perplexity: " << perplexity << "\n";
    return 0;
}
