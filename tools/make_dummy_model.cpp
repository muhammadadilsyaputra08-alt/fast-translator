// Generates a small dummy .fllm file for end-to-end testing on a real
// device — NOT a real trained model, just valid data that exercises the
// full load() -> generate() pipeline (header, checksum, all sections).
//
// Build: g++ -std=c++17 -I ../core/include ../core/src/fllm_parser.cpp make_dummy_model.cpp -o make_dummy_model
// Run:   ./make_dummy_model demo.fllm
#include "fllm_parser.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <output_path.fllm>\n";
        return 1;
    }

    fllm::FllmModel m;
    m.header.model_type = 1; // Instruct
    m.tokenizer_json = R"({"vocab":{"hello":1,"world":2,"halo":3,"dunia":4}})";
    m.embeddings = {0.12f, -0.34f, 0.56f, -0.78f, 0.11f, 0.22f};
    m.grammar_gbnf = "";
    m.kv_config = {/*minicache=*/true, /*rocketkv=*/false, /*compress_every=*/16, /*merge_ratio=*/5.0f, /*evict=*/512};
    m.medusa_config = {/*heads=*/4, /*hidden_dim=*/128, /*epsilon=*/0.8f, /*delta=*/0.5f};
    m.weights = {1, -1, 0, 1, 1, -1, 0, 0, 1, -1};

    bool ok = fllm::write_fllm(argv[1], m);
    if (!ok) {
        std::cerr << "failed to write " << argv[1] << "\n";
        return 1;
    }
    std::cout << "wrote dummy model to " << argv[1] << "\n";
    return 0;
}
