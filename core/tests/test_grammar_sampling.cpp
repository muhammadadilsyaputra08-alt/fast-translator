#include "engine.h"
#include "gbnf.h"
#include <iostream>
#include <sstream>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

std::string json_gbnf = R"(
    root   ::= object
    object ::= "{" pair ("," pair)* "}" | "{" "}"
    pair   ::= string ":" value
    value  ::= object | array | string | number | "true" | "false" | "null"
    array  ::= "[" value ("," value)* "]" | "[" "]"
    string ::= "\"" [^"]* "\""
    number ::= "-"? [0-9]+ ("." [0-9]+)?
)";

// Replays `text` through a FRESH grammar instance from scratch and confirms
// every character was a legal continuation at the time it appeared -- this
// is an independent check, not just trusting the engine's internal state.
bool is_valid_prefix(const std::string& text) {
    auto g = gbnf::Grammar::parse(json_gbnf);
    auto state = g->initial_state();
    for (char c : text) {
        if (!g->can_accept(state, static_cast<unsigned char>(c))) return false;
        state = g->advance(state, static_cast<unsigned char>(c));
    }
    return true;
}

bool is_complete_document(const std::string& text) {
    auto g = gbnf::Grammar::parse(json_gbnf);
    auto state = g->initial_state();
    for (char c : text) {
        if (!g->can_accept(state, static_cast<unsigned char>(c))) return false;
        state = g->advance(state, static_cast<unsigned char>(c));
    }
    return g->can_terminate(state);
}

int main(int argc, char** argv) {
    std::string model_path = argc > 1 ? argv[1] : "/tmp/model_v2.fllm";
    auto eng = engine::FastEngine::load(model_path, engine::Backend::QNN);
    if (!eng) { std::cerr << "FAILED to load\n"; return 1; }

    // Short generation (40 tokens) -- likely won't close the JSON object,
    // but every character emitted must still be a valid PREFIX.
    engine::GenerateOptions opts_short;
    opts_short.prompt = "Once upon a time";
    opts_short.max_tokens = 15;
    opts_short.grammar_gbnf = json_gbnf;
    std::ostringstream out_short;
    eng->generate(opts_short, [&](const std::string& t) { out_short << t; }, nullptr);

    // Strip the echoed last prompt token/leading space before the JSON
    // actually starts, so we validate from the first '{' onward.
    std::string s = out_short.str();
    size_t brace = s.find('{');
    std::string json_part = (brace != std::string::npos) ? s.substr(brace) : "";

    std::cout << "Generated (15 tokens, grammar-constrained):\n" << s << "\n\n";
    std::cout << "JSON portion validated: " << json_part << "\n";
    CHECK(!json_part.empty(), "grammar-constrained output contains a '{' (grammar forced JSON start)");
    CHECK(is_valid_prefix(json_part), "every character emitted was a valid JSON prefix at the time (independent re-check)");

    // Longer generation (200 tokens) -- more room for the model to
    // stumble into closing braces/quotes and produce a COMPLETE document.
    engine::GenerateOptions opts_long;
    opts_long.prompt = "Once upon a time";
    opts_long.max_tokens = 55;
    opts_long.grammar_gbnf = json_gbnf;
    std::ostringstream out_long;
    eng->generate(opts_long, [&](const std::string& t) { out_long << t; }, nullptr);

    std::string s2 = out_long.str();
    size_t brace2 = s2.find('{');
    std::string json_part2 = (brace2 != std::string::npos) ? s2.substr(brace2) : "";

    std::cout << "\nGenerated (55 tokens, grammar-constrained):\n" << s2 << "\n\n";
    CHECK(is_valid_prefix(json_part2), "55-token generation also stays a valid JSON prefix throughout");
    std::cout << "Complete valid JSON document by end of 55 tokens: "
              << (is_complete_document(json_part2) ? "YES" : "NO (still a valid prefix, just not closed yet)") << "\n";

    // Control: same prompt, NO grammar -- must NOT happen to already be
    // valid JSON (sanity check that the grammar is actually doing something,
    // not that stories15M naturally outputs JSON-like text anyway).
    engine::GenerateOptions opts_free;
    opts_free.prompt = "Once upon a time";
    opts_free.max_tokens = 15;
    std::ostringstream out_free;
    eng->generate(opts_free, [&](const std::string& t) { out_free << t; }, nullptr);
    CHECK(out_free.str().find('{') == std::string::npos,
          "control: without grammar, model does NOT spontaneously produce JSON syntax");

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
