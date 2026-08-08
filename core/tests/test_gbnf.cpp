#include "gbnf.h"
#include <iostream>
#include <string>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

// Feeds a whole string through the grammar, returns true if every char was
// accepted AND the grammar could legally terminate at the end.
bool matches(const gbnf::Grammar& g, const std::string& s) {
    auto state = g.initial_state();
    for (char c : s) {
        if (!g.can_accept(state, static_cast<uint8_t>(c))) return false;
        state = g.advance(state, static_cast<uint8_t>(c));
    }
    return g.can_terminate(state);
}

void test_literal() {
    auto g = gbnf::Grammar::parse(R"(root ::= "hello")");
    CHECK(g != nullptr, "literal grammar parses");
    CHECK(matches(*g, "hello"), "exact literal matches");
    CHECK(!matches(*g, "hell"), "prefix of literal does not terminate");
    CHECK(!matches(*g, "helloo"), "literal + extra char rejected");
    CHECK(!matches(*g, "world"), "different literal rejected");
}

void test_alternation() {
    auto g = gbnf::Grammar::parse(R"(root ::= "cat" | "dog" | "bird")");
    CHECK(matches(*g, "cat"), "alternation: first option matches");
    CHECK(matches(*g, "dog"), "alternation: second option matches");
    CHECK(matches(*g, "bird"), "alternation: third option matches");
    CHECK(!matches(*g, "fish"), "alternation: non-option rejected");
}

void test_char_class() {
    auto g = gbnf::Grammar::parse(R"(root ::= [0-9] [0-9])");
    CHECK(matches(*g, "42"), "char class: two digits matches");
    CHECK(!matches(*g, "4a"), "char class: digit+letter rejected");
    CHECK(!matches(*g, "4"), "char class: single digit incomplete, rejected");
}

void test_negated_class() {
    auto g = gbnf::Grammar::parse(R"(root ::= "\"" [^"]* "\"")");
    CHECK(matches(*g, "\"hello\""), "negated class: quoted string matches");
    CHECK(matches(*g, "\"\""), "negated class: empty quoted string matches");
    CHECK(!matches(*g, "\"he\"llo\""), "negated class: embedded quote breaks the string early") ;
}

void test_star_quantifier() {
    auto g = gbnf::Grammar::parse(R"(root ::= "a" [0-9]* "b")");
    CHECK(matches(*g, "ab"), "star: zero repetitions matches");
    CHECK(matches(*g, "a123b"), "star: multiple repetitions matches");
    CHECK(!matches(*g, "a12"), "star: missing terminator rejected");
}

void test_plus_quantifier() {
    auto g = gbnf::Grammar::parse(R"(root ::= [0-9]+)");
    CHECK(!matches(*g, ""), "plus: zero repetitions rejected");
    CHECK(matches(*g, "5"), "plus: one repetition matches");
    CHECK(matches(*g, "12345"), "plus: many repetitions matches");
}

void test_optional_quantifier() {
    auto g = gbnf::Grammar::parse(R"(root ::= "-"? [0-9]+)");
    CHECK(matches(*g, "42"), "optional: absent matches");
    CHECK(matches(*g, "-42"), "optional: present matches");
    CHECK(!matches(*g, "--42"), "optional: doubled rejected");
}

void test_rule_reference() {
    auto g = gbnf::Grammar::parse("root ::= greeting \" world\"\ngreeting ::= \"hello\" | \"hi\"");
    CHECK(matches(*g, "hello world"), "rule ref: first alt of referenced rule");
    CHECK(matches(*g, "hi world"), "rule ref: second alt of referenced rule");
    CHECK(!matches(*g, "bye world"), "rule ref: invalid referenced content rejected");
}

void test_grouped_alternation() {
    auto g = gbnf::Grammar::parse(R"(root ::= "x" ("a" | "b") "y")");
    CHECK(matches(*g, "xay"), "group: first alt matches");
    CHECK(matches(*g, "xby"), "group: second alt matches");
    CHECK(!matches(*g, "xcy"), "group: non-alt rejected");
}

// This is the ACTUAL default grammar used for Grammar.JSON in the Kotlin
// API (Types.kt) -- if this doesn't work, the production feature doesn't.
void test_json_grammar() {
    std::string json_gbnf = R"(
        root   ::= object
        object ::= "{" pair ("," pair)* "}" | "{" "}"
        pair   ::= string ":" value
        value  ::= object | array | string | number | "true" | "false" | "null"
        array  ::= "[" value ("," value)* "]" | "[" "]"
        string ::= "\"" [^"]* "\""
        number ::= "-"? [0-9]+ ("." [0-9]+)?
    )";
    auto g = gbnf::Grammar::parse(json_gbnf);
    CHECK(g != nullptr, "JSON grammar parses without error");
    CHECK(matches(*g, R"({"a":1})"), "JSON grammar: simple object matches");
    CHECK(matches(*g, R"({})"), "JSON grammar: empty object matches");
    CHECK(matches(*g, R"({"a":true,"b":null})"), "JSON grammar: multiple pairs with literals matches");
    CHECK(matches(*g, R"({"nested":{"x":[1,2,3]}})"), "JSON grammar: nested object+array matches");
    CHECK(!matches(*g, "not json at all"), "JSON grammar: plain text correctly rejected");
    CHECK(!matches(*g, R"({"a":1)"), "JSON grammar: unterminated object rejected");
}

// The core use case: at each generation step, can we correctly identify
// WHICH next characters are legal, to mask the token sampler?
void test_incremental_feasibility_for_sampling() {
    auto g = gbnf::Grammar::parse(R"(root ::= "{" "}")");
    auto state = g->initial_state();
    CHECK(g->can_accept(state, '{'), "sampling check: '{' accepted as first char");
    CHECK(!g->can_accept(state, '}'), "sampling check: '}' correctly rejected as first char");
    state = g->advance(state, '{');
    CHECK(g->can_accept(state, '}'), "sampling check: '}' accepted after '{'");
    CHECK(!g->can_accept(state, '{'), "sampling check: second '{' correctly rejected");
}

int main() {
    test_literal();
    test_alternation();
    test_char_class();
    test_negated_class();
    test_star_quantifier();
    test_plus_quantifier();
    test_optional_quantifier();
    test_rule_reference();
    test_grouped_alternation();
    test_json_grammar();
    test_incremental_feasibility_for_sampling();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
