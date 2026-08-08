#include "gbnf.h"
#include <unordered_map>
#include <iostream>
#include <cctype>
#include <set>

namespace gbnf {

namespace {

struct Token {
    enum class Kind { Ident, Arrow, Pipe, LParen, RParen, LBracket, RBracket,
                       Quantifier, StringLit, CharClass, End } kind;
    std::string text;
};

// Splits GBNF source into tokens. Comments (# to end of line) are skipped.
std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> toks;
    size_t i = 0, n = src.size();
    while (i < n) {
        char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '#') { while (i < n && src[i] != '\n') ++i; continue; }
        if (c == ':' && i + 2 < n && src[i+1] == ':' && src[i+2] == '=') {
            toks.push_back({Token::Kind::Arrow, "::="}); i += 3; continue;
        }
        if (c == '|') { toks.push_back({Token::Kind::Pipe, "|"}); ++i; continue; }
        if (c == '(') { toks.push_back({Token::Kind::LParen, "("}); ++i; continue; }
        if (c == ')') { toks.push_back({Token::Kind::RParen, ")"}); ++i; continue; }
        if (c == '?' || c == '*' || c == '+') {
            toks.push_back({Token::Kind::Quantifier, std::string(1, c)}); ++i; continue;
        }
        if (c == '"') {
            std::string s;
            ++i;
            while (i < n && src[i] != '"') {
                if (src[i] == '\\' && i + 1 < n) { s += src[i+1]; i += 2; }
                else { s += src[i]; ++i; }
            }
            ++i; // closing quote
            toks.push_back({Token::Kind::StringLit, s});
            continue;
        }
        if (c == '[') {
            std::string s;
            size_t start = i;
            ++i;
            while (i < n && src[i] != ']') {
                if (src[i] == '\\' && i + 1 < n) { s += src[i]; s += src[i+1]; i += 2; }
                else { s += src[i]; ++i; }
            }
            ++i; // closing bracket
            (void)start;
            toks.push_back({Token::Kind::CharClass, s});
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string s;
            while (i < n && (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_' || src[i] == '-')) {
                s += src[i]; ++i;
            }
            toks.push_back({Token::Kind::Ident, s});
            continue;
        }
        // Unknown character: skip it rather than hard-fail, so minor
        // formatting quirks in hand-written grammars don't break parsing.
        ++i;
    }
    toks.push_back({Token::Kind::End, ""});
    return toks;
}

} // namespace

// Owns the parser's mutable state while building a Grammar. Declared as a
// friend of Grammar so it can populate rules_/root_rule_id_ directly.
struct GrammarParser {
    std::vector<Token> toks;
    size_t pos = 0;
    std::unordered_map<std::string, int> rule_ids;
    std::vector<Alternatives> rules;

    const Token& peek() const { return toks[pos]; }
    const Token& advance() { return toks[pos++]; }

    int rule_id_for(const std::string& name) {
        auto it = rule_ids.find(name);
        if (it != rule_ids.end()) return it->second;
        int id = static_cast<int>(rules.size());
        rules.emplace_back(); // placeholder, filled in when defined
        rule_ids[name] = id;
        return id;
    }

    int new_synthetic_rule() {
        std::string name = "__anon" + std::to_string(rules.size());
        return rule_id_for(name);
    }

    // Parses a [..] char-class body (already extracted without brackets)
    // into ranges + negate flag.
    Element parse_char_class(const std::string& body) {
        Element el; el.kind = Element::Kind::CharRange;
        size_t i = 0, n = body.size();
        if (i < n && body[i] == '^') { el.negate = true; ++i; }
        while (i < n) {
            uint32_t lo;
            if (body[i] == '\\' && i + 1 < n) { lo = static_cast<uint8_t>(body[i+1]); i += 2; }
            else { lo = static_cast<uint8_t>(body[i]); ++i; }
            uint32_t hi = lo;
            if (i + 1 < n && body[i] == '-' && body[i+1] != ']') {
                ++i;
                if (body[i] == '\\' && i + 1 < n) { hi = static_cast<uint8_t>(body[i+1]); i += 2; }
                else { hi = static_cast<uint8_t>(body[i]); ++i; }
            }
            el.ranges.push_back({lo, hi});
        }
        return el;
    }

    Element literal_char_element(char c) {
        Element el; el.kind = Element::Kind::CharRange;
        el.ranges.push_back({static_cast<uint8_t>(c), static_cast<uint8_t>(c)});
        return el;
    }

    // Parses one "atom": a string literal (expanded into a sequence of
    // single-char elements chained via a synthetic rule if length > 1), a
    // char class, a rule reference, or a parenthesized group -- then
    // applies a trailing quantifier (?, *, +) if present by desugaring
    // into a synthetic rule.
    Sequence parse_atom() {
        Sequence seq;
        const Token& t = peek();

        if (t.kind == Token::Kind::StringLit) {
            advance();
            for (char c : t.text) seq.push_back(literal_char_element(c));
        } else if (t.kind == Token::Kind::CharClass) {
            advance();
            seq.push_back(parse_char_class(t.text));
        } else if (t.kind == Token::Kind::Ident) {
            advance();
            Element el; el.kind = Element::Kind::RuleRef;
            el.rule_id = rule_id_for(t.text);
            seq.push_back(el);
        } else if (t.kind == Token::Kind::LParen) {
            advance();
            Alternatives alts = parse_alternatives();
            advance(); // consume RParen (assumed present; malformed grammars just stop early)
            int group_rule = new_synthetic_rule();
            rules[group_rule] = alts;
            Element el; el.kind = Element::Kind::RuleRef;
            el.rule_id = group_rule;
            seq.push_back(el);
        } else {
            // Nothing consumable here; return empty so caller can stop.
            return seq;
        }

        if (peek().kind == Token::Kind::Quantifier) {
            char q = advance().text[0];
            // Wrap whatever we just parsed (seq, as a single sequence) into
            // a synthetic rule so quantifiers apply to the WHOLE atom, then
            // desugar:
            //   X?  ->  rule ::= X |            (empty alternative)
            //   X*  ->  rule ::= X rule |        (empty alternative)
            //   X+  ->  rule ::= X rule | X
            int inner_rule = new_synthetic_rule();
            rules[inner_rule] = {seq};

            int q_rule = new_synthetic_rule();
            Element ref; ref.kind = Element::Kind::RuleRef; ref.rule_id = inner_rule;
            Element self_ref; self_ref.kind = Element::Kind::RuleRef; self_ref.rule_id = q_rule;

            if (q == '?') {
                rules[q_rule] = { {ref}, {} };
            } else if (q == '*') {
                rules[q_rule] = { {ref, self_ref}, {} };
            } else { // '+'
                rules[q_rule] = { {ref, self_ref}, {ref} };
            }

            seq.clear();
            Element wrapped; wrapped.kind = Element::Kind::RuleRef; wrapped.rule_id = q_rule;
            seq.push_back(wrapped);
        }
        return seq;
    }

    Sequence parse_sequence() {
        Sequence seq;
        while (true) {
            Token::Kind k = peek().kind;
            if (k == Token::Kind::Pipe || k == Token::Kind::RParen || k == Token::Kind::End
                || k == Token::Kind::Arrow) {
                break;
            }
            // Stop the sequence if we hit an Ident that's actually the
            // start of the NEXT rule definition (Ident followed by Arrow).
            if (k == Token::Kind::Ident && pos + 1 < toks.size() && toks[pos + 1].kind == Token::Kind::Arrow) {
                break;
            }
            Sequence atom = parse_atom();
            if (atom.empty()) break;
            for (auto& e : atom) seq.push_back(e);
        }
        return seq;
    }

    Alternatives parse_alternatives() {
        Alternatives alts;
        alts.push_back(parse_sequence());
        while (peek().kind == Token::Kind::Pipe) {
            advance();
            alts.push_back(parse_sequence());
        }
        return alts;
    }

    int parse_rule_def() {
        const Token& name_tok = advance(); // Ident
        int id = rule_id_for(name_tok.text);
        advance(); // Arrow (::=)
        Alternatives alts = parse_alternatives();
        rules[id] = alts;
        return id;
    }
};

std::unique_ptr<Grammar> Grammar::parse(const std::string& gbnf_source) {
    GrammarParser p;
    p.toks = tokenize(gbnf_source);

    int first_rule_id = -1;
    while (p.peek().kind != Token::Kind::End) {
        if (p.peek().kind == Token::Kind::Ident) {
            int id = p.parse_rule_def();
            if (first_rule_id == -1) first_rule_id = id;
        } else {
            p.advance(); // skip stray token defensively
        }
    }

    if (first_rule_id == -1) {
        std::cerr << "[gbnf] no rules found in grammar\n";
        return nullptr;
    }

    auto g = std::unique_ptr<Grammar>(new Grammar());
    g->rules_ = std::move(p.rules);
    // Prefer a rule literally named "root" if present, else the first
    // rule defined (both are common GBNF conventions).
    auto it = p.rule_ids.find("root");
    g->root_rule_id_ = (it != p.rule_ids.end()) ? it->second : first_rule_id;
    return g;
}

} // namespace gbnf
