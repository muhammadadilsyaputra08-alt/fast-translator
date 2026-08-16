#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace gbnf {

// A single grammar element: either a set of accepted character ranges
// (used for literals and [abc]/[a-z]/[^...] classes) or a reference to
// another rule by id.
struct Element {
    enum class Kind { CharRange, RuleRef, End } kind;

    // For CharRange: pairs of (lo, hi) inclusive codepoints that are
    // accepted; `negate` inverts the whole set (for [^...]).
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    bool negate = false;

    // For RuleRef: index into Grammar::rules.
    int rule_id = -1;
};

using Sequence = std::vector<Element>;      // one alternative: elements in order
using Alternatives = std::vector<Sequence>; // a rule's `a | b | c`

// A parsed grammar: a set of named rules (quantifiers ?, *, + and grouped
// `(...)` are desugared into synthetic auxiliary rules at parse time, so
// the runtime state machine only ever deals with CharRange/RuleRef/End).
class Grammar {
public:
    // Parses GBNF source text. Returns nullptr on a syntax error (with a
    // message on stderr) rather than throwing, matching this codebase's
    // fail-soft-with-nullptr convention.
    static std::unique_ptr<Grammar> parse(const std::string& gbnf_source);

    // A "stack" represents one possible nested-rule-call position (a
    // pushdown-automaton frame stack); a State is the SET of stacks
    // simultaneously alive, since alternation/ambiguity can mean several
    // parse paths are valid at once until more input disambiguates them.
    struct Cursor { int rule_id; int alt_id; int pos; };
    using Stack = std::vector<Cursor>;
    struct State { std::vector<Stack> stacks; };

    // Initial state: positioned at the start of the root rule, with all
    // rule-refs already resolved down to the first actual char-matching
    // position(s) (i.e. advance_epsilon() already applied).
    State initial_state() const;

    // True if `c` can be consumed by at least one stack in `state`.
    bool can_accept(const State& state, uint32_t c) const;

    // Returns the state after consuming `c` (only the stacks that accepted
    // it survive, each advanced past that character and epsilon-closed).
    // Precondition: can_accept(state, c) is true.
    State advance(const State& state, uint32_t c) const;

    // True if the grammar could legally end here (some stack has reached
    // the end of the root rule).
    bool can_terminate(const State& state) const;

    int root_rule_id() const { return root_rule_id_; }

private:
    std::vector<Alternatives> rules_; // indexed by rule_id
    int root_rule_id_ = -1;

    // Epsilon-closure: expands every stack's top cursor while it points at
    // a RuleRef, branching into one stack per alternative of that rule,
    // until every stack is positioned at a CharRange (ready to consume) or
    // at End with an empty call stack (grammar-complete).
    // Epsilon-closure cache: for a given starting cursor stack, the set of
    // reachable "ready to consume a char" stacks is always the same (it's
    // a pure function of the stack alone) -- caching this is what makes
    // grammar-constrained sampling fast enough to run once per candidate
    // token across a ~32k vocabulary, since the same handful of grammar
    // positions recur constantly across different vocab tokens.
    mutable std::unordered_map<std::string, std::vector<Stack>> epsilon_cache_;
    static std::string signature(const Stack& s);
    std::vector<Stack> epsilon_close(Stack stack) const;

    friend struct GrammarParser;
};

} // namespace gbnf
