// src/core/asr_context_bias.h — CTC-WS phrase-boost trie (PLAN #98 Phase A).
//
// Aho-Corasick multi-pattern matcher over token-ID sequences. When the
// CTC/TDT decoder is scoring frame logits, the trie adds a configurable
// boost to tokens that continue an active hotword prefix match. This is
// the same "shallow fusion" approach described in NeMo's CTC-WS word
// boosting and the TurboBias paper (arxiv.org/abs/2508.07014).
//
// Usage:
//   1. Build a trie from hotword strings + a tokenizer function.
//   2. Before each frame's argmax, call apply_bias() with the current
//      trie state and the logits buffer — it adds the boost in-place.
//   3. After argmax, call advance() with the chosen token to move the
//      trie state forward (or reset on mismatch).
//
// Header-only, no ggml dependency. Pure CPU, ~200 LOC.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_context_bias {

// A single node in the Aho-Corasick trie.
struct TrieNode {
    std::unordered_map<int32_t, int> children; // token_id → child node index
    int fail = 0;                              // failure link (index into nodes[])
    float boost = 0.0f;                        // accumulated boost at this node (>0 if a hotword ends here)
    int depth = 0;                             // number of tokens from root to this node
};

// UTF-8 → code points with the case folded for ASCII Latin and Cyrillic (Ё → ё,
// ё ≠ е), as in Recheo's exact term replacement. Invalid UTF-8 (byte-fallback
// pieces) gives an empty string.
inline std::u32string fold_utf8(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t n = 0;
        char32_t cp = 0;
        if (c < 0x80) {
            n = 1;
            cp = c;
        } else if ((c >> 5) == 0x6) {
            n = 2;
            cp = c & 0x1F;
        } else if ((c >> 4) == 0xE) {
            n = 3;
            cp = c & 0x0F;
        } else if ((c >> 3) == 0x1E) {
            n = 4;
            cp = c & 0x07;
        } else {
            return {};
        }
        if (i + n > s.size())
            return {};
        for (size_t k = 1; k < n; k++) {
            const unsigned char d = (unsigned char)s[i + k];
            if ((d >> 6) != 0x2)
                return {};
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((cp >= U'A' && cp <= U'Z') || (cp >= 0x410 && cp <= 0x42F))
            cp += 0x20;
        else if (cp == 0x401)
            cp = 0x451;
        out.push_back(cp);
        i += n;
    }
    return out;
}

// A node of the case-folded text tries (Trie::fold_case).
struct CharNode {
    std::unordered_map<char32_t, int> next;
    int depth = 0; // code points from the root
};

struct Trie {
    std::vector<TrieNode> nodes;

    // Text mode, set by fold_case(); empty `text` = ID mode.
    std::vector<CharNode> text;                  // folded text of the forms
    std::vector<CharNode> vocab_nodes;           // folded text of the vocab
    std::vector<std::vector<int32_t>> vocab_ids; // tokens whose folded text ends at a vocab node
    std::vector<std::u32string> folded;          // folded text per token ID ("" never matches)

    Trie() { nodes.emplace_back(); } // root node at index 0

    // Insert a hotword as a sequence of token IDs with a given boost score.
    void insert(const std::vector<int32_t>& token_ids, float boost) {
        int cur = 0;
        for (int32_t tid : token_ids) {
            auto it = nodes[cur].children.find(tid);
            if (it == nodes[cur].children.end()) {
                int next = (int)nodes.size();
                nodes[cur].children[tid] = next;
                nodes.emplace_back();
                nodes.back().depth = nodes[cur].depth + 1;
                cur = next;
            } else {
                cur = it->second;
            }
        }
        nodes[cur].boost += boost; // accumulate if duplicate
    }

    // Build Aho-Corasick failure links. Must be called after all inserts
    // and before any matching.
    void build_failure_links() {
        std::queue<int> q;
        // Root's children fail to root
        for (auto& [tid, child] : nodes[0].children) {
            nodes[child].fail = 0;
            q.push(child);
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (auto& [tid, v] : nodes[u].children) {
                int f = nodes[u].fail;
                while (f != 0 && nodes[f].children.find(tid) == nodes[f].children.end())
                    f = nodes[f].fail;
                auto it = nodes[f].children.find(tid);
                nodes[v].fail = (it != nodes[f].children.end() && it->second != v) ? it->second : 0;
                // Propagate boost from suffix matches
                nodes[v].boost += nodes[nodes[v].fail].boost;
                q.push(v);
            }
        }
    }

    // Switch to text mode (Context-Assist #104). Call after all inserts. The
    // match then runs over the case-folded text of the forms: after the decoder
    // started a form, every vocab token whose folded text continues it is
    // boosted, whatever its case and however the text is split into tokens.
    // In ID mode a form spelled `Сре` (▁С ре) pulled a decoder that spells
    // `▁С Р Е` onto `ре` — off its own spelling and, on a character-level
    // vocab, into a merged word ("Сренженер"). The spelling stays the
    // decoder's; the exact replacement downstream restores the display form.
    // So every continuation gets the same boost, with no endpoint bonus: a bonus
    // for the token that ends the form at once (`ре`) again outweighs the
    // decoder's own shorter step (`Р`).
    // `<…>` pieces and `skip_id` (blank / EOS) never match.
    // ponytail: no failure links in text mode — a match that breaks restarts
    // at the root on the same token; add suffix links if overlapping forms
    // ("ab", "bc") turn out to matter.
    void fold_case(const std::vector<std::string>& id_to_token, int32_t skip_id) {
        text.assign(1, {});
        vocab_nodes.assign(1, {});
        vocab_ids.assign(1, {});
        folded.assign(id_to_token.size(), {});
        for (int32_t id = 0; id < (int32_t)id_to_token.size(); id++) {
            const auto& p = id_to_token[(size_t)id];
            if (id == skip_id || (p.size() >= 2 && p.front() == '<' && p.back() == '>'))
                continue;
            folded[(size_t)id] = fold_utf8(p);
            int cur = 0;
            for (char32_t c : folded[(size_t)id])
                cur = child(vocab_nodes, cur, c, &vocab_ids);
            if (cur != 0)
                vocab_ids[(size_t)cur].push_back(id);
        }
        std::vector<std::pair<int, int>> stack{{0, 0}}; // (ID node, text node)
        while (!stack.empty()) {
            const auto [id_node, text_node] = stack.back();
            stack.pop_back();
            for (const auto& [tid, id_child] : nodes[(size_t)id_node].children) {
                if (tid < 0 || tid >= (int32_t)folded.size() || folded[(size_t)tid].empty())
                    continue;
                int cur = text_node;
                for (char32_t c : folded[(size_t)tid])
                    cur = child(text, cur, c, nullptr);
                stack.push_back({id_child, cur});
            }
        }
    }

    bool text_mode() const { return !text.empty(); }
    bool empty() const { return nodes.size() <= 1; }

  private:
    static int child(std::vector<CharNode>& t, int cur, char32_t c, std::vector<std::vector<int32_t>>* ids) {
        const auto it = t[(size_t)cur].next.find(c);
        if (it != t[(size_t)cur].next.end())
            return it->second;
        const int next = (int)t.size();
        const int depth = t[(size_t)cur].depth + 1;
        t[(size_t)cur].next[c] = next;
        t.emplace_back();
        t.back().depth = depth;
        if (ids)
            ids->emplace_back();
        return next;
    }
};

// Runtime state for one ongoing decode. One per utterance.
struct MatchState {
    int node = 0; // current position in the trie (0 = root)

    void reset() { node = 0; }
};

// Advance the match state by one emitted token. Call this AFTER argmax
// with the chosen token_id to update the trie position.
inline void advance(const Trie& trie, MatchState& st, int32_t token_id) {
    if (trie.text_mode()) {
        // Continue the match by the token's folded text, else start over at the root.
        static const std::u32string none;
        const auto& f = token_id >= 0 && token_id < (int32_t)trie.folded.size() ? trie.folded[(size_t)token_id] : none;
        const auto walk = [&](int from) {
            int cur = from;
            for (char32_t c : f) {
                const auto it = trie.text[(size_t)cur].next.find(c);
                if (it == trie.text[(size_t)cur].next.end())
                    return 0;
                cur = it->second;
            }
            return cur;
        };
        st.node = f.empty() ? 0 : walk(st.node);
        if (st.node == 0 && !f.empty())
            st.node = walk(0);
        return;
    }
    int cur = st.node;
    while (cur != 0 && trie.nodes[cur].children.find(token_id) == trie.nodes[cur].children.end())
        cur = trie.nodes[cur].fail;
    auto it = trie.nodes[cur].children.find(token_id);
    st.node = (it != trie.nodes[cur].children.end()) ? it->second : 0;
}

// Apply bias to logits for the current frame. For each token_id that
// continues the current match (from the current node or any suffix via
// failure links), add `boost` to logits[token_id].
//
// This is the "shallow fusion" step, with two limits (Context-Assist #103):
//   - No start boost. Children of the root are never boosted: a hint helps
//     the decoder finish a hotword it started on its own, it does not start
//     one. Boosting first tokens on every frame (often single letters, ▁S,
//     ▁П, or common starts like ▁Про) pulled fillers, pauses and ordinary
//     words into hotword starts, and the predictor finished them with
//     training-data prompts ("Субтитры делал …", "Продолжение следует…").
//   - Blank gate. When `blank_id` is in range and blank already wins the
//     unbiased argmax, the step is left untouched: the bias picks between
//     words, it never turns a pause into a word. -1 disables the gate.
// A hotword of a single token therefore gets no boost at all.
// Text mode (fold_case): the same limits over the folded text — a token gets
// `default_boost` when its whole text continues a form from the current match
// (no endpoint bonus); a form started only by a lone "▁" is not started at all.
inline void apply_bias(const Trie& trie, const MatchState& st, float* logits, int n_vocab, float default_boost,
                       int blank_id) {
    if (trie.empty() || st.node == 0)
        return;
    if (blank_id >= 0 && blank_id < n_vocab && std::max_element(logits, logits + n_vocab) - logits == blank_id)
        return;

    if (trie.text_mode()) {
        if (trie.text[(size_t)st.node].depth < 2)
            return;
        // Walk the form text and the vocab text together; every vocab token met
        // on the way continues the form. Each token is reached once.
        std::vector<std::pair<int, int>> stack{{st.node, 0}}; // (text node, vocab node)
        while (!stack.empty()) {
            const auto [t, v] = stack.back();
            stack.pop_back();
            for (const auto& [c, t_child] : trie.text[(size_t)t].next) {
                const auto it = trie.vocab_nodes[(size_t)v].next.find(c);
                if (it == trie.vocab_nodes[(size_t)v].next.end())
                    continue;
                for (int32_t id : trie.vocab_ids[(size_t)it->second])
                    if (id < n_vocab)
                        logits[id] += default_boost;
                stack.push_back({t_child, it->second});
            }
        }
        return;
    }

    // Walk from the current node down the failure chain; the root is not boosted.
    for (int cur = st.node; cur != 0; cur = trie.nodes[cur].fail) {
        for (const auto& [tid, child_idx] : trie.nodes[cur].children) {
            if (tid >= 0 && tid < n_vocab) {
                // Boost = default_boost (for prefix continuation) + any
                // endpoint boost if a hotword completes at the child
                float b = default_boost;
                if (trie.nodes[child_idx].boost > 0.0f)
                    b += trie.nodes[child_idx].boost;
                logits[tid] += b;
            }
        }
    }
}

// Convenience: build a trie from a list of hotword strings using a
// tokenizer function. The tokenizer converts a string to token IDs.
using Tokenizer = std::function<std::vector<int32_t>(const std::string&)>;

inline Trie build_trie(const std::vector<std::string>& hotwords, const Tokenizer& tokenize,
                       float boost_per_word = 2.0f) {
    Trie trie;
    for (const auto& hw : hotwords) {
        if (hw.empty())
            continue;
        // Parse optional boost suffix: "word^5.0"
        float boost = boost_per_word;
        std::string word = hw;
        auto caret = hw.rfind('^');
        if (caret != std::string::npos && caret + 1 < hw.size()) {
            try {
                boost = std::stof(hw.substr(caret + 1));
                word = hw.substr(0, caret);
            } catch (...) {
                // Not a valid float suffix, treat the whole thing as the word
            }
        }
        auto ids = tokenize(word);
        if (!ids.empty()) {
            trie.insert(ids, boost);
        }
    }
    trie.build_failure_links();
    return trie;
}

// Form → SentencePiece token IDs: "▁" + the form with each run of spaces/tabs
// turned into one "▁"; greedy longest match against the vocab. Pieces of the
// form `<…>` (<unk>, byte fallback) and `blank_id` never match. Returns an empty
// vector when the vocab does not cover the whole form, or when the first piece
// is a lone "▁" (it would boost the word separator on every step): a form goes
// into the trie whole or not at all. Case is kept as is.
inline std::vector<int32_t> tokenize_spm_form(const std::vector<std::string>& id_to_token, int32_t blank_id,
                                              const std::string& form) {
    static const std::string kSep = "\xE2\x96\x81"; // U+2581 ▁
    std::string s = kSep;
    bool space = false;
    for (char c : form) {
        if (c == ' ' || c == '\t') {
            space = true;
            continue;
        }
        if (space && s.size() > kSep.size())
            s += kSep;
        space = false;
        s += c;
    }
    if (s.size() == kSep.size())
        return {};

    std::vector<int32_t> ids;
    for (size_t pos = 0; pos < s.size();) {
        size_t best_len = 0;
        int32_t best_id = -1;
        for (int32_t i = 0; i < (int32_t)id_to_token.size(); i++) {
            const auto& p = id_to_token[i];
            if (i == blank_id || p.size() <= best_len)
                continue;
            if (p.size() >= 2 && p.front() == '<' && p.back() == '>')
                continue;
            if (s.compare(pos, p.size(), p) == 0) {
                best_len = p.size();
                best_id = i;
            }
        }
        if (best_id < 0)
            return {};
        ids.push_back(best_id);
        pos += best_len;
    }
    if (id_to_token[ids[0]] == kSep)
        return {};
    return ids;
}

// Parse a comma-separated hotword string into a vector.
inline std::vector<std::string> parse_hotwords(const std::string& s) {
    std::vector<std::string> result;
    if (s.empty())
        return result;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos)
            end = s.size();
        // Trim whitespace
        size_t a = start, b = end;
        while (a < b && (s[a] == ' ' || s[a] == '\t'))
            a++;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t'))
            b--;
        if (b > a)
            result.push_back(s.substr(a, b - a));
        start = end + 1;
    }
    return result;
}

} // namespace core_context_bias
