#define PCRE2_CODE_UNIT_WIDTH 8 // use 8-bit api for regex lib (utf-8)
#include <pcre2.h>

#include "tokenizer.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace
{

// stateless helper functions

std::vector<std::string> pretokenize(const std::string &text)
{
    // convert plaintext into partitions based off the GPT-2 pre-tokenization regex
    // PCRE2 supports unicode properties such as \p{L} and \p{N}, unlike c++'s standard regex
    std::vector<std::string> pieces;

    int error_code;
    PCRE2_SIZE error_offset;
    pcre2_code *pattern = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(GPT2_PATTERN_TEXT),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &error_code,
        &error_offset,
        nullptr
    );

    if (pattern == nullptr)
        throw std::runtime_error("failed to compile GPT-2 pre-tokenization pattern");

    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(pattern, nullptr);
    if (match_data == nullptr)
    {
        pcre2_code_free(pattern);
        throw std::runtime_error("failed to allocate PCRE2 match data");
    }

    PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    PCRE2_SIZE offset = 0;


    while (offset < text.size())
    {
        // optimization pcre2 no utf recheck
        uint32_t match_options = PCRE2_ANCHORED;
        if (offset > 0) match_options |= PCRE2_NO_UTF_CHECK;
        
        // expensive
        int match_count = pcre2_match(
            pattern,
            subject,
            text.size(),
            offset,
            match_options,
            match_data,
            nullptr
        );

        if (match_count < 0)
        {
            pcre2_match_data_free(match_data);
            pcre2_code_free(pattern);
            throw std::runtime_error("GPT-2 pre-tokenization pattern did not preserve the input");
        }

        PCRE2_SIZE *match = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE start = match[0];
        PCRE2_SIZE end = match[1];

        if (start != offset || end == start)
        {
            pcre2_match_data_free(match_data);
            pcre2_code_free(pattern);
            throw std::runtime_error("invalid GPT-2 pre-tokenization match");
        }

        pieces.push_back(text.substr(start, end - start));
        offset = end;
    }

    pcre2_match_data_free(match_data);
    pcre2_code_free(pattern);

    return pieces;
}

std::vector<int> merge_pair(const std::vector<int> &token_ids, std::pair<int, int> pair, int new_token_id)
{
    // replace every non-overlapping adjacent pair with new_token_id from left to right
    std::vector<int> result;
    result.reserve(token_ids.size());
    std::size_t i = 0;

    while (i < token_ids.size())
    {
        if (i + 1 < token_ids.size() && token_ids[i] == pair.first && token_ids[i + 1] == pair.second)
        {
            result.push_back(new_token_id);
            i += 2;
        }
        else
        {
            result.push_back(token_ids[i]);
            i += 1;
        }
    }

    return result;
}

std::map<std::pair<int, int>, int> count_pairs(const std::vector<std::vector<int>> &pieces)
{
    // count adjacent pairs across multiple pre-tokenized partitions
    std::map<std::pair<int, int>, int> counts;

    for (const std::vector<int> &piece : pieces)
    {
        for (std::size_t i = 0; i + 1 < piece.size(); i++)
        {
            std::pair<int, int> pair = {piece[i], piece[i + 1]};
            counts[pair] += 1;
        }
    }

    return counts;
}

std::pair<int, int> select_pair(const std::map<std::pair<int, int>, int> &counts)
{
    // select the most frequent pair, breaking ties with the smallest numerical pair
    std::pair<int, int> best_pair = counts.begin()->first;
    int best_count = counts.begin()->second;

    for (const std::pair<const std::pair<int, int>, int> &entry : counts)
    {
        std::pair<int, int> pair = entry.first;
        int count = entry.second;

        if (count > best_count || (count == best_count && pair < best_pair))
        {
            best_pair = pair;
            best_count = count;
        }
    }

    return best_pair;
}

std::vector<int> bytes_to_ids(const std::string &piece)
{
    // string -> bytes
    std::vector<int> token_ids;
    token_ids.reserve(piece.size()); // optimization basic data

    for (unsigned char byte : piece)
        token_ids.push_back(byte);

    return token_ids;
}

} // namespace

BPETokenizer::BPETokenizer(int mergeable_vocab_size, const std::map<std::string, int> &special_tokens)
{
    if (mergeable_vocab_size < 256)
        throw std::invalid_argument("mergeable_vocab_size must be at least 256");

    this->mergeable_vocab_size = mergeable_vocab_size;
    this->special_tokens = special_tokens;
    vocab_size = mergeable_vocab_size + static_cast<int>(special_tokens.size());

    std::set<int> expected_special_ids;
    std::set<int> actual_special_ids;

    for (int token_id = mergeable_vocab_size; token_id < vocab_size; token_id++)
        expected_special_ids.insert(token_id);

    for (const std::pair<const std::string, int> &special_token : special_tokens)
    {
        if (special_token.first.empty())
            throw std::invalid_argument("special tokens cannot be empty");

        actual_special_ids.insert(special_token.second);
    }

    if (actual_special_ids != expected_special_ids)
        throw std::invalid_argument("special token IDs must begin after the mergeable vocabulary");

    reset_vocabulary();
}

void BPETokenizer::train(const std::string &text)
{
    // take a training corpus and learn BPE merge rules
    std::vector<std::string> string_pieces = pretokenize(text);
    std::vector<std::vector<int>> integer_pieces;

    integer_pieces.reserve(string_pieces.size());

    for (const std::string &piece : string_pieces)
        integer_pieces.push_back(bytes_to_ids(piece));

    merges.clear();
    reset_vocabulary();

    for (int new_token_id = 256; new_token_id < mergeable_vocab_size; new_token_id++)
    {
        std::map<std::pair<int, int>, int> counts = count_pairs(integer_pieces);
        if (counts.empty()) break;

        std::pair<int, int> selected_pair = select_pair(counts);

        for (std::vector<int> &piece : integer_pieces)
            piece = merge_pair(piece, selected_pair, new_token_id);

        merges[selected_pair] = new_token_id;
        vocabulary[new_token_id] = vocabulary[selected_pair.first] + vocabulary[selected_pair.second];
        token_to_id.emplace(vocabulary[new_token_id], new_token_id);
    }
}

std::vector<int> BPETokenizer::encode(const std::string &text, const std::set<std::string> &allowed_special) const
{
    // special tokens bypass pre-tokenization and BPE only when explicitly allowed
    for (const std::string &special_token : allowed_special)
    {
        if (!special_tokens.contains(special_token))
            throw std::invalid_argument("unknown special token: " + special_token);
    }

    // branch to ordinary encode if no speical tokens
    if (allowed_special.empty()) return encode_ordinary(text);

    std::vector<int> token_ids;
    token_ids.reserve(text.size());
    std::size_t ordinary_start = 0;
    std::size_t position = 0;

    while (position < text.size())
    {
        std::string matched_special;

        for (const std::string &special_token : allowed_special)
        {
            bool matches = text.compare(position, special_token.size(), special_token) == 0;

            if (matches && special_token.size() > matched_special.size())
                matched_special = special_token;
        }

        if (matched_special.empty())
        {
            position++;
            continue;
        }

        std::vector<int> ordinary_ids = encode_ordinary(text.substr(ordinary_start, position - ordinary_start));
        for (int token_id : ordinary_ids) token_ids.push_back(token_id);

        token_ids.push_back(special_tokens.at(matched_special));
        position += matched_special.size();
        ordinary_start = position;
    }

    std::vector<int> ordinary_ids = encode_ordinary(text.substr(ordinary_start));
    for (int token_id : ordinary_ids) token_ids.push_back(token_id);

    return token_ids;
}

std::string BPETokenizer::decode_bytes(const std::vector<int> &token_ids) const
{
    std::string text;

    for (int token_id : token_ids)
        text += vocabulary.at(token_id);

    return text;
}

TokenizerState BPETokenizer::to_state() const
{
    TokenizerState state;
    state.format = TOKENIZER_FORMAT;
    state.language = TOKENIZER_LANGUAGE;
    state.pre_tokenizer = PRETOKENIZER_NAME;
    state.mergeable_vocab_size = mergeable_vocab_size;
    state.vocab_size = vocab_size;
    state.special_tokens = special_tokens;
    state.regex = GPT2_PATTERN_TEXT;

    for (const std::pair<const std::pair<int, int>, int> &entry : merges)
        state.merges.push_back({entry.first.first, entry.first.second, entry.second});

    std::sort(state.merges.begin(), state.merges.end(), [](const MergeRule &left, const MergeRule &right) {
        return left.new_token < right.new_token;
    });

    return state;
}

BPETokenizer BPETokenizer::from_state(const TokenizerState &state)
{
    bool supported_language = state.language == "py" || state.language == "cpp" || state.language == "rust";

    if (state.format != TOKENIZER_FORMAT || !supported_language)
        throw std::invalid_argument("unsupported tokenizer artifact");

    if (state.pre_tokenizer != PRETOKENIZER_NAME || state.regex != GPT2_PATTERN_TEXT)
        throw std::invalid_argument("unsupported pre-tokenizer configuration");

    BPETokenizer tokenizer(state.mergeable_vocab_size, state.special_tokens);

    if (state.vocab_size != tokenizer.vocab_size)
        throw std::invalid_argument("tokenizer vocabulary sizes do not match");

    int expected_token = 256;

    for (const MergeRule &rule : state.merges)
    {
        if (rule.new_token != expected_token || rule.new_token >= tokenizer.mergeable_vocab_size)
            throw std::invalid_argument("merge token IDs are not sequential");

        if (rule.left_token < 0 || rule.right_token < 0 || rule.left_token >= rule.new_token || rule.right_token >= rule.new_token)
            throw std::invalid_argument("merge rule references an unavailable token");

        std::pair<int, int> pair = {rule.left_token, rule.right_token};

        if (tokenizer.merges.contains(pair))
            throw std::invalid_argument("duplicate merge pair");

        tokenizer.merges[pair] = rule.new_token;
        tokenizer.vocabulary[rule.new_token] = tokenizer.vocabulary[rule.left_token] + tokenizer.vocabulary[rule.right_token];
        tokenizer.token_to_id.emplace(tokenizer.vocabulary[rule.new_token], rule.new_token);
        expected_token++;
    }

    return tokenizer;
}

int BPETokenizer::get_mergeable_vocab_size() const
{
    return mergeable_vocab_size;
}

int BPETokenizer::get_vocab_size() const
{
    return vocab_size;
}

const std::map<std::pair<int, int>, int> &BPETokenizer::get_merges() const
{
    return merges;
}

const std::map<std::string, int> &BPETokenizer::get_special_tokens() const
{
    return special_tokens;
}

std::vector<int> BPETokenizer::encode_ordinary(const std::string &text) const
{
    // pretokenize entire text stream (break into groups)
    std::vector<std::string> pieces = pretokenize(text);

    // output (flattened token id's)
    std::vector<int> token_ids;

    // optimization basic data
    token_ids.reserve(text.size());

    // for each piece, encode_piece and add to our resultant reserve
    for (const std::string &piece : pieces)
    {
        // optimization: fast path
        auto token = token_to_id.find(piece);

        if (token != token_to_id.end())
        {
            token_ids.push_back(token->second);
        }
        else
        {
            // process normally
            std::vector<int> piece_token_ids = encode_piece(piece);
            for (int token_id : piece_token_ids) token_ids.push_back(token_id);
        }
    }

    return token_ids;
}

std::vector<int> BPETokenizer::encode_piece(const std::string &piece) const
{
    // we are now operating on a singular piece of plaintext (a pretokenized group)
    // we want to turn that plaintext into a flat vector that has our applied merge rules

    /*
        rank optimization:
         1. convert piece to byte token ids
         2. create ranks[] once => rank[i] = merge rank for (t[i],t[i+1]) 
         3. not mergeable => INF
         4. find lowest rank (scan left->right) (ties => leftmost; all INF => stop)
         5. merge only that e.g. A B {C D} E -> A B {X} E
            only pairs touching X changed: (B,X) and (X,E)
            so only recompute those ranks
         6. repeat until all ranks INF
    */


    // turn that string text into integers
    std::vector<int> token_ids = bytes_to_ids(piece);

    // repeatedly scan the current sequence and apply the earliest learned merge rule
    while (token_ids.size() >= 2)
    {
        std::pair<int, int> selected_pair;
        int selected_token_id = -1;

        for (std::size_t i = 0; i + 1 < token_ids.size(); i++)
        {
            std::pair<int, int> current_pair = {token_ids[i], token_ids[i + 1]};

            // optimization basic data
            auto merge_rule = merges.find(current_pair);
            if (merge_rule == merges.end()) continue;
            int current_token_id = merge_rule->second;


            if (selected_token_id == -1 || current_token_id < selected_token_id)
            {
                selected_pair = current_pair;
                selected_token_id = current_token_id;
            }
        }

        if (selected_token_id == -1) break;

        token_ids = merge_pair(token_ids, selected_pair, selected_token_id);
    }

    return token_ids;
}

void BPETokenizer::reset_vocabulary()
{
    vocabulary.assign(vocab_size, "");
    token_to_id.clear();
    token_to_id.reserve(mergeable_vocab_size);

    for (int token_id = 0; token_id < 256; token_id++)
    {
        vocabulary[token_id] = std::string(1, static_cast<char>(token_id));
        token_to_id.emplace(vocabulary[token_id], token_id);
    }

    for (const std::pair<const std::string, int> &special_token : special_tokens)
        vocabulary[special_token.second] = special_token.first;
}
