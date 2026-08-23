#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

inline constexpr const char *TOKENIZER_FORMAT = "bpe";
inline constexpr const char *TOKENIZER_LANGUAGE = "cpp";
inline constexpr const char *PRETOKENIZER_NAME = "gpt2";
inline constexpr const char *GPT2_PATTERN_TEXT = R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

struct MergeRule
{
    int left_token;
    int right_token;
    int new_token;
};

struct TokenizerState
{
    std::string format;
    std::string language;
    std::string pre_tokenizer;
    int mergeable_vocab_size;
    int vocab_size;
    std::map<std::string, int> special_tokens;
    std::string regex;
    std::vector<MergeRule> merges;
};

class BPETokenizer
{
public:
    BPETokenizer(int mergeable_vocab_size, const std::map<std::string, int> &special_tokens = {});

    void train(const std::string &text);
    std::vector<int> encode(const std::string &text, const std::set<std::string> &allowed_special = {}) const;
    std::string decode_bytes(const std::vector<int> &token_ids) const;

    TokenizerState to_state() const;
    static BPETokenizer from_state(const TokenizerState &state);

    int get_mergeable_vocab_size() const;
    int get_vocab_size() const;
    const std::map<std::pair<int, int>, int> &get_merges() const;
    const std::map<std::string, int> &get_special_tokens() const;

private:
    int mergeable_vocab_size;
    int vocab_size;
    std::map<std::string, int> special_tokens;

    // optimization: scratch
    struct EncodeScratch
    {
        std::vector<int> token_ids;
        std::vector<int> ranks;
    };

    std::vector<std::string> vocabulary;
    std::map<std::pair<int, int>, int> merges;

    // optimization: fastpath
    std::unordered_map<std::string, int> token_to_id;

    std::vector<int> encode_ordinary(const std::string &text) const;
    void encode_piece(const std::string &piece, std::vector<int> &output, EncodeScratch &scratch) const;
    void reset_vocabulary();
};
