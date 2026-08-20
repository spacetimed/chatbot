#define PCRE2_CODE_UNIT_WIDTH 8 // use 8-bit api for regex lib (utf-8)
#include <pcre2.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

const char *GPT2_PATTERN = R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

std::vector<std::string> pretokenize(const std::string &text)
{
    // convert a plaintext string into partitions based off GPT2 pre-train regex pattern
    // using PCRE2 because patterns like \p{L} \p{N} (unicode letters/numbers) not supported by c++'s standard regex

    // example:
    //  "Hello, world!" -> ["Hello", ",", " world", "!"]

    std::vector<std::string> pieces;

    // compile regex pattern, *pattern is a pointer to PCRE2's, compiled regex
    int error_code;
    PCRE2_SIZE error_offset;
    pcre2_code *pattern = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(GPT2_PATTERN),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP, // support aforementioned unicode properties
        &error_code,
        &error_offset,
        nullptr
    );

    // ensure compilation successful
    if (pattern == nullptr)
        throw std::runtime_error("failed to compile GPT-2 pre-tokenization pattern");

    // buffer to hold match information
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(pattern, nullptr);
    if (match_data == nullptr)
    {
        pcre2_code_free(pattern);
        throw std::runtime_error("failed to allocate PCRE2 match data");
    }

    // our input (plaintext) into the compiled regex
    PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    PCRE2_SIZE offset = 0;

    // consume each byte of input
    // this loop is basically doing python equivalent of:
    /*
    offset = 0
    while offset < len(text):
        match = pattern.match(text, pos=offset)
        if match is None: raise RuntimeError()
        pieces.append(text[match.start():match.end()])
        offset = match.end()
    */
    while (offset < text.size())
    {
        int match_count = pcre2_match(
            pattern,
            subject,
            text.size(),
            offset,
            PCRE2_ANCHORED,
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

    // free memory
    pcre2_match_data_free(match_data);
    pcre2_code_free(pattern);

    return pieces;
}

std::vector<int> merge_pair(
    const std::vector<int> &token_ids,
    std::pair<int, int> pair,
    int new_token_id
) 
{
    // replaces every non-overlapping adjacent pair with new_token_id (left-to-right) if that pair matches our input pattern

    std::vector<int> result;
    std::size_t i = 0;

    while (i < token_ids.size())
    {
        if ((i+1 < token_ids.size()) 
             && (token_ids[i] == pair.first) 
             && (token_ids[i+1] == pair.second))
        {
            result.push_back(new_token_id); // match! add merged token
            i += 2;
        } else {
            result.push_back(token_ids[i]); // no match. add token normally
            i += 1;
        }
    }

    return result;
}


std::map<std::pair<int, int>, int> count_pairs(
    const std::vector<std::vector<int>> &pieces
)
{
    // count pairs across multiple contiguous partitions (split by pre-tokenization)
    // returns a map of [pair] -> freq

    std::map<std::pair<int, int>, int> freq_map = {};

    for (const std::vector<int> &piece : pieces) 
    {
        for (std::size_t i = 0; i+1 < piece.size(); i++) {
            std::pair<int, int> pair = {piece[i], piece[i + 1]};
            freq_map[pair] += 1;
        }
    }

    return freq_map;
}

std::pair<int, int> select_pair(
    const std::map<std::pair<int, int>, int> &counts
)
{
    // select the pair with the highest numerical frequency
    // break ties based off smallest numerical value
    std::pair<int, int> best_pair = counts.begin()->first; // first map entry key (pair)
    int best_count = counts.begin()->second; // first map value (count)

    for (const auto &pair_entry : counts)
    {
        std::pair<int, int> pair = pair_entry.first;
        int count = pair_entry.second;

        if (count > best_count || (count == best_count && pair < best_pair))
        {
            best_pair = pair;
            best_count = count;
        }
    }

    return best_pair;
}

class BPETokenizer
{

public:
    BPETokenizer(int vocab_size)
    {
        mergeable_vocab_size = vocab_size;
    }

    void train(std::vector<std::vector<int>> pieces)
    {
        merges.clear();

        for (int new_token_id = 256; new_token_id < mergeable_vocab_size; new_token_id++)
        {
            std::map<std::pair<int, int>, int> counts = count_pairs(pieces);
            if (counts.empty()) break;
            std::pair<int, int> selected_pair = select_pair(counts);

            for (std::vector<int> &piece : pieces)
                piece = merge_pair(piece, selected_pair, new_token_id);

            merges[selected_pair] = new_token_id;
        }
    }

    const std::map<std::pair<int, int>, int> &get_merges() const
    {
        return merges;
    }

private:
    int mergeable_vocab_size;
    std::map<std::pair<int, int>, int> merges;
};

int main() {
    return 0;
}
