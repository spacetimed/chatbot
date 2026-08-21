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

std::vector<int> bytes_to_ids(const std::string &piece)
{
    std::vector<int> res;
    for (unsigned char byte : piece) // "A" -> 65
        res.push_back(byte); // convert byte from unsigned char to int
    return res;
}

class BPETokenizer
{

public:
    BPETokenizer(int vocab_size)
    {
        mergeable_vocab_size = vocab_size;
    }

    void train(const std::string &text)
    {

        // pretokenize raw text into boundary-respecting chunks of strings
        std::vector<std::string> str_pieces = pretokenize(text);

        // convert all string chunks into integer chunks
        std::vector<std::vector<int>> int_pieces;
        for (const std::string &piece : str_pieces)
            int_pieces.push_back(bytes_to_ids(piece));

        merges.clear();

        for (int new_token_id = 256; new_token_id < mergeable_vocab_size; new_token_id++)
        {
            std::map<std::pair<int, int>, int> counts = count_pairs(int_pieces);
            if (counts.empty()) break;
            std::pair<int, int> selected_pair = select_pair(counts);

            for (std::vector<int> &piece : int_pieces)
                piece = merge_pair(piece, selected_pair, new_token_id);

            merges[selected_pair] = new_token_id;
        }
    }

    const std::map<std::pair<int, int>, int> &get_merges() const
    {
        return merges;
    }

    std::vector<int> encode_piece(const std::string &piece) const
    {
        // pretokenize(input text) -> pieces -> encode_piece(piece), ...
        //   takes one pre-tokenized piece (string), converts it to token ID's using existing learned merge rules
        std::vector<int> token_ids = bytes_to_ids(piece); // convert string to list of id's

        // algorithm repeatedly scans entire sequence, checks all adjacent pairs against learned merge rules
        // tracks the lowest token ID (rule learned earliest) till end of string, merges that, repeats

        while (token_ids.size() >= 2)
        {
            std::pair<int, int> selected_pair;
            int selected_token_id = -1;

            // find the applicable pair whose rule was learned earliest
            // select all pairs, and find merge rule with lowest token ID. 
            for (std::size_t i = 0; i+1 < token_ids.size(); i++)
            {
                std::pair<int, int> current_pair = {token_ids[i], token_ids[i+1]};

                // if this current pair is not in our merge rules, skip
                if (!merges.contains(current_pair)) continue;

                int current_token_id = merges.at(current_pair);

                // smaller token ID means the rule was learned earlier
                // if first candidate, or smaller than current candidate
                if (selected_token_id == -1 || current_token_id < selected_token_id)
                {
                    selected_pair = current_pair;
                    selected_token_id = current_token_id;
                }
            }

            // no learned merge rule applied and we've reached end of input
            if (selected_token_id == -1) break;

            token_ids = merge_pair(token_ids, selected_pair, selected_token_id);
        }
        
        return token_ids;
    }


private:
    int mergeable_vocab_size;
    std::map<std::pair<int, int>, int> merges;
};

int main() 
{
    BPETokenizer tokenizer(260);
    tokenizer.train("hello hello hello");

    for (const auto &entry : tokenizer.get_merges())
    {
        std::pair<int, int> pair = entry.first;
        int new_token_id = entry.second;
        std::cout << pair.first << ", " << pair.second
                  << " -> " << new_token_id << "\n";
    }
}
