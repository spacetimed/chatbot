#include "tokenizer.hpp"

#include <iostream>

int main()
{
    std::map<std::string, int> special_tokens = {{"<|endoftext|>", 260}};
    BPETokenizer tokenizer(260, special_tokens);
    tokenizer.train("hello hello hello");

    for (const std::pair<const std::pair<int, int>, int> &entry : tokenizer.get_merges())
    {
        std::pair<int, int> pair = entry.first;
        int new_token_id = entry.second;
        std::cout << pair.first << ", " << pair.second << " -> " << new_token_id << "\n";
    }

    std::set<std::string> allowed_special = {"<|endoftext|>"};
    std::vector<int> token_ids = tokenizer.encode("hello<|endoftext|>hello", allowed_special);
    std::cout << tokenizer.decode_bytes(token_ids) << "\n";
}
