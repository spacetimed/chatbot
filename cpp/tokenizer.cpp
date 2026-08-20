#include <iostream>
#include <vector>
#include <utility>

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

int main() {
    std::vector<int> token_ids = {1, 2, 1, 2, 3};
    std::pair<int, int> pair = {1, 2};

    std::vector<int> result = merge_pair(
        token_ids,
        pair,
        256
    );

    for (int token_id : result)
    {
        std::cout << token_id << " ";
    }

    std::cout << "\n";
}