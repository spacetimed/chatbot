#include <iostream>
#include <vector>
#include <utility>
#include <map>

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

int main() {
    std::vector<std::vector<int>> pieces = {
        {1, 2, 1, 2, 3}
    };

    std::map<std::pair<int, int>, int> counts = count_pairs(pieces);

    std::cout << counts.at({1, 2}) << "\n";
}
