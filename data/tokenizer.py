from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        # pass

        # corpus = corpus.split()
        # seen = {}
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            seen = {}

            # 1. count adjacent pair
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in seen:
                    seen[pair] += 1
                else:
                    seen[pair] = 1

            if not seen:
                break
            
            # 2. most frequent pair, tie break
            best_pair = min(
            seen.keys(),
            key=lambda pair: (-seen[pair], pair))

            merges.append([best_pair[0], best_pair[1]])

            # 3. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges
    