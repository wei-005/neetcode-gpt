import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # pass
        sentences = positive + negative

        # 1. build vocabulary
        words = set()
        for sentence in sentences:
            words.update(sentence.split())

        word_to_id = {word: i 
                        for i, word in enumerate(sorted(words), start=1)
                     }

        # 2. encode each sentence
        tensors = []
        for sentence in sentences:
            ids = [word_to_id[word] for word in sentence.split()]
            tensors.append(torch.tensor(ids, dtype=torch.float))

        # 3. pad shorter sentences
        dataset = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0
        )

        return dataset