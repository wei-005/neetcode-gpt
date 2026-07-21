from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        # pass
        # self.stoi = defaultdict()
        # self.itos = defaultdict()

        chars = sorted(set(text))

        # self.stoi = {char: i for i, char in enumerate(chars)}
        # self.itos = {i: char for char, i in self.stoi.items()}
        stoi = {char: i for i, char in enumerate(chars)}
        itos = {i: char for char, i in stoi.items()}

        return (stoi, itos)
        
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        # pass
        return [stoi[char] for char in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        # pass
        return ''.join(itos[i] for i in ids)

