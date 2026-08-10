import json


class FormulaTokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges = []
        self.code_map = {}
        self.decode_map = {}

    @staticmethod
    def _normalize(text: str) -> str:
        # remove whitespace
        return "\n".join("".join(line.split()) for line in text.splitlines())

    @staticmethod
    def _merge_pair(tokens: list[str], pair: tuple[str, str], new_token: str):
        a, b = pair
        out = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_token)
                i += 2
            else:
                out.append(tokens[i])
                i += 1

        return out

    def train(self, text: str, n_merges: int):
        text = self._normalize(text)
        tokens = list(text)
        self.vocab = set(tokens)
        self.merges = []

        for _ in range(n_merges):
            pair_counts = {}

            for a, b in zip(tokens, tokens[1:]):
                # One training example per line, skip multi-lines
                if "\n" in a or "\n" in b:
                    continue
                if (a, b) in pair_counts.keys():
                    pair_counts[(a, b)] += 1
                else:
                    pair_counts[(a, b)] = 1

            if not pair_counts:
                break

            # Find most-frequent pair
            candidates = [
                (count, pair)
                for pair, count in pair_counts.items()
                if pair[0] + pair[1] not in self.vocab
            ]

            if not candidates:
                break

            _, (a, b) = max(candidates, key=lambda x: (x[0], x[1]))

            # new token is a+b
            new_token = a + b
            self.vocab.add(new_token)
            self.merges.append((a, b, new_token))

            tokens = self._merge_pair(tokens, (a, b), new_token)

        # Stable IDs
        base_tokens = sorted([t for t in self.vocab if len(t) == 1])

        merged_tokens = [new_token for _, _, new_token in self.merges]

        vocab = base_tokens + merged_tokens

        self.code_map = {token: i for i, token in enumerate(vocab)}

        self.decode_map = {i: token for token, i in self.code_map.items()}

    def tokenize(self, text: str) -> list[str]:
        text = self._normalize(text)
        tokens = list(text)

        for a, b, new_token in self.merges:
            tokens = self._merge_pair(tokens, (a, b), new_token)

        return tokens

    def encode(self, text: str) -> list[int]:
        return [self.code_map[token] for token in self.tokenize(text)]

    def decode(self, code: list[int]) -> str:
        return "".join(self.decode_map[i] for i in code)

    def save(self, path: str):
        data = {
            "merges": self.merges,
            "code_map": self.code_map,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls()

        tokenizer.merges = [tuple(x) for x in data["merges"]]
        tokenizer.code_map = data["code_map"]
        tokenizer.decode_map = {i: token for token, i in tokenizer.code_map.items()}
        tokenizer.vocab = set(tokenizer.code_map)

        return tokenizer
