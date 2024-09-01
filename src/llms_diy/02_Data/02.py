import importlib.metadata
import tiktoken
import re

from supplementary import create_dataloader_v1

TOKENIZER_REGEX = r'([,.:;?_!"()\']|--|\s)'
DECODER_REGEX = r'\s+([,.?!()\'])'


def simple_tokenizer(raw_text):
    preprocessed = re.split(TOKENIZER_REGEX, raw_text)
    preprocessed = [item for item in preprocessed if item]
    print("Number of tokens:", len(preprocessed))
    print(preprocessed[:38])
    return preprocessed


def get_vocab(preprocessed):
    unique_tokens = set(preprocessed)
    print("VocabSize/Number of unique tokens:", len(unique_tokens))
    all_words = sorted(unique_tokens)
    return { token: integer for integer, token in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(TOKENIZER_REGEX, text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(DECODER_REGEX, r'\1', text)
        return text

if __name__ == '__main__':
    print("tiktoken version: ", importlib.metadata.version("tiktoken"))

    with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total num chars: ", len(raw_text))
    print(raw_text[:99])

    preprocessed = simple_tokenizer(raw_text)
    vocab = get_vocab(preprocessed)

    tokenizer = SimpleTokenizerV1(vocab)
    test_string = "I found the couple at tea beneath their palm-trees"
    encoded = tokenizer.encode(test_string)
    print("original string: ", test_string)
    print("encoded: ", encoded)
    decoded = tokenizer.decode(encoded)
    print("decoded: ", decoded)

    BPETokenizer = tiktoken.get_encoding("gpt2")

    text = "TEST How do you like tea? In the sunlit terraces"
    integers = BPETokenizer.encode(text)
    print(integers)
    print(BPETokenizer.encode("asdfjk;lxyzmnop;"))

    dataloader = create_dataloader_v1(raw_text, batch_size=0,
                                      max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)