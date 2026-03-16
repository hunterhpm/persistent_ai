import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_FILE = MODEL_DIR / "vocab.json"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
NEWLINE_TOKEN = "<NL>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NEWLINE_TOKEN]

# Matches:
# - words with optional apostrophes: don't, user's
# - numbers/decimals: 12, 3.14
# - punctuation/symbols as separate tokens
TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE
)


def save_vocab(vocab, vocab_file=VOCAB_FILE):
    vocab_file = Path(vocab_file)
    vocab_file.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_vocab(vocab_file=VOCAB_FILE):
    vocab_file = Path(vocab_file)

    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_file}\n"
            "Make sure vocab.json exists in the models folder."
        )

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict):
        raise ValueError("Vocabulary file must contain a JSON object/dict.")

    return vocab


def tokenize_text(text):
    lines = text.splitlines()
    tokens = []

    for i, line in enumerate(lines):
        line_tokens = TOKEN_PATTERN.findall(line)
        tokens.extend(line_tokens)

        if i < len(lines) - 1:
            tokens.append(NEWLINE_TOKEN)

    return tokens


def build_vocab_from_text(text):
    tokens = tokenize_text(text)
    unique_tokens = sorted(set(tokens))

    vocab = {}
    next_id = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = next_id
        next_id += 1

    for token in unique_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

    return vocab


def encode(text, vocab, add_bos=False, add_eos=False):
    unk_id = vocab[UNK_TOKEN]
    tokens = tokenize_text(text)

    token_ids = []

    if add_bos:
        token_ids.append(vocab[BOS_TOKEN])

    token_ids.extend(vocab.get(token, unk_id) for token in tokens)

    if add_eos:
        token_ids.append(vocab[EOS_TOKEN])

    return token_ids


def decode(token_ids, vocab):
    if not token_ids:
        return ""

    id_to_token = {idx: token for token, idx in vocab.items()}

    output_parts = []
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "}", "%"}
    no_space_after = {"(", "[", "{", "$", "#"}
    special_skip = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}

    for token_id in token_ids:
        token = id_to_token.get(int(token_id), UNK_TOKEN)

        if token in special_skip:
            continue

        if token == NEWLINE_TOKEN:
            output_parts.append("\n")
            continue

        if not output_parts:
            output_parts.append(token)
            continue

        prev = output_parts[-1]

        if prev.endswith("\n"):
            output_parts.append(token)
        elif token in no_space_before:
            output_parts[-1] = prev + token
        elif prev and prev[-1] in no_space_after:
            output_parts[-1] = prev + token
        else:
            output_parts.append(" " + token)

    return "".join(output_parts)
