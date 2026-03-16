from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from tokenizer_utils import build_vocab_from_text, save_vocab

CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.txt"


def build_combined_corpus():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RAW_DATA_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in raw data directory: {RAW_DATA_DIR}"
        )

    combined_parts = []
    used_files = []

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            combined_parts.append(text)
            used_files.append((file_path.name, len(text)))

    if not combined_parts:
        raise ValueError(f"Found .txt files in {RAW_DATA_DIR}, but all were empty.")

    combined_text = "\n\n".join(combined_parts) + "\n"
    CORPUS_FILE.write_text(combined_text, encoding="utf-8")

    return combined_text, used_files


def main():
    text, used_files = build_combined_corpus()

    vocab = build_vocab_from_text(text)
    save_vocab(vocab)

    print("Combined files:")
    for name, char_count in used_files:
        print(f"  - {name}: {char_count} chars")

    print(f"\nCorpus file: {CORPUS_FILE}")
    print(f"Characters in corpus: {len(text)}")
    print(f"Vocabulary size: {len(vocab)}")
    print("Saved vocab to models/vocab.json")


if __name__ == "__main__":
    main()
