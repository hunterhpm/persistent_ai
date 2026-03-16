from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import PROCESSED_DATA_DIR
from tokenizer_utils import build_vocab_from_text, save_vocab, load_vocab, encode
from model import TinyTransformerLanguageModel

# -----------------------------
# Paths
# -----------------------------
CORPUS_FILE = PROCESSED_DATA_DIR / "corpus.txt"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "tiny_word_transformer.pt"
LATEST_CHECKPOINT_FILE = MODEL_DIR / "tiny_word_transformer_latest.pt"
BEST_CHECKPOINT_FILE = MODEL_DIR / "tiny_word_transformer_best.pt"

# -----------------------------
# Hyperparameters
# -----------------------------
SEQ_LEN = 48
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
EPOCHS = 30
LR = 0.0005
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_IF_AVAILABLE = True


def build_training_data(token_ids, seq_len):
    inputs = []
    targets = []

    for i in range(len(token_ids) - seq_len):
        x = token_ids[i : i + seq_len]
        y = token_ids[i + 1 : i + seq_len + 1]
        inputs.append(x)
        targets.append(y)

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(
        targets, dtype=torch.long
    )


def save_checkpoint(path, model, optimizer, epoch, avg_loss, vocab_size):
    torch.save(
        {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
        },
        path,
    )


def main():
    print(f"Using device: {DEVICE}")

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_FILE}")

    text = CORPUS_FILE.read_text(encoding="utf-8")

    vocab = build_vocab_from_text(text)
    save_vocab(vocab)

    vocab = load_vocab()
    vocab_size = len(vocab)

    token_ids = encode(text, vocab, add_bos=True, add_eos=True)

    print(f"Corpus token count: {len(token_ids)}")
    print(f"Vocab size: {vocab_size}")

    X, Y = build_training_data(token_ids, SEQ_LEN)

    dataset_size = X.size(0)
    print(f"Training sequences: {dataset_size}")

    if dataset_size == 0:
        raise ValueError(
            "Not enough corpus data to build training sequences. "
            f"Need more than {SEQ_LEN} encoded tokens."
        )

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    start_epoch = 0
    best_loss = float("inf")

    if RESUME_IF_AVAILABLE and LATEST_CHECKPOINT_FILE.exists():
        print(f"Resuming from checkpoint: {LATEST_CHECKPOINT_FILE}")
        checkpoint = torch.load(LATEST_CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("avg_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch + 1}")

    print("Starting training...")

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            total_loss = 0.0

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_x = X[start:end].to(DEVICE)
                batch_y = Y[start:end].to(DEVICE)

                optimizer.zero_grad()

                logits = model(batch_x)
                loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(
                1, (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
            )
            print(f"Epoch {epoch + 1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

            save_checkpoint(
                LATEST_CHECKPOINT_FILE,
                model,
                optimizer,
                epoch,
                avg_loss,
                vocab_size,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    BEST_CHECKPOINT_FILE,
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    vocab_size,
                )
                print(f"New best checkpoint saved: {BEST_CHECKPOINT_FILE}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print(f"Latest checkpoint preserved at: {LATEST_CHECKPOINT_FILE}")

    final_source = (
        BEST_CHECKPOINT_FILE
        if BEST_CHECKPOINT_FILE.exists()
        else LATEST_CHECKPOINT_FILE
    )

    if final_source.exists():
        checkpoint = torch.load(final_source, map_location="cpu")
        torch.save(
            {
                "model_state_dict": checkpoint["model_state_dict"],
                "vocab_size": checkpoint["vocab_size"],
                "d_model": checkpoint["d_model"],
                "nhead": checkpoint["nhead"],
                "num_layers": checkpoint["num_layers"],
                "dim_feedforward": checkpoint["dim_feedforward"],
                "dropout": checkpoint["dropout"],
                "seq_len": checkpoint["seq_len"],
            },
            MODEL_FILE,
        )
        print(f"Exported inference model to: {MODEL_FILE}")
    else:
        print("No checkpoint found to export.")


if __name__ == "__main__":
    main()
