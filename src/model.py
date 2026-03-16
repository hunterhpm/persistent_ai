import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        pos_emb = self.position_embedding(positions)
        return x + pos_emb


class TinyTransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(self, x):
        _, seq_len = x.size()

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x = self.token_embedding(x) * (self.d_model**0.5)
        x = self.positional_encoding(x)

        causal_mask = self._generate_causal_mask(seq_len, x.device)

        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.output_head(x)

        return logits
