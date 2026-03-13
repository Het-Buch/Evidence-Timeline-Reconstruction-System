import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=2000):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pe = torch.zeros(max_len, d_model, device=device)

        position = torch.arange(0, max_len, device=device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim=512,
        d_model=256,
        n_heads=8,
        num_layers=4,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        # GPU device automatically selected
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

        # Move entire model to GPU automatically
        self.to(self.device)

    def forward(self, x):

        # Ensure input also moves to GPU
        x = x.to(self.device)

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.transformer(x)

        logits = self.classifier(x)

        return logits