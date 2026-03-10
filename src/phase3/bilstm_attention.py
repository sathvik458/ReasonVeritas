import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    """
    Bi-LSTM + Additive Attention for Fake News Detection
    """

    def __init__(self, vocab_size, embed_dim=300, hidden_size=128, num_classes=2):
        super(BiLSTMAttention, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Bi-LSTM Encoder
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attn_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attn_vector = nn.Linear(hidden_size * 2, 1, bias=False)

        # Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, mask=None):

        # (B,T) → (B,T,E)
        x = self.embedding(input_ids)

        # (B,T,E) → (B,T,2H)
        H, _ = self.bilstm(x)

        # Attention scores
        score = torch.tanh(self.attn_linear(H))
        score = self.attn_vector(score).squeeze(-1)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(score, dim=1)

        # Context vector
        context = torch.sum(H * alpha.unsqueeze(-1), dim=1)

        logits = self.fc(context)

        return logits, alpha