
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class AccountBiGRU(nn.Module):

    def __init__(
        self,
        input_size=5,
        hidden_size=64,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Regularization applied to the final representation
        self.dropout = nn.Dropout(dropout)

        # One output logit per account
        self.classifier = nn.Linear(
            hidden_size * self.num_directions,
            1
        )

    def forward(self, sequences, lengths):

        # --------------------------------------------------
        # 1. Validate input dimensions
        # --------------------------------------------------

        if sequences.ndim != 3:
            raise ValueError(
                "Expected sequences with shape [batch, time, features]"
            )

        if lengths.ndim != 1:
            raise ValueError(
                "Expected lengths with shape [batch]"
            )

        batch_size, max_length, feature_count = sequences.shape

        if feature_count != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {feature_count}"
            )

        if len(lengths) != batch_size:
            raise ValueError(
                "Number of lengths must match batch size"
            )

        # pack_padded_sequence requires positive lengths,
        # sorted in descending order when enforce_sorted=True.
        if torch.any(lengths <= 0):
            raise ValueError("Sequence lengths must be positive")

        if torch.any(lengths > max_length):
            raise ValueError(
                "A sequence length exceeds the padded sequence size"
            )

        if torch.any(lengths[:-1] < lengths[1:]):
            raise ValueError(
                "Sequences must be sorted by descending length"
            )

        # --------------------------------------------------
        # 2. Pack variable-length sequences
        # --------------------------------------------------

        packed_sequences = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # --------------------------------------------------
        # 3. Process sequences with the BiGRU
        # --------------------------------------------------

        _, hidden = self.gru(packed_sequences)

        # hidden shape:
        # [num_layers * 2, batch_size, hidden_size]

        # Reshape to separate layers and directions
        hidden = hidden.view(
            self.num_layers,
            self.num_directions,
            batch_size,
            self.hidden_size
        )

        # Get the final layer's forward and backward states
        final_layer_hidden = hidden[-1]

        forward_hidden = final_layer_hidden[0]
        backward_hidden = final_layer_hidden[1]

        # Concatenate both directions
        account_representation = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        # Shape: [batch_size, hidden_size * 2]

        # --------------------------------------------------
        # 4. Classification
        # --------------------------------------------------

        account_representation = self.dropout(
            account_representation
        )

        logits = self.classifier(account_representation)

        # Shape: [batch_size]
        return logits.squeeze(-1)
