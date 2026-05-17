import torch
import torch.nn as nn
import torch.optim as optim


class ThermalLSTMPlant(nn.Module):
    """
    LSTM plant predictor.

    Input:  (batch, seq_len, 2) with features [T, U]
    Output: (batch, 1) predicted next temperature (normalized)
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        return self.head(last)


class LSTMFuzzyPlantModel:
    """
    Online LSTM predictor with normalization and single-step training.
    """

    def __init__(
        self,
        seq_len=30,
        temp_ref=37.0,
        temp_scale=10.0,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        lr=0.001,
        weight_decay=1e-5,
        device=None,
        verbose=True,
    ):
        self.sequence_length = int(seq_len)
        self.temp_ref = float(temp_ref)
        self.temp_scale = float(temp_scale)
        self.verbose = verbose

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = ThermalLSTMPlant(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        if self.verbose:
            print("\n" + "=" * 70)
            print("[LSTMFuzzyPlantModel] INITIALIZATION")
            print("=" * 70)
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Temperature normalization: (T - {self.temp_ref}) / {self.temp_scale}")
            print(f"  Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
            print(f"  Device: {self.device}")
            print("=" * 70 + "\n")

    def normalize_temp(self, temp):
        return (temp - self.temp_ref) / self.temp_scale

    def denormalize_temp(self, norm_temp):
        return norm_temp * self.temp_scale + self.temp_ref

    def _sequence_to_tensor(self, seq_tu):
        if len(seq_tu) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} steps, got {len(seq_tu)}"
            )
        seq_norm = [[self.normalize_temp(t), float(u)] for t, u in seq_tu]
        return torch.tensor([seq_norm], dtype=torch.float32, device=self.device)

    def predict_next(self, seq_tu):
        x = self._sequence_to_tensor(seq_tu)
        self.model.eval()
        with torch.no_grad():
            norm_pred = self.model(x).item()
        return self.denormalize_temp(norm_pred)

    def train_step(self, seq_tu, target_temp):
        x = self._sequence_to_tensor(seq_tu)
        y = torch.tensor(
            [[self.normalize_temp(target_temp)]],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.train()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())
