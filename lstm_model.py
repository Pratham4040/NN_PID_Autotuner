import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class ThermalLSTMPlant(nn.Module):
    """
    LSTM-based plant model for thermal dynamics.

    Input:  sequence (batch, seq_len, 2) with features [T, U]
    Output: next temperature T_k (normalized)
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

        # Xavier init for the output head helps stabilize early training.
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        print("[ThermalLSTMPlant] Initialized LSTM plant model")
        print(f"[ThermalLSTMPlant] Hidden size: {hidden_size}, layers: {num_layers}")

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        return self.head(last)


class LSTMPlantModel:
    """
    Online LSTM plant model wrapper with normalization, training, and diagnostics.

    Training convention:
    - Each sequence contains the last N pairs (T_i, U_i)
    - Target is the next temperature T_{i+1}
    - U_i represents the heater command applied after measuring T_i
    """

    def __init__(
        self,
        seq_len=20,
        temp_ref=37.0,
        temp_scale=10.0,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        verbose=True,
    ):
        self.verbose = verbose
        self.sequence_length = int(seq_len)
        self.is_sequence_model = True

        self.model = ThermalLSTMPlant(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1e-5,
        )
        self.loss_fn = nn.MSELoss()

        self.train_data = deque(maxlen=2000)
        self.val_data = deque(maxlen=500)

        self.train_losses = deque(maxlen=200)
        self.val_losses = deque(maxlen=200)
        self.total_samples_seen = 0
        self.training_steps = 0

        self.temp_ref = temp_ref
        self.temp_scale = temp_scale

        if self.verbose:
            print("\n" + "=" * 70)
            print("[LSTMPlantModel] INITIALIZATION")
            print("=" * 70)
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Temperature normalization: (T - {temp_ref}) / {temp_scale}")
            print(f"  Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
            print(f"  Scheduler: ReduceLROnPlateau (patience=50)")
            print(f"  Buffer sizes: train={self.train_data.maxlen}, val={self.val_data.maxlen}")
            print("=" * 70 + "\n")

    def normalize_temp(self, temp):
        return (temp - self.temp_ref) / self.temp_scale

    def denormalize_temp(self, norm_temp):
        return norm_temp * self.temp_scale + self.temp_ref

    def _normalize_sequence(self, seq_tu):
        # seq_tu is a list of (temp, u) pairs with length sequence_length.
        return [[self.normalize_temp(t), float(u)] for t, u in seq_tu]

    def add_sequence_sample(self, seq_tu, target_temp):
        """
        Add a training sample based on a rolling sequence.

        seq_tu: list of (T, U) pairs, length == sequence_length
        target_temp: next temperature to predict
        """
        if len(seq_tu) != self.sequence_length:
            if self.verbose:
                print(
                    f"[Data] Skipping sample, need {self.sequence_length} steps, got {len(seq_tu)}"
                )
            return

        norm_seq = self._normalize_sequence(seq_tu)
        norm_target = self.normalize_temp(target_temp)

        # 80/20 train/val split
        if self.total_samples_seen % 5 == 0:
            self.val_data.append((norm_seq, norm_target))
            if self.verbose and len(self.val_data) % 50 == 0:
                print(f"[Data] Validation set size: {len(self.val_data)}")
        else:
            self.train_data.append((norm_seq, norm_target))

        self.total_samples_seen += 1
        if self.verbose and self.total_samples_seen % 100 == 0:
            print(
                f"[Data] Total samples collected: {self.total_samples_seen} "
                f"(train={len(self.train_data)}, val={len(self.val_data)})"
            )

    def train_step(self, batch_size=32, num_epochs=1):
        if len(self.train_data) < 50:
            if self.verbose and self.total_samples_seen % 10 == 0:
                print(f"[Train] Waiting for more data... ({len(self.train_data)}/50)")
            return None

        batch_size = min(batch_size, len(self.train_data))
        indices = np.random.choice(len(self.train_data), batch_size, replace=False)
        batch = [self.train_data[i] for i in indices]

        x = torch.tensor([d[0] for d in batch], dtype=torch.float32)
        y = torch.tensor([[d[1]] for d in batch], dtype=torch.float32)

        self.model.train()
        total_loss = 0.0
        for _ in range(num_epochs):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_epochs
        self.train_losses.append(avg_loss)
        self.training_steps += 1

        if self.training_steps % 10 == 0 and len(self.val_data) >= 20:
            val_loss = self._compute_validation_loss()
            self.val_losses.append(val_loss)

            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]

            if self.verbose:
                lr_changed = " (LR REDUCED!)" if new_lr < old_lr else ""
                print(
                    f"[Train] Step {self.training_steps:04d} | "
                    f"Train Loss: {avg_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {new_lr:.2e}{lr_changed}"
                )
        elif self.verbose and self.training_steps % 50 == 0:
            print(f"[Train] Step {self.training_steps:04d} | Loss: {avg_loss:.6f}")

        return avg_loss

    def _compute_validation_loss(self):
        self.model.eval()
        with torch.no_grad():
            x_val = torch.tensor([d[0] for d in self.val_data], dtype=torch.float32)
            y_val = torch.tensor([[d[1]] for d in self.val_data], dtype=torch.float32)
            pred_val = self.model(x_val)
            val_loss = self.loss_fn(pred_val, y_val).item()
        return val_loss

    def predict_from_sequence(self, seq_tu):
        """
        Predict the next temperature from a full sequence.

        seq_tu must be a list of (T, U) pairs with length sequence_length.
        """
        if len(seq_tu) != self.sequence_length:
            raise ValueError(
                f"predict_from_sequence expects {self.sequence_length} steps, got {len(seq_tu)}"
            )

        x = torch.tensor([self._normalize_sequence(seq_tu)], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            norm_pred = self.model(x).item()
            abs_pred = self.denormalize_temp(norm_pred)
        return abs_pred

    def get_training_quality(self, loss_threshold=0.001, min_steps=0):
        if not self.train_losses:
            return False, float("inf"), "No training loss yet"

        recent_loss = np.mean(list(self.train_losses)[-20:])

        if recent_loss < loss_threshold:
            return True, recent_loss, f"Model converged (loss={recent_loss:.6f})"

        return False, recent_loss, f"Loss still too high ({recent_loss:.6f} > {loss_threshold})"

    def print_diagnostics(self):
        is_ready, loss, status = self.get_training_quality()

        print("\n" + "=" * 70)
        print("[LSTMPlantModel] DIAGNOSTICS")
        print("=" * 70)
        print(f"  Training steps:     {self.training_steps}")
        print(f"  Samples collected:  {self.total_samples_seen}")
        print(f"  Train data size:    {len(self.train_data)}")
        print(f"  Val data size:      {len(self.val_data)}")

        if self.train_losses:
            print(f"  Recent train loss:  {np.mean(list(self.train_losses)[-20:]):.6f}")
        if self.val_losses:
            print(f"  Recent val loss:    {np.mean(list(self.val_losses)[-10:]):.6f}")

        print(f"  Current LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Model ready:        {is_ready}")
        print(f"  Status:             {status}")
        print("=" * 70 + "\n")
