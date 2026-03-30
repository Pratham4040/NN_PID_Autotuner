import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PlantNN(nn.Module):
    """
    Enhanced Neural Network for Plant Dynamics Modeling
    
    Architecture: 4 → 64 → 64 → 64 → 1 with BatchNorm and Dropout
    - Input: [T[k-1], T[k-2], U[k-1], U[k-2]] (normalized)
    - Output: T[k] (normalized)
    
    Key Improvements:
    1. Deeper network (3 hidden layers) for complex dynamics
    2. Batch normalization for training stability
    3. Dropout for regularization
    4. ReLU activations (better gradients than Tanh)
    """
    
    def __init__(self, hidden_size=64, dropout_rate=0.15):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size, 1)
        )
        
        # Xavier initialization for better gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        print("[PlantNN] ✓ Initialized 4→64→64→64→1 architecture")
        print(f"[PlantNN] ✓ Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, x):
        return self.net(x)


class NeuralPlantModel:
    """
    Complete Neural Plant Model with Training, Prediction, and Diagnostics
    
    Features:
    - Automatic temperature normalization
    - Mini-batch training with learning rate scheduling
    - Training quality metrics
    - Validation set for overfitting detection
    """
    
    def __init__(self, temp_ref=37.0, temp_scale=10.0, verbose=True):
        """
        Args:
            temp_ref: Reference temperature for normalization (typically setpoint)
            temp_scale: Expected temperature deviation range (±10°C is reasonable)
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        
        # Model architecture
        self.model = PlantNN(hidden_size=64, dropout_rate=0.15)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Learning rate scheduler: reduce LR when loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=50, 
            min_lr=1e-5
        )
        # Note: verbose parameter removed for PyTorch compatibility
        
        self.loss_fn = nn.MSELoss()
        
        # Data buffers
        self.train_data = deque(maxlen=2000)  # Rolling window of training samples
        self.val_data = deque(maxlen=500)     # Validation set
        
        # Normalization parameters
        self.temp_ref = temp_ref
        self.temp_scale = temp_scale
        
        # Training metrics
        self.train_losses = deque(maxlen=200)
        self.val_losses = deque(maxlen=200)
        self.total_samples_seen = 0
        self.training_steps = 0
        
        if self.verbose:
            print("\n" + "="*70)
            print("[NeuralPlantModel] INITIALIZATION")
            print("="*70)
            print(f"  Temperature normalization: (T - {temp_ref}) / {temp_scale}")
            print(f"  Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
            print(f"  Scheduler: ReduceLROnPlateau (patience=50)")
            print(f"  Buffer sizes: train={self.train_data.maxlen}, val={self.val_data.maxlen}")
            print("="*70 + "\n")
    
    def normalize_temp(self, temp):
        """Convert absolute temperature to normalized value (~[-1, 1] range)"""
        return (temp - self.temp_ref) / self.temp_scale
    
    def denormalize_temp(self, norm_temp):
        """Convert normalized temperature back to absolute value"""
        return norm_temp * self.temp_scale + self.temp_ref
    
    def add_sample(self, t1, t2, u1, u2, target):
        """
        Add a new training sample
        
        Args:
            t1, t2: Previous temperatures T[k-1], T[k-2]
            u1, u2: Previous heater powers U[k-1], U[k-2]
            target: Current temperature T[k]
        """
        # Normalize temperatures, keep power as-is (already in [0,1])
        norm_sample = (
            [self.normalize_temp(t1), self.normalize_temp(t2), u1, u2],
            self.normalize_temp(target)
        )
        
        # 80/20 train/val split
        if self.total_samples_seen % 5 == 0:  # Every 5th sample goes to validation
            self.val_data.append(norm_sample)
            if self.verbose and len(self.val_data) % 50 == 0:
                print(f"[Data] Validation set size: {len(self.val_data)}")
        else:
            self.train_data.append(norm_sample)
        
        self.total_samples_seen += 1
        
        if self.verbose and self.total_samples_seen % 100 == 0:
            print(f"[Data] Total samples collected: {self.total_samples_seen} "
                  f"(train={len(self.train_data)}, val={len(self.val_data)})")
    
    def train_step(self, batch_size=32, num_epochs=1):
        """
        Perform one training step with mini-batch
        
        Args:
            batch_size: Number of samples per batch
            num_epochs: Number of passes through the batch
            
        Returns:
            Average training loss (None if insufficient data)
        """
        if len(self.train_data) < 50:
            if self.verbose and self.total_samples_seen % 10 == 0:
                print(f"[Train] Waiting for more data... ({len(self.train_data)}/50)")
            return None
        
        # Sample mini-batch randomly
        batch_size = min(batch_size, len(self.train_data))
        indices = np.random.choice(len(self.train_data), batch_size, replace=False)
        batch = [self.train_data[i] for i in indices]
        
        x = torch.tensor([d[0] for d in batch], dtype=torch.float32)
        y = torch.tensor([[d[1]] for d in batch], dtype=torch.float32)
        
        # Training mode (enables dropout and batch norm)
        self.model.train()
        
        total_loss = 0.0
        for epoch in range(num_epochs):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_epochs
        self.train_losses.append(avg_loss)
        self.training_steps += 1
        
        # Compute validation loss periodically
        if self.training_steps % 10 == 0 and len(self.val_data) >= 20:
            val_loss = self._compute_validation_loss()
            self.val_losses.append(val_loss)
            
            # Check if LR will change
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)  # Adjust LR based on validation performance
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if self.verbose:
                lr_changed = " (LR REDUCED!)" if new_lr < old_lr else ""
                print(f"[Train] Step {self.training_steps:04d} | "
                      f"Train Loss: {avg_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {new_lr:.2e}{lr_changed}")
        elif self.verbose and self.training_steps % 50 == 0:
            print(f"[Train] Step {self.training_steps:04d} | Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def _compute_validation_loss(self):
        """Compute loss on validation set (used for LR scheduling and overfitting detection)"""
        self.model.eval()  # Disable dropout and batch norm updates
        
        with torch.no_grad():
            x_val = torch.tensor([d[0] for d in self.val_data], dtype=torch.float32)
            y_val = torch.tensor([[d[1]] for d in self.val_data], dtype=torch.float32)
            
            pred_val = self.model(x_val)
            val_loss = self.loss_fn(pred_val, y_val).item()
        
        return val_loss
    
    def predict(self, t1, t2, u1, u2):
        """
        Predict next temperature given current state
        
        Args:
            t1, t2: Previous temperatures (absolute values)
            u1, u2: Previous heater powers
            
        Returns:
            Predicted temperature (absolute value)
        """
        x = torch.tensor([[
            self.normalize_temp(t1),
            self.normalize_temp(t2),
            u1, u2
        ]], dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            norm_pred = self.model(x).item()
            abs_pred = self.denormalize_temp(norm_pred)
        
        return abs_pred
    
    def get_training_quality(self, loss_threshold=0.01, min_steps=100):
        """
        Check if model has trained sufficiently for parameter extraction
        
        Args:
            loss_threshold: Maximum acceptable validation loss
            min_steps: Minimum training steps before considering ready
            
        Returns:
            (is_ready, current_loss, status_message)
        """
        if self.training_steps < min_steps:
            return False, float('inf'), f"Need more training steps ({self.training_steps}/{min_steps})"
        
        if len(self.val_losses) < 10:
            recent_loss = np.mean(list(self.train_losses)[-20:]) if self.train_losses else float('inf')
            return False, recent_loss, "Insufficient validation data"
        
        recent_val_loss = np.mean(list(self.val_losses)[-10:])
        recent_train_loss = np.mean(list(self.train_losses)[-20:])
        
        # Check for overfitting
        overfitting_ratio = recent_val_loss / (recent_train_loss + 1e-8)
        
        if overfitting_ratio > 2.0:
            return False, recent_val_loss, f"Overfitting detected (val/train = {overfitting_ratio:.2f})"
        
        if recent_val_loss < loss_threshold:
            return True, recent_val_loss, f"Model converged (loss={recent_val_loss:.6f})"
        
        return False, recent_val_loss, f"Loss still too high ({recent_val_loss:.6f} > {loss_threshold})"
    
    def print_diagnostics(self):
        """Print detailed training diagnostics"""
        is_ready, loss, status = self.get_training_quality()
        
        print("\n" + "="*70)
        print("[NeuralPlantModel] DIAGNOSTICS")
        print("="*70)
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
        print("="*70 + "\n")