# src/strategies/replay.py
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.data = deque()

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        # x: [B, C, H, W], y: [B]
        for i in range(x.size(0)):
            self.data.append((x[i].cpu(), y[i].cpu()))
            while len(self.data) > self.capacity:
                self.data.popleft()

    def sample_like(self, batch_size: int, device, ratio: float = 0.2):
        k = int(batch_size * ratio)
        if k <= 0 or len(self.data) == 0:
            return None, None
        k = min(k, len(self.data))
        samples = random.sample(self.data, k)
        xs = torch.stack([s[0] for s in samples]).to(device)
        ys = torch.stack([s[1] for s in samples]).to(device)
        return xs, ys

    def state_dict(self):
        """Return a lossless FIFO snapshot of the replay buffer."""
        return {
            "capacity": int(self.capacity),
            # These tensors already live on CPU. torch.save is synchronous, so
            # retaining their references avoids doubling a multi-GB buffer.
            "data": [(x.detach().cpu(), y.detach().cpu()) for x, y in self.data],
        }

    def load_state_dict(self, state):
        """Restore a lossless FIFO snapshot without changing its ordering."""
        if set(state) != {"capacity", "data"}:
            raise ValueError("invalid replay-buffer state")
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError("replay-buffer capacity must be positive")
        data = list(state["data"])
        if len(data) > capacity:
            raise ValueError("replay-buffer state exceeds its capacity")
        restored = deque()
        for item in data:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("replay-buffer entries must be (x, y) pairs")
            x, y = item
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise TypeError("replay-buffer entries must contain tensors")
            restored.append((x.detach().cpu(), y.detach().cpu()))
        self.capacity = capacity
        self.data = restored
