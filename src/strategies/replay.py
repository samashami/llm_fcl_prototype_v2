# src/strategies/replay.py
import torch
from collections import deque
import random
from typing import Optional

class ReplayBuffer:
    def __init__(self, capacity: int = 2000, seed: Optional[int] = None):
        self.capacity = capacity
        self.data = deque()
        # Preserve the historical global-RNG sampling path unless a private
        # stream is explicitly requested by the attribution protocol.
        self._rng = None if seed is None else random.Random(int(seed))

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        # x: [B, C, H, W], y: [B]
        for i in range(x.size(0)):
            self.data.append((x[i].cpu(), y[i].cpu()))
            while len(self.data) > self.capacity:
                self.data.popleft()

    def sample_like(self, batch_size: int, device, ratio: float = 0.2):
        k = int(batch_size * ratio)
        return self.sample_count(k, device=device)

    def sample_count(self, count: int, device):
        """Sample at most ``count`` items using the configured RNG stream."""
        k = int(count)
        if k <= 0 or len(self.data) == 0:
            return None, None
        k = min(k, len(self.data))
        population = list(self.data)
        samples = (
            random.sample(population, k)
            if self._rng is None
            else self._rng.sample(population, k)
        )
        xs = torch.stack([s[0] for s in samples]).to(device)
        ys = torch.stack([s[1] for s in samples]).to(device)
        return xs, ys

    def state_dict(self):
        """Return a lossless FIFO snapshot of the replay buffer."""
        state = {
            "capacity": int(self.capacity),
            # These tensors already live on CPU. torch.save is synchronous, so
            # retaining their references avoids doubling a multi-GB buffer.
            "data": [(x.detach().cpu(), y.detach().cpu()) for x, y in self.data],
        }
        if self._rng is not None:
            state["rng_state"] = self._rng.getstate()
        return state

    def load_state_dict(self, state):
        """Restore a lossless FIFO snapshot without changing its ordering."""
        if set(state) not in ({"capacity", "data"}, {"capacity", "data", "rng_state"}):
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
        if "rng_state" in state:
            if self._rng is None:
                self._rng = random.Random()
            self._rng.setstate(state["rng_state"])
        else:
            self._rng = None
