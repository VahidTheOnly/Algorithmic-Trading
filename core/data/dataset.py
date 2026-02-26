from torch.utils.data import Dataset
import torch


class SlidingWindowDataset(Dataset):
    def __init__(self, data: torch.Tensor, look_back: int, look_ahead: int, input_indices: list, target_indices: list):
        super().__init__()

        self.data = data
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.input_indices = input_indices
        self.target_indices = target_indices


    def __len__(self):
        return len(self.data) - self.look_back - self. look_ahead + 1
    
    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        
        t = idx + self.look_back
        x = self.data[idx : t, self.input_indices]
        y = self.data[t : t + self.look_ahead, self.target_indices]
        return x, y
