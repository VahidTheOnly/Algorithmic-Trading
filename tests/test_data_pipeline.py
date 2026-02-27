import torch
import pytest
from core.data.dataset import SlidingWindowDataset


def test_dataset_length():
    data_length = 10
    look_back = 3
    look_ahead = 2
    data = torch.randn(data_length, 5)

    dataset = SlidingWindowDataset(
        data=data, 
        look_back=look_back, 
        look_ahead=look_ahead, 
        input_indices=[0, 1, 2, 3, 4], 
        target_indices=[3]
    )
    
    assert len(dataset) == 6, f"Expected 6 windows, but got {len(dataset)}"
    
    
def test_dataset_shapes():
    data = torch.randn(100, 5)
    look_back = 10
    look_ahead = 5

    dataset = SlidingWindowDataset(
        data=data, 
        look_back=look_back, 
        look_ahead=look_ahead, 
        input_indices=[0, 1, 2, 3, 4], 
        target_indices=[3]
    )

    x_0, y_0 = dataset[0]
    assert x_0.shape == (10, 5), f"Expected x shape (10, 5), got {x_0.shape}"
    assert y_0.shape == (5, 1), f"Expected y shape (5, 1), got {y_0.shape}"


def test_data_leakage():
    data = torch.arange(100).view(100, 1).float()
    look_back = 10
    look_ahead = 5

    dataset = SlidingWindowDataset(
        data=data, 
        look_back=look_back, 
        look_ahead=look_ahead, 
        input_indices=[0], 
        target_indices=[0]
    )

    x_0, y_0 = dataset[0]
    assert y_0[0, 0] == x_0[-1, 0] + 1
