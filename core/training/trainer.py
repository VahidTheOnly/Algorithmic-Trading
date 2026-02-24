import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, optimizer, criterion, device, max_grad_norm=1.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.scaler = torch.amp.GradScaler()

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def eval_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
        return loss.item()
    
    def save_checkpoint(self, filepath):
        state = dict()
        state['model_state'] = self.model.state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        state['scaler_state'] = self.scaler.state_dict()
        torch.save(state, filepath)

    def load_checkpoint(self, filepath):
        state = torch.load(filepath, weights_only=True)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scaler.load_state_dict(state['scaler_state'])
