import torch

class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch.nn.CrossEntropyLoss()
    
    def forward(self, y_pred, y_true):
        return self.func(y_pred, y_true)

class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch.nn.BCELoss()
    
    def forawd(self, y_pred, y_true):
        return self.func(y_pred, y_true)
