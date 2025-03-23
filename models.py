import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        Args:
            input_dim (int): total input size (125 + positional_encoding_dim).
            hidden_dim (int): size of hidden layers in the MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # single logit output
        )

    def forward(self, x):
        """
        Args:
            x (FloatTensor): shape (batch_size, input_dim)
        Returns:
            logit (FloatTensor): shape (batch_size, 1)
        """
        logit = self.net(x)
        return logit

    def predict_hard(self, x):
        """
        Returns the *hard* 0/1 predictions. 
        You can call this at inference time.
        """
        logit = self.forward(x)
        # Sigmoid, then threshold at 0.5
        probs = torch.sigmoid(logit)  # in [0,1]
        preds = (probs > 0.5).long()  # 0 or 1
        return preds

    def predict_proba(self, x):
        """
        Returns the probability (sigmoid) – you can interpret 
        this as confidence for Dire winning.
        """
        logit = self.forward(x)
        return torch.sigmoid(logit)

    def uncertainty_score(self, x):
        """
        If you want a measure of uncertainty, you can return
        the raw logit or the distance from 0.5 after sigmoid, etc.
        Here we simply return the raw logits.
        """
        logit = self.forward(x)
        return logit
