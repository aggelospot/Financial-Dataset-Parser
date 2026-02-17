import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from utils import evaluate_bankruptcy_model
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    average_precision_score, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit








# ==========================================
# A Generalized Torch Model class

class GeneralizedTorchModel:
    """
    A PyTorch-based binary classification model
    that mimics the .fit(), .predict(), .predict_proba() interface
    of scikit-learn.

    You can configure it to behave like logistic regression,
    or a small MLP, by changing 'architecture'.
    """

    def __init__(
        self,
        input_dim,
        architecture='logistic',
        hidden_dim=32,
        epochs=10,
        lr=1e-3,
        random_state=42
    ):
        """
        input_dim: Number of features in input
        architecture: 'logistic' or 'mlp'
        hidden_dim: Size of hidden layer if using 'mlp'
        epochs: Number of training epochs
        lr: Learning rate
        random_state: Seed for reproducibility
        """
        self.input_dim = input_dim
        self.architecture = architecture.lower()
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if self.architecture == 'logistic':
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, 1)
            )
            self.use_logits = True
        elif self.architecture == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )
            self.use_logits = True
        else:
            raise ValueError("Unsupported architecture. Choose 'logistic' or 'mlp'.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, X, y):
        """
        X: numpy array of shape [n_samples, n_features]
        y: numpy array of shape [n_samples,], binary labels (0 or 1)
        """
        self.model.train()

        # Convert to tensors
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_t)  # shape: [n_samples, 1]
            loss = self.criterion(logits, y_t)
            loss.backward()
            self.optimizer.step()

            # print or track training loss
            # if (epoch+1) % 5 == 0:
            #     print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        return self

    def predict_proba(self, X):
        """
        Returns predicted probability for the positive class.
        X: numpy array [n_samples, n_features]
        """
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            logits = self.model(X_t)
            if self.use_logits:
                probs = torch.sigmoid(logits)
            else:
                probs = logits
        return probs.view(-1).numpy()  # 1D array of shape [n_samples,]

    def predict(self, X):
        """
        Returns hard class predictions (0 or 1).
        X: numpy array [n_samples, n_features]
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

# ==========================================
# Updated run_bankruptcy_experiments


# ==========================================
# Example usage

