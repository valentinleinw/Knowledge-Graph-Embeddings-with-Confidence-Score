import torch
import torch.nn as nn
from pykeen.losses import Loss

class ConfidenceWeightedLoss(Loss):
    def forward(self, predictions: torch.FloatTensor, targets: torch.FloatTensor, confidences: torch.FloatTensor) -> torch.FloatTensor:
        """
        Custom loss function that applies confidence scores to the standard loss.
        """
        
        if confidences is None:
            raise ValueError("Confidence scores not provided in metadata.")
        
        predictions = predictions.float()
        targets = targets.float()
        
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")  # No reduction so we can apply weights
        base_loss = loss_fn(predictions, targets)  # Standard loss
        weighted_loss = base_loss * confidences  # Apply confidence scores
        return weighted_loss.mean()  # Take the mean to keep the loss scalar