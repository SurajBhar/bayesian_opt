import torch
from torchmetrics.classification import (
    Precision,
    F1Score,
    MatthewsCorrCoef,
    AUROC,
    Recall,
    ConfusionMatrix,
)

class Metrics:
    """Provides methods to calculate evaluation metrics for model performance."""

    @staticmethod
    def calculate_balanced_accuracy_manual(y_pred, y_true, num_classes):
        """Calculates the balanced accuracy manually across given predictions and true labels.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Balanced accuracy score (manual implementation).
        """
        correct_per_class = torch.zeros(num_classes, device=y_pred.device)
        total_per_class = torch.zeros(num_classes, device=y_pred.device)
        for c in range(num_classes):
            true_positives = ((y_pred == c) & (y_true == c)).sum()
            condition_positives = (y_true == c).sum()
            correct_per_class[c] = true_positives.float()
            total_per_class[c] = condition_positives.float()
        recall_per_class = correct_per_class / total_per_class.clamp(min=1)
        return recall_per_class.mean().item()

    @staticmethod
    def calculate_balanced_accuracy_torchmetrics(y_pred, y_true, num_classes):
        """Calculates the balanced accuracy using TorchMetrics Recall with averaging.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Balanced accuracy score (TorchMetrics implementation).
        """
        recall_metric = Recall(task="multiclass", num_classes=num_classes, average="macro").to(y_pred.device)
        return recall_metric(y_pred, y_true).item()

    @staticmethod
    def calculate_precision(y_pred, y_true, num_classes):
        """Calculates the macro-averaged precision.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Macro-averaged precision score.
        """
        precision_metric = Precision(task="multiclass", num_classes=num_classes, average="macro").to(y_pred.device)
        return precision_metric(y_pred, y_true).item()

    @staticmethod
    def calculate_f1_score(y_pred, y_true, num_classes):
        """Calculates the macro-averaged F1 score.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Macro-averaged F1 score.
        """
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(y_pred.device)
        return f1_metric(y_pred, y_true).item()

    @staticmethod
    def calculate_mcc(y_pred, y_true, num_classes):
        """Calculates the Matthews Correlation Coefficient (MCC).

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: MCC score.
        """
        mcc_metric = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(y_pred.device)
        return mcc_metric(y_pred, y_true).item()

    @staticmethod
    def calculate_auroc(y_probs, y_true, num_classes):
        """Calculates the macro-averaged AUROC score.

        Args:
            y_probs (torch.Tensor): Predicted probabilities from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            float: Macro-averaged AUROC score.
        """
        auroc_metric = AUROC(task="multiclass", num_classes=num_classes, average="macro").to(y_probs.device)
        return auroc_metric(y_probs, y_true).item()

    @staticmethod
    def calculate_confusion_matrix(y_pred, y_true, num_classes):
        """Calculates the confusion matrix.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Actual labels from the dataset.
            num_classes (int): Number of different classes in the dataset.

        Returns:
            torch.Tensor: Confusion matrix.
        """
        cm_metric = ConfusionMatrix(num_classes=num_classes).to(y_pred.device)
        return cm_metric(y_pred, y_true).cpu().numpy()
