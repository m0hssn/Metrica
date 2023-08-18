from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy
from torchmetrics.classification import Recall
from torchmetrics.classification import Precision
from torchmetrics.classification import AUROC
from torchmetrics.classification import MulticlassAveragePrecision

class Metrica:
    def __init__(self, num_classes=3, task="multiclass"):
        """
        Initializes the Metrica class with the specified number of classes and task type.
        Args:
            num_classes (int): The number of classes for the classification task.
            task (str): The type of classification task, either "multiclass" or "binary".
        """
        self.num_classes = num_classes
        self.task = task
        self.y_hat = []  # List to store predicted values
        self.y = []  # List to store ground truth values

    def upgrade(self, y_hat: torch.tensor, y: torch.tensor):
        """
        Method to upgrade the metric by adding new predicted and ground truth values.
        Args:
            y_hat (torch.tensor): Predicted values
            y (torch.tensor): Ground truth values
        """
        self.y.append(y)
        self.y_hat.append(y_hat)

    def reset(self):
        """
        Method to reset the metric by clearing the stored predicted and ground truth values.
        """
        self.y_hat = []
        self.y = []

    def f1_score(self):
        """
        Method to calculate the F1 score for the stored predicted and ground truth values.
        Returns:
            float: The calculated F1 score.
        """
        y = torch.cat(self.y).to('cpu')
        y_hat = torch.cat(self.y_hat).to('cpu')

        metric = MulticlassF1Score(num_classes=self.num_classes)

        return metric(y_hat, y)

    def accuracy(self):
        """
        Method to calculate the accuracy for the stored predicted and ground truth values.
        Returns:
            float: The calculated accuracy.
        """
        y = torch.cat(self.y).to('cpu')
        y_hat = torch.cat(self.y_hat).to('cpu')

        metric = Accuracy(num_classes=self.num_classes, task=self.task)

        return metric(y_hat, y)

    def recall_score(self, average='macro'):
         """
         Method to calculate the recall score for the stored predicted and ground truth values.

         Args:
             average (str): The type of averaging to use when calculating the recall score. Can be "macro", "micro", or "weighted".

         Returns:
             float: The calculated recall score.
         """

         y = torch.cat(self.y).to('cpu')
         y_hat = torch.cat(self.y_hat).to('cpu')

         metric = Recall(num_classes=self.num_classes, task=self.task, average=average)

         return metric(y_hat, y)

    def precision_score(self, average='macro'):
         """
         Method to calculate the precision score for the stored predicted and ground truth values.

         Args:
             average (str): The type of averaging to use when calculating the precision score. Can be "macro", "micro", or "weighted".

         Returns:
             float: The calculated precision score.
         """

         y = torch.cat(self.y).to('cpu')
         y_hat = torch.cat(self.y_hat).to('cpu')

         metric = Precision(num_classes=self.num_classes, task=self.task, average=average)

         return metric(y_hat, y)
    
    def AUROC(self, average='macro'):
         """
         Method to calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) for the stored predicted and ground truth values.

         Args:
             average (str): The type of averaging to use when calculating the AUROC. Can be "macro", "micro", or "weighted".

         Returns:
             float: The calculated AUROC.
         """

         y = torch.cat(self.y).to('cpu')
         y_hat = torch.cat(self.y_hat).to('cpu')

         metric = AUROC(num_classes=self.num_classes, task=self.task, average=average)

         return metric(y_hat, y)
    
    def AUPRC(self, average='macro'):
         """
         Method to calculate the Area Under the Precision-Recall Curve (AUPRC) for the stored predicted and ground truth values.

         Args:
             average (str): The type of averaging to use when calculating the AUPRC. Can be "macro", "micro", or "weighted".

         Returns:
             float: The calculated AUPRC.
         """

         y = torch.cat(self.y).to('cpu')
         y_hat = torch.cat(self.y_hat).to('cpu')

         metric = MulticlassAveragePrecision(num_classes=self.num_classes, average=average, thresholds=None)

         return metric(y_hat, y)
    
    def print_metrics(self):
          """
          Method to print all metrics for stored predicted and ground truth values in one line
          """
          print(f'F1 Score: {self.f1_score()}, Accuracy: {self.accuracy()}, Recall Score: {self.recall_score()}, Precision Score: {self.precision_score()}, AUROC: {self.AUROC()}, AUPRC: {self.AUPRC()}')
