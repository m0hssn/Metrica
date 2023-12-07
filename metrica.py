class Metrica:
    def __init__(self, num_classes=3, task="multiclass"):
        self.num_classes = num_classes
        self.task = task
        self.reset_metrics()

    def reset_metrics(self):
        # Initialize necessary metrics
        self.metric_f1 = MulticlassF1Score(num_classes=self.num_classes)
        self.metric_acc = Accuracy(num_classes=self.num_classes, task=self.task)
        self.metric_recall = Recall(num_classes=self.num_classes, task=self.task)
        self.metric_precision = Precision(num_classes=self.num_classes, task=self.task)
        self.metric_auroc = AUROC(num_classes=self.num_classes, task=self.task)
        self.metric_auprc = MulticlassAveragePrecision(num_classes=self.num_classes, thresholds=None)
        self.metric_mcc = MatthewsCorrcoef(num_classes=self.num_classes)
        self.metric_fbeta = FBeta(num_classes=self.num_classes, beta=0.5)
        
        # Initialize variables for TP, TN, FP, FN
        self.TP = torch.zeros(self.num_classes)
        self.TN = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)

    def upgrade(self, y_hat: torch.tensor, y: torch.tensor):
        # Update metrics
        self.metric_f1.update(y_hat, y)
        self.metric_acc.update(y_hat, y)
        self.metric_recall.update(y_hat, y)
        self.metric_precision.update(y_hat, y)
        self.metric_auroc.update(y_hat, y)
        self.metric_auprc.update(y_hat, y)
        self.metric_mcc.update(y_hat, y)
        self.metric_fbeta.update(y_hat, y)
        
        # Calculate TP, TN, FP, FN
        for i in range(self.num_classes):
            self.TP[i] += ((y_hat == i) & (y == i)).sum()
            self.TN[i] += ((y_hat != i) & (y != i)).sum()
            self.FP[i] += ((y_hat == i) & (y != i)).sum()
            self.FN[i] += ((y_hat != i) & (y == i)).sum()

    def f1_score(self, average='macro'):
        return self.metric_f1.compute(average=average)

    def accuracy(self):
        return self.metric_acc.compute()

    def recall_score(self, average='macro'):
        return self.metric_recall.compute(average=average)

    def precision_score(self, average='macro'):
        return self.metric_precision.compute(average=average)

    def AUROC(self, average='macro'):
        return self.metric_auroc.compute(average=average)

    def AUPRC(self, average='macro'):
        return self.metric_auprc.compute(average=average)
    
    def MCC(self):
        return self.metric_mcc.compute()
    
    def fbeta_score(self, average='macro'):
        return self.metric_fbeta.compute(average=average)
    
    def print_metrics(self):
        print(f'F1 Score: {self.f1_score()}, Accuracy: {self.accuracy()}, Recall Score: {self.recall_score()}, Precision Score: {self.precision_score()}, AUROC: {self.AUROC()}, AUPRC: {self.AUPRC()}, MCC: {self.MCC()}, Fbeta Score: {self.fbeta_score()}')
        print(f'TP: {self.TP}, TN: {self.TN}, FP: {self.FP}, FN: {self.FN}')
