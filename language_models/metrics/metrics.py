import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Metrics:
    def __init__(self, actual, predictions):
        self.actual = actual
        self.predictions = np.argmax(predictions, axis=1)

    def accuracy(self):
        accuracy = accuracy_score(y_true=self.actual, y_pred=self.predictions)
        return accuracy

    def recall(self):
        recall = recall_score(y_true=self.actual, y_pred=self.predictions)
        return recall

    def precision(self):
        precision = precision_score(y_true=self.actual, y_pred=self.predictions)
        return precision
    
    def f1(self):
        f1 = f1_score(y_true=self.actual, y_pred=self.predictions)
        return f1

    def builder(self):
        accuracy = self.accuracy()
        recall = self.recall()
        precision = self.precision()
        f1 = self.f1()
        result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        return result