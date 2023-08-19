import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass


class ACCURACY(Evaluation):
    """
    Evaluation Strategy that uses Accuracy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            acc = accuracy_score(y_true, y_pred)
            logging.info("Accuracy: {}".format(acc))
            return acc
        except Exception as e:
            logging.error("Error in calculating accuracy: {}".format(e))
            raise e




# class ClassificationReport(Evaluation):
#     """
#     Evaluation Strategy that uses Classification Report
#     """
#     def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
#         try:
#             logging.info("Calculating Classification Report")
#             cr = classification_report(y_true, y_pred)
#             logging.info("Classification Report: {}".format(cr))
#             return cr
#         except Exception as e:
#             logging.error("Error in calculating Classification Report: {}".format(e))
#             raise e