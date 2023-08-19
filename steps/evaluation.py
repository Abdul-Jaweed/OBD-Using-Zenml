import logging
from zenml import step
import pandas as pd

from src.model_eval import ACCURACY
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated


import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


# @step
# def evaluate_model(model: ClassifierMixin,
#                    X_test: pd.DataFrame,
#                    y_test: pd.DataFrame) -> Tuple[
#                        Annotated[float, "accuracy"],
#                     #    Annotated[float, "classification_report"]
#                    ]:
#     """
#     Evaluates the model on the ingested data.

#     Args:
#         df: the ingested data
#     """
#     try:
#         prediction = model.predict(X_test)
        
#         acc_class = ACCURACY()
#         accuracy = acc_class.calculate_scores(y_test, prediction)
        
#         # cr_class = ClassificationReport()
#         # classification_report = cr_class.calculate_scores(y_test, prediction)
        
#         return accuracy
#     except Exception as e:
#         logging.error("Error in evaluating model: {}".format(e))
#         raise e





@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[Annotated[float, "accuracy"]]:
    """
    Evaluates the model on the ingested data.

    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        
        acc_class = ACCURACY()
        accuracy = acc_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("accuracy", accuracy)
        
        return (accuracy,)  # Return the accuracy as a single-element tuple
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e
