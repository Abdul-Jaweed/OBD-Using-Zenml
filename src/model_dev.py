import logging
from abc import ABC, abstractmethod
# from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        
        pass
    

class LogisticRegressionModel(Model):
    """
    Logistic Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        """
        
        try:
            reg = LogisticRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e