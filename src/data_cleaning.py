import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split




class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        pass
    


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        
        try:
            data = data.drop("CO2", axis=1)
            
            
          # Adding the code to remove the null values or fill the null values
            return data
      
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
          

# drop(
#                 [
#                     "Lane_Departure",
#                     "Forward_Collision",
#                     "Tire_Pressure_FL",
#                     "Tire_Pressure_FR",
#                     "Blind_Spot_Detection",
#                     "Tire_Pressure_RL",
#                     "Tire_Pressure_RR"
#                 ],
#                 axis=1)




class DataDivideStrategy(DataStrategy):
    
    """
    Strategy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        
        try:
            
            X = data.drop(["Target"], axis=1)
            y = data["Target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))



class DataCleaning:
    """
    Class for cleaning data which precesse  the data and divides it into train and test
    """
    
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Handle data
        """
        
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            
            logging.error("Error in handing data:{}".format(e))
            raise e
        


# if __name__ == "__main__":
#     data = pd.read_csv("G:\VEHK\zenml\data\dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()