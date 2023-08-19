from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == '__main__':
    # Run the pipeline
    
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
    # training_pipeline(data_path="data\dataset.csv")
    
   # training_pipeline(data_path=r"file:C:\Users\Admin\AppData\Roaming\zenml\local_stores\e113a655-b713-4bb6-a0c1-507907ec1507\mlruns")



import os

user_home = os.path.expanduser("~")
data_path = os.path.join(user_home, "AppData", "Roaming", "zenml", "local_stores", "e113a655-b713-4bb6-a0c1-507907ec1507", "mlruns")

training_pipeline(data_path=f"file:{data_path}")

