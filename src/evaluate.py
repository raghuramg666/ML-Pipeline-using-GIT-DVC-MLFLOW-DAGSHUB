import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow


from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/raghupy1998/MachineLearningPipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="raghupy1998"
os.environ["MLFLOW_TRACKING_PASSWORD"]="55e7bfd4655d4f3880453e2c1917e4f042a3bb19"

##Load the parameters from params y.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]


    mlflow.set_tracking_uri("https://dagshub.com/raghupy1998/MachineLearningPipeline.mlflow")

    ##Load the model from the disk inorder to load it
    model=pickle.load(open(model_path,'rb'))
    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)


    ##Log the metrics
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy:{accuracy}")
if __name__=="__main__":
    evaluate(params['data'],params['model'])