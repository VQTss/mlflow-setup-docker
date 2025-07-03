import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import RocCurveDisplay, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold

from loguru import logger
import os

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5001")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "MLFlowUser"
os.environ["AWS_SECRET_ACCESS_KEY"] = "MyFlowPass"


print("Numpy: {}".format(np.__version__))
print("Pandas: {}".format(pd.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("Scitkit-Learn: {}".format(sklearn.__version__))
print("MLFlow: {}".format(mlflow.__version__))
print(f"Active run: {mlflow.active_run()}")

data_path="./data/creditcard.csv"
df=pd.read_csv(data_path)
df=df.drop("Time",axis=1)
print(df.head())
print(f"Shape: {df.shape}")

# Randomly sampling 50% of all the normal data points
# in the data frame and picking out all of the anomalies from the data
# frame as separate data frames. 
normal=df[df.Class==0].sample(frac=0.5,random_state=2020).reset_index(drop=True)
anomaly=df[df.Class==1]

print(f"Normal: {normal.shape}")
print(f"Anomalies: {anomaly.shape}")

# split the normal and anomaly sets into train-test

normal_train,normal_test=train_test_split(normal,test_size=0.2,random_state=2020)
anomaly_train,anomaly_test=train_test_split(anomaly,test_size=0.2,random_state=2020)

# From there split train into train validate

normal_train,normal_validate=train_test_split(normal_train,test_size=0.25,random_state=2020)
anomaly_train,anomaly_validate=train_test_split(anomaly_train,test_size=0.25,random_state=2020)

# Create the whole sets

x_train =pd.concat((normal_train,anomaly_train))
x_test=pd.concat((normal_test,anomaly_test))
x_validate=pd.concat((normal_validate, anomaly_validate))

y_train=np.array(x_train["Class"])
y_test=np.array(x_test["Class"])
y_validate=np.array(x_validate["Class"])

x_train=x_train.drop("Class",axis=1)
x_test=x_test.drop("Class",axis=1)
x_validate=x_validate.drop("Class",axis=1)

print("Training sets:\nx_train: {} \ny_train: {}".format(x_train.shape, y_train.shape))
print("Testing sets:\nx_test: {} \ny_test: {}".format(x_test.shape, y_test.shape))
print("Validation sets:\nx_validate: {} \ny_validate: {}".format(x_validate.shape, y_validate.shape))

# Scale the data

scaler= StandardScaler()
scaler.fit(pd.concat((normal,anomaly)).drop("Class",axis=1))

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_validate=scaler.transform(x_validate)


def train(sk_model,x_train,y_train):
    sk_model=sk_model.fit(x_train,y_train)
    
    train_acc=sk_model.score(x_train,y_train)
    mlflow.log_metric("train_acc",train_acc)
    
    logger.info(f"Train Accuracy: {train_acc:.3%}")

def evaluate(sk_model,x_test,y_test):
    eval_acc=sk_model.score(x_test,y_test)
    
    preds=sk_model.predict(x_test)
    auc_score=roc_auc_score(y_test,preds)
    
    mlflow.log_metric("eval_acc",eval_acc)
    mlflow.log_metric("auc_score",auc_score)
    
    print(f"Auc Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")
    
    roc_plot = RocCurveDisplay.from_estimator(sk_model, x_test, y_test, name='Scikit-learn ROC Curve')
    
    plt.savefig("sklearn_roc_plot.png")
    plt.show()
    plt.clf()
    
    conf_matrix=confusion_matrix(y_test, preds)
    ax=sns.heatmap(conf_matrix,annot=True,fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.savefig("sklearn_conf_matrix.png")
    
    mlflow.log_artifact("sklearn_roc_plot.png")
    mlflow.log_artifact("sklearn_conf_matrix.png")

sk_model= LogisticRegression(random_state=None, max_iter=400, solver='newton-cg')

mlflow.set_experiment("scikit_learn_experiment")
with mlflow.start_run():
    train(sk_model,x_train,y_train)
    evaluate(sk_model,x_test,y_test)

    # Provide input example to infer model signature

    input_example = x_train[0].reshape(1, -1) 
    mlflow.sklearn.log_model(sk_model, "log_reg_model", input_example=input_example)


    
    logger.info(f"Model run: {mlflow.active_run().info.run_id}")

mlflow.end_run()