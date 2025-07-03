import mlflow
import os
import matplotlib.pyplot as plt
import numpy as np

# Set the tracking URI to your remote MLflow server
mlflow.set_tracking_uri("http://localhost:5001")

print(mlflow.get_tracking_uri())

# Set S3/MinIO credentials (optional if already set as environment variables)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "MLFlowUser"
os.environ["AWS_SECRET_ACCESS_KEY"] = "MyFlowPass"

# Create a new experiment (if it doesn’t exist, this will create it)
experiment_name = "test_experiment2"
mlflow.set_experiment(experiment_name)

# Start a new MLflow run
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)
    mlflow.log_metric("accuracy", 0.95)

    # Generate a sample plot and save it as a PNG
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure()
    plt.plot(x, y)
    plt.title("Sample Sine Wave")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the figure
    img_path = "sample_plot.png"
    plt.savefig(img_path)
    plt.close()

    # Log the PNG file as an artifact
    mlflow.log_artifact(img_path)

    # Cleanup
    os.remove(img_path)
    print(f"Artifact {img_path} logged successfully!")