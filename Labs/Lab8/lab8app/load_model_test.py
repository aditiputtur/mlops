import mlflow
import mlflow.sklearn

# Set the tracking URI to the MLflow server (adjust URL if necessary)
mlflow.set_tracking_uri('http://127.0.0.1:5000')

try:
    model = mlflow.sklearn.load_model("models:/hello/2")  # Adjust path if necessary
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

