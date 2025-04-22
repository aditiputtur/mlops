import mlflow
import mlflow.sklearn
from metaflow import FlowSpec, step

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        # Simulate reading the data for scoring
        self.data = self.load_data()  # Replace with your data loading logic
        self.next(self.load_model)
    
    @step
    def load_model(self):
        # Check if the model is registered and fetch the latest version
        model_name = "rf_model"
        try:
            client = mlflow.tracking.MlflowClient()
            registered_model = client.get_registered_model(model_name)
            latest_version = registered_model.latest_versions[0].version
            print(f"Loading model: {model_name}, version {latest_version}")
            
            # Load the model
            self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            print("Model loaded successfully.")
        except mlflow.exceptions.MlflowException:
            print(f"Model {model_name} not found. Registering a new model.")
            # If model is not found, register a new model (you need a trained model object here)
            self.model = self.register_model(model_name)

        self.next(self.make_predictions)  # Transition to next step after loading or registering the model

    @step
    def make_predictions(self):
        # Assuming `self.model` is your trained model
        print("Making predictions...")
        self.predictions = self.model.predict(self.data)  # Replace with your scoring logic
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow completed.")
        print(f"Predictions: {self.predictions[:5]}")  # Display first 5 predictions

    def load_data(self):
        # Placeholder method for loading the scoring data
        # Replace with your actual data loading code
        return [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]]  # Example input data
    
    def register_model(self, model_name):
        # If the model isn't registered, you would need to train and register it.
        # Here, we are assuming you have a trained model `rf_model`
        print("Training and registering a new model...")

        # Example model registration (replace with your actual model)
        model = self.train_model()
        
        # Log the model in MLFlow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)
        
        return model

    def train_model(self):
        # Placeholder method for model training
        # Replace with your actual training code
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Dummy training data
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        
        model = RandomForestRegressor()
        model.fit(X, y)
        return model


if __name__ == "__main__":
    flow = ScoringFlow()
    flow.run()

