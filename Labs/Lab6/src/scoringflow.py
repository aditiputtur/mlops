import mlflow
import mlflow.sklearn
from metaflow import FlowSpec, step

class ScoringFlow(FlowSpec):
    
    # Start step - Loads the data for scoring
    @step
    def start(self):
        # Simulate loading scoring data (replace this with your data loading logic)
        self.data = self.load_data()  # Replace with actual data loading function
        self.next(self.load_model)  # Transition to the next step, which is load_model
    
    # Load the registered model or register a new one if not found
    @step
    def load_model(self):
        model_name = "rf_model"  # Name of the registered model to load
        
        try:
            # Using the MLflow client to get the registered model by name
            client = mlflow.tracking.MlflowClient()
            registered_model = client.get_registered_model(model_name)
            # Get the latest model version
            latest_version = registered_model.latest_versions[0].version
            print(f"Loading model: {model_name}, version {latest_version}")
            
            # Load the model using MLflow's load_model function
            self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            print("Model loaded successfully.")
        except mlflow.exceptions.MlflowException:
            # If model isn't found, it will be registered (we need to train and register a new model)
            print(f"Model {model_name} not found. Registering a new model.")
            self.model = self.register_model(model_name)  # Call the register_model method

        # Transition to the next step after loading or registering the model
        self.next(self.make_predictions)
    
    # Make predictions using the loaded model
    @step
    def make_predictions(self):
        print("Making predictions...")
        # Assuming `self.model` is a trained model, use it to make predictions
        self.predictions = self.model.predict(self.data)  # Replace with actual prediction logic
        self.next(self.end)  # Transition to the end step after predictions are made
    
    # End step - Marks the end of the flow
    @step
    def end(self):
        print("Scoring flow completed.")
        # Print the first 5 predictions as an example output
        print(f"Predictions: {self.predictions[:5]}")

    # Placeholder method for loading the scoring data (replace with actual data)
    def load_data(self):
        # Example data (replace with your data loading code)
        return [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]]  # Example scoring data

    # If model isn't found, this method will train and register a new model
    def register_model(self, model_name):
        print("Training and registering a new model...")
        
        # Train a new model (this is just a placeholder, replace with your actual model training)
        model = self.train_model()
        
        # Log the trained model using MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)  # Log the model under the given name
        
        return model

    # Placeholder method to train a model (replace with actual training code)
    def train_model(self):
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Dummy training data (replace with your actual training data)
        X = np.random.rand(100, 3)  # Random features
        y = np.random.rand(100)     # Random target values
        
        # Initialize and train the RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X, y)
        return model  # Return the trained model


# Entry point for running the flow
if __name__ == "__main__":
    flow = ScoringFlow()  # Instantiate the flow class
    flow.run()  # Run the flow

