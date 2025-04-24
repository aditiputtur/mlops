from metaflow import FlowSpec, step, kubernetes, resources, conda, retry, timeout, catch
import mlflow
import mlflow.sklearn

class ScoringFlow(FlowSpec):

    # Start step - Loads the data for scoring
    @step
    def start(self):
        # Simulate loading scoring data (replace this with your data loading logic)
        self.data = self.load_data()
        self.next(self.load_model)

    # Load the registered model or register a new one if not found
    @conda(libraries={"scikit-learn": "1.2.2", "pandas": "1.5.3", "mlflow": "2.11.3"})
    @kubernetes(cpu=2, memory=4096)
    @resources(cpu=2, memory=4096)
    @step
    def load_model(self):
        model_name = "rf_model"
        
        try:
            client = mlflow.tracking.MlflowClient()
            registered_model = client.get_registered_model(model_name)
            latest_version = registered_model.latest_versions[0].version
            print(f"Loading model: {model_name}, version {latest_version}")
            self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
            print("Model loaded successfully.")
        except mlflow.exceptions.MlflowException:
            print(f"Model {model_name} not found. Registering a new model.")
            self.model = self.register_model(model_name)

        self.next(self.make_predictions)

    # Make predictions using the loaded model
    @conda(libraries={"scikit-learn": "1.2.2", "pandas": "1.5.3", "mlflow": "2.11.3"})
    @kubernetes(cpu=2, memory=4096)
    @resources(cpu=2, memory=4096)
    @step
    def make_predictions(self):
        print("Making predictions...")
        self.predictions = self.model.predict(self.data)
        self.next(self.end)

    # End step
    @step
    def end(self):
        print("Scoring flow completed.")
        print(f"Predictions: {self.predictions[:5]}")

    # Data loading placeholder
    def load_data(self):
        return [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]]

    # Train and register a model
    def register_model(self, model_name):
        print("Training and registering a new model...")
        model = self.train_model()
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)
        return model

    def train_model(self):
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        model = RandomForestRegressor()
        model.fit(X, y)
        return model


if __name__ == "__main__":
    ScoringFlow()
