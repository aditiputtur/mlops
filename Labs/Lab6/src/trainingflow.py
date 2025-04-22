from metaflow import FlowSpec, step
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class TrainingFlow(FlowSpec):

    @step
    def start(self):
        # Ingest data
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.next(self.split_data)

    @step
    def split_data(self):
        # Split the data for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.next(self.train_model)

    @step
    def train_model(self):
        # Train the model (RandomForest in this case)
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.score(self.X_test, self.y_test)

        # Log the model and metrics in MLFlow
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForest")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, "rf_model")

        self.next(self.end)

    @step
    def end(self):
        print("Training complete and model registered with MLFlow.")

if __name__ == "__main__":
    TrainingFlow()

