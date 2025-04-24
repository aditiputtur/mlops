from metaflow import FlowSpec, step, conda_base, kubernetes, resources, retry, timeout, catch, conda

@conda_base(libraries={'numpy': '1.23.5', 'scikit-learn': '1.2.2'}, python='3.9.16')
class TrainingFlowGCP(FlowSpec):

    @step
    def start(self):
        # Data loading and preprocessing
        from sklearn.datasets import load_diabetes
        import pandas as pd

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        self.X = X
        self.y = y
        print("Starting the flow...")
        self.next(self.train_model)

    @conda(libraries={"scikit-learn": "1.2.2", "pandas": "1.5.3", "mlflow": "2.11.3"})
    @kubernetes(cpu=2, memory=4096)
    @resources(cpu=2, memory=4096)
    @retry(times=3)
    @timeout(seconds=600)
    @catch(var='error')
    @step
    def train_model(self):
        print("Training model...")
        from sklearn.linear_model import LinearRegression
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri("http://mlflow.default.svc.cluster.local:5000")  # Update if needed
        mlflow.set_experiment("gcp-training")

        model = LinearRegression()

        with mlflow.start_run():
            model.fit(self.X, self.y)
            score = model.score(self.X, self.y)

            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("score", score)
            mlflow.sklearn.log_model(model, "model")

        self.model_score = score
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed.")
