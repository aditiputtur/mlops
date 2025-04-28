import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import json

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Define the FastAPI app
app = FastAPI()

# Load the model once when the app starts
model = mlflow.sklearn.load_model("models:/hello/2")  # Adjust path if necessary

class InputData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
async def predict(data: InputData):
    input_features = [data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
                      data.magnesium, data.total_phenols, data.flavanoids, data.nonflavanoid_phenols,
                      data.proanthocyanins, data.color_intensity, data.hue,
                      data.od280_od315_of_diluted_wines, data.proline]
    try:
        prediction = model.predict([input_features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"detail": f"Prediction failed: {str(e)}"}
