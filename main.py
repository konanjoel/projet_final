from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
#import Evaluate
# Définition de la classe pour le corps de la requête
class DonneesEntree(BaseModel):
    step: int
    # Ajoutez les autres caractéristiques selon votre modèle

# Charger le modèle ML

modele = joblib.load('./Modeles/model_arima')

# Créer une instance de l'application FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bienvenu sur l'api de KONAN JOEL!"}

# Définir le point de terminaison pour les prédictions
@app.post("/predict")
async def predict(data: DonneesEntree):
    # Faire la prédiction avec le modèle
    prediction = modele.forecast(data.step)
    # Vous pouvez également retourner des probabilités ou d'autres informations selon votre modèle

    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1",
                port=8000, reload=True)

