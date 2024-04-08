
import preprocess
import connectiondb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib

# Fonction pour le chargement du modele
def charger_modele(chemin_fichier):
    
    modele = joblib.load(chemin_fichier)

    return modele

# Fonction pour l'evaluation du modél
def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    print(y_pred)
    return score


donne = connectiondb.Connexion()
data1, ts_train, ts_test = preprocess.preprocessing(donne)
X_train, X_test, y_train, y_test = preprocess.preprocessingML(data1)

modele = charger_modele('./Modeles/randomforest')
score_test = evaluate_model(modele, X_test, y_test)
print(f'le score obtenu est : {score_test}')
