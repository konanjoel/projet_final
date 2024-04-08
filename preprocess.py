#import connectiondb
import pandas as pd 
from sklearn.model_selection import train_test_split

def preprocessing(dataset):
    data = pd.read_csv(dataset, index_col=None)
    # Convertir la colonne de dates en type datetime
    data['date'] = pd.to_datetime(data['date'])
    # Supprimer la partie "+00:00" du format de date
    data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M')
    data['date'] = pd.to_datetime(data['date'])
    # Définir la colonne 'date' comme index
    data.set_index('date', inplace=True)

    # Calculer la moyenne des valeurs à chaque intervalle de 3 heures
    data1 = data.resample('3H').mean()
    # Split data into train (80%) and test (20%) sets
    ind_split = int(len(data1) * 0.8)

    ts_train = data1[:ind_split]
    ts_test = data1[ind_split:]
    return data1, ts_train, ts_test 


value_column = 'temperature_2m'

def preprocessingML(data):
    for i in range(1, 8):
        data[f"lag_{i}"] = data[value_column].shift(i)
    data.dropna(inplace=True)
    from sklearn.model_selection import train_test_split
    X = data.drop(value_column, axis=1)
    y = data[value_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    return X_train, X_test, y_train, y_test
    

###################################################################################################
