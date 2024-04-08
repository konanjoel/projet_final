
import pandas as pd
import sqlite3
from io import StringIO

def Connexion():
    con = sqlite3.connect('./Donnee/temperature_abidjan.db')
    data_test = pd.read_sql('SELECT * FROM temperature', con)
    con.close()
    
    # Convertir le DataFrame en une chaîne CSV
    csv_buffer = StringIO()
    data_test.to_csv(csv_buffer, index=False)
    
    # Réinitialiser la position du curseur dans le buffer
    csv_buffer.seek(0)
    
    return csv_buffer




