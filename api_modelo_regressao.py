import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class request_body(BaseModel):
    horas_estudo : float

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the file path
file_path = os.path.join(current_dir, 'modelo_regresao.pkl')

# Load the model
modelo = joblib.load(file_path)

@app.post('/predict')
def predict (data : request_body):
  # Preparar os dados para predição
  input_feature = [[data.horas_estudo]]
  
  # Fazer a predição
  prediction = modelo.predict(input_feature)[0].astype(int)
  
  return {"pontuacao_teste" : prediction}