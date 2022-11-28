from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import csv
import codecs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def make_predict(df_input):
    with open('model.pickle', 'rb') as file:
        pickle_obj = pickle.load(file)
    
    df = (df_input
          .eval("mileage = mileage.str.extract(r'(\d+\.*\d*)(?=\skmpl)').astype('float')")
          .eval("engine = engine.str.extract(r'(^\d+\.*\d*)(?=\sCC$)').astype('float')")
          .eval("max_power = max_power.str.extract(r'(\d+\.*\d*)(?=\sbhp)').astype('float')")
          .eval("seats = seats.replace('', 'nan').astype('float')")
          .fillna(pickle_obj['medians'])
          .drop(['name', 'torque'], axis = 1))
    
    labels_numbers = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    
    df_numbers = pd.DataFrame(
        pickle_obj['standard_scaler'].transform(df[labels_numbers]),
        columns = labels_numbers).drop('seats', axis = 1)
    df_cats = pd.DataFrame(
        pickle_obj['onehot_encoder'].transform(df[pickle_obj['cats_to_encode']]).toarray())
    df_ready = pd.concat([df_numbers, df_cats], axis = 'columns').rename(lambda x: str(x), axis = 'columns')
    
    return pickle_obj['model'].predict(df_ready)

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return make_predict(pd.DataFrame([dict(item)]))[0]

@app.post("/predict_items")
def upload(file: UploadFile = File(...)):
    csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    res = []
    for rows in csvReader:             
        res.append(rows)
    df_res = pd.concat([
        pd.DataFrame(res), 
        pd.DataFrame(list(make_predict(pd.DataFrame(res))), columns = ['predict'])
        ], axis = 'columns')
    
    stream = io.StringIO()
    df_res.to_csv(stream, index = False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response

