from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import pickle


class InputModel(BaseModel):
    WindDirection: float
    month: float
    day: float
    Hour: float
    meanSpeed: float


app = FastAPI()
pickle_in = open("trained_model.pkl", "rb")
predictor = pickle.load(pickle_in)

@app.get('/test')
def fun():
    return {'result': "Working"}


@app.post('/wind_power_predict')
def fun(data: InputModel):
    data = data.dict()
    prediction = predictor.predict([[data['WindDirection'], data['month'], data['day'], data['Hour'], data['meanSpeed']]])
    return {'prediction': prediction[0]}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


#uvicorn main_api:app --host 0.0.0.0 --port 10000
