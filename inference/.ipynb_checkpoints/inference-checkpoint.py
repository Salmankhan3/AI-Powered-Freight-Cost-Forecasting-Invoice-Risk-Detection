import joblib
import pandas as pd

model_path='model/best_model.pkl'

def load_model(model_path: str = model_path):
    with open(model_path,'rb') as f:
        model=joblib.load(f)
        return model

def freight_cost_prediction(model,input_data):
    input_data=pd.DataFrame(input_data)
    input_data['prediction']=model.predict(input_data).round()
    return input_data


if __name__ == '__main__':
    input_data={
    'Dollars':[18500,9000,7800]
      }
    model=load_model()
    prediction=freight_cost_prediction(model,input_data)
    print(prediction)