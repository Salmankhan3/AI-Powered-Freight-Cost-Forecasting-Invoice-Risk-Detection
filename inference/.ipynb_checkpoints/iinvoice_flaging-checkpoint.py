import joblib
import pandas as pd


model_path='model/best_classifier.pkl'
def load_model(model_path : str = model_path):
    with open(model_path,'rb') as f:
        model=joblib.load(f)
        return model

def invoice_flaging(input,model):
    input_data=pd.DataFrame(input)
    prediction=model.predict(input_data)
    return prediction

if __name__ == '__main__':
    model=load_model()
    input={
        'invoice_quantity': [48,34773, 70,104],
        'invoice_dollar' : [352,225706.96,634.11,987.34],
        'Freight' :[1.73,1196.25,2.85,4.64 ],
        'total_item_quantity' : [162,34773,2357,104],
        'total_item_dollars' :[2476.99,225706.96,26584.61,987.34]
    }
    prediction=invoice_flaging(input,model)
    print(prediction)
    
    
        