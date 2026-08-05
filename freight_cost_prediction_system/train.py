import data_preprocessing as dp
import Model 
import joblib 
from pathlib import Path
def main():
    db_path=r"Data/inventory2.db"
    model_dir=Path('model')
    model_dir.mkdir(exist_ok=True)
    # Load and Prepre data
    df=dp.load_vendor_invoice_data(db_path)
    X,y=dp.prepare_feature(df)
    X_train,X_test,y_train,y_test=dp.split_data(X,y)
    
    # Train Model
    lin_model=Model.lin_model(X_train,y_train)
    tree_model=Model.tree_model(X_train,y_train)
    Random_forest=Model.Random_forest(X_train,y_train)
    # Evaluate Model
    result=[]
    result.append(Model.evaluate_model(lin_model,X_test,y_test,"Linear Regression"))
    result.append(Model.evaluate_model(tree_model,X_test,y_test,"Decision tree Regressor"))
    result.append(Model.evaluate_model(Random_forest,X_test,y_test,"Random Forest Regressor"))
    # Best Model based on r2 score
    best_model_info=max(result,key=lambda x: x['r2_score'])
    best_model_name=best_model_info['model_name']
    best_model={
        'Linear Regression' : lin_model,
        'Decision tree Regressor' : tree_model,
        'Random Forest Regressor' : Random_forest
    }[best_model_name]
    # Save Best Model
    path=model_dir / "best_model.pkl"
    joblib.dump(best_model,path)
    print(f"Best Model Saved {best_model_name}")
    print(f"Model Saved to {path}")
    
if __name__ == "__main__":
    print("Starting the training process")
    main()

        
    
    