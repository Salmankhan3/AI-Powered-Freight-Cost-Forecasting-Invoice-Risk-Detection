from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensamble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error,mean_absolute_error,r2_score
def lin_model(X_train,y_train):
    lin_model=LinearRegression()
    lin_model.fit(X_train,y_train)
    return lin_model
def tree_model(X_train,y_train):
    tree_model=DecisionTreeRegressor(criterion='squared_error',splitter='best', min_impurity_decrease=0.0,)
    tree_model.fit(X_train,y_train)
    return tree_model

def Random_forest(X_train,y_train):
    random_forest_model=RandomForestRegressor(criterion='squared_error',max_depth=None,n_estimators=400)
    random_forest_model.fit(X_train,y_train)
    return random_forest_model

def evaluate_model(model,X_test,y_test,model_name: str)-> dict:
    prediction=model.predict(X_test)
    MSE=mean_squared_error(y_test,prediction)
    RMSE=root_mean_squared_error(y_test,prediction)
    R2_Score=r2_score(y_test,prediction)*100
    print(f"Model {model_name} Performance")
    print(f"Mean Square Error {MSE}")
    print(f"Root Mean Square Error {RMSE}")
    print(f"R2 Score {R2_Score}")
    return dict{
        model_name: model_name,
        "MSE": MSE,
        "RMSE" : RMSE,
        r2_score : R2_Score
    }