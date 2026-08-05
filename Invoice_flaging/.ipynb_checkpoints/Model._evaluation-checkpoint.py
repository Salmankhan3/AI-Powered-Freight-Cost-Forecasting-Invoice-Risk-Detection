from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=2,min_samples_split=2,min_samples_leaf=2)
model.fit(x_train,y_train)

def evaluate_model(model, X_test, y_test,model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n {model_name} Perfomance \n")
    print(f"Acuraccy : {accuracy * 100 : .2f}%")
    print(classification_report(y_test, y_pred))
    