from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score






def evaluate_model(model, X_test, y_test,model_name):
    print(f"test data {X_test}\n actual Label {y_test}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n {model_name} Perfomance \n")
    print(f"Acuraccy : {accuracy * 100 : .2f}%")
    print(classification_report(y_test, y_pred))
    