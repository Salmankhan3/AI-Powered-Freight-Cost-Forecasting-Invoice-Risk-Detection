from Data_Preprocessing import load_invoice_data,create_invoice_risk_label,apply_label,split_data,scale_feature
from Model_evaluation import evaluate_model
import joblib
from sklearn.ensemble import RandomForestClassifier

features=['invoice_quantity',
         'invoice_dollar',
          'Freight',
          'total_item_quantity',
          'total_item_dollars'
         ]
target=['flag_invoice']

def main():
    # Load data
    df=load_invoice_data()
    df=apply_label(df)
    x_train,x_test,y_train,y_test=split_data(df,features,target)
    x_train_scaled,x_test_scaled=scale_feature(x_train,x_test)
    model=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=2,min_samples_split=2,min_samples_leaf=2)
    model.fit(x_train,y_train)
    evaluate_model(model,x_test, y_test,'Random Forest Classifier')
    print("Saving Model")
    joblib.dump(model,'model/best_classifier.pkl')

if __name__ == "__main__":
    print("Trainig Model...")
    main()
    print("Finished training...")