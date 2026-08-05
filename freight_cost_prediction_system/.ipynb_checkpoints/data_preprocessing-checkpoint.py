import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
# Load Data from sqlite database
def load_vendor_invoice_data(db_path):
    conn=sqlite3.connect(r'C:\Users\sk021\Desktop\Project 2\Data\inventory2.db')
    df=pd.read_sql_query("select * from vendor_invoice",conn)
    conn.close
    return df
def prepare_feature(df):
    X=df['Dollars']
    y=df['Freight']
    return X,y
def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
df=load_vendor_invoice_data('Data/inventory2.db')
X,y=prepare_feature(df)
X_train,X_test,y_train,y_test=split_data(X,y)
