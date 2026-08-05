import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
def load_invoice_data():
    conn=sqlite3.connect(r'C:\Users\sk021\Desktop\Project 2\Data\inventory2.db')
    df=pd.read_sql_query(""" 
with purchasessss_aggregated_df as (
select p.PONumber,
count (distinct p.Brand ) as total_brands,
sum (p.quantity) as total_item_quantity,
sum (p.dollars) as total_item_dollars,
avg (julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_reciveing_delay
from purchases p
group by p.PONumber
)
select 
vi.PONumber,vi.Quantity as invoice_quantity,vi.Dollars as invoice_dollar,vi.Freight,
(julianday(vi.InvoiceDate) - julianday(vi.PODate)) as day_po_invoice,(julianday(vi.PayDate) -  julianday(vi.InvoiceDate)) as day_to_pay,
pa.total_brands,pa.total_item_quantity,pa.total_item_dollars,pa.avg_reciveing_delay
from vendor_invoice vi
left join 
purchasessss_aggregated_df pa
on
vi.PONumber=pa.PONumber

""",conn)
    conn.close()
    return df


# lable invoices
def create_invoice_risk_label(row):
     # invoice total mismath with item-level total
    if (abs(row['invoice_dollar'] - row['total_item_dollars']) > 5):
        return 1
    # abnormally high recieving delay
    if row['avg_reciveing_delay'] > 10:
        return 1
    return 0

# add lable column to dataframe
def apply_label(df):
    df['flag_invoice']=df.apply(create_invoice_risk_label,axis=1)
    return df



# split data
def split_data(df,feature,target):
    x=df[feature]
    y=df[target]
    return train_test_split(x,y,test_size=0.3,random_state=42)

# Scale feature

def scale_feature(x_train,x_test):
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    print("Saving scaler")
    joblib.dump(scaler,'model/scaler.pkl')
    print("scaler saved")
    return x_train_scaled,x_test_scaled
    
    




