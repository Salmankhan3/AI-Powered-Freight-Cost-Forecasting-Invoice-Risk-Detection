# 🚚  Ai-Powered Freight Cost Forecasting and Invoice Risk Detection

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

##  📌 Project Overview

The Freight Cost Forecasting and Invoice Risk Detection system is a machine learning application. It helps companies automate the analysis of freight invoices.

The application does two things. First it predicts the expected freight cost for vendor invoices. It uses procurement data to make these predictions. Second it identifies invoices that may need approval. It does this by looking for freight or billing patterns.

The project uses technologies, including machine learning regression, machine learning classification SQLite database, interactive data visualization and Streamlit web application.

This solution helps finance, procurement and logistics teams. It allows them to identify high-risk invoices before payment. They can reduce overbilling and improve forecasting accuracy. It also supports compliance audits.

---

The project integrates:

- 📈 Machine Learning Regression
- 🚩 Machine Learning Classification
- 🗄 SQLite Database
- 📊 Interactive Data Visualization
- 🌐 Streamlit Web Application
  
#  📑 Table of Contents
- Project Overview

- Business Problem

- Solution

- Features

- Machine Learning Models

- Project Architecture

- Technology Stack

- Database Design

- Streamlit Application

- Screenshots

- Installation

- Project Structure

- Future Improvements

- Author

---

# 🎯  Business Problem

Freight invoices often have mistakes. These mistakes can include freight charges, duplicate billing, unexpected transportation costs, pricing anomalies and manual approval bottlenecks.

Companies spend a lot of time reviewing invoices. This project automates the analysis of freight invoices using machine learning.

---

# 💡  Solution

The system has two parts.

## 1. Freight Cost Prediction

This part predicts the expected freight cost. It uses two factors: the quantity of the invoice and the dollar amount of the invoice.

The model estimates the freight amount before the invoice is approved.

## 2. Invoice Risk Detection

This part classifies invoices as safe for auto-approval. Requiring manual approval. It uses past invoice patterns to make these classifications.

---

# Features

## Freight Cost Prediction

- Predict freight charges

- Use a regression machine learning model

- Make predictions in time

- Keep a history of predictions

- Export data to CSV

## Invoice Risk Detection

- Use a binary classification model

- Detect abnormal invoices

- Flag risky invoices

- Recommend manual approval

- Keep a history of sessions

- Export data to CSV

##  📊 Dashboard

- Interactive Streamlit UI

- Navigate using the sidebar

- See a summary of business impact

- Get success and warning alerts

- View prediction history

##  🗄  Database Integration

The SQLite database stores historical invoice data, vendor information, purchase orders, freight transactions and prediction history.

Data is retrieved using SQL queries.

##  📉  Data Visualization

The project performs exploratory data analysis using charts and plots.

---

#  🤖  Machine Learning Models

## Regression Model

The goal of this model is to predict the expected freight cost.

It uses two factors: the quantity of the invoice and the dollar amount of the invoice.

The output is the predicted freight cost.

## Classification Model

The goal of this model is to predict whether an invoice requires approval.

It uses factors, including quantity, invoice dollars, freight cost, total quantity and total item dollars.

The output is either auto-approved or manual review.

---

# 🏗 Project Architecture

```
                Historical Invoice Data
                         │
                         ▼
                SQLite Database
                         │
                  SQL Queries
                         │
                Data Processing
         (Pandas + NumPy)
                         │
               Feature Engineering
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
Regression Model                Classification Model
(Freight Prediction)          (Risk Detection)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                 Streamlit Dashboard
                         │
                User Predictions
```

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|-------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Database | SQLite3 |
| SQL | SQLite Queries |
| Model Serialization | Joblib |
| IDE | VS Code |

---

# 📂 Database

The application uses **SQLite3** as its local relational database.

Major Tables

| Table |
|---------|
| purchases |
| purchase_prices |
| vendor_invoice |
| begin_inventory |
| end_inventory |

# Streamlit Application

## Home Page
![Home](Screenshots/Screenshot%20%28282%29.png)
![Home](Screenshots/Screenshot%20%28281%29.png)

## Freight Cost Prediction

![Prediction](Screenshots/Screenshot%20%28276%29.png)
## Prediction Result

![Prediction Result](Screenshots/Screenshot%20%28278%29.png)

## Prediction History

![Prediction History](Screenshots/Screenshot%20%28279%29.png)

## Invoice Risk Detection

![Invoice Risk Detection](Screenshots/Screenshot%20%28280%29.png)

## Invoice Approved

![Invoice Approved](Screenshots/Screenshot%20%28281%29.png)
---

# 📁 Project Structure

```
AI-Powered-Freight-Cost-Forecasting-Invoice-Risk-Detection
│
├── Data
│   ├── inventory.csv
│   └── ...
│
├── Models
│   ├── freight_model.pkl
│   ├── invoice_classifier.pkl
│
├── Notebook
│
├── Images
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- SQLite3
- Joblib

---
---

# 🚀 Future Improvements

- Vendor Risk Scoring
- Time Series Freight Forecasting
- Explainable AI (SHAP)
- Model Monitoring Dashboard
- REST API using FastAPI
- Cloud Deployment
- User Authentication
- PDF Report Generation
- Email Alerts
- Power BI Integration

# Business Benefits

The Freight Cost Forecasting and Invoice Risk Detection system provides several benefits, including:

- Reducing freight overbilling

- Improving freight cost forecasting

- Detecting invoice anomalies

- Reducing manual review effort

- Improving procurement efficiency

- Enhancing audit readiness

- Data-driven decision making

---

# 👨‍💻 Author

**Salman Khan**

Machine Learning | Data Science | Python | SQL | Streamlit

---
