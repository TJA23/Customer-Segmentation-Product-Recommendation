# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations in E-Commerce

## 📌 Overview

The **Shopper Spectrum** project aims to derive actionable insights from e-commerce transaction data using unsupervised learning techniques and collaborative filtering. By applying **RFM (Recency, Frequency, Monetary)** analysis and clustering, we segment customers into distinct behavioral groups. We also deploy an **item-based recommendation system** to enhance personalized shopping experiences.

This solution is deployed through a **Streamlit web application** offering real-time predictions and recommendations.

---

## 🧠 Problem Statement

E-commerce platforms generate massive amounts of transaction data daily. This project analyzes that data to:
- Identify customer segments for targeted marketing.
- Recommend relevant products using customer behavior patterns.
- Support retention, dynamic pricing, and inventory decisions.

---

## 🔍 Project Objectives

- Segment customers using **RFM-based clustering**.
- Build a **collaborative filtering recommendation engine**.
- Deploy an interactive **Streamlit app** for end-user predictions.

---

## 📂 Dataset Description

| Column       | Description                                  |
|--------------|----------------------------------------------|
| InvoiceNo    | Transaction number                           |
| StockCode    | Unique product/item code                     |
| Description  | Name of the product                          |
| Quantity     | Number of products purchased                 |
| InvoiceDate  | Date and time of transaction (2022–2023)     |
| UnitPrice    | Price per product                            |
| CustomerID   | Unique identifier for each customer          |
| Country      | Country where the customer is based          |

---

## 🛠️ Tech Stack & Libraries

- **Python**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Machine Learning**: `KMeans`, `StandardScaler`, `cosine_similarity`
- **Web Deployment**: `Streamlit`
- **Data Transformation**: `Pivot Tables`, `Feature Engineering`

---

## 🔧 Project Workflow

### Step 1: Data Preprocessing
- Removed rows with missing `CustomerID`
- Filtered out cancelled invoices (`InvoiceNo` starting with 'C')
- Dropped negative/zero quantity and price values

### Step 2: Exploratory Data Analysis
- Analyzed transaction volume by country
- Identified top-selling products
- Visualized RFM distributions
- Used Elbow Method & Silhouette Score to determine optimal clusters

### Step 3: RFM Segmentation (Clustering)
- **Recency**: Days since last purchase
- **Frequency**: Number of transactions
- **Monetary**: Total spending
- Used **KMeans Clustering** for customer segmentation:
  - High-Value
  - Regular
  - Occasional
  - At-Risk

### Step 4: Product Recommendation Engine
- Built **item-based collaborative filtering** model
- Calculated **cosine similarity** between products
- Returned top 5 product recommendations

---

## 📱 Streamlit App Features

### 🔹 1. Product Recommendation Module
- Input: Product Name
- Output: 5 similar product suggestions

### 🔹 2. Customer Segmentation Module
- Input: Recency, Frequency, Monetary
- Output: Predicted customer segment label

---

## 📊 Visualizations

- Time-based purchasing trends
- RFM distribution plots
- Cluster scatter plots & 3D views
- Cosine similarity heatmap

---

## 📦 Deliverables

-  Jupyter Notebook (.ipynb) with:
  - Clean code
  - Visualizations
  - Clustering & Recommendation logic

-  Streamlit Web Application:
  - Real-time cluster prediction
  - Top-5 product recommendations

---

##  Real-Time Use Cases

- Personalized marketing campaigns
- Customer retention programs
- Dynamic pricing strategies
- Inventory forecasting





## 📌 License

This project is for academic and learning purposes only. Contact the author for commercial use.
