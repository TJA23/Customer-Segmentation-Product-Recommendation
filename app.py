import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

# ---------------- Streamlit UI ------------------
st.title("🛒 Shopper Spectrum: Customer Intelligence App")
st.markdown("Improve your product strategy and customer engagement using data!")

# Load preprocessed dataset and models if available
@st.cache_data

def load_data():
    df = pd.read_csv("cleaned_transactions.csv")
    similarity_matrix = pd.read_csv("product_similarity_matrix.csv", index_col=0)
    with open("kmeans_rfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return df, similarity_matrix, model

try:
    df, product_similarity_matrix, kmeans_model = load_data()
except:
    df, product_similarity_matrix, kmeans_model = pd.DataFrame(), pd.DataFrame(), None

scaler = StandardScaler()

st.sidebar.title("Select Module")
module = st.sidebar.radio("Choose a feature:", ["Product Recommendation", "Customer Segmentation", "Train & Export Models"])

# ---------------- Recommendation Module ------------------
if module == "Product Recommendation":
    st.subheader("🔍 Find Similar Products")
    if product_similarity_matrix.empty:
        st.error("Product similarity matrix not found. Please generate it in 'Train & Export Models'.")
    else:
        product_name = st.text_input("Enter Product Name:")
        if st.button("Get Recommendations"):
            if product_name in product_similarity_matrix.index:
                sim_scores = product_similarity_matrix.loc[product_name]
                top_5 = sim_scores.sort_values(ascending=False).head(6).iloc[1:]
                st.success("Top 5 Similar Products:")
                for i, item in enumerate(top_5.index, 1):
                    st.markdown(f"**{i}.** {item}")
            else:
                st.error("Product not found. Please try a different product name.")

# ---------------- Customer Segmentation Module ------------------
if module == "Customer Segmentation":
    st.subheader("🧠 Predict Customer Segment")
    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spending)", min_value=0.0)

    if st.button("Predict Cluster"):
        if kmeans_model is None:
            st.error("KMeans model not found. Please generate it in 'Train & Export Models'.")
        else:
            input_data = np.array([[recency, frequency, monetary]])
            input_scaled = scaler.fit_transform(input_data)  # Should be same scaler as training
            cluster_label = kmeans_model.predict(input_scaled)[0]

            segment_map = {
                0: "High-Value",
                1: "Regular",
                2: "Occasional",
                3: "At-Risk"
            }

            segment = segment_map.get(cluster_label, "Unknown")
            st.success(f"This customer belongs to the **{segment}** segment.")

# ---------------- Model Training Module ------------------
if module == "Train & Export Models":
    st.subheader("⚙️ Train Models & Generate Files")
    upload = st.file_uploader("Upload Transaction CSV", type="csv")

    if upload:
        df = pd.read_csv(upload)

        st.success("File uploaded. Cleaning data...")

        # Preprocessing
        df = df.dropna(subset=['CustomerID'])
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

        # Feature Engineering - RFM
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'UnitPrice': lambda x: (df.loc[x.index, 'Quantity'] * x).sum()
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']

        # Train Clustering Model
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # Save model
        with open("kmeans_rfm_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        rfm.to_csv("rfm_clustered.csv")

        # Product similarity matrix
        purchase_matrix = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', fill_value=0)
        similarity = cosine_similarity(purchase_matrix.T)
        similarity_df = pd.DataFrame(similarity, index=purchase_matrix.columns, columns=purchase_matrix.columns)
        similarity_df.to_csv("product_similarity_matrix.csv")

        df.to_csv("cleaned_transactions.csv", index=False)
        st.success("Model and matrices generated and saved.")

st.markdown("---")
st.caption("© 2025 Shopper Spectrum | Capstone Project")
