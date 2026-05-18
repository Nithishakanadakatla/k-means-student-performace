import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Student ML App", layout="wide")

st.title("🎓 Student Performance Clustering System")
st.write("K-Means Clustering with Evaluation Metrics & Prediction")

MODEL_FILE = "kmeans_model.pkl"
SCALER_FILE = "scaler.pkl"
MAP_FILE = "cluster_map.pkl"

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    features = ['StudyHours', 'Attendance', 'PreviousMarks',
                'InternalAssessment', 'ExtracurricularScore']

    X = df[features]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select K
    k = st.slider("Select number of clusters (K)", 2, 5, 3)

    # -----------------------------
    # Train Model
    # -----------------------------
    if st.button("🚀 Train Model"):

        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Save model + scaler
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(kmeans, f)

        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

        # -----------------------------
        # Evaluation Metrics
        # -----------------------------
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, df['Cluster'])

        st.subheader("📊 Model Evaluation")

        col1, col2 = st.columns(2)
        col1.metric("Inertia (WCSS)", f"{inertia:.2f}")
        col2.metric("Silhouette Score", f"{sil_score:.3f}")

        st.info("Higher Silhouette = Better | Lower Inertia = Better")

        # -----------------------------
        # Cluster Naming
        # -----------------------------
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=features)
        centers_df['Cluster'] = range(k)

        centers_df['PerformanceScore'] = (
            centers_df['StudyHours'] +
            centers_df['Attendance'] +
            centers_df['PreviousMarks'] +
            centers_df['InternalAssessment']
        )

        centers_df = centers_df.sort_values(by='PerformanceScore')

        labels = ['At-Risk', 'Average', 'Topper', 'Excellent', 'Elite']
        centers_df['Label'] = labels[:k]

        cluster_label_map = dict(zip(centers_df['Cluster'], centers_df['Label']))
        df['Category'] = df['Cluster'].map(cluster_label_map)

        # Save mapping
        with open(MAP_FILE, "wb") as f:
            pickle.dump(cluster_label_map, f)

        st.subheader("🏷 Cluster Labels")
        st.dataframe(centers_df[['Cluster', 'Label']])

        st.subheader("📈 Segmented Data")
        st.dataframe(df)

        # -----------------------------
        # Plotly Visualization
        # -----------------------------
        st.subheader("📊 Visualization")

        fig = px.scatter(
            df,
            x='StudyHours',
            y='PreviousMarks',
            color='Category',
            title="Student Segmentation",
            hover_data=['Attendance', 'InternalAssessment']
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result", csv, "segmented_students.csv")

# -----------------------------
# Prediction Section
# -----------------------------
st.header("🔮 Predict Student Category")

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(MAP_FILE):

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    with open(MAP_FILE, "rb") as f:
        cluster_map = pickle.load(f)

    col1, col2, col3 = st.columns(3)

    with col1:
        study = st.number_input("Study Hours", 1, 15, 5)

    with col2:
        attend = st.number_input("Attendance", 50, 100, 75)

    with col3:
        marks = st.number_input("Previous Marks", 0, 100, 60)

    col4, col5 = st.columns(2)

    with col4:
        internal = st.number_input("Internal Assessment", 0, 100, 65)

    with col5:
        extra = st.number_input("Extracurricular Score", 0, 100, 50)

    if st.button("🎯 Predict"):
        input_data = np.array([[study, attend, marks, internal, extra]])
        input_scaled = scaler.transform(input_data)

        cluster = model.predict(input_scaled)[0]
        label = cluster_map.get(cluster, "Unknown")

        st.success(f"Cluster: {cluster}")
        st.info(f"🎓 Category: {label}")

else:
    st.warning("⚠️ Train model first to enable predictions.")