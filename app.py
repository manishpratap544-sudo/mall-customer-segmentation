
import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Clustering",
    page_icon="📊",
    layout="wide"
)

# Model files must be in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    cluster_names = joblib.load(BASE_DIR / "cluster_names.pkl")
    return model, scaler, cluster_names

try:
    model, scaler, cluster_names = load_models()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()

# Cluster label helper
def get_cluster_name(cluster_id):
    if isinstance(cluster_names, dict):
        return cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    if isinstance(cluster_names, (list, tuple)):
        if 0 <= cluster_id < len(cluster_names):
            return cluster_names[cluster_id]

    return f"Cluster {cluster_id}"

# Sidebar
st.sidebar.title("📊 Customer Analytics")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Predict Customer", "Cluster Chart", "About"]
)

# Dashboard
if page == "Dashboard":
    st.title("📊 Customer Clustering Dashboard")
    st.write("Machine learning customer segmentation using K-Means.")

    col1, col2 = st.columns(2)
    col1.metric("Algorithm", "K-Means")
    col2.metric("Number of Clusters", model.n_clusters)

    st.info("Open Predict Customer to test your trained model.")

# Customer prediction
elif page == "Predict Customer":
    st.title("🔮 Predict Customer Cluster")

    with st.form("customer_form"):
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        spending = st.slider("Spending Score", 1, 100, 50)
        income = st.number_input(
            "Annual Income (k$)", min_value=0.0, value=50.0
        )

        submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        try:
            # Standard Python lists avoid requiring Pandas in your code
            customer = [[age, spending, income]]
            scaled_customer = scaler.transform(customer)
            cluster_id = int(model.predict(scaled_customer)[0])
            cluster_name = get_cluster_name(cluster_id)

            st.success("Prediction completed!")
            col1, col2 = st.columns(2)
            col1.metric("Cluster ID", cluster_id)
            col2.metric("Cluster Name", str(cluster_name))

            st.subheader("Customer Profile")
            st.write(f"Age: {age}")
            st.write(f"Spending Score: {spending}")
            st.write(f"Annual Income: {income} k$")

        except Exception as error:
            st.error(f"Prediction error: {error}")

# Simple chart using Streamlit only
elif page == "Cluster Chart":
    st.title("📈 Cluster Visualization")

    st.write("Enter the number of customers to display in this sample chart.")
    number = st.slider("Sample size", 1, 100, 10)

    chart_data = {
        "Customer": [f"Customer {i + 1}" for i in range(number)],
        "Spending Score": [
            (i * 17 + 25) % 100 + 1 for i in range(number)
        ],
    }

    st.bar_chart(chart_data, x="Customer", y="Spending Score")

    st.caption(
        "This is a demonstration chart, not actual customer data. "
        "Upload or connect your dataset to visualize real results."
    )

# About
else:
    st.title("ℹ️ About")
    st.write("Customer segmentation using a trained K-Means model.")
    st.write("Libraries used: Streamlit, Joblib, Scikit-learn.")
