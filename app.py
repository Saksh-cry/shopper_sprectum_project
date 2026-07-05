import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#2563EB,#1E3A8A);
}

section[data-testid="stSidebar"] *{
    color:white;
}

.title{
font-size:38px;
font-weight:bold;
color:#2563EB;
}

.subtitle{
font-size:18px;
color:gray;
}

.metric-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 10px rgba(0,0,0,0.15);
text-align:center;
}

.big-font{
font-size:22px;
font-weight:bold;
}

.footer{
text-align:center;
padding:20px;
font-size:15px;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_online_retail.csv")

df = load_data()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

customer_segments = pd.read_csv("customer_segments.csv")

# -------------------------------
# LOAD MODELS
# -------------------------------

kmeans = pickle.load(open("kmeans.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
similarity = pickle.load(open("similarity.pkl","rb"))
product_list = pickle.load(open("product_list.pkl","rb"))

# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.image(
"https://img.icons8.com/color/480/shopping-cart.png",
width=120
)

st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
"Navigation",
[
"🏠 Home",
"📊 Dashboard",
"👥 Customer Segmentation",
"🛍 Product Recommendation",
"📈 Sales Analytics",
"ℹ️ About"
]
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.markdown("<h1 class='title'>🛒 Shopper Spectrum</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Customer Segmentation & Product Recommendation System</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ===========================
    # KPI CARDS
    # ===========================

    total_orders = df["InvoiceNo"].nunique()
    total_customers = df["CustomerID"].nunique()
    total_products = df["Description"].nunique()
    total_revenue = df["TotalAmount"].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🧾 Orders", f"{total_orders:,}")

    with c2:
        st.metric("👥 Customers", f"{total_customers:,}")

    with c3:
        st.metric("🛍 Products", f"{total_products:,}")

    with c4:
        st.metric("💰 Revenue", f"₹ {total_revenue:,.0f}")

    st.markdown("---")

    left, right = st.columns([2,1])

    with left:

        st.subheader("📌 Project Overview")

        st.write("""

This project applies **Machine Learning** techniques to analyze customer purchasing behavior in an e-commerce business.

### Main Objectives

✔ Customer Segmentation using **RFM Analysis**

✔ Customer Clustering using **KMeans**

✔ Product Recommendation using **Collaborative Filtering**

✔ Interactive Business Dashboard

✔ Sales Analytics

✔ Professional Streamlit Application

        """)

        st.success("✔ Dataset Successfully Loaded")
        st.success("✔ Machine Learning Models Loaded")
        st.success("✔ Ready for Prediction")

    with right:

        st.subheader("📂 Dataset Information")

        info = pd.DataFrame({

            "Feature":[

                "Transactions",
                "Customers",
                "Products",
                "Countries"

            ],

            "Value":[

                len(df),
                df["CustomerID"].nunique(),
                df["Description"].nunique(),
                df["Country"].nunique()

            ]

        })

        st.dataframe(info,use_container_width=True)

    st.markdown("---")

    st.subheader("🚀 Technologies Used")

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.info("🐍 Python")

    with col2:
        st.info("📊 Pandas")

    with col3:
        st.info("🤖 Scikit-Learn")

    with col4:
        st.info("🎈 Streamlit")

    st.markdown("---")

    st.subheader("✨ Project Features")

    a,b,c = st.columns(3)

    with a:
        st.success("""
### 👥 Customer Segmentation

- RFM Analysis
- KMeans Clustering
- High Value Customers
- Loyal Customers
- Regular Customers
- At Risk Customers
""")

    with b:
        st.success("""
### 🛍 Product Recommendation

- Collaborative Filtering
- Cosine Similarity
- Top 5 Products
- Fast Recommendation
""")

    with c:
        st.success("""
### 📊 Analytics Dashboard

- KPI Cards
- Sales Trends
- Country Analysis
- Product Analysis
- Customer Insights
""")

    st.markdown("---")

    st.subheader("📈 Revenue Distribution")

    fig = px.histogram(
        df,
        x="TotalAmount",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
    """
    <div class="footer">
    Shopper Spectrum © 2026 <br>
    Developed using Python • Machine Learning • Streamlit
    </div>
    """,
    unsafe_allow_html=True
    )

    # ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Dashboard":

    st.title("📊 Business Dashboard")
    st.markdown("### E-Commerce Sales Analytics")

    # ----------------------------
    # KPI
    # ----------------------------

    revenue = df["TotalAmount"].sum()
    customers = df["CustomerID"].nunique()
    products = df["Description"].nunique()
    orders = df["InvoiceNo"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Revenue", f"₹ {revenue:,.0f}")
    c2.metric("👥 Customers", customers)
    c3.metric("🛍 Products", products)
    c4.metric("📦 Orders", orders)

    st.markdown("---")

    # ----------------------------
    # Monthly Sales
    # ----------------------------

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("Month")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="TotalAmount",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Top Products
    # ----------------------------

    col1, col2 = st.columns(2)

    with col1:

        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="Quantity",
            y="Description",
            orientation="h",
            title="Top 10 Selling Products"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        top_country = (
            df.groupby("Country")["TotalAmount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_country,
            x="Country",
            y="TotalAmount",
            title="Revenue by Country"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # Customer Segments
    # ----------------------------

    seg = (
        customer_segments["Customer_Segment"]
        .value_counts()
        .reset_index()
    )

    seg.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            seg,
            names="Segment",
            values="Customers",
            title="Customer Segments"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            seg,
            x="Segment",
            y="Customers",
            color="Segment",
            title="Segment Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # Top Customers
    # ----------------------------

    top_customer = (
        df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_customer,
        x="CustomerID",
        y="TotalAmount",
        title="Top 10 Customers by Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.success("Dashboard Loaded Successfully ✅")

    # ==========================================================
# CUSTOMER SEGMENTATION
# =========================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")
    st.write("Enter customer RFM values to predict the customer segment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (Days)", min_value=0, value=30)

    with col2:
        frequency = st.number_input("Frequency", min_value=1, value=5)

    with col3:
        monetary = st.number_input("Monetary Value", min_value=0.0, value=1000.0)

    if st.button("🔍 Predict Customer Segment"):

        sample = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        sample_scaled = scaler.transform(sample)
        cluster = int(kmeans.predict(sample_scaled)[0])

        cluster_names = {
            0: "Regular Customer",
            1: "At-Risk Customer",
            2: "High Value Customer",
            3: "Loyal Customer"
        }

        segment = cluster_names.get(cluster, "Unknown")

        st.success(f"Predicted Segment: **{segment}**")

        if segment == "High Value Customer":
            st.info("💎 Recommendation: Provide VIP offers, premium membership and exclusive discounts.")

        elif segment == "Loyal Customer":
            st.info("🎁 Recommendation: Reward with loyalty points and cashback offers.")

        elif segment == "Regular Customer":
            st.info("🛍 Recommendation: Send combo offers and product recommendations.")

        else:
            st.warning("📢 Recommendation: Send win-back campaigns and special discount coupons.")

        summary = pd.DataFrame({
            "Feature": ["Recency", "Frequency", "Monetary", "Cluster", "Segment"],
            "Value": [
                recency,
                frequency,
                monetary,
                cluster,
                segment
            ]
        })

        summary["Value"] = summary["Value"].astype(str)
        st.table(summary)


df = pd.read_csv("cleaned_online_retail.csv", encoding="latin1")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

df["Description"] = df["Description"].fillna("Unknown Product")
df["Country"] = df["Country"].fillna("Unknown")

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Product Recommendation Check

if isinstance(product_list, np.ndarray):
    product_list = product_list.tolist()

product_list = sorted(list(set(product_list)))

st.write("Total Products:", len(product_list))