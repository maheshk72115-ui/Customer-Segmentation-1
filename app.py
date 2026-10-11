
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "customer_segmentation_model.pkl"
DATA_PATH = BASE_DIR / "segmented_customers.csv"
EVALUATION_PATH = BASE_DIR / "model_evaluation.csv"

st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Customer Personality Analysis using Clustering "
    "for targeted marketing."
)

# --------------------------------------------------
# LOAD SAVED MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_customer_data():
    return pd.read_csv(DATA_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run the model-saving cell "
        "in your notebook and copy the .pkl file into this folder."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "segmented_customers.csv not found. Run the saving cell "
        "in your notebook and copy the CSV into this folder."
    )
    st.stop()

try:
    artifact = load_model()
    df = load_customer_data()
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]
medians = artifact["medians"]

if "Segment" not in df.columns:
    st.error("The saved customer data must contain a Segment column.")
    st.stop()

# Convert segment IDs into readable labels for display
df["Segment"] = df["Segment"].astype(str).apply(
    lambda x: f"Cluster {x}"
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

segments = sorted(df["Segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Select customer segments",
    options=segments,
    default=segments
)

filtered_df = df[df["Segment"].isin(selected_segments)].copy()

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
total_customers = len(filtered_df)
total_segments = filtered_df["Segment"].nunique()

numeric_cols = filtered_df.select_dtypes(
    include=np.number
).columns.tolist()

income_col = next(
    (c for c in numeric_cols if c.lower() in
     ["income", "annualincome", "annual_income"]),
    None
)

spending_col = next(
    (c for c in numeric_cols if c.lower() in
     ["spendingscore", "spending_score", "total_spending"]),
    None
)

k1, k2, k3 = st.columns(3)

k1.metric("Customers in View", f"{total_customers:,}")
k2.metric("Customer Segments", total_segments)

if income_col and not filtered_df.empty:
    k3.metric(
        "Average Income",
        f"{filtered_df[income_col].mean():,.2f}"
    )
elif spending_col and not filtered_df.empty:
    k3.metric(
        "Average Spending",
        f"{filtered_df[spending_col].mean():,.2f}"
    )
else:
    k3.metric("Features Available", len(features))

st.divider()

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Segment Analysis",
    "Feature Explorer",
    "Correlation Analysis",
    "Model Evaluation",
    "Predict New Customers"
])

# --------------------------------------------------
# TAB 1: SEGMENT ANALYSIS
# --------------------------------------------------
with tab1:
    st.subheader("Customer Segment Distribution")

    if filtered_df.empty:
        st.warning("No customers match the selected filters.")
    else:
        counts = (
            filtered_df["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        left, right = st.columns(2)

        with left:
            fig = px.bar(
                counts,
                x="Segment",
                y="Customers",
                color="Segment",
                title="Customers in Each Segment",
                text="Customers"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.pie(
                counts,
                names="Segment",
                values="Customers",
                title="Segment Percentage",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        if income_col and spending_col:
            st.subheader("Income vs Spending")

            fig = px.scatter(
                filtered_df,
                x=income_col,
                y=spending_col,
                color="Segment",
                title="Customer Groups by Income and Spending",
                hover_data=[
                    c for c in ["CustomerID", "ID", "Age"]
                    if c in filtered_df.columns
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Numeric Values by Segment")

        if numeric_cols:
            profile = (
                filtered_df.groupby("Segment")[numeric_cols]
                .mean()
                .round(2)
            )
            st.dataframe(profile, use_container_width=True)
        else:
            st.info("No numeric features are available.")

# --------------------------------------------------
# TAB 2: FEATURE EXPLORER
# --------------------------------------------------
with tab2:
    st.subheader("Explore Customer Features")

    available_numeric = filtered_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not available_numeric:
        st.info("No numeric features are available to plot.")
    else:
        x_feature = st.selectbox(
            "Select X-axis feature",
            available_numeric,
            key="feature_x"
        )

        y_options = [
            c for c in available_numeric if c != x_feature
        ]

        if y_options:
            y_feature = st.selectbox(
                "Select Y-axis feature",
                y_options,
                key="feature_y"
            )

            fig = px.scatter(
                filtered_df,
                x=x_feature,
                y=y_feature,
                color="Segment",
                title=f"{y_feature} vs {x_feature}"
            )
            st.plotly_chart(fig, use_container_width=True)

        selected_feature = st.selectbox(
            "View feature distribution",
            available_numeric,
            key="feature_distribution"
        )

        fig = px.histogram(
            filtered_df,
            x=selected_feature,
            color="Segment",
            marginal="box",
            title=f"Distribution of {selected_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(
            filtered_df[available_numeric].describe().round(2),
            use_container_width=True
        )

# --------------------------------------------------
# TAB 3: CORRELATION ANALYSIS
# --------------------------------------------------
with tab3:
    st.subheader("Correlation Heatmap")

    corr_data = filtered_df.select_dtypes(
        include=np.number
    )

    if corr_data.shape[1] >= 2:
        corr = corr_data.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are needed.")

# --------------------------------------------------
# TAB 4: MODEL EVALUATION
# --------------------------------------------------
with tab4:
    st.subheader("Clustering Model Evaluation")

    if EVALUATION_PATH.exists():
        evaluation_df = pd.read_csv(EVALUATION_PATH)
        st.dataframe(evaluation_df, use_container_width=True)

        st.caption(
            "For clustering: a higher Silhouette Score is generally "
            "better; lower Davies-Bouldin is generally better; "
            "higher Calinski-Harabasz is generally better. "
            "Compare these with cluster sizes and business usefulness."
        )
    else:
        st.warning(
            "model_evaluation.csv was not saved. The dashboard still "
            "works, but the evaluation table will appear after you "
            "save the evaluation results from your notebook."
        )

    st.write("**Model used for new predictions:** K-Means")

# --------------------------------------------------
# TAB 5: PREDICT NEW CUSTOMERS
# --------------------------------------------------
with tab5:
    st.subheader("Predict Segments for New Customers")

    st.write(
        "Upload a CSV containing the same preprocessed feature columns "
        "used when training your existing K-Means model."
    )
    st.write("Required feature columns:", ", ".join(features))

    uploaded_file = st.file_uploader(
        "Upload preprocessed customer CSV",
        type=["csv"],
        key="new_customer_upload"
    )

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            missing = [c for c in features if c not in new_df.columns]

            if missing:
                st.error(
                    "Required feature columns are missing: "
                    + ", ".join(missing)
                )
            else:
                X_new = new_df[features].copy()

                for col in features:
                    X_new[col] = pd.to_numeric(
                        X_new[col], errors="coerce"
                    )
                    X_new[col] = X_new[col].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    X_new[col] = X_new[col].fillna(
                        medians.get(col, 0)
                    )

                X_new_scaled = scaler.transform(X_new)
                predictions = model.predict(X_new_scaled)

                result_df = new_df.copy()
                result_df["Predicted_Segment"] = [
                    f"Cluster {x}" for x in predictions
                ]

                st.success("Segmentation completed!")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Segmented Customers",
                    data=csv_bytes,
                    file_name="predicted_customer_segments.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not process uploaded file: {e}")

# --------------------------------------------------
# FOOTER / DOWNLOAD
# --------------------------------------------------
st.divider()
st.caption(
    "Customer Segmentation Project | Streamlit Deployment"
)

st.download_button(
    "Download Current Customer Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_segments.csv",
    mime="text/csv"
)