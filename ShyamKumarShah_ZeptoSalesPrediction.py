"""
Shyam Kumar Shah
Zepto Sales Prediction — Complete Project Code

Contains the Flask backend and Streamlit frontend in one Python source file.

Backend:
    python ShyamKumarShah_ZeptoSalesPrediction.py

Frontend (second terminal):
    streamlit run ShyamKumarShah_ZeptoSalesPrediction.py
"""

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    def get_script_run_ctx():
        return None

# ============================================================
# BACKEND — Flask + Machine Learning
# ============================================================

"""
Zepto Sales Prediction — Flask Backend
Trains multiple regression models on startup and exposes
prediction + analytics REST endpoints.
"""

import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

DATA_PATH = "zepto_sales_dataset.csv"
MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoders.pkl"


# ──────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df.columns = df.columns.str.strip()

    cat_cols = [
        "Product Name",
        "Category",
        "City",
        "Influencer Active"
    ]

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(
            df[col].astype(str)
        )
        encoders[col] = le

    return df, encoders


def build_feature_matrix(df):
    feature_cols = [
        "Product Name_enc",
        "Category_enc",
        "City_enc",
        "Original Price",
        "Current Price",
        "Discount",
        "Orders",
        "Influencer Active_enc"
    ]

    X = df[feature_cols].values
    y = df["Total Revenue"].values

    return X, y, feature_cols


# ──────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42
        ),

        "Linear Regression": LinearRegression()
    }

    results = {}
    trained = {}

    for name, mdl in models.items():

        mdl.fit(X_train, y_train)

        preds = mdl.predict(X_test)

        cv_scores = cross_val_score(
            mdl,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )

        results[name] = {
            "MAE": round(
                float(mean_absolute_error(y_test, preds)),
                2
            ),

            "RMSE": round(
                float(np.sqrt(
                    mean_squared_error(y_test, preds)
                )),
                2
            ),

            "R2": round(
                float(r2_score(y_test, preds)),
                4
            ),

            "CV_R2_mean": round(
                float(cv_scores.mean()),
                4
            ),

            "CV_R2_std": round(
                float(cv_scores.std()),
                4
            )
        }

        trained[name] = mdl

    return trained, results


# ──────────────────────────────────────────────
# Startup: load data and train models
# ──────────────────────────────────────────────

df_global, encoders_global = load_and_preprocess()

X_all, y_all, FEATURE_COLS = build_feature_matrix(
    df_global
)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42
)

trained_models, model_metrics = train_models(
    X_train,
    y_train,
    X_test,
    y_test
)


# Pick best model based on test R²
best_model_name = max(
    model_metrics,
    key=lambda k: model_metrics[k]["R2"]
)

best_model = trained_models[best_model_name]


# ──────────────────────────────────────────────
# Prediction interval calculation
# ──────────────────────────────────────────────
#
# Instead of treating Gradient Boosting's individual
# boosting stages as independent trees, we calculate
# an empirical prediction range using residuals from
# the test set.
#

test_predictions = best_model.predict(X_test)

test_residuals = y_test - test_predictions

residual_lower = float(
    np.percentile(test_residuals, 5)
)

residual_upper = float(
    np.percentile(test_residuals, 95)
)


# Save model and encoders
joblib.dump(best_model, MODEL_PATH)
joblib.dump(encoders_global, ENCODER_PATH)


print(
    f"[Startup] Best model: {best_model_name}  "
    f"R²={model_metrics[best_model_name]['R2']}"
)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def safe_encode(encoder, value):
    """
    Encode a categorical value safely.
    Returns 0 if the value was not present during training.
    """

    classes = list(encoder.classes_)

    if value in classes:
        return int(
            encoder.transform([value])[0]
        )

    return 0


# ──────────────────────────────────────────────
# Home
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():

    return jsonify({
        "service": "Zepto Sales Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/analytics",
            "/model_metrics",
            "/categories",
            "/feature_importance",
            "/health"
        ]
    })


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "best_model": best_model_name
    })


# ──────────────────────────────────────────────
# Categories
# ──────────────────────────────────────────────

@app.route("/categories", methods=["GET"])
def categories():

    df = df_global

    return jsonify({

        "products": sorted(
            df["Product Name"]
            .unique()
            .tolist()
        ),

        "categories": sorted(
            df["Category"]
            .unique()
            .tolist()
        ),

        "cities": sorted(
            df["City"]
            .unique()
            .tolist()
        ),

        "influencer": [
            "Yes",
            "No"
        ]
    })


# ──────────────────────────────────────────────
# Model metrics
# ──────────────────────────────────────────────

@app.route("/model_metrics", methods=["GET"])
def get_model_metrics():

    return jsonify({

        "metrics": model_metrics,

        "best_model": best_model_name
    })


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():

    """
    POST JSON body:

    {
        "product_name": "Maggi Noodles",
        "category": "Instant Food",
        "city": "Mumbai",
        "original_price": 110,
        "current_price": 120,
        "discount": 5,
        "orders": 200,
        "influencer_active": "Yes"
    }
    """

    try:

        data = request.get_json(force=True)

        if not data:
            return jsonify({
                "error": "No JSON body provided"
            }), 400


        required = [
            "product_name",
            "category",
            "city",
            "original_price",
            "current_price",
            "discount",
            "orders",
            "influencer_active"
        ]


        missing = [
            field
            for field in required
            if field not in data
        ]


        if missing:

            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400


        enc = encoders_global


        # ──────────────────────────────────────
        # Encode input features
        # ──────────────────────────────────────

        product_encoded = safe_encode(
            enc["Product Name"],
            str(data["product_name"])
        )

        category_encoded = safe_encode(
            enc["Category"],
            str(data["category"])
        )

        city_encoded = safe_encode(
            enc["City"],
            str(data["city"])
        )

        influencer_encoded = safe_encode(
            enc["Influencer Active"],
            str(data["influencer_active"])
        )


        # ──────────────────────────────────────
        # Create model input
        # ──────────────────────────────────────

        features = np.array([[
            product_encoded,
            category_encoded,
            city_encoded,
            float(data["original_price"]),
            float(data["current_price"]),
            float(data["discount"]),
            float(data["orders"]),
            influencer_encoded
        ]])


        # ──────────────────────────────────────
        # Make prediction
        # ──────────────────────────────────────

        prediction = float(
            best_model.predict(features)[0]
        )


        # ──────────────────────────────────────
        # Empirical 90% prediction range
        # ──────────────────────────────────────

        lower_bound = max(
            0,
            prediction + residual_lower
        )

        upper_bound = max(
            0,
            prediction + residual_upper
        )


        # ──────────────────────────────────────
        # Response
        # ──────────────────────────────────────

        return jsonify({

            "predicted_revenue": round(
                prediction,
                2
            ),

            "model_used": best_model_name,

            "confidence_interval": {

                "lower_90": round(
                    lower_bound,
                    2
                ),

                "upper_90": round(
                    upper_bound,
                    2
                )
            }
        })


    except ValueError as exc:

        print(
            f"[Prediction Value Error] {exc}"
        )

        return jsonify({
            "error": f"Invalid input value: {exc}"
        }), 400


    except Exception as exc:

        print(
            f"[Prediction Error] "
            f"{type(exc).__name__}: {exc}"
        )

        return jsonify({
            "error": str(exc)
        }), 500


# ──────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────

@app.route("/analytics", methods=["GET"])
def analytics():

    df = df_global


    # ──────────────────────────────────────────
    # Revenue by category
    # ──────────────────────────────────────────

    rev_by_cat = (

        df.groupby("Category")["Total Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

        .reset_index()

        .rename(
            columns={
                "Total Revenue": "Revenue"
            }
        )

        .assign(
            Revenue=lambda x:
            x["Revenue"].round(2)
        )
    )


    # ──────────────────────────────────────────
    # Revenue by city
    # ──────────────────────────────────────────

    rev_by_city = (

        df.groupby("City")["Total Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

        .reset_index()

        .rename(
            columns={
                "Total Revenue": "Revenue"
            }
        )

        .assign(
            Revenue=lambda x:
            x["Revenue"].round(2)
        )
    )


    # ──────────────────────────────────────────
    # Top 10 products
    # ──────────────────────────────────────────

    top_products = (

        df.groupby("Product Name")["Total Revenue"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(10)

        .reset_index()

        .rename(
            columns={
                "Total Revenue": "Revenue"
            }
        )

        .assign(
            Revenue=lambda x:
            x["Revenue"].round(2)
        )
    )


    # ──────────────────────────────────────────
    # Orders by city
    # ──────────────────────────────────────────

    orders_city = (

        df.groupby("City")["Orders"]

        .sum()

        .sort_values(
            ascending=False
        )

        .reset_index()
    )


    # ──────────────────────────────────────────
    # Average discount by category
    # ──────────────────────────────────────────

    avg_discount = (

        df.groupby("Category")["Discount"]

        .mean()

        .round(2)

        .reset_index()

        .rename(
            columns={
                "Discount":
                "Avg Discount %"
            }
        )
    )


    # ──────────────────────────────────────────
    # Influencer impact
    # ──────────────────────────────────────────

    influencer_impact = (

        df.groupby(
            "Influencer Active"
        )["Total Revenue"]

        .agg([
            "mean",
            "sum",
            "count"
        ])

        .round(2)

        .reset_index()

        .rename(
            columns={
                "mean": "Avg Revenue",
                "sum": "Total Revenue",
                "count": "Count"
            }
        )
    )


    # ──────────────────────────────────────────
    # Correlations
    # ──────────────────────────────────────────

    corr_rev_orders = round(
        float(
            df["Total Revenue"]
            .corr(df["Orders"])
        ),
        4
    )

    corr_rev_discount = round(
        float(
            df["Total Revenue"]
            .corr(df["Discount"])
        ),
        4
    )

    corr_rev_price = round(
        float(
            df["Total Revenue"]
            .corr(df["Current Price"])
        ),
        4
    )


    # ──────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────

    summary = {

        "total_records":
            int(len(df)),

        "total_revenue":
            round(
                float(
                    df["Total Revenue"].sum()
                ),
                2
            ),

        "avg_revenue":
            round(
                float(
                    df["Total Revenue"].mean()
                ),
                2
            ),

        "max_revenue":
            round(
                float(
                    df["Total Revenue"].max()
                ),
                2
            ),

        "min_revenue":
            round(
                float(
                    df["Total Revenue"].min()
                ),
                2
            ),

        "total_orders":
            int(
                df["Orders"].sum()
            ),

        "unique_products":
            int(
                df["Product Name"].nunique()
            ),

        "unique_cities":
            int(
                df["City"].nunique()
            ),

        "unique_categories":
            int(
                df["Category"].nunique()
            )
    }


    # ──────────────────────────────────────────
    # Return analytics
    # ──────────────────────────────────────────

    return jsonify({

        "summary":
            summary,

        "revenue_by_category":
            rev_by_cat.to_dict(
                orient="records"
            ),

        "revenue_by_city":
            rev_by_city.to_dict(
                orient="records"
            ),

        "top_products":
            top_products.to_dict(
                orient="records"
            ),

        "orders_by_city":
            orders_city.to_dict(
                orient="records"
            ),

        "avg_discount_by_cat":
            avg_discount.to_dict(
                orient="records"
            ),

        "influencer_impact":
            influencer_impact.to_dict(
                orient="records"
            ),

        "correlations": {

            "revenue_vs_orders":
                corr_rev_orders,

            "revenue_vs_discount":
                corr_rev_discount,

            "revenue_vs_price":
                corr_rev_price
        }
    })


# ──────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────

@app.route("/feature_importance", methods=["GET"])
def feature_importance():

    if not hasattr(
        best_model,
        "feature_importances_"
    ):

        return jsonify({
            "error":
            "Current best model does not support feature importances"
        }), 400


    readable = [

        "Product Name",
        "Category",
        "City",
        "Original Price",
        "Current Price",
        "Discount",
        "Orders",
        "Influencer Active"
    ]


    importances = (
        best_model
        .feature_importances_
        .tolist()
    )


    pairs = sorted(
        zip(readable, importances),
        key=lambda x: x[1],
        reverse=True
    )


    return jsonify({

        "feature_importances": [

            {
                "feature": feature,
                "importance": round(
                    importance,
                    4
                )
            }

            for feature, importance in pairs
        ]
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__" and get_script_run_ctx() is None:

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        debug=True,
        host="0.0.0.0",
        port=port
    )

# ============================================================
# FRONTEND — Streamlit Dashboard
# ============================================================

def run_streamlit_frontend():
    """
    Zepto Sales Prediction — Streamlit Frontend
    Connects to the Flask backend (app.py) and renders interactive dashboards + prediction UI.

    Run:
        streamlit run frontend.py
    """

    import streamlit as st
    import requests
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go


    # ============================================================
    # CONFIG
    # ============================================================

    API_BASE = "http://localhost:5000"

    st.set_page_config(
        page_title="Zepto Sales Prediction",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # ============================================================
    # CUSTOM CSS
    # ============================================================

    st.markdown(
        """
    <style>

        /* =========================
           PAGE
           ========================= */

        .main .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }


        /* =========================
           SIDEBAR
           ========================= */

        section[data-testid="stSidebar"] {
            background-color: #f0ebfc !important;
            min-width: 260px !important;
            max-width: 260px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 260px !important;
        }

        section[data-testid="stSidebar"] p {
            color: #222222 !important;
        }

        section[data-testid="stSidebar"] span {
            color: #222222 !important;
        }

        section[data-testid="stSidebar"] label {
            color: #222222 !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #222222 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: #d8cef0 !important;
        }


        /* Navigation radio buttons */

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #222222 !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }


        /* =========================
           MAIN HEADER
           ========================= */

        .main-header {
            background: linear-gradient(
                135deg,
                #6C3DE1 0%,
                #9B59B6 50%,
                #8E44AD 100%
            );

            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
            text-align: center;
        }

        .main-header h1 {
            font-size: 2.2rem;
            margin: 0;
            font-weight: 700;
            color: white !important;
        }

        .main-header p {
            font-size: 1rem;
            margin: 0.4rem 0 0;
            color: white !important;
            opacity: 0.9;
        }


        /* =========================
           KPI CARDS
           ========================= */

        .metric-card {
            background: #f7f8fa;
            border: 1px solid #e2e4e9;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            text-align: center;
        }

        .metric-card h3 {
            font-size: 1.6rem;
            color: #6C3DE1 !important;
            margin: 0;
            font-weight: 700;
        }

        .metric-card p {
            font-size: 0.8rem;
            color: #555555 !important;
            margin: 0.2rem 0 0;
        }


        /* =========================
           PREDICTION BOX
           ========================= */

        .prediction-box {
            background: linear-gradient(
                135deg,
                #6C3DE1,
                #9B59B6
            );

            color: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1rem;
        }

        .prediction-box h2 {
            font-size: 2rem;
            margin: 0;
            color: white !important;
        }

        .prediction-box p {
            margin: 0.3rem 0 0;
            color: white !important;
        }


        /* =========================
           SECTION TITLES
           ========================= */

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #6C3DE1 !important;
            border-left: 4px solid #6C3DE1;
            padding-left: 0.7rem;
            margin: 1.2rem 0 0.8rem;
        }

    </style>
    """,
        unsafe_allow_html=True,
    )


    # ============================================================
    # API FUNCTIONS
    # ============================================================

    @st.cache_data(ttl=300)
    def fetch_analytics():

        try:
            r = requests.get(
                f"{API_BASE}/analytics",
                timeout=10
            )

            r.raise_for_status()

            return r.json()

        except Exception as e:

            return {
                "error": str(e)
            }


    @st.cache_data(ttl=300)
    def fetch_categories():

        try:
            r = requests.get(
                f"{API_BASE}/categories",
                timeout=10
            )

            r.raise_for_status()

            return r.json()

        except Exception:

            return {}


    @st.cache_data(ttl=300)
    def fetch_model_metrics():

        try:
            r = requests.get(
                f"{API_BASE}/model_metrics",
                timeout=10
            )

            r.raise_for_status()

            return r.json()

        except Exception as e:

            return {
                "error": str(e)
            }


    @st.cache_data(ttl=300)
    def fetch_feature_importance():

        try:
            r = requests.get(
                f"{API_BASE}/feature_importance",
                timeout=10
            )

            r.raise_for_status()

            return r.json()

        except Exception as e:

            return {
                "error": str(e)
            }


    def call_predict(payload):

        try:

            r = requests.post(
                f"{API_BASE}/predict",
                json=payload,
                timeout=10
            )

            r.raise_for_status()

            return r.json()

        except requests.exceptions.ConnectionError:

            return {
                "error":
                "Cannot connect to backend. "
                "Make sure app.py is running on port 5000."
            }

        except Exception as e:

            return {
                "error": str(e)
            }


    def check_backend():

        try:

            r = requests.get(
                f"{API_BASE}/health",
                timeout=5
            )

            return r.status_code == 200

        except Exception:

            return False


    # ============================================================
    # KPI CARD
    # ============================================================

    def metric_card(col, label, value):

        col.markdown(
            f'<div class="metric-card">'
            f'<h3>{value}</h3>'
            f'<p>{label}</p>'
            f'</div>',
            unsafe_allow_html=True
        )


    # ============================================================
    # SIDEBAR
    # ============================================================

    with st.sidebar:

        # Logo
        st.markdown(
            '<div style="text-align:center;padding:10px 0 15px 0;">'
            '<div style="font-size:42px;margin-bottom:5px;">🛒</div>'
            '<div style="color:#222222;font-size:18px;font-weight:700;">'
            'Zepto Sales Dashboard'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "📊 Analytics",
                "🤖 Predict Revenue",
                "📈 Model Performance"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Backend status
        is_alive = check_backend()

        if is_alive:

            st.markdown(
                '<div style="color:#222222;font-weight:600;">'
                '🟢 Backend status: Online'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div style="color:#222222;font-weight:600;">'
                '🔴 Backend status: Offline'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div style="color:#666666;font-size:13px;margin-top:12px;">'
            'Start backend:'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="'
            'background:#ffffff;'
            'border:1px solid #d8cef0;'
            'border-radius:6px;'
            'padding:7px 9px;'
            'margin-top:4px;'
            'font-family:monospace;'
            'font-size:12px;'
            'color:#333333;'
            '">'
            'python app.py'
            '</div>',
            unsafe_allow_html=True
        )


    # ============================================================
    # OVERVIEW
    # ============================================================

    def page_overview():

        st.markdown(
            '<div class="main-header">'
            '<h1>🛒 Zepto Sales Intelligence</h1>'
            '<p>AI-powered sales prediction and analytics for Zepto quick-commerce operations</p>'
            '</div>',
            unsafe_allow_html=True
        )

        analytics = fetch_analytics()

        if "error" in analytics:

            st.error(
                f"⚠️ Could not load analytics: "
                f"{analytics['error']}"
            )

            st.info(
                "Make sure the Flask backend is running: "
                "`python app.py`"
            )

            return

        s = analytics["summary"]


        # KPI ROW 1

        c1, c2, c3, c4 = st.columns(4)

        metric_card(
            c1,
            "Total Revenue",
            f"₹{s['total_revenue']:,.0f}"
        )

        metric_card(
            c2,
            "Total Orders",
            f"{s['total_orders']:,}"
        )

        metric_card(
            c3,
            "Avg Revenue / Record",
            f"₹{s['avg_revenue']:,.0f}"
        )

        metric_card(
            c4,
            "Unique Products",
            str(s["unique_products"])
        )


        st.markdown("")


        # KPI ROW 2

        c5, c6, c7 = st.columns(3)

        metric_card(
            c5,
            "Cities Covered",
            str(s["unique_cities"])
        )

        metric_card(
            c6,
            "Product Categories",
            str(s["unique_categories"])
        )

        metric_card(
            c7,
            "Total Records",
            f"{s['total_records']:,}"
        )


        st.markdown("")


        # CHARTS

        col_l, col_r = st.columns(2)


        # Revenue by Category

        with col_l:

            st.markdown(
                '<div class="section-title">'
                'Revenue by Category'
                '</div>',
                unsafe_allow_html=True
            )

            df_cat = pd.DataFrame(
                analytics["revenue_by_category"]
            )

            fig = px.bar(
                df_cat,
                x="Revenue",
                y="Category",
                orientation="h",
                color="Revenue",
                color_continuous_scale="Purples",
                labels={
                    "Revenue":
                    "Total Revenue (₹)"
                }
            )

            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0
                ),
                height=320
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # Revenue by City

        with col_r:

            st.markdown(
                '<div class="section-title">'
                'Revenue by City'
                '</div>',
                unsafe_allow_html=True
            )

            df_city = pd.DataFrame(
                analytics["revenue_by_city"]
            )

            fig2 = px.pie(
                df_city,
                values="Revenue",
                names="City",
                color_discrete_sequence=
                px.colors.sequential.Purples_r,
                hole=0.4
            )

            fig2.update_layout(
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0
                ),
                height=320
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


    # ============================================================
    # ANALYTICS
    # ============================================================

    def page_analytics():

        st.markdown(
            '<h2 style="color:#6C3DE1;">📊 Deep Analytics</h2>',
            unsafe_allow_html=True
        )

        analytics = fetch_analytics()

        if "error" in analytics:

            st.error(
                f"⚠️ {analytics['error']}"
            )

            return


        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Top Products",
                "Orders Analysis",
                "Discount Impact",
                "Influencer Effect"
            ]
        )


        # TOP PRODUCTS

        with tab1:

            st.markdown(
                '<div class="section-title">'
                'Top 10 Products by Revenue'
                '</div>',
                unsafe_allow_html=True
            )

            df_top = pd.DataFrame(
                analytics["top_products"]
            )

            fig = px.bar(
                df_top,
                x="Product Name",
                y="Revenue",
                color="Revenue",
                color_continuous_scale="Purples",
                text_auto=".2s",
                labels={
                    "Revenue":
                    "Total Revenue (₹)"
                }
            )

            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
                margin=dict(
                    t=20,
                    b=60
                ),
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ORDERS ANALYSIS

        with tab2:

            st.markdown(
                '<div class="section-title">'
                'Orders by City'
                '</div>',
                unsafe_allow_html=True
            )

            df_ord = pd.DataFrame(
                analytics["orders_by_city"]
            )

            fig2 = px.bar(
                df_ord,
                x="City",
                y="Orders",
                color="Orders",
                color_continuous_scale="Purples",
                text_auto=True
            )

            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20),
                height=380
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


            st.markdown(
                '<div class="section-title">'
                'Correlations with Total Revenue'
                '</div>',
                unsafe_allow_html=True
            )

            corr = analytics["correlations"]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "vs. Orders",
                f"{corr['revenue_vs_orders']:.4f}"
            )

            c2.metric(
                "vs. Discount",
                f"{corr['revenue_vs_discount']:.4f}"
            )

            c3.metric(
                "vs. Price",
                f"{corr['revenue_vs_price']:.4f}"
            )


        # DISCOUNT IMPACT

        with tab3:

            st.markdown(
                '<div class="section-title">'
                'Avg Discount % by Category'
                '</div>',
                unsafe_allow_html=True
            )

            df_disc = pd.DataFrame(
                analytics["avg_discount_by_cat"]
            )

            fig3 = px.bar(
                df_disc,
                x="Category",
                y="Avg Discount %",
                color="Avg Discount %",
                color_continuous_scale="RdYlGn",
                text_auto=".1f"
            )

            fig3.update_layout(
                coloraxis_showscale=True,
                margin=dict(t=20),
                height=380
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )


        # INFLUENCER EFFECT

        with tab4:

            st.markdown(
                '<div class="section-title">'
                'Influencer Active vs. Revenue'
                '</div>',
                unsafe_allow_html=True
            )

            df_infl = pd.DataFrame(
                analytics["influencer_impact"]
            )

            col_l, col_r = st.columns(2)


            with col_l:

                fig4 = px.bar(
                    df_infl,
                    x="Influencer Active",
                    y="Avg Revenue",
                    color="Influencer Active",
                    color_discrete_map={
                        "Yes": "#6C3DE1",
                        "No": "#C8B8F5"
                    },
                    text_auto=".2s",
                    title="Avg Revenue"
                )

                fig4.update_layout(
                    showlegend=False,
                    margin=dict(t=40),
                    height=320
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )


            with col_r:

                fig5 = px.pie(
                    df_infl,
                    values="Count",
                    names="Influencer Active",
                    color_discrete_map={
                        "Yes": "#6C3DE1",
                        "No": "#C8B8F5"
                    },
                    title="Records: Influencer Yes vs No",
                    hole=0.4
                )

                fig5.update_layout(
                    margin=dict(t=40),
                    height=320
                )

                st.plotly_chart(
                    fig5,
                    use_container_width=True
                )


    # ============================================================
    # PREDICT REVENUE
    # ============================================================

    def page_predict():

        st.markdown(
            '<h2 style="color:#6C3DE1;">'
            '🤖 Predict Total Revenue'
            '</h2>',
            unsafe_allow_html=True
        )

        st.markdown(
            "Fill in the product details below to get "
            "an AI-powered revenue prediction."
        )

        cats = fetch_categories()

        if not cats:

            st.error(
                "Could not load category data from "
                "the backend. Is app.py running?"
            )

            return


        with st.form("prediction_form"):

            col1, col2 = st.columns(2)


            with col1:

                product = st.selectbox(
                    "Product Name",
                    cats.get("products", [])
                )

                category = st.selectbox(
                    "Category",
                    cats.get("categories", [])
                )

                city = st.selectbox(
                    "City",
                    cats.get("cities", [])
                )

                influencer = st.selectbox(
                    "Influencer Active",
                    cats.get(
                        "influencer",
                        ["Yes", "No"]
                    )
                )


            with col2:

                original_price = st.number_input(
                    "Original Price (₹)",
                    min_value=1,
                    max_value=10000,
                    value=100,
                    step=1
                )

                current_price = st.number_input(
                    "Current Price (₹)",
                    min_value=1,
                    max_value=10000,
                    value=110,
                    step=1
                )

                discount = st.slider(
                    "Discount (%)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1
                )

                orders = st.number_input(
                    "Number of Orders",
                    min_value=1,
                    max_value=10000,
                    value=150,
                    step=1
                )


            submitted = st.form_submit_button(
                "🔮 Predict Revenue",
                use_container_width=True
            )


        if submitted:

            payload = {
                "product_name": product,
                "category": category,
                "city": city,
                "original_price": original_price,
                "current_price": current_price,
                "discount": discount,
                "orders": orders,
                "influencer_active": influencer
            }


            with st.spinner("Predicting..."):

                result = call_predict(
                    payload
                )


            if "error" in result:

                st.error(
                    f"Prediction failed: "
                    f"{result['error']}"
                )

            else:

                rev = result["predicted_revenue"]

                model = result.get(
                    "model_used",
                    "ML Model"
                )

                ci = result.get(
                    "confidence_interval",
                    {}
                )


                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2>₹ {rev:,.2f}</h2>'
                    f'<p>Predicted Total Revenue — '
                    f'<strong>{model}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )


                if ci.get("lower_90") is not None:

                    st.markdown("")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "90% CI — Lower Bound",
                        f"₹ {ci['lower_90']:,.2f}"
                    )

                    c2.metric(
                        "90% CI — Upper Bound",
                        f"₹ {ci['upper_90']:,.2f}"
                    )


                st.markdown(
                    '<div class="section-title">'
                    'Input Summary'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    pd.DataFrame(
                        [payload]
                    ).T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )


    # ============================================================
    # MODEL PERFORMANCE
    # ============================================================

    def page_model_performance():

        st.markdown(
            '<h2 style="color:#6C3DE1;">'
            '📈 Model Performance'
            '</h2>',
            unsafe_allow_html=True
        )

        metrics_data = fetch_model_metrics()

        if "error" in metrics_data:

            st.error(
                f"⚠️ {metrics_data['error']}"
            )

            return


        best = metrics_data.get(
            "best_model",
            ""
        )

        mets = metrics_data.get(
            "metrics",
            {}
        )


        st.info(
            f"🏆 **Best model selected by test R²:** {best}"
        )


        # Metrics table

        df_metrics = (
            pd.DataFrame(mets)
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Model"
                }
            )
        )


        # Convert numeric columns for safe styling across pandas 2.x and 3.x
        numeric_cols = ["MAE", "RMSE", "R2", "CV_R2_mean", "CV_R2_std"]
        df_metrics[numeric_cols] = df_metrics[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        st.dataframe(
            df_metrics.style
            .highlight_max(
                subset=["R2", "CV_R2_mean"],
                color="#d8c8f8",
                axis=0,
            )
            .highlight_min(
                subset=["MAE", "RMSE"],
                color="#d8c8f8",
                axis=0,
            ),
            use_container_width=True,
        )


        col_l, col_r = st.columns(2)


        # R2

        with col_l:

            st.markdown(
                '<div class="section-title">'
                'R² Score Comparison'
                '</div>',
                unsafe_allow_html=True
            )

            fig_r2 = px.bar(
                df_metrics,
                x="Model",
                y="R2",
                color="Model",
                color_discrete_sequence=[
                    "#6C3DE1",
                    "#9B59B6",
                    "#C8B8F5"
                ],
                text_auto=".4f"
            )

            fig_r2.update_layout(
                showlegend=False,
                yaxis_range=[0, 1.05],
                margin=dict(t=20),
                height=320
            )

            st.plotly_chart(
                fig_r2,
                use_container_width=True
            )


        # MAE RMSE

        with col_r:

            st.markdown(
                '<div class="section-title">'
                'MAE & RMSE Comparison'
                '</div>',
                unsafe_allow_html=True
            )

            df_err = df_metrics.melt(
                id_vars="Model",
                value_vars=["MAE", "RMSE"],
                var_name="Metric",
                value_name="Value",
            )
            df_err["Value"] = pd.to_numeric(df_err["Value"], errors="coerce")

            fig_err = px.bar(
                df_err,
                x="Model",
                y="Value",
                color="Metric",
                barmode="group",
                color_discrete_map={
                    "MAE": "#6C3DE1",
                    "RMSE": "#9B59B6"
                },
                text_auto=".0f"
            )

            fig_err.update_layout(
                margin=dict(t=20),
                height=320
            )

            st.plotly_chart(
                fig_err,
                use_container_width=True
            )


        # Feature importance

        st.markdown(
            '<div class="section-title">'
            'Feature Importances (Best Model)'
            '</div>',
            unsafe_allow_html=True
        )

        fi_data = fetch_feature_importance()

        if "error" not in fi_data:

            df_fi = pd.DataFrame(
                fi_data["feature_importances"]
            )

            fig_fi = px.bar(
                df_fi,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Purples",
                text_auto=".4f",
                labels={
                    "importance": "Importance",
                    "feature": "Feature"
                }
            )

            fig_fi.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(
                    autorange="reversed"
                ),
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=0
                ),
                height=380
            )

            st.plotly_chart(
                fig_fi,
                use_container_width=True
            )

        else:

            st.warning(
                "Feature importances not available "
                "for the current best model."
            )


        # Cross validation

        st.markdown(
            '<div class="section-title">'
            'Cross-Validation R² (5-Fold)'
            '</div>',
            unsafe_allow_html=True
        )

        fig_cv = go.Figure()


        for _, row in df_metrics.iterrows():

            fig_cv.add_trace(
                go.Bar(
                    name=row["Model"],
                    x=[row["Model"]],
                    y=[row["CV_R2_mean"]],
                    error_y=dict(
                        type="data",
                        array=[
                            row["CV_R2_std"]
                        ],
                        visible=True
                    ),
                    text=[
                        f"{row['CV_R2_mean']:.4f}"
                    ],
                    textposition="outside"
                )
            )


        fig_cv.update_layout(
            showlegend=False,
            yaxis_range=[0, 1.1],
            margin=dict(t=20),
            height=320,
            yaxis_title="Mean CV R²"
        )


        st.plotly_chart(
            fig_cv,
            use_container_width=True
        )


    # ============================================================
    # ROUTER
    # ============================================================

    if page == "🏠 Overview":

        page_overview()

    elif page == "📊 Analytics":

        page_analytics()

    elif page == "🤖 Predict Revenue":

        page_predict()

    elif page == "📈 Model Performance":

        page_model_performance()

if get_script_run_ctx() is not None:
    run_streamlit_frontend()
