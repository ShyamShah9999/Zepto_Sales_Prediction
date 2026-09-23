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

if __name__ == "__main__":

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