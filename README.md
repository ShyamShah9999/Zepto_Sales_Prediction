# 🛒 Zepto Sales Prediction

An end-to-end machine learning project that predicts **Total Revenue** for Zepto quick-commerce products using a Flask REST backend and a Streamlit interactive frontend.

---

## 📁 Project Structure

```
Zepto Sales & Analysis Project/
├── zepto_sales_dataset.csv   # Source dataset
├── app.py                    # Flask backend — ML training + REST API
├── frontend.py               # Streamlit frontend — dashboard + prediction UI
├── requirements.txt          # Python dependencies
├── model.pkl                 # Serialised best model (auto-generated on first run)
├── encoders.pkl              # Label encoders (auto-generated on first run)
└── README.md                 # This file
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Revenue Prediction** | Predict Total Revenue for any product/city/order combination |
| **90% Confidence Interval** | Tree-ensemble percentile bounds on every prediction |
| **Multi-model comparison** | Random Forest, Gradient Boosting, Linear Regression |
| **Auto model selection** | Best model by test R² is selected and persisted automatically |
| **Interactive analytics** | Revenue by category, city, product; discount & influencer impact |
| **Feature importances** | Which inputs drive revenue the most |
| **Cross-validation scores** | 5-fold CV R² with standard deviation for each model |
| **REST API** | Clean JSON API usable by any HTTP client |

---

## 🗃️ Dataset

**File:** `zepto_sales_dataset.csv`

| Column | Type | Description |
|---|---|---|
| `Product Name` | string | Product SKU (e.g. Maggi Noodles, Amul Milk) |
| `Category` | string | Product category (Snacks, Beverages, Dairy, …) |
| `City` | string | Delivery city (Delhi, Mumbai, Bangalore, …) |
| `Original Price` | int | MRP before any discount (₹) |
| `Current Price` | int | Selling price (₹) |
| `Discount` | int | Discount percentage applied (0–50 %) |
| `Orders` | int | Number of orders placed |
| `Total Revenue` | int | Revenue generated — **prediction target** |
| `Influencer Active` | Yes/No | Whether an influencer campaign was active |

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Start the Flask backend

```bash
python app.py
```

The server starts on **http://localhost:5000**.  
Models are trained automatically on startup (takes ~5 s).

### 3 — Launch the Streamlit frontend (new terminal)

```bash
streamlit run frontend.py
```

Open **http://localhost:8501** in your browser.

---

## 🔌 REST API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Returns backend status and the active best model.

```json
{ "status": "ok", "best_model": "Random Forest" }
```

---

### `GET /categories`
Returns distinct values for all categorical fields (use to populate dropdowns).

```json
{
  "products":   ["Aashirvaad Atta", "Amul Milk 500ml", ...],
  "categories": ["Beverages", "Confectionery", ...],
  "cities":     ["Bangalore", "Chennai", ...],
  "influencer": ["Yes", "No"]
}
```

---

### `POST /predict`
Predict total revenue for a given product scenario.

**Request body:**
```json
{
  "product_name":     "Maggi Noodles",
  "category":         "Instant Food",
  "city":             "Mumbai",
  "original_price":   110,
  "current_price":    120,
  "discount":         5,
  "orders":           200,
  "influencer_active": "Yes"
}
```

**Response:**
```json
{
  "predicted_revenue": 24150.30,
  "model_used": "Random Forest",
  "confidence_interval": {
    "lower_90": 21200.00,
    "upper_90": 27300.00
  }
}
```

---

### `GET /analytics`
Pre-aggregated dataset analytics: summary KPIs, revenue by category/city, top products, influencer impact, correlations.

---

### `GET /model_metrics`
Returns MAE, RMSE, R², CV R² mean & std for all three models plus the selected best model name.

---

### `GET /feature_importance`
Returns ranked feature importances for the best model (when it supports `feature_importances_`).

---

## 🧠 Machine Learning Pipeline

```
Raw CSV → Label Encoding → Feature Matrix
              ↓
   Train / Test Split (80/20, random_state=42)
              ↓
   Three models trained in parallel:
   • Random Forest (200 trees)
   • Gradient Boosting (150 trees, lr=0.1)
   • Linear Regression
              ↓
   5-fold Cross Validation on training set
              ↓
   Best model selected by test R² → serialised to model.pkl
```

**Features used:**
- Product Name (encoded)
- Category (encoded)
- City (encoded)
- Original Price
- Current Price
- Discount %
- Orders
- Influencer Active (encoded)

**Target:** `Total Revenue`

---

## 🖥️ Frontend Pages

| Page | Contents |
|---|---|
| 🏠 **Overview** | KPI cards, revenue by category (bar), revenue by city (pie) |
| 📊 **Analytics** | Top products, orders by city, discount impact, influencer analysis |
| 🤖 **Predict Revenue** | Interactive form → live prediction with confidence interval |
| 📈 **Model Performance** | Metrics table, R² bar, MAE/RMSE grouped bar, feature importance, CV chart |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.0, Flask-CORS |
| ML | scikit-learn (RandomForest, GradientBoosting, LinearRegression) |
| Data | pandas, NumPy |
| Frontend | Streamlit 1.37, Plotly Express |
| Serialisation | joblib |

---

## 📊 Model Performance (typical)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | ~1,200 | ~2,100 | ~0.97 |
| Gradient Boosting | ~1,500 | ~2,500 | ~0.96 |
| Linear Regression | ~3,800 | ~5,200 | ~0.82 |

*Results may vary slightly depending on the dataset split.*

---

## 📝 License

This project is for educational and analytical purposes.  
Dataset: Zepto quick-commerce sales records.
