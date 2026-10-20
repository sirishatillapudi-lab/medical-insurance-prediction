import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🩺", layout="centered")

st.markdown("""
<style>
.main {background: linear-gradient(180deg, #f5faff 0%, #ffffff 65%);}
h1 {color:#123b63;}
div.stButton > button {background:#087f8c;color:white;border-radius:10px;border:0;padding:0.65rem 1.1rem;font-weight:700;}
div.stButton > button:hover {background:#06636c;color:white;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Medical Insurance Cost Prediction")
st.write("Estimate medical insurance charges from the details entered below.")
st.caption("Educational demonstration only. This estimate is not an insurance quote or medical/financial advice.")

@st.cache_resource
def load_model():
    data_path = Path(__file__).resolve().parent / "insurance.csv"
    data = pd.read_csv(data_path)
    required = ["age", "bmi", "children", "smoker", "region", "charges"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("insurance.csv is missing columns: " + ", ".join(missing))
    X = data[["age", "bmi", "children", "smoker", "region"]]
    y = data["charges"]
    numeric = ["age", "bmi", "children"]
    categorical = ["smoker", "region"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=2, n_jobs=-1))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions), len(data)

try:
    model, mae, r2, row_count = load_model()
except Exception as exc:
    st.error("The model or dataset could not be loaded. Check that insurance.csv is in the same folder as app.py.")
    st.exception(exc)
    st.stop()

with st.form("prediction_form"):
    st.subheader("Enter your details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1, format="%.1f")
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
    with c2:
        smoker = st.selectbox("Smoker", ["no", "yes"], format_func=lambda x: "🚭 No" if x == "no" else "🚬 Yes")
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    submitted = st.form_submit_button("🔍 Predict Insurance Cost", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "age": int(age), "bmi": float(bmi), "children": int(children),
        "smoker": smoker, "region": region
    }])
    estimate = float(model.predict(input_data)[0])
    st.success("Prediction completed")
    st.metric("Estimated Medical Insurance Charges", f"${estimate:,.2f}")
    st.info("This is an approximate model output from the included demonstration dataset. Real insurer prices may differ.")

with st.expander("📊 Model information"):
    st.write(f"Training dataset rows: **{row_count:,}**")
    st.write(f"Test-set MAE: **${mae:,.2f}**")
    st.write(f"Test-set R²: **{r2:.3f}**")
    st.caption("The included CSV is a reproducible demonstration dataset, not verified real customer or insurer data.")

st.divider()
st.caption("Built with Python, Streamlit, pandas and scikit-learn.")
