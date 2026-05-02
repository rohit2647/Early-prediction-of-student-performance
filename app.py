import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("StudentPerformanceFactors.csv")

# ==============================
# PREPROCESSING
# ==============================
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# MODELS
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVM": SVR()
}

for model in models.values():
    model.fit(X_train, y_train)

# ==============================
# UI
# ==============================
st.title("🎓 Student Performance Predictor")

# Model selection (outside form)
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ==============================
# FORM INPUT (KEY FIX 🔥)
# ==============================
st.sidebar.title("Student Input")

with st.sidebar.form("input_form"):

    input_data = {}

    for col in X.columns:
        if col in le_dict:
            options = list(le_dict[col].classes_)
            selected = st.selectbox(col, options)
            input_data[col] = le_dict[col].transform([selected])[0]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            input_data[col] = st.slider(col, min_val, max_val)

    submit = st.form_submit_button("Predict")

# ==============================
# PREDICTION (only after submit)
# ==============================
if submit:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Predicted Exam Score")
    st.success(f"{prediction:.2f}")

    # ==============================
    # MODEL COMPARISON
    # ==============================
    st.subheader("📈 Model Comparison")

    results = []

    for name, m in models.items():
        pred = m.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        results.append([name, r2, mae])

    results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MAE"])
    st.dataframe(results_df)

    # ==============================
    # GRAPH: R2
    # ==============================
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="R2 Score", data=results_df, ax=ax)
    st.pyplot(fig)

    # ==============================
    # GRAPH: ACTUAL VS PREDICTED
    # ==============================
    st.subheader("📉 Actual vs Predicted")

    pred_test = model.predict(X_test)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, pred_test)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)