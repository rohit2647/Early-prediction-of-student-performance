import streamlit as st
import pandas as pd
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
@st.cache_data
def load_data():
    return pd.read_csv("StudentPerformanceFactors.csv")

df = load_data()


# ==============================
# PREPROCESSING
# ==============================
@st.cache_data
def preprocess(df):
    df = df.copy()

    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, X_train, X_test, y_train, y_test, scaler, le_dict


X, X_train, X_test, y_train, y_test, scaler, le_dict = preprocess(df)


# ==============================
# TRAIN MODELS (FAST)
# ==============================
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(kernel='linear')  # faster
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models


models = train_models(X_train, y_train)


# ==============================
# UI
# ==============================
st.title("🎓 Early Prediction of Student Performance")

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]


# ==============================
# INPUT FORM (BEST PRACTICE)
# ==============================
st.sidebar.header("Student Input")

with st.sidebar.form("input_form"):

    input_data = {}

    for col in X.columns:

        # categorical
        if col in le_dict:
            options = list(le_dict[col].classes_)
            selected = st.selectbox(col, options)
            input_data[col] = le_dict[col].transform([selected])[0]

        # numerical
        else:
            if col in ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]:
                min_val = 0
            else:
                min_val = df[col].min()

            max_val = df[col].max()

            if pd.api.types.is_integer_dtype(df[col]):
                input_data[col] = st.slider(
                    col, int(min_val), int(max_val), int(min_val), step=1
                )
            else:
                input_data[col] = st.slider(
                    col, float(min_val), float(max_val), float(min_val),
                    step=0.5, format="%.1f"
                )

    submit = st.form_submit_button("Predict")


# ==============================
# PREDICTION
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
    results = []

    for name, m in models.items():
        pred = m.predict(X_test)
        results.append([
            name,
            r2_score(y_test, pred),
            mean_absolute_error(y_test, pred)
        ])

    results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MAE"])

    st.subheader("📈 Model Comparison")
    st.dataframe(results_df)


    # ==============================
    # GRAPH: R2 SCORE
    # ==============================
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="R2 Score", data=results_df, ax=ax)
    st.pyplot(fig)


    # ==============================
    # ACTUAL VS PREDICTED
    # ==============================
    st.subheader("📉 Actual vs Predicted")

    pred_test = model.predict(X_test)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, pred_test)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")

    st.pyplot(fig2)