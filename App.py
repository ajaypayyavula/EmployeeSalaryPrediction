
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)


# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fb, #eaf0fa);
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #172554, #2563eb);
    color: white;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
.result-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #dbe4f0;
    text-align: center;
    box-shadow: 0 5px 20px rgba(20, 40, 80, 0.08);
}
.salary {
    color: #15803d;
    font-size: 34px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ================= DATASET PATH =================
DATA_PATH = Path(__file__).parent / "data" / "sampledataset.csv"


# ================= LOAD AND CLEAN DATA =================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # First, try reading a normal CSV file
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Fix files where an entire row is stored as one column
    if data.shape[1] == 1:
        parsed_rows = []

        with open(
            DATA_PATH, "r", encoding="utf-8-sig", newline=""
        ) as file:
            for row in csv.reader(file):
                if len(row) == 1 and "," in row[0]:
                    inner_row = next(csv.reader([row[0]]))
                    parsed_rows.append(inner_row)
                else:
                    parsed_rows.append(row)

        if len(parsed_rows) < 2:
            raise ValueError("The dataset does not contain enough rows.")

        # Make all rows the same length
        column_count = max(len(row) for row in parsed_rows)

        parsed_rows = [
            row + [""] * (column_count - len(row))
            for row in parsed_rows
        ]

        data = pd.DataFrame(
            parsed_rows[1:],
            columns=parsed_rows[0]
        )

    # Clean and standardize column names
    cleaned_columns = []

    for column in data.columns:
        name = str(column).strip().strip('"').strip("'")
        name = name.replace(" ", "_").replace("-", "_")
        cleaned_columns.append(name)

    data.columns = cleaned_columns

    # Match common column name variations
    aliases = {
        "name": "Name",
        "age": "Age",
        "salary": "Salary",
        "department": "Department",
        "years_at_company": "Years_at_Company",
        "yearsatcompany": "Years_at_Company",
        "years_company": "Years_at_Company",
        "experience": "Years_at_Company"
    }

    rename_columns = {}

    for column in data.columns:
        key = column.lower().replace("__", "_")
        if key in aliases:
            rename_columns[column] = aliases[key]

    data = data.rename(columns=rename_columns)

    required_columns = [
        "Age",
        "Salary",
        "Department",
        "Years_at_Company"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Detected columns: {data.columns.tolist()}. "
            "Please check the CSV file format."
        )

    # Convert numerical columns
    for column in ["Age", "Salary", "Years_at_Company"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["Department"] = (
        data["Department"]
        .astype(str)
        .str.strip()
    )

    # Remove invalid records
    data = data.dropna(
        subset=[
            "Age",
            "Salary",
            "Department",
            "Years_at_Company"
        ]
    )

    data = data[
        data["Department"].ne("")
        & data["Department"].str.lower().ne("nan")
    ]

    if len(data) < 5:
        raise ValueError(
            "At least 5 valid employee records are required."
        )

    return data.reset_index(drop=True)


# ================= MAIN APPLICATION =================
try:
    df = load_data()

    # ================= MACHINE LEARNING =================
    features = [
        "Age",
        "Department",
        "Years_at_Company"
    ]

    X = df[features]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    def create_model():
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "department",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Department"]
                )
            ],
            remainder="passthrough"
        )

        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        return model_pipeline

    # Evaluate the model using held-out test records
    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)

    test_predictions = evaluation_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )
    r2 = r2_score(y_test, test_predictions)

    # Train the final model on all available records
    model = create_model()
    model.fit(X, y)

    # ================= HEADER =================
    st.markdown("""
    <div class="hero">
        <h1>💼 Employee Salary Prediction System</h1>
        <p>Machine Learning Based Salary Estimation</p>
        <p>Enter employee details to estimate the salary.</p>
    </div>
    """, unsafe_allow_html=True)

    # ================= SUMMARY CARDS =================
    col1, col2, col3 = st.columns(3)

    col1.metric("Employee Records", len(df))
    col2.metric(
        "Departments",
        df["Department"].nunique()
    )
    col3.metric("Algorithm", "Linear Regression")

    st.divider()

    # ================= PREDICTION SECTION =================
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("👤 Employee Details")
        st.write("Enter the employee information below.")

        age_min = int(df["Age"].min())
        age_max = int(df["Age"].max())

        years_min = max(
            0, int(df["Years_at_Company"].min())
        )
        years_max = max(
            years_min, int(df["Years_at_Company"].max())
        )

        # Prevent invalid slider ranges
        if age_min == age_max:
            age_min = max(0, age_min - 1)
            age_max = age_max + 1

        if years_min == years_max:
            years_max = years_min + 1

        age_default = int(df["Age"].median())
        years_default = int(
            df["Years_at_Company"].median()
        )

        age_default = max(
            age_min, min(age_default, age_max)
        )
        years_default = max(
            years_min, min(years_default, years_max)
        )

        age = st.slider(
            "Age",
            min_value=age_min,
            max_value=age_max,
            value=age_default
        )

        departments = sorted(
            df["Department"].unique().tolist()
        )

        department = st.selectbox(
            "Department",
            departments
        )

        years = st.slider(
            "Years at Company",
            min_value=years_min,
            max_value=years_max,
            value=years_default
        )

        predict_clicked = st.button(
            "✨ Predict Salary",
            type="primary",
            use_container_width=True
        )

        if predict_clicked:
            employee = pd.DataFrame({
                "Age": [age],
                "Department": [department],
                "Years_at_Company": [years]
            })

            prediction = float(
                model.predict(employee)[0]
            )

            st.session_state["salary_prediction"] = prediction

            st.session_state["prediction_inputs"] = {
                "age": age,
                "department": department,
                "years": years
            }

    with right:
        st.subheader("📊 Prediction Result")

        if "salary_prediction" in st.session_state:
            result = st.session_state["salary_prediction"]
            details = st.session_state["prediction_inputs"]

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:18px;color:#475569;">
                    Estimated Salary
                </div>
                <div class="salary">₹{result:,.2f}</div>
                <div style="color:#64748b;margin-top:12px;">
                    Salary estimated by the trained model
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("**Employee Details Used**")
            st.write(f"Age: {details['age']}")
            st.write(f"Department: {details['department']}")
            st.write(
                f"Years at Company: {details['years']}"
            )

            st.caption(
                "This is a model estimate, not a guaranteed actual salary."
            )
        else:
            st.info(
                "Select employee details and click "
                "'Predict Salary' to view the result."
            )

    st.divider()

    # ================= PROJECT TABS =================
    tab1, tab2, tab3 = st.tabs([
        "📋 Dataset & Insights",
        "📈 Visualizations",
        "🧠 Model Performance"
    ])

    # ================= DATASET TAB =================
    with tab1:
        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        info1, info2, info3 = st.columns(3)

        info1.metric("Rows", df.shape[0])
        info2.metric("Columns", df.shape[1])
        info3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.write("**Missing Values by Column**")

        st.dataframe(
            df.isnull().sum().rename("Missing Values"),
            use_container_width=True
        )

    # ================= VISUALIZATION TAB =================
    with tab2:
        st.subheader("Salary Distribution")

        fig1, ax1 = plt.subplots(figsize=(8, 4))

        sns.histplot(
            data=df,
            x="Salary",
            kde=len(df) > 1,
            ax=ax1
        )

        ax1.set_xlabel("Salary")
        ax1.set_ylabel("Number of Employees")
        ax1.set_title("Employee Salary Distribution")

        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("Years at Company vs Salary")

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        sns.scatterplot(
            data=df,
            x="Years_at_Company",
            y="Salary",
            hue="Department",
            ax=ax2
        )

        ax2.set_xlabel("Years at Company")
        ax2.set_ylabel("Salary")
        ax2.set_title("Experience and Salary Relationship")

        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Average Salary by Department")

        department_salary = (
            df.groupby("Department")["Salary"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(department_salary)

    # ================= MODEL PERFORMANCE TAB =================
    with tab3:
        st.subheader("Linear Regression Evaluation")

        st.caption(
            "Metrics are calculated using held-out test records."
        )

        m1, m2, m3 = st.columns(3)

        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")
        m3.metric("R²", f"{r2:.4f}")

        st.write(
            "**MAE:** Average absolute difference "
            "between actual and predicted salaries."
        )

        st.write(
            "**RMSE:** Error metric that gives more "
            "weight to larger prediction errors."
        )

        st.write(
            "**R²:** Measures how well the model explains "
            "salary variation on the test data."
        )

        # Actual vs predicted chart
        comparison = pd.DataFrame({
            "Actual Salary": y_test.to_numpy(),
            "Predicted Salary": test_predictions
        })

        st.subheader("Actual vs Predicted Salary")

        st.dataframe(
            comparison.round(2),
            use_container_width=True
        )

    # ================= FOOTER =================
    st.divider()

    st.caption(
        "Employee Salary Prediction System | "
        "Python • Streamlit • Pandas • Scikit-learn"
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Create a folder named 'data' next to app.py "
        "and place sampledataset.csv inside it."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")
    st.info(
        "If the problem continues, share the complete error "
        "message and the first few lines of your CSV file."
    )