import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer ML Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🧬 Breast Cancer Classification Dashboard")

st.markdown(
    """
    ### Machine Learning Based Cancer Classification

    This application performs **data analysis, visualization,
    preprocessing, model training and evaluation** using machine
    learning algorithms.
    """
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Application Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

if uploaded_file is None:

    st.info("👈 Please upload your CSV dataset from the sidebar.")

    st.markdown(
        """
        ### What this dashboard can do

        - Explore your dataset
        - Detect missing values
        - Visualize important features
        - Train multiple machine learning models
        - Compare model performance
        - Display confusion matrix
        - Display ROC curve
        - Download predictions
        """
    )

    st.stop()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading the dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", data.shape[0])

with col2:
    st.metric("Columns", data.shape[1])

with col3:
    st.metric("Missing Values", data.isnull().sum().sum())

with col4:
    st.metric("Duplicate Rows", data.duplicated().sum())

st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Data Analysis",
        "📈 Visualization",
        "🤖 Machine Learning",
        "📥 Predictions"
    ]
)

# =========================================================
# TAB 1 - DATA ANALYSIS
# =========================================================

with tab1:

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Data Types")

        dtype_df = pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

    with col2:

        st.write("### Missing Values")

        missing = data.isnull().sum()

        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })

        missing_df = missing_df[
            missing_df["Missing Values"] > 0
        ]

        if missing_df.empty:
            st.success("✅ No missing values found.")

        else:
            st.dataframe(
                missing_df,
                use_container_width=True
            )

    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe(include="all").T,
        use_container_width=True
    )

    st.subheader("Duplicate Records")

    if data.duplicated().sum() == 0:
        st.success("✅ No duplicate records found.")
    else:
        st.warning(
            f"⚠️ {data.duplicated().sum()} duplicate rows found."
        )

# =========================================================
# TAB 2 - VISUALIZATION
# =========================================================

with tab2:

    st.subheader("📈 Data Visualization")

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) == 0:

        st.warning(
            "No numerical columns available for visualization."
        )

    else:

        selected_feature = st.selectbox(
            "Select a numerical feature",
            numeric_columns
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write("### Distribution")

            fig, ax = plt.subplots()

            ax.hist(
                data[selected_feature].dropna(),
                bins=30
            )

            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

        with col2:

            st.write("### Box Plot")

            fig, ax = plt.subplots()

            ax.boxplot(
                data[selected_feature].dropna()
            )

            ax.set_ylabel(selected_feature)

            st.pyplot(fig)

        st.subheader("Correlation Analysis")

        correlation = data[numeric_columns].corr()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        image = ax.imshow(
            correlation,
            aspect="auto"
        )

        ax.set_xticks(
            range(len(correlation.columns))
        )

        ax.set_yticks(
            range(len(correlation.columns))
        )

        ax.set_xticklabels(
            correlation.columns,
            rotation=90
        )

        ax.set_yticklabels(
            correlation.columns
        )

        fig.colorbar(image)

        st.pyplot(fig)

# =========================================================
# TAB 3 - MACHINE LEARNING
# =========================================================

with tab3:

    st.header("🤖 Machine Learning")

    st.write(
        "Select the target column and train multiple "
        "classification models."
    )

    # Target column

    target_column = st.selectbox(
        "🎯 Select Target Column",
        data.columns
    )

    # Remove missing target rows

    ml_data = data.dropna(
        subset=[target_column]
    ).copy()

    X = ml_data.drop(
        columns=[target_column]
    )

    y = ml_data[target_column]

    # -----------------------------------------------------
    # ENCODE TARGET
    # -----------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )

    # -----------------------------------------------------
    # KEEP NUMERICAL FEATURES
    # -----------------------------------------------------

    X = X.select_dtypes(
        include=np.number
    )

    if X.shape[1] == 0:

        st.error(
            "No numerical features found. "
            "Please use a dataset containing numerical features."
        )

        st.stop()

    # Fill missing values

    X = X.fillna(
        X.median()
    )

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    test_size = st.slider(
        "Test Data Percentage",
        min_value=10,
        max_value=40,
        value=20
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size / 100,
        random_state=42,
        stratify=y_encoded
    )

    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=2000),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "Naive Bayes":
            GaussianNB()
    }

    selected_models = st.multiselect(
        "Select Models",
        list(models.keys()),
        default=list(models.keys())
    )

    if st.button(
        "🚀 Train Models",
        type="primary"
    ):

        if len(selected_models) == 0:

            st.warning(
                "Please select at least one model."
            )

            st.stop()

        results = []

        trained_models = {}

        st.subheader("🏆 Model Performance")

        for model_name in selected_models:

            model = models[model_name]

            # KNN and Logistic Regression work well
            # with scaled data

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({

                "Model": model_name,

                "Accuracy": round(
                    accuracy * 100,
                    2
                ),

                "Precision": round(
                    precision * 100,
                    2
                ),

                "Recall": round(
                    recall * 100,
                    2
                ),

                "F1 Score": round(
                    f1 * 100,
                    2
                )
            })

            trained_models[
                model_name
            ] = model

        results_df = pd.DataFrame(
            results
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        best_model_name = results_df.loc[
            results_df["Accuracy"].idxmax(),
            "Model"
        ]

        st.success(
            f"🏆 Best Model: **{best_model_name}**"
        )

        best_model = trained_models[
            best_model_name
        ]

        best_predictions = best_model.predict(
            X_test_scaled
        )

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader(
            "📊 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            best_predictions
        )

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        ax.set_title(
            f"Confusion Matrix - {best_model_name}"
        )

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        # -------------------------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------------------------

        st.subheader(
            "📋 Classification Report"
        )

        report = classification_report(
            y_test,
            best_predictions,
            target_names=[
                str(x)
                for x in label_encoder.classes_
            ],
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # ROC CURVE
        # -------------------------------------------------

        if len(
            np.unique(y_test)
        ) == 2:

            probabilities = best_model.predict_proba(
                X_test_scaled
            )[:, 1]

            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            st.subheader(
                "📈 ROC Curve"
            )

            fig, ax = plt.subplots()

            ax.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.3f}"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                "ROC Curve"
            )

            ax.legend()

            st.pyplot(fig)

        # Save predictions in session

        prediction_output = X_test.copy()

        prediction_output[
            "Actual"
        ] = label_encoder.inverse_transform(
            y_test
        )

        prediction_output[
            "Predicted"
        ] = label_encoder.inverse_transform(
            best_predictions
        )

        st.session_state[
            "prediction_output"
        ] = prediction_output

# =========================================================
# TAB 4 - DOWNLOAD
# =========================================================

with tab4:

    st.header("📥 Download Predictions")

    if "prediction_output" in st.session_state:

        prediction_output = st.session_state[
            "prediction_output"
        ]

        st.dataframe(
            prediction_output.head(20),
            use_container_width=True
        )

        csv = prediction_output.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="breast_cancer_predictions.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Train a machine learning model first "
            "to generate predictions."
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Breast Cancer Classification | "
    "Machine Learning & Data Analytics Project"
)
