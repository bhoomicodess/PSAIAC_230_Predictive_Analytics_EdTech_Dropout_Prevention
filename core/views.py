import base64
import os
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .ml_engine import (
    load_dataset,
    detect_target_column,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_data_directory():

    data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "data"
    )

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    return data_dir


def get_dataset_path():

    return os.path.join(
        get_data_directory(),
        "uploaded_dataset.csv"
    )


def get_processed_dataset_path():

    return os.path.join(
        get_data_directory(),
        "processed_dataset.csv"
    )


def generate_roc_curve_image(y_true, probabilities, model_name):

    try:

        if probabilities is None:
            return None

        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")

        unique_classes = sorted(pd.Series(y_true).unique().tolist())

        if len(unique_classes) == 2:

            positive_class = 1 if 1 in unique_classes else unique_classes[-1]
            positive_index = 1 if probabilities.shape[1] > 1 else 0

            if probabilities.shape[1] > 1:
                score_values = probabilities[:, positive_index]
            else:
                score_values = probabilities[:, 0]

            fpr, tpr, _ = roc_curve(y_true, score_values, pos_label=positive_class)
            roc_auc = roc_auc_score(y_true, score_values)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        else:

            for class_label in unique_classes:
                if class_label >= probabilities.shape[1]:
                    continue

                y_true_binary = (y_true == class_label).astype(int)
                class_probabilities = probabilities[:, class_label]
                fpr, tpr, _ = roc_curve(
                    y_true_binary,
                    class_probabilities,
                )
                class_auc = roc_auc_score(
                    y_true_binary,
                    class_probabilities,
                )
                plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {class_auc:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception:

        return None


# =========================================================
# 1. UPLOAD + DATASET ANALYSIS
# =========================================================

def upload_dataset(request):

    context = {}

    if request.method == "POST":

        uploaded_file = request.FILES.get(
            "dataset"
        )

        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if not uploaded_file:

            context["error"] = (
                "Please select a CSV file."
            )

            return render(
                request,
                "core/upload.html",
                context
            )

        if not uploaded_file.name.lower().endswith(
            ".csv"
        ):

            context["error"] = (
                "Please upload a CSV file."
            )

            return render(
                request,
                "core/upload.html",
                context
            )

        try:

            # -------------------------------------------------
            # Save uploaded dataset
            # -------------------------------------------------

            dataset_path = get_dataset_path()

            with open(
                dataset_path,
                "wb+"
            ) as destination:

                for chunk in uploaded_file.chunks():

                    destination.write(
                        chunk
                    )

            # -------------------------------------------------
            # Load dataset
            # -------------------------------------------------

            df = load_dataset(
                dataset_path
            )

            # -------------------------------------------------
            # Detect target column
            # -------------------------------------------------

            target = detect_target_column(
                df
            )

            # -------------------------------------------------
            # Dataset information
            # -------------------------------------------------

            context["rows"] = int(
                df.shape[0]
            )

            context["columns"] = int(
                df.shape[1]
            )

            context["column_names"] = list(
                df.columns
            )

            # -------------------------------------------------
            # Data types
            # -------------------------------------------------

            context["data_types"] = {

                column: str(dtype)

                for column, dtype
                in df.dtypes.items()

            }

            # -------------------------------------------------
            # Missing values
            # -------------------------------------------------

            context["missing_values"] = {

                column: int(value)

                for column, value
                in df.isnull().sum().items()

            }

            # -------------------------------------------------
            # Target
            # -------------------------------------------------

            context["target"] = target

            context["success"] = True

        except Exception as e:

            context["error"] = (
                f"Could not analyse dataset: {str(e)}"
            )

    return render(
        request,
        "core/upload.html",
        context
    )


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

def data_preprocessing(request):

    context = {}

    dataset_path = get_dataset_path()

    # -----------------------------------------------------
    # Check uploaded dataset
    # -----------------------------------------------------

    if not os.path.exists(
        dataset_path
    ):

        context["error"] = (
            "Please upload a dataset first."
        )

        return render(
            request,
            "core/preprocessing.html",
            context
        )

    try:

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )

        # -------------------------------------------------
        # Detect target
        # -------------------------------------------------

        target = detect_target_column(
            df
        )

        # -------------------------------------------------
        # Original information
        # -------------------------------------------------

        original_rows = len(df)

        original_columns = len(
            df.columns
        )

        # -------------------------------------------------
        # Missing values
        # -------------------------------------------------

        missing_before = int(
            df.isnull().sum().sum()
        )

        for column in df.columns:

            if df[column].isnull().sum() > 0:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    df[column] = (
                        df[column].fillna(
                            df[column].median()
                        )
                    )

                else:

                    mode = (
                        df[column].mode()
                    )

                    if not mode.empty:

                        df[column] = (
                            df[column].fillna(
                                mode.iloc[0]
                            )
                        )

                    else:

                        df[column] = (
                            df[column].fillna(
                                "Unknown"
                            )
                        )

        missing_after = int(
            df.isnull().sum().sum()
        )

        # -------------------------------------------------
        # Remove duplicates
        # -------------------------------------------------

        duplicates_before = int(
            df.duplicated().sum()
        )

        df = df.drop_duplicates()

        duplicates_removed = (
            duplicates_before
        )

        # -------------------------------------------------
        # Encode categorical features
        # -------------------------------------------------

        categorical_columns = []

        for column in df.columns:

            if column == target:

                continue

            if df[column].dtype == "object":

                categorical_columns.append(
                    column
                )

                df[column] = pd.factorize(
                    df[column].astype(str)
                )[0]

        # -------------------------------------------------
        # Encode target column
        # -------------------------------------------------

        target_mapping = {}

        if target in df.columns:

            if df[target].dtype == "object":

                target_values = (
                    df[target].astype(str)
                )

                encoded_target, unique_values = (
                    pd.factorize(
                        target_values
                    )
                )

                df[target] = (
                    encoded_target
                )

                target_mapping = {

                    int(index): str(value)

                    for index, value
                    in enumerate(
                        unique_values
                    )

                }

        # -------------------------------------------------
        # Save processed dataset
        # -------------------------------------------------

        processed_path = (
            get_processed_dataset_path()
        )

        df.to_csv(
            processed_path,
            index=False
        )

        # -------------------------------------------------
        # Frontend information
        # -------------------------------------------------

        context["success"] = True

        context["original_rows"] = (
            original_rows
        )

        context["original_columns"] = (
            original_columns
        )

        context["processed_rows"] = int(
            df.shape[0]
        )

        context["processed_columns"] = int(
            df.shape[1]
        )

        context["target"] = target

        context["missing_before"] = (
            missing_before
        )

        context["missing_after"] = (
            missing_after
        )

        context["duplicates_removed"] = (
            duplicates_removed
        )

        context["categorical_columns"] = (
            categorical_columns
        )

        context["target_mapping"] = (
            target_mapping
        )

        context["processed_columns_list"] = (
            list(df.columns)
        )

    except Exception as e:

        context["error"] = (
            f"Could not preprocess dataset: {str(e)}"
        )

    return render(
        request,
        "core/preprocessing.html",
        context
    )


# =========================================================
# 3. MODEL TRAINING
# =========================================================

def model_training(request):

    context = {}

    processed_path = (
        get_processed_dataset_path()
    )

    # -----------------------------------------------------
    # Check processed dataset
    # -----------------------------------------------------

    if not os.path.exists(
        processed_path
    ):

        context["error"] = (
            "Please upload and preprocess "
            "the dataset first."
        )

        return render(
            request,
            "core/model_training.html",
            context
        )

    try:

        # -------------------------------------------------
        # Load processed dataset
        # -------------------------------------------------

        df = pd.read_csv(
            processed_path
        )

        # -------------------------------------------------
        # Detect target
        # -------------------------------------------------

        target = detect_target_column(
            df
        )

        if target not in df.columns:

            context["error"] = (
                "Target column could not be found."
            )

            return render(
                request,
                "core/model_training.html",
                context
            )

        # -------------------------------------------------
        # Separate features and target
        # -------------------------------------------------

        X = df.drop(
            columns=[target]
        ).copy()

        y = df[target].copy()

        # =================================================
        # IMPORTANT FIX
        # =================================================
        # Convert every text/categorical feature
        # into numbers before sending it to ML models.
        #
        # This prevents errors such as:
        #
        # could not convert string to float: 'BBB'
        #
        # =================================================

        for column in X.columns:

            if not pd.api.types.is_numeric_dtype(
                X[column]
            ):

                X[column] = pd.factorize(
                    X[column].astype(str)
                )[0]

        # -------------------------------------------------
        # Convert target to numeric
        # -------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            y
        ):

            target_encoder = LabelEncoder()

            y = target_encoder.fit_transform(
                y.astype(str)
            )

        # -------------------------------------------------
        # Replace remaining missing values
        # -------------------------------------------------

        X = X.fillna(
            0
        )

        y = pd.Series(
            y
        ).fillna(
            0
        )

        # -------------------------------------------------
        # Make sure target is integer
        # -------------------------------------------------

        y = y.astype(
            int
        )

        # -------------------------------------------------
        # Train / Test split
        # -------------------------------------------------

        unique_classes = (
            len(
                pd.Series(y).unique()
            )
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=(
                    y
                    if unique_classes > 1
                    else None
                )
            )
        )

        # =================================================
        # MACHINE LEARNING MODELS
        # =================================================

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

        }

        results = []

        # =================================================
        # TRAIN MODELS
        # =================================================

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

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

<<<<<<< HEAD
            roc_auc = None
            roc_curve_image = None

            try:

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(X_test)

                    if len(pd.Series(y_test).unique()) == 2:
                        if probabilities.shape[1] > 1:
                            positive_class = 1 if 1 in pd.Series(y_test).unique() else sorted(pd.Series(y_test).unique())[-1]
                            roc_auc = roc_auc_score(
                                y_test,
                                probabilities[:, 1] if positive_class == 1 else probabilities[:, 0],
                            )
                        else:
                            roc_auc = roc_auc_score(
                                y_test,
                                probabilities[:, 0],
                            )
                    else:
                        roc_auc = roc_auc_score(
                            y_test,
                            probabilities,
                            multi_class="ovr",
                            average="macro",
                        )

                    roc_curve_image = generate_roc_curve_image(
                        y_test,
                        probabilities,
                        model_name,
                    )

            except Exception:

                roc_auc = None
                roc_curve_image = None

=======
>>>>>>> origin/main
            results.append({

                "name": model_name,

                "accuracy": round(
                    accuracy * 100,
                    2
                ),

                "precision": round(
                    precision * 100,
                    2
                ),

                "recall": round(
                    recall * 100,
                    2
                ),

                "f1": round(
                    f1 * 100,
                    2
                ),

<<<<<<< HEAD
                "roc_auc": round(
                    roc_auc * 100,
                    2
                ) if roc_auc is not None else None,

                "roc_curve_image": roc_curve_image,

=======
>>>>>>> origin/main
            })

        # =================================================
        # SELECT BEST MODEL
        # =================================================

        best_model = max(
            results,
            key=lambda x: x["accuracy"]
        )

        # =================================================
        # SEND RESULTS TO FRONTEND
        # =================================================

        context["success"] = True

        context["target"] = target

        context["total_records"] = (
            len(df)
        )

        context["training_records"] = (
            len(X_train)
        )

        context["testing_records"] = (
            len(X_test)
        )

        context["feature_count"] = (
            X.shape[1]
        )

        context["results"] = (
            results
        )

        context["best_model"] = (
            best_model["name"]
        )

        context["best_accuracy"] = (
            best_model["accuracy"]
        )

        context["best_precision"] = (
            best_model["precision"]
        )

        context["best_recall"] = (
            best_model["recall"]
        )

        context["best_f1"] = (
            best_model["f1"]
        )

    except Exception as e:

        context["error"] = (
            f"Could not train models: {str(e)}"
        )

    return render(
        request,
        "core/model_training.html",
        context
    )