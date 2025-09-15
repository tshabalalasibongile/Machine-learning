#1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------
# 2. Data Loading and Preprocessing
# ------------------------------------------------------
def load_and_preprocess(data_path):
    """Load dataset, drop irrelevant columns, and encode target."""
    df = pd.read_csv(data_path)

    # Drop unnecessary columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    # Encode target: Malignant (M) = 1, Benign (B) = 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Features (X) and Target (y)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    return X, y

# ------------------------------------------------------
# 3. Train-Test Split and Model Training
# ------------------------------------------------------
def train_decision_tree(X, y):
    """Split data, train decision tree, and return model + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# ------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Generate predictions, accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])

    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return y_pred, cm, acc

# ------------------------------------------------------
# 5. Visualizations
# ------------------------------------------------------
def visualize_results(model, X, cm):
    """Plot decision tree, confusion matrix, and feature importances."""
    # Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns,
              class_names=["Benign", "Malignant"], rounded=True, fontsize=8)
    plt.title("Figure 1: Decision Tree Visualization", fontsize=14)
    plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Figure 2: Confusion Matrix Heatmap")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), X.columns[indices], rotation=45, ha="right")
    plt.title("Figure 3: Top 10 Feature Importances")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------
# 6. Run Pipeline
# ------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\tshab\Downloads\data.csv"

    # Step 1: Load + preprocess
    X, y = load_and_preprocess(DATA_PATH)

    # Step 2: Train model
    model, X_train, X_test, y_train, y_test = train_decision_tree(X, y)

    # Step 3: Evaluate model
    y_pred, cm, acc = evaluate_model(model, X_test, y_test)

    # Step 4: Visualize results
    visualize_results(model, X, cm)
