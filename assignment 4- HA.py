# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. LOAD DATA
DATA_PATH = r"C:\Users\tshab\Downloads\data.csv"  # Update path
df = pd.read_csv(DATA_PATH)

# Drop the 'id' column (not predictive)
df = df.drop("id", axis=1)

# Check first rows
print(df.head())
print(df.info())

# 3. DATA PREPARATION
# Features (X) = all numeric columns except target
X = df.drop("diagnosis", axis=1)
# Target (y) = diagnosis (M = Malignant, B = Benign)
y = df["diagnosis"]

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. TRAIN DECISION TREE
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 6. EVALUATION
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. VISUALIZE TREE
plt.figure(figsize=(20,12))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Decision Tree Classifier for Breast Cancer Diagnosis")
plt.show()
