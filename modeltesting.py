# Install required packages before running:
# pip install alt-profanity-check scikit-learn matplotlib pandas seaborn
import streamlit as st
from profanity_check import predict, predict_prob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Step 1: Load or Create Test Dataset ---
# You can replace this small dataset with your own (e.g., offensive_dataset.csv)
data = pd.DataFrame({
    "text": [
        "You are amazing",
        "You idiot",
        "I love this place",
        "You are dumb",
        "Have a nice day",
        "Go to hell",
        "She is kind",
        "You’re useless",
        "This is wonderful",
        "You’re disgusting"
    ],
    "label": [0,1,0,1,0,1,0,1,0,1]
})

X = data["text"]
y = data["label"]

# --- Step 2: Convert text into feature vectors using alt_profanity_check predictions ---
# We'll use profanity probabilities as features
X_features = [[p] for p in predict_prob(X)]

# --- Step 3: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=42)

# --- Step 4: Train Multiple Scikit-learn Models ---
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([name, acc, prec, rec, f1])

# --- Step 5: Display Metrics as Bar Chart ---
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print(df_results)

plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=df_results)
plt.title("Model Comparison based on Accuracy")
plt.ylim(0,1)
st.pyplot()

# --- Step 6: Confusion Matrix for the Best Model ---
best_model_name = df_results.iloc[df_results["F1 Score"].idxmax()]["Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot()
