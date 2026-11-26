import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("emails.csv")          # columns: text, spam
df = df.dropna(subset=["text", "spam"])

# 2. Train/test split (stratified keeps the spam ratio equal)
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["spam"]
)

y_train = train["spam"].values.astype(float)
y_test = test["spam"].values.astype(float)

# 3. Text -> numeric features (TF-IDF replaces feature scaling)
vectorizer = TfidfVectorizer(
    stop_words="english", max_features=3000, ngram_range=(1, 2), min_df=2
)
X_train = vectorizer.fit_transform(train["text"]).toarray()   # fit on train only
X_test = vectorizer.transform(test["text"]).toarray()

# 4. Add bias column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# 5. Normal Equation (ridge least squares; plain inverse would be singular here)
lam = 1.0
I = np.eye(X_train.shape[1])
I[0, 0] = 0                              # don't penalize the bias
theta = np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ y_train)

pred1 = (X_test @ theta >= 0.5).astype(int)

# 6. Batch Gradient Descent (logistic regression)
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

theta_gd = np.zeros(X_train.shape[1])
lr = 1.0                                 # TF-IDF values are small, so a larger lr works
n = len(y_train)

for i in range(2000):
    error = sigmoid(X_train @ theta_gd) - y_train
    gradient = (X_train.T @ error) / n
    theta_gd -= lr * gradient

pred2 = (sigmoid(X_test @ theta_gd) >= 0.5).astype(int)

# 7. Scikit-learn Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train[:, 1:], y_train)
pred3 = model.predict(X_test[:, 1:])

# 8. Evaluation Metrics
def evaluate(y, pred):
    y = y.astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]: [[{tn} {fp}] [{fn} {tp}]]\n")

print("Normal Equation (ridge least squares):")
evaluate(y_test, pred1)

print("Gradient Descent (logistic):")
evaluate(y_test, pred2)

print("Scikit-learn (logistic):")
evaluate(y_test, pred3)