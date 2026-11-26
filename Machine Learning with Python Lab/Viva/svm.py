
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate dataset
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)

# 2. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. Linear SVM
linear = SVC(kernel='linear', C=1.0, random_state=42)
linear.fit(X_train, y_train)

# 5. RBF SVM
rbf = SVC(kernel='rbf', random_state=42)
rbf.fit(X_train, y_train)

# 6. Polynomial SVM
poly = SVC(kernel='poly', degree=3, random_state=42)
poly.fit(X_train, y_train)

# 7. Grid Search for best parameters
params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(random_state=42),
    params, cv=5, scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 8. Evaluate models
models = [linear, rbf, poly, grid]

for model in models:
    pred = model.predict(X_val)

    print("\nModel:", model)
    print("Accuracy:", accuracy_score(y_val, pred))
    print(classification_report(y_val, pred))

# 9. Confusion matrix
from sklearn.metrics import confusion_matrix

pred = grid.predict(X_val)
print("Confusion Matrix:\n", confusion_matrix(y_val, pred))

# 10. Decision boundary visualization
x_min, x_max = X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5
y_min, y_max = X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

for model, name in zip(
    models[:3], ["Linear SVM", "RBF SVM", "Polynomial SVM"]
):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.title(name)
    plt.show()
