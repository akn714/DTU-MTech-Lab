import time
from data import load_data
import numpy as np
from pca_scratch import MyPCA
from preprocessing import get_preprocessor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def main():
    # Step 1: Load and Split Data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor(X)

    # -------------------------------------------------------------
    # Model 1: Baseline KNN (High-Dimensional Transformed Space)
    # -------------------------------------------------------------
    print("=" * 60)
    print("MODEL 1: Baseline KNN")
    print("=" * 60)
    baseline_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    t0 = time.time()
    baseline_pipeline.fit(X_train, y_train)
    t_train_base = time.time() - t0

    t0 = time.time()
    y_pred_base = baseline_pipeline.predict(X_test)
    t_pred_base = time.time() - t0

    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Training Time: {t_train_base:.4f} s")
    print(f"Prediction Time: {t_pred_base:.4f} s")
    print(f"Baseline KNN Model Accuracy: {acc_base:.4f}")
    print(
        "\nClassification Report (Baseline):\n",
        classification_report(y_test, y_pred_base),
    )

    # Feature space analysis
    X_train_proc = baseline_pipeline.named_steps["preprocessor"].transform(
        X_train
    )
    X_test_proc = baseline_pipeline.named_steps["preprocessor"].transform(
        X_test
    )
    print(f"Total features after One-Hot Encoding: {X_train_proc.shape[1]}")

    # -------------------------------------------------------------
    # Determine k with sklearn.PCA (95% variance)
    # -------------------------------------------------------------
    pca_sklearn = PCA(n_components=0.95, random_state=42)
    pca_sklearn.fit(X_train_proc)
    k = pca_sklearn.n_components_
    print(f"\nsklearn PCA selected {k} components to explain 95% variance.")

    # -------------------------------------------------------------
    # Model 2: Scratch PCA + KNN
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 2: Scratch PCA + KNN")
    print("=" * 60)
    my_pca = MyPCA(n_components=k)
    my_pca.fit(X_train_proc)

    X_train_pca_scratch = my_pca.transform(X_train_proc)
    X_test_pca_scratch = my_pca.transform(X_test_proc)

    knn_scratch = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_scratch.fit(X_train_pca_scratch, y_train)
    t_train_scratch = time.time() - t0

    t0 = time.time()
    y_pred_scratch = knn_scratch.predict(X_test_pca_scratch)
    t_pred_scratch = time.time() - t0

    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    print(f"Training Time: {t_train_scratch:.4f} s")
    print(f"Prediction Time: {t_pred_scratch:.4f} s")
    print(f"Scratch PCA KNN Model Accuracy: {acc_scratch:.4f}")
    print(
        "\nClassification Report (Scratch PCA):\n",
        classification_report(y_test, y_pred_scratch),
    )

    # -------------------------------------------------------------
    # Model 3: Sklearn PCA + KNN
    # -------------------------------------------------------------
    print("=" * 60)
    print("MODEL 3: sklearn PCA + KNN")
    print("=" * 60)
    X_train_pca_sk = pca_sklearn.transform(X_train_proc)
    X_test_pca_sk = pca_sklearn.transform(X_test_proc)

    knn_sk = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    knn_sk.fit(X_train_pca_sk, y_train)
    t_train_sk = time.time() - t0

    t0 = time.time()
    y_pred_sk = knn_sk.predict(X_test_pca_sk)
    t_pred_sk = time.time() - t0

    acc_sk = accuracy_score(y_test, y_pred_sk)
    print(f"Training Time: {t_train_sk:.4f} s")
    print(f"Prediction Time: {t_pred_sk:.4f} s")
    print(f"sklearn PCA KNN Model Accuracy: {acc_sk:.4f}")
    print(
        "\nClassification Report (sklearn PCA):\n",
        classification_report(y_test, y_pred_sk),
    )


if __name__ == "__main__":
    main()