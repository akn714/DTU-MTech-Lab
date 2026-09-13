from data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda():
    X, y = load_data()
    df = X.copy()
    df["income"] = y

    sns.set_theme(style="whitegrid")

    # 1. Missing Data Analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=missing_pct.index, y=missing_pct.values, palette="mako")
    plt.ylabel("Missing Percentage (%)")
    plt.title("Missing Data Percentage per Feature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("missing_values.png")
    plt.show()

    # 2. Numerical Distributions
    num_features = ["age", "hours-per-week", "capital-gain"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, col in enumerate(num_features):
        if col in df.columns:
            sns.histplot(
                data=df,
                x=col,
                hue="income",
                kde=True,
                ax=axes[i],
                bins=25,
                palette="Set2",
            )
            axes[i].set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig("numerical_distributions.png")
    plt.show()

    # 3. Categorical Distributions
    cat_features = ["workclass", "occupation"]
    for col in cat_features:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            order = df[col].value_counts().index
            sns.countplot(
                data=df, x=col, hue="income", order=order, palette="Set1"
            )
            plt.title(f"Count of {col} by Income Class")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"cat_{col}_distribution.png")
            plt.show()


if __name__ == "__main__":
    run_eda()