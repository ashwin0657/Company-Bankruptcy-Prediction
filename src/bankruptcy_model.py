"""
Corporate Bankruptcy Prediction and Financial Health Intelligence Pipeline.

This pipeline performs:
1. Automated data hygiene, duplicate removal, and missing value checks.
2. Financial domain feature engineering (liquidity gap, solvency buffer, leverage risk index).
3. Supervised classification modeling (HistGradientBoosting, Random Forest, Logistic Regression)
   with focus on Precision-Recall AUC (PR-AUC) under extreme class imbalance.
4. Continuous health proxy regression (Random Forest Regressor, Ridge Regression) for Net Income / Assets.
5. Unsupervised risk grouping with K-Means clustering and silhouette score optimization.
6. Generation of production-ready diagnostic charts, summary metrics, and Excel analytics.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def safe_name(x: str) -> str:
    """Sanitize strings for column and file naming."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(x)).strip("_")[:70]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific corporate distress features.
    Computes financial margin buffers, liquidity ratios, and a composite leverage risk index.
    """
    X = X.copy()

    def col(name: str):
        return name if name in X.columns else None

    # Key financial ratio differences reflecting financial distress indicators
    pairs = [
        ("Total Asset Growth Rate", "Cash Flow to Total Assets", "CashFlow_Asset_Gap"),
        ("Debt ratio %", "Net worth/Assets", "Debt_to_Equity_Buffer"),
        ("Current Liability to Current Assets", "Quick Ratio", "Liquidity_Gap"),
        ("Operating Expense Rate", "Operating Profit Rate", "Operating_Efficiency_Gap"),
        ("Gross Profit to Sales", "Operating Gross Margin", "Gross_Operating_Margin_Gap"),
        ("Net Income to Total Assets", "ROA(A) before interest and % after tax", "ROA_Gap"),
        ("Borrowing dependency", "Current Liability to Assets", "Leverage_Liquidity_Gap"),
    ]
    for a, b, new_name in pairs:
        if col(a) and col(b):
            X[new_name] = X[a] - X[b]

    # Composite risk indicator: normalized leverage & debt burden
    risk_cols = [
        "Debt ratio %",
        "Borrowing dependency",
        "Current Liability to Assets",
        "Current Liability to Current Assets",
        "Liability to Equity",
        "Interest Coverage Ratio (Interest expense to EBIT)",
    ]
    present = [c for c in risk_cols if c in X.columns]
    if present:
        z = pd.DataFrame(index=X.index)
        for c in present:
            s = X[c].astype(float)
            median_val = s.median()
            mad = s.abs().sub(median_val).mean()
            scale = mad if mad != 0 else (s.std() or 1.0)
            z[c] = (s - median_val) / scale
        X["Leverage_Risk_Index"] = z.mean(axis=1)

    return X


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):
    """Fit and evaluate classification models on standard metrics."""
    model = clone(model)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
        proba = 1 / (1 + np.exp(-score))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "PR_AUC": average_precision_score(y_test, proba),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "_model": model,
        "_proba": proba,
    }


def resolve_input_path(input_path: str | None = None) -> Path:
    """Find the dataset path across common locations."""
    if input_path and Path(input_path).exists():
        return Path(input_path)

    search_candidates = [
        Path("data/P2.csv"),
        Path("P2.csv"),
        Path("../data/P2.csv"),
        Path("data/data.csv"),
    ]
    for candidate in search_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find input dataset. Please specify --input <path_to_P2.csv>."
    )


def run_pipeline(input_csv: str | Path | None = None, output_dir: str | Path = "outputs"):
    """Execute end-to-end bankruptcy modeling and analytics pipeline."""
    input_file = resolve_input_path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = output_dir / "charts"
    charts.mkdir(exist_ok=True)

    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    target = "Bankrupt?"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in {input_file}.")

    # 1. Data Quality & Preprocessing
    quality = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing cells",
            "Duplicate rows",
            "Target positive (bankrupt) count",
            "Target positive rate",
            "Constant columns",
        ],
        "Value": [
            len(df),
            df.shape[1],
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            int(df[target].sum()),
            float(df[target].mean()),
            int(sum(df[c].nunique(dropna=False) <= 1 for c in df.columns)),
        ],
    })

    # Clean duplicates & constant columns
    clean = df.drop_duplicates().copy()
    constant_cols = [c for c in clean.columns if c != target and clean[c].nunique(dropna=False) <= 1]
    clean = clean.drop(columns=constant_cols)

    for c in clean.columns:
        if c != target:
            clean[c] = pd.to_numeric(clean[c], errors="coerce")

    X = clean.drop(columns=[target])
    y = clean[target].astype(int)

    X = add_engineered_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = list(X.columns)
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 2. Classification Strategy
    classifiers = {
        "Logistic Regression": Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep),
            ("model", RandomForestClassifier(
                n_estimators=120,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )),
        ]),
    }

    results = []
    fitted = {}
    for name, model in classifiers.items():
        r = evaluate_classifier(name, model, X_train, X_test, y_train, y_test)
        fitted[name] = r["_model"]
        results.append({k: v for k, v in r.items() if not k.startswith("_")})

    cls_results = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    best_name = cls_results.iloc[0]["Model"]

    best_proba = next(
        r["_proba"]
        for r in [
            evaluate_classifier(n, classifiers[n], X_train, X_test, y_train, y_test)
            for n in [best_name]
        ]
    )
    risk_output = X_test.copy()
    risk_output.insert(0, "Actual_Bankrupt", y_test.values)
    risk_output["Bankruptcy_Risk_Probability"] = best_proba
    risk_output["Risk_Band"] = pd.cut(
        best_proba,
        bins=[-np.inf, 0.20, 0.50, 0.80, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)

    # 3. Continuous Financial Health Regression
    regression_target = " Net Income to Total Assets"
    regression_results = pd.DataFrame()
    reg_pred = pd.DataFrame()

    if regression_target in clean.columns:
        reg_df = clean.dropna(subset=[regression_target]).copy()
        y_reg = reg_df[regression_target]
        X_reg = reg_df.drop(columns=[target, regression_target])
        X_reg = add_engineered_features(X_reg)
        xr_train, xr_test, yr_train, yr_test = train_test_split(
            X_reg, y_reg, test_size=0.20, random_state=42
        )

        reg_models = {
            "Ridge Regression": Pipeline([
                ("prep", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])),
                ("model", Ridge(alpha=1.0)),
            ]),
            "Random Forest Regressor": Pipeline([
                ("prep", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=42
                )),
            ]),
        }

        rr = []
        for name, model in reg_models.items():
            model.fit(xr_train, yr_train)
            p = model.predict(xr_test)
            rr.append({
                "Model": name,
                "MAE": mean_absolute_error(yr_test, p),
                "RMSE": mean_squared_error(yr_test, p) ** 0.5,
                "R2": r2_score(yr_test, p),
            })
        regression_results = pd.DataFrame(rr).sort_values("R2", ascending=False)

        best_reg_name = regression_results.iloc[0]["Model"]
        best_reg = reg_models[best_reg_name]
        best_reg.fit(xr_train, yr_train)
        p = best_reg.predict(xr_test)
        reg_pred = pd.DataFrame({
            "Actual_Net_Income_to_Total_Assets": yr_test.values,
            "Predicted_Net_Income_to_Total_Assets": p,
            "Absolute_Error": np.abs(yr_test.values - p),
        })

    # 4. Clustering Strategy
    cluster_X = X.copy()
    for c in cluster_X.columns:
        lo, hi = cluster_X[c].quantile([0.01, 0.99])
        cluster_X[c] = cluster_X[c].clip(lo, hi)

    cluster_arr = SimpleImputer(strategy="median").fit_transform(cluster_X)
    cluster_arr = StandardScaler().fit_transform(cluster_arr)

    sil = []
    models = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(cluster_arr)
        score = silhouette_score(
            cluster_arr, labels, sample_size=min(1500, len(cluster_arr)), random_state=42
        )
        sil.append({"K": k, "Silhouette": score})
        models[k] = (km, labels)

    sil_df = pd.DataFrame(sil)
    best_k = int(sil_df.loc[sil_df["Silhouette"].idxmax(), "K"])
    km, labels = models[best_k]

    cluster_summary = clean.copy()
    cluster_summary["Cluster"] = labels
    cluster_counts = cluster_summary["Cluster"].value_counts().sort_index().rename("Count").to_frame()
    cluster_counts["Share"] = cluster_counts["Count"] / len(cluster_summary)
    cluster_bankruptcy = cluster_summary.groupby("Cluster")[target].mean().rename("Bankruptcy_Rate")
    cluster_counts = cluster_counts.join(cluster_bankruptcy)
    cluster_means = cluster_summary.groupby("Cluster")[[c for c in numeric_features if c in cluster_summary.columns]].mean()

    # 5. Feature Importance
    rf_pipe = fitted["Random Forest"]
    rf_model = rf_pipe.named_steps["model"]
    feature_importance = pd.DataFrame({
        "Feature": numeric_features,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # 6. Diagnostic Visualizations
    plt.figure(figsize=(8, 5))
    counts = y.value_counts().sort_index()
    plt.bar(["Non-bankrupt (0)", "Bankrupt (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#2563eb", "#dc2626"])
    plt.title("Target Class Distribution", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Companies")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(charts / "01_target_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(cls_results["Model"], cls_results["PR_AUC"], color="#3b82f6")
    plt.title("Classification PR-AUC Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("Precision-Recall AUC")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(charts / "02_classification_pr_auc.png", dpi=180)
    plt.close()

    top = feature_importance.head(15).sort_values("Importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["Feature"], top["Importance"], color="#10b981")
    plt.title("Top 15 Random Forest Predictive Features", fontsize=13, fontweight="bold")
    plt.xlabel("Gini Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(charts / "03_feature_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sil_df["K"], sil_df["Silhouette"], marker="o", color="#8b5cf6", linewidth=2)
    plt.title("K-Means Silhouette Scores by Cluster Count", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(sil_df["K"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(charts / "04_cluster_silhouette.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(cluster_counts.index.astype(str), cluster_counts["Bankruptcy_Rate"], color="#f59e0b")
    plt.title("Bankruptcy Rate Across Discovered Clusters", fontsize=13, fontweight="bold")
    plt.xlabel("Cluster Group")
    plt.ylabel("Bankruptcy Rate")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(charts / "05_cluster_bankruptcy_rate.png", dpi=180)
    plt.close()

    if not reg_pred.empty:
        plt.figure(figsize=(7, 6))
        plt.scatter(
            reg_pred["Actual_Net_Income_to_Total_Assets"],
            reg_pred["Predicted_Net_Income_to_Total_Assets"],
            alpha=0.45,
            color="#0284c7",
        )
        mn = min(reg_pred.iloc[:, 0].min(), reg_pred.iloc[:, 1].min())
        mx = max(reg_pred.iloc[:, 0].max(), reg_pred.iloc[:, 1].max())
        plt.plot([mn, mx], [mn, mx], color="#ef4444", linestyle="--", label="Ideal Identity (y=x)")
        plt.title("Regression: Actual vs Predicted Profitability", fontsize=13, fontweight="bold")
        plt.xlabel("Actual Net Income to Total Assets")
        plt.ylabel("Predicted Net Income to Total Assets")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(charts / "06_regression_actual_vs_predicted.png", dpi=180)
        plt.close()

    # 7. Multi-Sheet Excel Report Export
    excel_path = output_dir / "P2_AI_First_Analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        quality.to_excel(writer, sheet_name="Data_Quality", index=False)
        cls_results.to_excel(writer, sheet_name="Classification", index=False)
        regression_results.to_excel(writer, sheet_name="Regression", index=False)
        sil_df.to_excel(writer, sheet_name="Clustering_Silhouette", index=False)
        cluster_counts.to_excel(writer, sheet_name="Cluster_Summary")
        cluster_means.to_excel(writer, sheet_name="Cluster_Means")
        feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
        risk_output.reset_index(drop=True).to_excel(writer, sheet_name="Risk_Scoring", index=False)
        if not reg_pred.empty:
            reg_pred.to_excel(writer, sheet_name="Regression_Predictions", index=False)

    # 8. Action Points Output
    action_lines = [
        f"1. Deploy the {best_name} classification pipeline as the first-line financial-distress screening model.",
        "2. Use probability bands rather than a binary flag so finance/risk teams can prioritize reviews.",
        "3. Monitor recall and PR-AUC because bankrupt firms are a small minority of the dataset.",
        "4. Review the top model features with finance SMEs before converting them into automated interventions.",
        f"5. Use {best_k} customer/company clusters for differentiated monitoring and AI-service treatment.",
    ]
    if not regression_results.empty:
        action_lines.append(
            f"6. Use {regression_results.iloc[0]['Model']} as an auxiliary model for continuous profitability/health monitoring."
        )
    action_lines.append(
        "7. Traditional time-series forecasting is not appropriate here because the supplied dataset has no date/time field; add longitudinal financial records before deploying forecasting."
    )
    (output_dir / "Action_Points.txt").write_text("\n".join(action_lines), encoding="utf-8")

    # 9. Summary Metadata
    summary = {
        "rows": int(len(clean)),
        "columns_after_cleaning": int(clean.shape[1]),
        "bankruptcy_rate": float(y.mean()),
        "best_classification_model": str(best_name),
        "best_classification_PR_AUC": float(cls_results.iloc[0]["PR_AUC"]),
        "best_classification_ROC_AUC": float(cls_results.iloc[0]["ROC_AUC"]),
        "best_classification_F1": float(cls_results.iloc[0]["F1"]),
        "best_cluster_k": best_k,
        "best_cluster_silhouette": float(sil_df.loc[sil_df["K"] == best_k, "Silhouette"].iloc[0]),
        "best_regression_model": None if regression_results.empty else str(regression_results.iloc[0]["Model"]),
        "best_regression_R2": None if regression_results.empty else float(regression_results.iloc[0]["R2"]),
    }
    pd.Series(summary).to_json(output_dir / "summary.json", indent=2)

    print("\n=== PIPELINE RUN COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Corporate Bankruptcy Prediction & Financial Distress Intelligence"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input dataset (defaults to auto-detecting data/P2.csv or P2.csv)",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Path to directory for output artifacts (defaults to 'outputs')",
    )
    args = parser.parse_args()
    run_pipeline(input_csv=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
