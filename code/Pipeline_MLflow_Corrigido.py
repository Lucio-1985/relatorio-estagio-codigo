# -*- coding: utf-8 -*-
"""
Pipeline de preparação de dados e modelação (BXB2 e BCXBC2).

Lê o ficheiro Fused_All.csv, escolhe o par de máquinas GOOD/BAD da
configuração e do ano definidos mais abaixo, prepara os dados e treina sete
modelos de classificação, registando os resultados no MLflow.

Tudo o que é calculado a partir dos dados (limites do IQR, colunas
removidas por correlação, escala MinMax) usa só o treino. O teste não é
filtrado nem balanceado e só recebe as transformações já ajustadas.

Autor: LS
"""

# ======================================================
# Bibliotecas
# ======================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ======================================================
# Configuração: estas três linhas mudam entre execuções
# ======================================================
YEAR_FILTER = 2025
TARGET_SPLICE = "BXB2"   # "BXB2" ou "BCXBC2"
TARGET_CROSS_SECTION = 1.05  # 1.05 (BXB2) ou 2.2 (BCXBC2)

# ======================================================
# Máquinas escolhidas para cada configuração e ano
# O script escolhe o par pela chave (configuração, ano).
# ======================================================
FIXED_ASSETS = {
    ("BXB2", 2024):   {"good": 14190, "bad": 14171},
    ("BXB2", 2025):   {"good": 14190, "bad": 14206},
    ("BCXBC2", 2024): {"good": 14167, "bad": 14206},
}

CSV_FILE = "Fused_All.csv"
OUT_DIR = os.path.abspath(f"{TARGET_SPLICE}_outputs_{YEAR_FILTER}")

MIN_ROWS = 1000
CORR_THRESHOLD = 0.90
TEST_SIZE = 0.30
RANDOM_STATE = 42
IQR_K = 1.5

META_COLS = ["AssetID", "Date", "Time", "Splice Comment", "CrossSection"]

DROP_EARLY_COLS = [
    "CH-I-T-Pos", "CH-I-T-Neg", "CH-O-T-Neg", "CH-O-T-Pos",
    "WH-I-T-Pos", "WH-I-T-Neg", "Wh-O-T-Neg", "Wh-O-T-Pos", "Sequence"
]

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# Configuração do MLflow (base de dados SQLite)
# ======================================================
MLFLOW_DB = os.path.abspath("mlflow.db")
MLFLOW_ART = os.path.abspath("mlruns_artifacts")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(f"{TARGET_SPLICE}_{YEAR_FILTER}_Pipeline")

os.makedirs(MLFLOW_ART, exist_ok=True)


# ======================================================
# Funções auxiliares
# ======================================================
def audit(label, df):
    print("\n" + "=" * 80)
    print(label)
    print("Linhas:", df.shape[0], "| Colunas:", df.shape[1])
    print(df["is_good"].value_counts())
    print("=" * 80)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=200)
    plt.show()
    plt.close()


def normalize_splice(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if "X" in s and "TEST" not in s:
        p = s.split("X")
        if len(p) == 2:
            s = "X".join(sorted(p))
    return s


def compute_iqr_bounds(df, cols, k=1.5):
    """Calcula os limites inferior e superior do IQR de cada coluna.
    Usar só com o conjunto de treino."""
    bounds = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        bounds[c] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_iqr_filter(df, bounds):
    """Remove as linhas fora dos limites. Usar só no treino;
    o teste não é filtrado."""
    mask = pd.Series(True, index=df.index)
    for c, (lo, hi) in bounds.items():
        mask &= (df[c] >= lo) & (df[c] <= hi)
    return df.loc[mask].copy()


def check_class_impact(before, after, label):
    """Mostra quantas linhas de cada classe o filtro IQR removeu."""
    vc_before = before["is_good"].value_counts(normalize=True)
    vc_after = after["is_good"].value_counts(normalize=True)
    print(f"\n--- Impacto por classe ({label}) ---")
    print("Antes  :", dict(vc_before.round(3)))
    print("Depois :", dict(vc_after.round(3)))
    print("Linhas removidas por classe:")
    print((before["is_good"].value_counts() - after["is_good"].value_counts()).fillna(before["is_good"].value_counts()))


# ======================================================
# 0. Leitura dos dados e filtros iniciais
# ======================================================
df = pd.read_csv(CSV_FILE, low_memory=False)
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

df["Splice Comment"] = df["Splice Comment"].apply(normalize_splice)

df = df[
    (df["Date"].dt.year == YEAR_FILTER) &
    (df["Splice Comment"] == normalize_splice(TARGET_SPLICE)) &
    (df["CrossSection"] == TARGET_CROSS_SECTION)
]

audit("Após filtros iniciais", df)

# ======================================================
# 1. Escolha das máquinas
# ======================================================
fixo = FIXED_ASSETS.get((TARGET_SPLICE, YEAR_FILTER))
if fixo is None:
    raise ValueError(
        f"Não há par de máquinas definido em FIXED_ASSETS para "
        f"({TARGET_SPLICE}, {YEAR_FILTER}). Acrescenta o par "
        f"ao dicionário FIXED_ASSETS."
    )

good_asset, bad_asset = fixo["good"], fixo["bad"]

asset_summary = df.groupby(["AssetID", "is_good"]).size().unstack(fill_value=0)
n_good = asset_summary.loc[good_asset, 1] if good_asset in asset_summary.index and 1 in asset_summary.columns else 0
n_bad = asset_summary.loc[bad_asset, 0] if bad_asset in asset_summary.index and 0 in asset_summary.columns else 0

if n_good < MIN_ROWS or n_bad < MIN_ROWS:
    raise ValueError(
        f"Par fixo insuficiente para ({TARGET_SPLICE}, {YEAR_FILTER}): "
        f"GOOD={good_asset} tem {n_good} ciclos, BAD={bad_asset} tem {n_bad} ciclos "
        f"(mínimo exigido: {MIN_ROWS}). Escolhe outro par de máquinas."
    )

print(f"Máquina GOOD: {good_asset} ({n_good} ciclos) | Máquina BAD: {bad_asset} ({n_bad} ciclos)")

df = df[df["AssetID"].isin([good_asset, bad_asset])]
audit("Após seleção de assets", df)

df_assets = df.copy()
sns.countplot(data=df_assets, x="is_good")
plt.title("Distribuição GOOD vs BAD – após seleção de assets")
savefig("assets_class_distribution.png")

# ======================================================
# 2. Remoção de colunas que não são medições
# ======================================================
df = df.drop(columns=[c for c in DROP_EARLY_COLS if c in df.columns], errors="ignore")

# Remove as colunas de referência e de estatísticas (Ref, Stat), antes da
# divisão. Ficam só as medições diretas de produto e de sensor.
df = df.drop(columns=[c for c in df.columns if "Ref" in c or "Stat" in c], errors="ignore")

FEATURES = [
    c for c in df.columns
    if c not in META_COLS + ["is_good"]
    and pd.api.types.is_numeric_dtype(df[c])
]

# ======================================================
# 3. Divisão treino/teste (antes de qualquer ajuste aos dados)
# ======================================================
df_train_raw, df_test_raw = train_test_split(
    df, stratify=df["is_good"], test_size=TEST_SIZE, random_state=RANDOM_STATE
)

audit("TREINO – logo após a divisão (antes de qualquer pré-processamento)", df_train_raw)
audit("TESTE  – logo após a divisão (fica sem alterações até ao fim)", df_test_raw)

# ======================================================
# 4. Filtro IQR (só no treino)
# ======================================================
iqr_bounds = compute_iqr_bounds(df_train_raw, FEATURES, k=IQR_K)

df_train = apply_iqr_filter(df_train_raw, iqr_bounds)
check_class_impact(df_train_raw, df_train, "IQR no treino")
audit("TREINO – após IQR", df_train)

df_test = df_test_raw.copy()
audit("TESTE – sem alteração (IQR não se aplica ao teste)", df_test)

# ======================================================
# 5. Balanceamento mensal por subamostragem (só no treino)
# ======================================================
df_train["Month"] = df_train["Date"].dt.month

chunks = []
# em cada mês, iguala o número de ciclos GOOD e BAD; meses com uma só
# classe são ignorados
for m, g in df_train.groupby("Month"):
    vc = g["is_good"].value_counts()
    if len(vc) < 2:
        continue
    n_sample = min(vc[0], vc[1])
    chunks.append(pd.concat([
        g[g["is_good"] == 0].sample(n=n_sample, random_state=RANDOM_STATE),
        g[g["is_good"] == 1].sample(n=n_sample, random_state=RANDOM_STATE)
    ]))

df_train = pd.concat(chunks).reset_index(drop=True)
audit("TREINO – após balanceamento temporal (mensal)", df_train)

audit("TESTE – proporção real, não balanceada", df_test)

df_balanced = df_train.copy()
sns.countplot(data=df_balanced, x="is_good")
plt.title("Distribuição GOOD vs BAD (treino) – após subamostragem mensal")
savefig("balanced_class_distribution.png")

for col in FEATURES:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    sns.histplot(df_balanced[df_balanced["is_good"] == 1],
                 x="Month", y=col, bins=12, cmap="YlOrRd", cbar=True, ax=ax[0])
    sns.histplot(df_balanced[df_balanced["is_good"] == 0],
                 x="Month", y=col, bins=12, cmap="YlOrRd", cbar=True, ax=ax[1])
    ax[0].set_title(f"GOOD – {col}")
    ax[1].set_title(f"BAD – {col}")
    plt.suptitle(f"Temporal Density (treino) – {col}")
    savefig(f"temporal_density_balanced_{col}.png")

FEATURES = [
    c for c in df_train.columns
    if c not in META_COLS + ["is_good", "Month"]
    and pd.api.types.is_numeric_dtype(df_train[c])
]

# ======================================================
# 6. Corte por correlação (calculado no treino, aplicado também ao teste)
# ======================================================
corr = df_train[FEATURES].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))

cols_to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]

df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors="ignore")
df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], errors="ignore")

FEATURES = [
    c for c in df_train.columns
    if c not in META_COLS + ["is_good", "Month"]
    and pd.api.types.is_numeric_dtype(df_train[c])
]

audit("TREINO – após corte de correlação", df_train)
audit("TESTE – mesmas colunas do treino aplicadas", df_test)
print("Nº de features finais:", len(FEATURES))
print(FEATURES)

sns.heatmap(df_train[FEATURES].corr(), cmap="RdBu_r", center=0)
plt.title("Correlação final (treino) – features selecionadas")
savefig("heatmap_after_correlation.png")

# ======================================================
# 7. Normalização MinMax (ajustada no treino, aplicada ao teste)
# ======================================================
scaler = MinMaxScaler()
df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
df_test[FEATURES] = scaler.transform(df_test[FEATURES])

audit("TREINO – após normalização MinMax", df_train)
audit("TESTE – após transform com o scaler do treino", df_test)

# ======================================================
# Gráficos após a normalização
# ======================================================
sns.heatmap(df_train[FEATURES].corr(), cmap="RdBu_r", center=0)
plt.title("Heatmap (treino) – features normalizadas")
savefig("heatmap_final.png")

for col in FEATURES[:min(4, len(FEATURES))]:
    sns.boxplot(data=df_train, x="is_good", y=col)
    plt.title(f"{col} – normalizado (treino)")
    savefig(f"box_final_{col}.png")

# ======================================================
# Modelos
# ======================================================
models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=False, cache_size=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=100
    )
}

X_train = df_train[FEATURES]
y_train = df_train["is_good"]

X_test = df_test[FEATURES]
y_test = df_test["is_good"]

print("\nProporção GOOD/BAD no teste (real, não balanceada):")
print(y_test.value_counts(normalize=True).round(3))

results = []

# ======================================================
# Treino e avaliação de cada modelo
# ======================================================
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        print("\n==============================")
        print("MODELO:", name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)  # recall da classe GOOD (valor 1)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("Recall :", rec)
        print("F1     :", f1)
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("iqr_fit_on", "train_only")
        mlflow.log_param("scaler_fit_on", "train_only")
        mlflow.log_param("good_asset", good_asset)
        mlflow.log_param("bad_asset", bad_asset)

        results.append([name, acc, rec, f1])

        # matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        plt.title(name)
        path_cm = os.path.join(OUT_DIR, f"cm_{name}.png")
        plt.savefig(path_cm)
        plt.close()
        mlflow.log_artifact(path_cm)

        # importância das variáveis
        # árvores e XGBoost: feature_importances_ do próprio modelo
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=FEATURES)
        elif name == "LogisticRegression":
            # valor absoluto dos coeficientes (variáveis em MinMax); não mostra o sentido do efeito
            imp = pd.Series(model.coef_[0], index=FEATURES).abs()
        else:
            imp = None

        if imp is not None:
            imp = imp.sort_values()
            plt.figure(figsize=(12, 6))
            imp.plot(kind="barh")
            plt.title("Feature Importance")
            plt.yticks(fontsize=11)
            plt.subplots_adjust(left=0.35)
            plt.tight_layout()
            path_fi = os.path.join(OUT_DIR, f"fi_{name}.png")
            plt.savefig(path_fi, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(path_fi)

        # gráfico real vs. previsto
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, c=(y_test != y_pred), cmap="coolwarm", alpha=0.6)
        plt.xlabel("Real")
        plt.ylabel("Previsto")
        plt.title("Real vs Predito (Erro)")
        plt.plot([0, 1], [0, 1], 'k--')
        path_plot = os.path.join(OUT_DIR, f"real_vs_pred_{name}.png")
        plt.savefig(path_plot)
        plt.close()
        mlflow.log_artifact(path_plot)

        # SHAP (só para modelos de árvores)
        if hasattr(model, "feature_importances_"):
            try:
                bg = X_train.sample(200, random_state=RANDOM_STATE)
                exp = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(exp)
                shap.summary_plot(sv, exp, show=False)
                path_shap1 = os.path.join(OUT_DIR, f"shap_summary_{name}.png")
                plt.savefig(path_shap1)
                plt.close()
                mlflow.log_artifact(path_shap1)
            except Exception as e:
                print("SHAP falhou para", name, "-", e)

        mlflow.sklearn.log_model(model, name)

# ======================================================
# Resultados
# ======================================================
res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "F1"])
# Ordena por F1 no mesmo teste usado para reportar o desempenho; serve só
# para ver os resultados, não para escolher um modelo.
res_df = res_df.sort_values("F1", ascending=False)

print("\n===== RESULTADOS =====")
print(res_df)

path_res = os.path.join(OUT_DIR, "results.csv")
res_df.to_csv(path_res, index=False)

with mlflow.start_run(run_name="Resumo"):
    mlflow.log_artifact(path_res)

# primeiro da ordenação acima; não quer dizer que seja o melhor para dados futuros
best = res_df.iloc[0]
print("\nPRIMEIRO DA ORDENAÇÃO (F1-score, exploratória):")
print(best)