
# -*- coding: utf-8 -*-
"""
DIAGRAMAS DE CAIXA (ASSINATURA BRUTA) + IMPORTÂNCIA DE VARIÁVEIS
Gera outputs com nome claro por execução: <SPLICE>_<ANO>/

CORREÇÃO APLICADA (vs. versão anterior):
    A seleção de máquinas deixou de ser "a primeira da lista"
    (good_candidates[0]/bad_candidates[0]) e passou a ser FIXA, com o
    par confirmado por diagnostico_maquinas.py para cada (config, ano).
    A seleção pela ordem escolhia, por vezes, uma máquina com poucos
    dados em vez da correta (ex.: BXB2 2024 escolhia a 14178 em vez da
    14190). O script agora falha com erro claro se o par fixo não tiver
    pelo menos MIN_ROWS ciclos de cada classe, em vez de escolher outra
    máquina silenciosamente.

Corre este script 3 vezes, mudando só as 3 linhas de CONFIG no topo:
  1) TARGET_SPLICE="BXB2",   TARGET_CROSS_SECTION=1.05, YEAR_FILTER=2024
  2) TARGET_SPLICE="BCXBC2", TARGET_CROSS_SECTION=2.2,  YEAR_FILTER=2024
  3) TARGET_SPLICE="BXB2",   TARGET_CROSS_SECTION=1.05, YEAR_FILTER=2025

Cada execução cria uma pasta própria, ex.: resultados_Q1_Q3/BXB2_2024/
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ======================================================
# CONFIG — muda só estas 3 linhas entre as 3 execuções
# ======================================================
CSV_FILE = "Fused_All.csv"
TARGET_SPLICE = "BCXBC2"          # "BXB2" ou "BCXBC2"
TARGET_CROSS_SECTION = 2.2     # 1.05 (BXB2) ou 2.2 (BCXBC2)
YEAR_FILTER = 2024              # 2024 ou 2025

MIN_ROWS = 1000
CORR_THRESHOLD = 0.90
RANDOM_STATE = 42

# ======================================================
# MÁQUINAS FIXAS — confirmadas com diagnostico_maquinas.py
# Não editar por execução; o script escolhe sozinho pela chave (config, ano).
# ======================================================
FIXED_ASSETS = {
    ("BXB2", 2024):   {"good": 14190, "bad": 14171},
    ("BXB2", 2025):   {"good": 14190, "bad": 14206},
    ("BCXBC2", 2024): {"good": 14167, "bad": 14206},
}

# Nome da pasta gerado automaticamente — não precisas de mudar isto
OUT_DIR = os.path.join("resultados_Q1_Q3", f"{TARGET_SPLICE}_{YEAR_FILTER}")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_CANDIDATAS = ["CH-Product", "CH-Sensor", "WH-Product", "WH-Sensor",
                       "dH-Product", "dH-Sensor", "t"]


def normalize_splice(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if "X" in s and "TEST" not in s:
        p = s.split("X")
        if len(p) == 2:
            s = "X".join(sorted(p))
    return s


# ======================================================
# LOAD + FILTROS
# ======================================================
df = pd.read_csv(CSV_FILE, low_memory=False)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df["Splice Comment"] = df["Splice Comment"].apply(normalize_splice)

df = df[
    (df["Date"].dt.year == YEAR_FILTER) &
    (df["CrossSection"] == TARGET_CROSS_SECTION) &
    (df["Splice Comment"] == normalize_splice(TARGET_SPLICE))
]
print(f"[{TARGET_SPLICE} {YEAR_FILTER}] Após filtros:", df.shape)

# ======================================================
# SELEÇÃO DE ASSETS (FIXA, NÃO PELA ORDEM)
# ======================================================
fixo = FIXED_ASSETS.get((TARGET_SPLICE, YEAR_FILTER))
if fixo is None:
    raise ValueError(
        f"Não há par de máquinas definido em FIXED_ASSETS para "
        f"({TARGET_SPLICE}, {YEAR_FILTER}). Corre primeiro o "
        f"diagnostico_maquinas.py e acrescenta o par aqui."
    )

good_asset, bad_asset = fixo["good"], fixo["bad"]

asset_summary = df.groupby(["AssetID", "is_good"]).size().unstack(fill_value=0)
n_good = asset_summary.loc[good_asset, 1] if good_asset in asset_summary.index and 1 in asset_summary.columns else 0
n_bad = asset_summary.loc[bad_asset, 0] if bad_asset in asset_summary.index and 0 in asset_summary.columns else 0

if n_good < MIN_ROWS or n_bad < MIN_ROWS:
    raise ValueError(
        f"Par fixo insuficiente para ({TARGET_SPLICE}, {YEAR_FILTER}): "
        f"GOOD={good_asset} tem {n_good} ciclos, BAD={bad_asset} tem {n_bad} ciclos "
        f"(mínimo exigido: {MIN_ROWS}). Corre o diagnóstico e decide manualmente."
    )

print(f"[{TARGET_SPLICE} {YEAR_FILTER}] Máquina GOOD: {good_asset} ({n_good} ciclos) | "
      f"Máquina BAD: {bad_asset} ({n_bad} ciclos)")

df = df[df["AssetID"].isin([good_asset, bad_asset])].dropna(subset=FEATURES_CANDIDATAS)

# Rótulo legível para as legendas dos gráficos (pedido explícito do professor)
df["Máquina"] = df["AssetID"].map({
    good_asset: f"Máquina {good_asset} (GOOD)",
    bad_asset: f"Máquina {bad_asset} (BAD)",
})

# ======================================================
# PARTE 1 — DIAGRAMAS DE CAIXA (DADOS EM BRUTO, ANTES DE IQR/BALANCEAMENTO/NORMALIZAÇÃO)
# Caracterização da assinatura de processo (Q1)
# ======================================================
box_dir = os.path.join(OUT_DIR, "boxplots_assinatura_bruta")
os.makedirs(box_dir, exist_ok=True)

for feat in FEATURES_CANDIDATAS:
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="Máquina", y=feat)
    plt.title(f"{TARGET_SPLICE} {YEAR_FILTER} — {feat} (dados brutos, por máquina)")
    plt.xlabel("")
    plt.tight_layout()
    path = os.path.join(box_dir, f"box_{TARGET_SPLICE}_{YEAR_FILTER}_{feat}.png")
    plt.savefig(path, dpi=200)
    plt.close()

print(f"[{TARGET_SPLICE} {YEAR_FILTER}] Boxplots gravados em: {box_dir}")

# ======================================================
# PARTE 2 — PIPELINE ATÉ AO MODELO (split -> IQR -> balanceamento -> correlação -> normalização)
# Igual ao Pipeline_MLflow_Corrigido.py, resumido aqui só para chegar à
# importância de variáveis com o conjunto final de cada execução.
# ======================================================
DROP_EARLY_COLS = ["CH-I-T-Pos", "CH-I-T-Neg", "CH-O-T-Neg", "CH-O-T-Pos",
                    "WH-I-T-Pos", "WH-I-T-Neg", "Wh-O-T-Neg", "Wh-O-T-Pos", "Sequence"]
META_COLS = ["AssetID", "Date", "Time", "Splice Comment", "CrossSection", "Máquina"]

df = df.drop(columns=[c for c in DROP_EARLY_COLS if c in df.columns], errors="ignore")

df_train_raw, df_test_raw = train_test_split(
    df, stratify=df["is_good"], test_size=0.3, random_state=RANDOM_STATE
)

def iqr_bounds(d, cols, k=1.5):
    b = {}
    for c in cols:
        q1, q3 = d[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        b[c] = (q1 - k * iqr, q3 + k * iqr)
    return b

def apply_iqr(d, bounds):
    mask = pd.Series(True, index=d.index)
    for c, (lo, hi) in bounds.items():
        mask &= (d[c] >= lo) & (d[c] <= hi)
    return d.loc[mask].copy()

bounds = iqr_bounds(df_train_raw, FEATURES_CANDIDATAS)
df_train = apply_iqr(df_train_raw, bounds)
df_test = df_test_raw.copy()  # teste intacto

df_train["Month"] = df_train["Date"].dt.month
chunks = []
for m, g in df_train.groupby("Month"):
    vc = g["is_good"].value_counts()
    if len(vc) < 2:
        continue
    n = min(vc[0], vc[1])
    chunks.append(pd.concat([
        g[g["is_good"] == 0].sample(n=n, random_state=RANDOM_STATE),
        g[g["is_good"] == 1].sample(n=n, random_state=RANDOM_STATE)
    ]))
df_train = pd.concat(chunks).reset_index(drop=True)

corr = df_train[FEATURES_CANDIDATAS].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
FEATURES = [c for c in FEATURES_CANDIDATAS if c not in drop_corr]
print(f"[{TARGET_SPLICE} {YEAR_FILTER}] Features finais ({len(FEATURES)}):", FEATURES)

scaler = MinMaxScaler()
df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
df_test[FEATURES] = scaler.transform(df_test[FEATURES])

X_train, y_train = df_train[FEATURES], df_train["is_good"]
X_test, y_test = df_test[FEATURES], df_test["is_good"]

# ======================================================
# PARTE 3 — IMPORTÂNCIA DE VARIÁVEIS (nome claro, por modelo e por execução)
# ======================================================
fi_dir = os.path.join(OUT_DIR, "importancia_variaveis")
os.makedirs(fi_dir, exist_ok=True)

models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=100),
}

resumo_importancias = []

for nome, modelo in models.items():
    modelo.fit(X_train, y_train)

    if hasattr(modelo, "feature_importances_"):
        imp = pd.Series(modelo.feature_importances_, index=FEATURES)
    elif nome == "LogisticRegression":
        imp = pd.Series(modelo.coef_[0], index=FEATURES).abs()
    else:
        continue  # SVM, KNN sem importância nativa — ficam de fora deste ranking

    imp = imp.sort_values(ascending=False)
    for var, val in imp.items():
        resumo_importancias.append({
            "execucao": f"{TARGET_SPLICE}_{YEAR_FILTER}",
            "modelo": nome,
            "variavel": var,
            "importancia": val,
            "posicao": list(imp.index).index(var) + 1,
        })

    plt.figure(figsize=(8, 5))
    imp.sort_values().plot(kind="barh")
    plt.title(f"{TARGET_SPLICE} {YEAR_FILTER} — Importância das Variáveis ({nome})")
    plt.tight_layout()
    path = os.path.join(fi_dir, f"fi_{TARGET_SPLICE}_{YEAR_FILTER}_{nome}.png")
    plt.savefig(path, dpi=200)
    plt.close()

pd.DataFrame(resumo_importancias).to_excel(
    os.path.join(fi_dir, f"importancia_variaveis_{TARGET_SPLICE}_{YEAR_FILTER}.xlsx"), index=False
)

print(f"[{TARGET_SPLICE} {YEAR_FILTER}] Importância de variáveis gravada em: {fi_dir}")
print(f"\nCONCLUÍDO — tudo em: {os.path.abspath(OUT_DIR)}")
