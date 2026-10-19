# -*- coding: utf-8 -*-
"""
TESTE TEMPORAL BLOQUEADO E SENSIBILIDADE AO IQR (BXB2, 2025)
Autor: LS

DECISÕES FIXADAS ANTES DE VER QUALQUER RESULTADO (não alterar entre execuções)
  Caso ............ BXB2-2025, máquinas GOOD=14190 e BAD=14206 (par fixo)
  Modelo .......... Gradient Boosting, random_state=42, parâmetros por omissão
  Variáveis ....... as 7 do pipeline principal (CH, WH, dH de Product e Sensor, t)
  Mês de teste .... dezembro de 2025. Regra, escolhida só pela cobertura de dados
                    (nunca pela métrica): último mês do ano em que as DUAS máquinas
                    têm ciclos desde o 1.º ao último dia do mês e >= 20 dias com dados.
                    Dezembro cumpre: GOOD 21 dias e BAD 26 dias, de 1 a 31/12.
  Treino .......... todos os ciclos com Date < 2025-12-01 (mesma fronteira para as
                    duas máquinas). Não há identificador de sessão/lote nos dados:
                    a fronteira é a data do ciclo, pelo que nenhum ciclo é dividido.
  Pré-processamento (idêntico ao pipeline principal, ajustado SÓ no treino):
                    limites IQR (k=1,5) -> balanceamento mensal por subamostragem
                    -> corte de correlação |r|>0,90 -> MinMax.
  Teste ........... dezembro completo, SEM filtragem IQR e SEM balanceamento.
  Com IQR ......... execução de referência (IQR no treino).
  Sem IQR ......... igual à anterior, desativando APENAS a remoção IQR do treino.
  Referência ...... regra da maioria: classe maioritária do treino antes do
                    balanceamento, aplicada como regra constante no teste.
  Sem tuning, uma única execução por condição.

Uso:  python holdout_temporal.py --csv Fused_All.csv
Saída: pasta holdout_outputs/
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="Fused_All.csv")
ap.add_argument("--out", default="holdout_outputs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

YEAR, SPLICE, CROSS_SECTION = 2025, "BXB2", 1.05
GOOD_ASSET, BAD_ASSET = 14190, 14206
FEATURES = ["CH-Product", "CH-Sensor", "WH-Product", "WH-Sensor",
            "dH-Product", "dH-Sensor", "t"]
RANDOM_STATE, IQR_K, CORR_THRESHOLD = 42, 1.5, 0.90
TEST_START, TEST_END = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")


def normalize_splice(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if "X" in s and "TEST" not in s:
        p = s.split("X")
        if len(p) == 2:
            s = "X".join(sorted(p))
    return s


def iqr_bounds(d, cols, k):
    b = {}
    for c in cols:
        q1, q3 = d[c].quantile([0.25, 0.75])
        i = q3 - q1
        if pd.isna(i) or i == 0:
            continue
        b[c] = (q1 - k * i, q3 + k * i)
    return b


def apply_bounds(d, b):
    m = pd.Series(True, index=d.index)
    for c, (lo, hi) in b.items():
        m &= (d[c] >= lo) & (d[c] <= hi)
    return d.loc[m].copy()


def balance_monthly(d):
    d = d.copy()
    d["Month"] = d["Date"].dt.month
    chunks, excl = [], []
    for m, g in d.groupby("Month"):
        vc = g["is_good"].value_counts()
        if len(vc) < 2:
            excl.append(int(m))
            continue
        n = min(vc[0], vc[1])
        chunks.append(pd.concat([
            g[g["is_good"] == 0].sample(n=n, random_state=RANDOM_STATE),
            g[g["is_good"] == 1].sample(n=n, random_state=RANDOM_STATE)]))
    return pd.concat(chunks).reset_index(drop=True), excl


def run(train_raw, test, usar_iqr, nome):
    r = {"execucao": nome, "IQR_no_treino": "sim" if usar_iqr else "não",
         "n_treino_antes": len(train_raw),
         "n_GOOD_treino_antes": int((train_raw["is_good"] == 1).sum()),
         "n_BAD_treino_antes": int((train_raw["is_good"] == 0).sum())}
    maj_class = 1 if r["n_GOOD_treino_antes"] >= r["n_BAD_treino_antes"] else 0
    r["classe_maioritaria_treino"] = "GOOD" if maj_class == 1 else "BAD"

    tr = train_raw
    if usar_iqr:
        tr = apply_bounds(train_raw, iqr_bounds(train_raw, FEATURES, IQR_K))
    r["n_treino_apos_IQR"] = len(tr)
    tr, excl = balance_monthly(tr)
    r["meses_excluidos_balanceamento"] = ",".join(map(str, excl)) or "nenhum"
    r["n_treino_final"] = len(tr)
    r["n_GOOD_treino_final"] = int((tr["is_good"] == 1).sum())
    r["n_BAD_treino_final"] = int((tr["is_good"] == 0).sum())

    corr = tr[FEATURES].corr().abs()
    up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    drop = [c for c in up.columns if any(up[c] > CORR_THRESHOLD)]
    feats = [f for f in FEATURES if f not in drop]
    r["variaveis_removidas_corr"] = ",".join(drop) or "nenhuma"
    r["n_variaveis"] = len(feats)

    sc = MinMaxScaler().fit(tr[feats])
    Xtr = sc.transform(tr[feats])
    Xte = sc.transform(test[feats])
    m = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(Xtr, tr["is_good"])
    y, p = test["is_good"].values, m.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    rg, rb = tp / (tp + fn), tn / (tn + fp)
    r.update({"n_teste": len(y), "n_GOOD_teste": int(tp + fn), "n_BAD_teste": int(tn + fp),
              "baseline_maioria_teste": (np.mean(y == maj_class)),
              "accuracy": (tp + tn) / len(y), "accuracy_equilibrada": (rg + rb) / 2,
              "recall_GOOD": rg, "recall_BAD": rb,
              "precisao_BAD": tn / (tn + fn) if (tn + fn) else np.nan,
              "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return r


df = pd.read_csv(args.csv, low_memory=False)
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df["Splice Comment"] = df["Splice Comment"].apply(normalize_splice)
df = df[(df["Date"].dt.year == YEAR) & (df["Splice Comment"] == SPLICE)
        & (df["CrossSection"] == CROSS_SECTION)
        & (df["AssetID"].isin([GOOD_ASSET, BAD_ASSET]))]
df = df.dropna(subset=FEATURES + ["is_good"]).copy()

cob = (df.assign(mes=df["Date"].dt.to_period("M"))
         .groupby(["mes", "AssetID"])
         .agg(ciclos=("t", "size"), dias_com_dados=("Date", lambda s: s.dt.normalize().nunique()),
              primeiro=("Date", "min"), ultimo=("Date", "max")).reset_index())
cob.to_csv(os.path.join(args.out, "T_cobertura_mensal.csv"), index=False)

train_raw = df[df["Date"] < TEST_START].copy()
test = df[(df["Date"] >= TEST_START) & (df["Date"] < TEST_END)].copy()
assert train_raw["Date"].max() < test["Date"].min()
print("Treino:", train_raw["Date"].min().date(), "->", train_raw["Date"].max().date(), len(train_raw))
print("Teste :", test["Date"].min().date(), "->", test["Date"].max().date(), len(test))

res = pd.DataFrame([
    run(train_raw, test, True, "Com IQR no treino (referência)"),
    run(train_raw, test, False, "Sem IQR no treino"),
])
res.insert(1, "treino_datas", f"{train_raw['Date'].min().date()} a {train_raw['Date'].max().date()}")
res.insert(2, "teste_datas", f"{test['Date'].min().date()} a {test['Date'].max().date()}")
res.to_csv(os.path.join(args.out, "T_holdout_temporal.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(res.round(4).T.to_string())
