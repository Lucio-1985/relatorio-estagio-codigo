# -*- coding: utf-8 -*-
"""
Q2 — ESTABILIDADE TEMPORAL (BXB2, 2025) — Gradient Boosting
Autor: LS

Produz, a partir do Fused_All.csv, os números da Secção 5.2 do relatório
(Tabelas 5.3 a 5.8) e grava tudo em CSV na pasta de saída.

Configuração: máquinas GOOD=14190 e BAD=14206; Gradient Boosting
(random_state=42); semestres P1 = jan-jun e P2 = jul-dez; teste de 30 %
(estratificado) reutilizado nos cenários que o usam.

Cenários temporais: dados dos dois semestres sem filtragem IQR e sem
balanceamento (o Gradient Boosting não depende da escala das variáveis).
Rolling window: 3 meses de treino, 2 de teste, avanço de 1 mês; limites
IQR e escala Min-Max ajustados no treino de cada janela; o IQR filtra apenas o
treino da janela e o teste fica sem filtragem (como no pipeline principal). Modelo estático: treinado na primeira janela e
aplicado ao teste das janelas seguintes.

Uso:
    python q2_temporal.py                    # usa Fused_All.csv na pasta atual
    python q2_temporal.py --csv C:\\caminho\\Fused_All.csv

Saída: pasta q2_outputs/ (CSV + figuras). Demora alguns minutos.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
# CONFIGURAÇÃO (nada aqui deve ser mudado entre execuções)
# ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="Fused_All.csv")
ap.add_argument("--out", default="q2_outputs")
args = ap.parse_args()

CSV_FILE = args.csv
OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)

YEAR = 2025
SPLICE = "BXB2"
CROSS_SECTION = 1.05
GOOD_ASSET, BAD_ASSET = 14190, 14206          # par fixo de BXB2-2025
FEATURES = ["CH-Product", "CH-Sensor", "WH-Product", "WH-Sensor",
            "dH-Product", "dH-Sensor", "t"]
RANDOM_STATE = 42
TEST_SIZE = 0.30
IQR_K = 1.5
TRAIN_MONTHS, TEST_MONTHS = 3, 2

META_COLS = ["AssetID", "Date", "Time", "Splice Comment", "CrossSection"]
DROP_EARLY_COLS = ["CH-I-T-Pos", "CH-I-T-Neg", "CH-O-T-Neg", "CH-O-T-Pos",
                   "WH-I-T-Pos", "WH-I-T-Neg", "Wh-O-T-Neg", "Wh-O-T-Pos", "Sequence"]


def out(name):
    return os.path.join(OUT_DIR, name)


def normalize_splice(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if "X" in s and "TEST" not in s:
        p = s.split("X")
        if len(p) == 2:
            s = "X".join(sorted(p))
    return s


def metricas(y, pred):
    """Métricas por classe a partir da matriz de confusão (BAD=0, GOOD=1)."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec_bad = tn / (tn + fp) if (tn + fp) else np.nan
    rec_good = tp / (tp + fn) if (tp + fn) else np.nan
    return {
        "n_teste": int(len(y)),
        "n_GOOD_teste": int(tp + fn), "n_BAD_teste": int(tn + fp),
        "accuracy": (tp + tn) / len(y),
        "accuracy_equilibrada": np.nanmean([rec_good, rec_bad]),
        "recall_GOOD": rec_good, "recall_BAD": rec_bad,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "baseline_maioria": max(tp + fn, tn + fp) / len(y),
    }


def gb():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


# ------------------------------------------------------------------
# CARREGAR E FILTRAR
# ------------------------------------------------------------------
t0 = time.time()
print("A carregar", CSV_FILE, "...")
df_all = pd.read_csv(CSV_FILE, low_memory=False)
df_all = df_all.loc[:, ~df_all.columns.astype(str).str.contains("^Unnamed")]
df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
df_all = df_all.dropna(subset=["Date"])
df_all["Splice Comment"] = df_all["Splice Comment"].apply(normalize_splice)

df = df_all[(df_all["Date"].dt.year == YEAR)
            & (df_all["Splice Comment"] == normalize_splice(SPLICE))
            & (df_all["CrossSection"] == CROSS_SECTION)].copy()
print("Após filtros iniciais:", df.shape)

resumo_assets = df.groupby(["AssetID", "is_good"]).size().unstack(fill_value=0)
print(resumo_assets)
assert GOOD_ASSET in resumo_assets.index and BAD_ASSET in resumo_assets.index, \
    "Par fixo de máquinas não encontrado nos dados."

df = df[df["AssetID"].isin([GOOD_ASSET, BAD_ASSET])].copy()
df_full = df.copy()                       # com TODAS as colunas (Ref, tolerâncias...)
df = df.dropna(subset=FEATURES + ["is_good"])
print("Após seleção de máquinas + dropna nas 7 variáveis:", df.shape,
      "(removidas por NaN:", len(df_full) - len(df), ")")
df["periodo"] = np.where(df["Date"].dt.month <= 6, "P1", "P2")
df["mes"] = df["Date"].dt.to_period("M")
df_full["periodo"] = np.where(df_full["Date"].dt.month <= 6, "P1", "P2")

# ------------------------------------------------------------------
# T1 — desenho dos casos (+ meses com dados)
# ------------------------------------------------------------------
linhas = []
for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
    g = df[df["AssetID"] == aid]
    linhas.append({"AssetID": aid, "classe": lab, "n_ciclos": len(g),
                   "n_P1": int((g["periodo"] == "P1").sum()),
                   "n_P2": int((g["periodo"] == "P2").sum()),
                   "meses_com_dados": ", ".join(sorted(g["mes"].astype(str).unique()))})
pd.DataFrame(linhas).to_csv(out("T1_desenho_casos.csv"), index=False)
print(pd.DataFrame(linhas).to_string(index=False))

# ------------------------------------------------------------------
# TABELAS 5.3 / 5.4 / 5.5 — KS, deslocamento (em IQR de P1), média e desvio
# (dados originais, sem IQR e sem normalização)
# ------------------------------------------------------------------
res = []
for (aid, classe), g in df.groupby(["AssetID", "is_good"]):
    p1, p2 = g[g["periodo"] == "P1"], g[g["periodo"] == "P2"]
    for f in FEATURES:
        a, b = p1[f], p2[f]
        if len(a) == 0 or len(b) == 0:
            continue
        q1a, meda, q3a = a.quantile([.25, .5, .75])
        q1b, medb, q3b = b.quantile([.25, .5, .75])
        iqr_a = q3a - q1a
        res.append({
            "AssetID": aid, "classe": "GOOD" if classe == 1 else "BAD", "variavel": f,
            "n_P1": len(a), "n_P2": len(b),
            "KS_D": ks_2samp(a, b).statistic,
            "mediana_P1": meda, "mediana_P2": medb, "IQR_P1": iqr_a, "IQR_P2": q3b - q1b,
            "desloc_mediana_em_IQR_P1": (medb - meda) / iqr_a if iqr_a > 0 else np.nan,
            "media_P1": a.mean(), "media_P2": b.mean(),
            "var_media_pct": 100 * (b.mean() - a.mean()) / a.mean() if a.mean() else np.nan,
            "dp_P1": a.std(), "dp_P2": b.std(),
            "var_dp_pct": 100 * (b.std() - a.std()) / a.std() if a.std() else np.nan,
        })
t_ks = pd.DataFrame(res)
t_ks.to_csv(out("T_KS_deslocamento_media_dp.csv"), index=False)
print("\nKS / deslocamento / média / dp:")
print(t_ks[["classe", "variavel", "KS_D", "desloc_mediana_em_IQR_P1",
            "var_media_pct", "var_dp_pct"]].round(3).to_string(index=False))

# ------------------------------------------------------------------
# CORRELAÇÕES por período: (a) as duas máquinas em conjunto (Tabela 5.6) e (b) por máquina
# ------------------------------------------------------------------
c1j = df[df["periodo"] == "P1"][FEATURES].corr()["t"]
c2j = df[df["periodo"] == "P2"][FEATURES].corr()["t"]
t_corr_j = pd.DataFrame({"variavel": FEATURES, "r_t_P1": c1j.values, "r_t_P2": c2j.values}).query("variavel != 't'")
t_corr_j.to_csv(out("T_correlacoes_t_duas_maquinas.csv"), index=False)
print("\nCorrelação de t com as restantes variáveis (duas máquinas em conjunto):")
print(t_corr_j.round(2).to_string(index=False))

corr_rows = []
for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
    g = df[df["AssetID"] == aid]
    c1 = g[g["periodo"] == "P1"][FEATURES].corr()
    c2 = g[g["periodo"] == "P2"][FEATURES].corr()
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            corr_rows.append({"maquina": lab, "AssetID": aid, "var_a": a, "var_b": b,
                              "r_P1": c1.loc[a, b], "r_P2": c2.loc[a, b],
                              "diferenca": c2.loc[a, b] - c1.loc[a, b]})
t_corr = pd.DataFrame(corr_rows)
t_corr.to_csv(out("T_correlacoes_P1_P2.csv"), index=False)
print("\nCorrelações de t (máquina GOOD e BAD):")
print(t_corr[(t_corr.var_b == "t") | (t_corr.var_a == "t")].round(2).to_string(index=False))

# Dependência algébrica CH/WH/dH: dH = CH - WH ?
alg = []
for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
    g = df[df["AssetID"] == aid]
    for tipo in ["Product", "Sensor"]:
        d = g[f"dH-{tipo}"]
        r = g[f"CH-{tipo}"] - g[f"WH-{tipo}"]
        alg.append({"maquina": lab, "tipo": tipo,
                    "corr(dH, CH-WH)": d.corr(r),
                    "erro_abs_medio(dH-(CH-WH))": (d - r).abs().mean(),
                    "erro_abs_max": (d - r).abs().max(),
                    "dH_mediana": d.median()})
pd.DataFrame(alg).to_csv(out("T_dependencia_algebrica_dH.csv"), index=False)
print("\nDependência algébrica dH vs CH-WH:")
print(pd.DataFrame(alg).round(4).to_string(index=False))

# ------------------------------------------------------------------
# REFERÊNCIA INTERNA (Ref-Product) — mediana e média por período e máquina
# ------------------------------------------------------------------
ref_cols = [c for c in df_full.columns if c.endswith("Ref-Product") or c in ("t-Ref",)]
ref_rows = []
for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
    g = df_full[df_full["AssetID"] == aid]
    for c in ref_cols:
        a = g.loc[g["periodo"] == "P1", c].dropna()
        b = g.loc[g["periodo"] == "P2", c].dropna()
        if len(a) and len(b):
            ref_rows.append({"maquina": lab, "coluna": c,
                             "mediana_P1": a.median(), "mediana_P2": b.median(),
                             "var_mediana_pct": 100 * (b.median() - a.median()) / a.median()
                             if a.median() else np.nan,
                             "media_P1": a.mean(), "media_P2": b.mean()})
pd.DataFrame(ref_rows).to_csv(out("T_referencia_interna.csv"), index=False)
print("\nReferência interna:")
print(pd.DataFrame(ref_rows).round(3).to_string(index=False))

# ------------------------------------------------------------------
# TOLERÂNCIA (CH e WH): IQR do desvio (Product - Ref-Product) vs. banda registada
# NÃO se calculam proporções de ciclos fora de tolerância (plano C16).
# ------------------------------------------------------------------
tol_rows = []
band_rows = []
for pref, ref_col, band_cols in [
        ("CH", "CH-Ref-Product", ["CH-I-T-Pos", "CH-O-T-Pos"]),
        ("WH", "WH-Ref-Product", ["WH-I-T-Pos", "Wh-O-T-Pos"])]:
    if ref_col not in df_full.columns:
        continue
    for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
        g = df_full[df_full["AssetID"] == aid]
        for bc in band_cols:
            if bc in g.columns:
                vc = g[bc].value_counts(dropna=False)
                for val, n in vc.items():
                    band_rows.append({"maquina": lab, "coluna": bc, "valor": val, "n_ciclos": n})
        for per in ["P1", "P2"]:
            gp = g[g["periodo"] == per]
            dev = ((gp[f"{pref}-Product"] - gp[ref_col]) * 1000).dropna()   # mm -> µm
            if len(dev) == 0:
                continue
            q1, q3 = dev.quantile([.25, .75])
            ad = dev.abs()
            iq_abs = ad.quantile(.75) - ad.quantile(.25)
            row = {"maquina": lab, "variavel": f"{pref}-Product", "periodo": per,
                   "n": len(dev), "mediana_desvio_um": dev.median(),
                   "IQR_desvio_um": q3 - q1, "IQR_desvio_absoluto_um": iq_abs,
                   "P75_desvio_absoluto_um": ad.quantile(.75)}
            for bc in band_cols:
                if bc in gp.columns:
                    for val in sorted(gp[bc].dropna().unique()):
                        if bc.upper().startswith(pref) and "-O-" in bc.upper():
                            row[f"IQR|desvio|/banda {bc}={val:g}"] = iq_abs / val
                            row[f"IQR(desvio)/banda {bc}={val:g}"] = (q3 - q1) / val
            tol_rows.append(row)
pd.DataFrame(tol_rows).to_csv(out("T_tolerancia_IQR_desvio.csv"), index=False)
pd.DataFrame(band_rows).to_csv(out("T_valores_banda_tolerancia.csv"), index=False)
print("\nValores de banda de tolerância existentes (por máquina):")
print(pd.DataFrame(band_rows).to_string(index=False))
print("\nIQR do desvio vs banda:")
print(pd.DataFrame(tol_rows).round(4).to_string(index=False))

# ------------------------------------------------------------------
# TABELA 5.7 — QUATRO CENÁRIOS (Gradient Boosting; teste de cada período reutilizado)
# Dados dos dois semestres sem filtragem IQR e sem escala.
# ------------------------------------------------------------------
p1 = df[df["periodo"] == "P1"]
p2 = df[df["periodo"] == "P2"]
p1_tr, p1_te = train_test_split(p1, test_size=TEST_SIZE, stratify=p1["is_good"], random_state=RANDOM_STATE)
p2_tr, p2_te = train_test_split(p2, test_size=TEST_SIZE, stratify=p2["is_good"], random_state=RANDOM_STATE)

linhas = []
for _ in [0]:
    cen = {"P1 -> P1": (p1_tr, p1_te), "P2 -> P2": (p2_tr, p2_te),
           "P2 -> P1": (p2_tr, p1_te), "P1 -> P2": (p1_tr, p2_te)}
    for nome, (tr, te) in cen.items():
        m = gb().fit(tr[FEATURES], tr["is_good"])
        r = metricas(te["is_good"], m.predict(te[FEATURES]))
        r.update({"cenario": nome, "n_treino": len(tr),
                  "n_GOOD_treino": int((tr["is_good"] == 1).sum()),
                  "n_BAD_treino": int((tr["is_good"] == 0).sum())})
        linhas.append(r)
t_cen = pd.DataFrame(linhas)
t_cen = t_cen[["cenario", "n_treino", "n_GOOD_treino", "n_BAD_treino", "n_teste",
               "n_GOOD_teste", "n_BAD_teste", "accuracy", "accuracy_equilibrada", "recall_GOOD",
               "recall_BAD", "baseline_maioria", "TN", "FP", "FN", "TP"]]
t_cen.to_csv(out("T_cenarios_temporais.csv"), index=False)
print("\nCenários temporais:")
print(t_cen.round(3).to_string(index=False))

# ------------------------------------------------------------------
# TABELA 5.8 — ROLLING WINDOW vs MODELO ESTÁTICO
# IQR por janela: limites calculados no treino da janela e aplicados apenas ao
# treino da janela; o teste não é filtrado (regra do pipeline principal).
# Escala (MinMax) ajustada SEMPRE só no treino de cada janela.
# O modelo estático é treinado na PRIMEIRA janela e aplicado a todas as seguintes,
# com o mesmo pré-processamento dessa primeira janela.
# ------------------------------------------------------------------
dfr = df.copy()


def janelas(d0, d1):
    cur = d0
    while True:
        tr0 = cur
        tr1 = tr0 + relativedelta(months=TRAIN_MONTHS) - relativedelta(days=1)
        te0 = tr1 + relativedelta(days=1)
        te1 = te0 + relativedelta(months=TEST_MONTHS) - relativedelta(days=1)
        if te1 > d1:
            break
        yield tr0, tr1, te0, te1
        cur = cur + relativedelta(months=1)


def limites_iqr(d, cols, k=IQR_K):
    b = {}
    for c in cols:
        q1, q3 = d[c].quantile([.25, .75])
        i = q3 - q1
        if pd.isna(i) or i == 0:
            continue
        b[c] = (q1 - k * i, q3 + k * i)
    return b


def filtra(d, b):
    m = pd.Series(True, index=d.index)
    for c, (lo, hi) in b.items():
        m &= (d[c] >= lo) & (d[c] <= hi)
    return d.loc[m]


def prepara(tr_raw, te_raw, filtrar_teste):
    b = limites_iqr(tr_raw, FEATURES)
    tr = filtra(tr_raw, b)
    te = filtra(te_raw, b) if filtrar_teste else te_raw
    sc = MinMaxScaler().fit(tr[FEATURES])
    return tr, te, sc, b


d0, d1 = dfr["Date"].min(), dfr["Date"].max()
print("\nIntervalo de datas usado no rolling:", d0.date(), "->", d1.date())
janelas_lista = list(janelas(d0, d1))
print("Nº de janelas possíveis:", len(janelas_lista))

linhas_roll = []
for variante, filtrar_teste in [("IQR_so_treino", False)]:
    modelo_est, sc_est = None, None
    for i, (tr0, tr1, te0, te1) in enumerate(janelas_lista, start=1):
        tr_raw = dfr[(dfr["Date"] >= tr0) & (dfr["Date"] <= tr1)]
        te_raw = dfr[(dfr["Date"] >= te0) & (dfr["Date"] <= te1)]
        if len(tr_raw) == 0 or len(te_raw) == 0:
            print(f"[{variante}] janela {i}: sem dados, ignorada")
            continue
        if tr_raw["is_good"].nunique() < 2 or te_raw["is_good"].nunique() < 2:
            print(f"[{variante}] janela {i}: falta uma classe, ignorada")
            continue
        tr, te, sc, b = prepara(tr_raw, te_raw, filtrar_teste)
        if len(tr) == 0 or len(te) == 0 or tr["is_good"].nunique() < 2 or te["is_good"].nunique() < 2:
            print(f"[{variante}] janela {i}: vazia/uma classe após IQR, ignorada")
            continue
        Xtr = pd.DataFrame(sc.transform(tr[FEATURES]), columns=FEATURES)
        Xte = pd.DataFrame(sc.transform(te[FEATURES]), columns=FEATURES)
        m = gb().fit(Xtr, tr["is_good"].values)
        r_roll = metricas(te["is_good"].values, m.predict(Xte))
        # modelo estático = o da primeira janela válida
        if modelo_est is None:
            modelo_est, sc_est = m, sc
            r_est = r_roll
        else:
            Xte_e = pd.DataFrame(sc_est.transform(te[FEATURES]), columns=FEATURES)
            r_est = metricas(te["is_good"].values, modelo_est.predict(Xte_e))
        for tipo, r in [("retreino_mensal", r_roll), ("estatico_1a_janela", r_est)]:
            rr = dict(r)
            rr.update({"variante": variante, "modelo": tipo, "janela": i,
                       "treino": f"{tr0.date()} -> {tr1.date()}",
                       "teste": f"{te0.date()} -> {te1.date()}",
                       "n_treino_apos_IQR": len(tr),
                       "n_teste_antes_IQR": len(te_raw)})
            linhas_roll.append(rr)
        print(f"[{variante}] janela {i} {te0.date()}->{te1.date()}: "
              f"retreino bal_acc={r_roll['accuracy_equilibrada']:.3f} | "
              f"estático bal_acc={r_est['accuracy_equilibrada']:.3f} "
              f"| n_treino={len(tr)} n_teste={len(te)}  ({time.time()-t0:.0f}s)")

t_roll = pd.DataFrame(linhas_roll)
t_roll.to_csv(out("T_rolling_janelas.csv"), index=False)

resumo = (t_roll.groupby(["variante", "modelo"])["accuracy_equilibrada"]
          .agg(["count", "mean", "std", "min", "max"]).reset_index())
resumo.to_csv(out("T_rolling_resumo.csv"), index=False)
print("\nRESUMO rolling (janelas de 2 meses de teste):")
print(resumo.round(3).to_string(index=False))

# ------------------------------------------------------------------
# Estatísticas mensais (mediana e IQR) por máquina — dados das Figuras 5.x (F3)
# ------------------------------------------------------------------
mens = []
for aid, lab in [(GOOD_ASSET, "GOOD"), (BAD_ASSET, "BAD")]:
    g = df[df["AssetID"] == aid]
    for f in FEATURES:
        s = g.groupby("mes")[f].quantile([.25, .5, .75]).unstack()
        for mes, row in s.iterrows():
            mens.append({"maquina": lab, "AssetID": aid, "variavel": f, "mes": str(mes),
                         "q1": row[.25], "mediana": row[.5], "q3": row[.75],
                         "n": int(((g["mes"] == mes)).sum())})
pd.DataFrame(mens).to_csv(out("T_mensal_mediana_IQR.csv"), index=False)

for f in ["t", "dH-Product"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    for aid, lab, cor in [(GOOD_ASSET, "GOOD", "tab:green"), (BAD_ASSET, "BAD", "tab:red")]:
        s = df[df["AssetID"] == aid].groupby("mes")[f].quantile([.25, .5, .75]).unstack()
        x = s.index.astype(str)
        ax.plot(x, s[.5], marker="o", color=cor, label=f"Máquina {aid} ({lab})")
        ax.fill_between(x, s[.25], s[.75], alpha=.2, color=cor)
    ax.set_title(f"BXB2 2025 — mediana e IQR mensal — {f}")
    ax.set_xlabel("Mês"); ax.set_ylabel(f); ax.grid(True, alpha=.3); ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(out(f"F3_mensal_{f}.png"), dpi=200); plt.close()

# Figura dos cenários
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.27
t_fig = t_cen.reset_index(drop=True)
xs = np.arange(len(t_fig))
ax.bar(xs - w, t_fig["accuracy_equilibrada"], w, label="Acc. equilibrada")
ax.bar(xs, t_fig["recall_GOOD"], w, label="Recall GOOD")
ax.bar(xs + w, t_fig["recall_BAD"], w, label="Recall BAD")
ax.set_xticks(xs); ax.set_xticklabels(t_fig["cenario"]); ax.set_ylim(0, 1.2)
ax.set_title("BXB2 2025 — cenários temporais"); ax.legend(ncol=3, loc="upper center"); ax.grid(True, axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(out("F4_desempenho_cenarios.png"), dpi=200); plt.close()

# Registo de versões
import sklearn, scipy
with open(out("versoes_ambiente.txt"), "w") as fh:
    fh.write(f"python {sys.version.split()[0]}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
             f"scipy {scipy.__version__}\nscikit-learn {sklearn.__version__}\n")

print(f"\nCONCLUÍDO em {time.time()-t0:.0f}s. Resultados em: {os.path.abspath(OUT_DIR)}")
