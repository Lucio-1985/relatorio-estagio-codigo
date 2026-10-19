# -*- coding: utf-8 -*-
"""
Q1 / Q3 — REPRODUÇÃO DO PIPELINE PRINCIPAL (BXB2-2024, BXB2-2025, BCXBC2-2024)
Autor: LS

Executa o MESMO pipeline do Anexo (FASE 0-7: filtros -> máquinas fixas -> remoção
de Ref/Stat -> split 70/30 -> IQR só no treino -> balanceamento mensal só no treino
-> corte de correlação no treino -> Min-Max com fit no treino -> 7 modelos) e grava,
para cada caso e modelo, TUDO o que o plano do tutor pede (C01-C04, C08, C11, C13,
C21, O1) em ficheiros CSV, para que as tabelas do relatório sejam copiadas daí:

  * contagens por etapa e por classe (linhagem); linhas removidas pelo IQR por classe
  * meses incluídos/excluídos do treino balanceado (efeito do `continue`)
  * baseline da classe maioritária (definida no treino, antes do balanceamento)
  * por modelo: accuracy, accuracy equilibrada, recall GOOD, recall BAD, precision GOOD/BAD,
    F1 (GOOD=1) e F1 (BAD=0), matriz de confusão (BAD=0, GOOD=1), tempo
  * previsões do teste guardadas (para recalcular qualquer métrica sem re-treinar)
  * importâncias de variáveis com o TIPO de importância (impureza, ganho, |coeficiente|)
  * % do teste fora de [0,1] depois do Min-Max (o scaler foi ajustado no treino)
  * versões das bibliotecas e hiperparâmetros EFETIVOS de cada modelo (get_params)

NÃO altera o algoritmo histórico: ordem, sementes e modelos são os do Anexo.

Uso (Windows, na pasta onde está o Fused_All.csv):
    python q1_q3_reproducao.py --skip-svm            # rápido (minutos): 6 modelos
    python q1_q3_reproducao.py --only-svm            # lento (horas): só o SVM, acrescenta ao CSV
    python q1_q3_reproducao.py --svm-test            # compara 3 configurações do SVM no BCXBC2
    python q1_q3_reproducao.py --cases BCXBC2-2024   # só um caso
Opções: --csv CAMINHO  --out PASTA  --no-mlflow

Ambos os modos (--skip-svm e --only-svm) acrescentam linhas ao mesmo ficheiro
metricas_modelos.csv, pelo que não é preciso juntar nada à mão.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception as e:                                   # pragma: no cover
    XGBClassifier = None
    print("[AVISO] xgboost não disponível:", e)

# ------------------------------------------------------------------
# ARGUMENTOS
# ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="Fused_All.csv")
ap.add_argument("--out", default="q1q3_outputs")
ap.add_argument("--cases", default="all",
                help="all ou lista separada por vírgulas: BXB2-2024,BXB2-2025,BCXBC2-2024")
ap.add_argument("--skip-svm", action="store_true")
ap.add_argument("--only-svm", action="store_true")
ap.add_argument("--svm-test", action="store_true",
                help="compara configurações do SVM no BCXBC2-2024 e sai")
ap.add_argument("--svm-config", default="B", choices=["A", "B", "C"],
                help="A=anexo atual (probability=True,max_iter=5000); "
                     "B=execução histórica (probability=False, cache_size=1000, max_iter por omissão); "
                     "C=probability=False, cache_size=1000, max_iter=5000")
ap.add_argument("--no-mlflow", action="store_true")
args = ap.parse_args()

# ------------------------------------------------------------------
# CONFIGURAÇÃO (igual ao pipeline do Anexo)
# ------------------------------------------------------------------
CASES = {
    "BXB2-2024":   dict(splice="BXB2",   year=2024, cross=1.05, good=14190, bad=14171),
    "BXB2-2025":   dict(splice="BXB2",   year=2025, cross=1.05, good=14190, bad=14206),
    "BCXBC2-2024": dict(splice="BCXBC2", year=2024, cross=2.2,  good=14167, bad=14206),
}
MIN_ROWS = 1000
CORR_THRESHOLD = 0.90
TEST_SIZE = 0.30
RANDOM_STATE = 42
IQR_K = 1.5
META_COLS = ["AssetID", "Date", "Time", "Splice Comment", "CrossSection"]
DROP_EARLY_COLS = ["CH-I-T-Pos", "CH-I-T-Neg", "CH-O-T-Neg", "CH-O-T-Pos",
                   "WH-I-T-Pos", "WH-I-T-Neg", "Wh-O-T-Neg", "Wh-O-T-Pos", "Sequence"]

SVM_CONFIGS = {
    "A": dict(probability=True, cache_size=1000, max_iter=5000),
    "B": dict(probability=False, cache_size=1000),
    "C": dict(probability=False, cache_size=1000, max_iter=5000),
}

os.makedirs(args.out, exist_ok=True)
METRICS_CSV = os.path.join(args.out, "metricas_modelos.csv")


def p(*parts):
    return os.path.join(args.out, *parts)


def normalize_splice(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if "X" in s and "TEST" not in s:
        q = s.split("X")
        if len(q) == 2:
            s = "X".join(sorted(q))
    return s


def compute_iqr_bounds(d, cols, k=IQR_K):
    b = {}
    for c in cols:
        q1, q3 = d[c].quantile([0.25, 0.75])
        i = q3 - q1
        if pd.isna(i) or i == 0:
            continue
        b[c] = (q1 - k * i, q3 + k * i)
    return b


def apply_iqr_filter(d, b):
    m = pd.Series(True, index=d.index)
    for c, (lo, hi) in b.items():
        m &= (d[c] >= lo) & (d[c] <= hi)
    return d.loc[m].copy()


def cls_counts(d):
    vc = d["is_good"].value_counts()
    return int(vc.get(1, 0)), int(vc.get(0, 0))


def make_models(svm_cfg):
    m = {
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(**SVM_CONFIGS[svm_cfg]),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=100)
    return m


def metricas(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    rec_good = tp / (tp + fn) if (tp + fn) else np.nan
    rec_bad = tn / (tn + fp) if (tn + fp) else np.nan
    prec_good = tp / (tp + fp) if (tp + fp) else np.nan
    prec_bad = tn / (tn + fn) if (tn + fn) else np.nan

    def f1(pr, rc):
        return 2 * pr * rc / (pr + rc) if (pr + rc) and not np.isnan(pr + rc) else np.nan

    return {
        "accuracy": (tp + tn) / n,
        "accuracy_equilibrada": (rec_good + rec_bad) / 2,
        "recall_GOOD": rec_good, "recall_BAD": rec_bad,
        "precision_GOOD": prec_good, "precision_BAD": prec_bad,
        "F1_GOOD": f1(prec_good, rec_good), "F1_BAD": f1(prec_bad, rec_bad),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ------------------------------------------------------------------
# CARREGAR (uma vez)
# ------------------------------------------------------------------
t_start = time.time()
print("A carregar", args.csv, "...")
raw = pd.read_csv(args.csv, low_memory=False)
raw = raw.loc[:, ~raw.columns.astype(str).str.contains("^Unnamed")]
raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
raw = raw.dropna(subset=["Date"])
raw["Splice Comment"] = raw["Splice Comment"].apply(normalize_splice)

# ------------------------------------------------------------------
# MLFLOW (opcional; base própria para não misturar com a execução histórica)
# ------------------------------------------------------------------
mlflow = None
if not args.no_mlflow:
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri("sqlite:///" + os.path.abspath(p("mlflow_reproducao.db")))
    except Exception as e:
        print("[AVISO] MLflow indisponível, a continuar sem registo:", e)
        mlflow = None

# versões
import sklearn, scipy
with open(p("versoes_ambiente.txt"), "w") as fh:
    fh.write(f"python {sys.version.split()[0]}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
             f"scipy {scipy.__version__}\nscikit-learn {sklearn.__version__}\n")
    if XGBClassifier is not None:
        import xgboost
        fh.write(f"xgboost {xgboost.__version__}\n")
    if mlflow is not None:
        fh.write(f"mlflow {mlflow.__version__}\n")


# ------------------------------------------------------------------
# PREPARAÇÃO DE UM CASO (FASE 0-7 do pipeline)
# ------------------------------------------------------------------
def prepara_caso(nome):
    c = CASES[nome]
    d = raw[(raw["Date"].dt.year == c["year"])
            & (raw["Splice Comment"] == normalize_splice(c["splice"]))
            & (raw["CrossSection"] == c["cross"])]
    lin = {"caso": nome}
    g, b = cls_counts(d)
    lin.update({"filtros_total": len(d), "filtros_GOOD": g, "filtros_BAD": b})

    summ = d.groupby(["AssetID", "is_good"]).size().unstack(fill_value=0)
    n_good = summ.loc[c["good"], 1] if c["good"] in summ.index and 1 in summ.columns else 0
    n_bad = summ.loc[c["bad"], 0] if c["bad"] in summ.index and 0 in summ.columns else 0
    assert n_good >= MIN_ROWS and n_bad >= MIN_ROWS, f"{nome}: par fixo insuficiente"
    d = d[d["AssetID"].isin([c["good"], c["bad"]])]
    g, b = cls_counts(d)
    lin.update({"assets_total": len(d), "assets_GOOD": g, "assets_BAD": b})

    d = d.drop(columns=[x for x in DROP_EARLY_COLS if x in d.columns], errors="ignore")
    d = d.drop(columns=[x for x in d.columns if "Ref" in x or "Stat" in x], errors="ignore")
    feats0 = [x for x in d.columns if x not in META_COLS + ["is_good"]
              and pd.api.types.is_numeric_dtype(d[x])]
    lin["n_variaveis_candidatas"] = len(feats0)
    lin["variaveis_candidatas"] = "|".join(feats0)
    lin["NaN_nas_candidatas"] = int(d[feats0].isna().any(axis=1).sum())

    tr_raw, te_raw = train_test_split(d, stratify=d["is_good"], test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE)
    g, b = cls_counts(tr_raw); lin.update({"treino_split_total": len(tr_raw),
                                           "treino_split_GOOD": g, "treino_split_BAD": b})
    g, b = cls_counts(te_raw); lin.update({"teste_total": len(te_raw), "teste_GOOD": g, "teste_BAD": b})

    bounds = compute_iqr_bounds(tr_raw, feats0, IQR_K)
    tr = apply_iqr_filter(tr_raw, bounds)
    g, b = cls_counts(tr); lin.update({"treino_IQR_total": len(tr), "treino_IQR_GOOD": g, "treino_IQR_BAD": b})
    g0, b0 = lin["treino_split_GOOD"], lin["treino_split_BAD"]
    lin["IQR_removido_GOOD_pct"] = 100 * (g0 - g) / g0
    lin["IQR_removido_BAD_pct"] = 100 * (b0 - b) / b0
    lin["IQR_removido_total_pct"] = 100 * (len(tr_raw) - len(tr)) / len(tr_raw)
    lin["classe_maioritaria_treino_pre_balanc"] = "GOOD" if g >= b else "BAD"

    # baseline da classe maioritária (definida no treino, antes do balanceamento) sobre o teste
    maj = 1 if g >= b else 0
    lin["baseline_maioria_accuracy"] = float((te_raw["is_good"] == maj).mean())

    tr = tr.copy()
    tr["Month"] = tr["Date"].dt.month
    chunks, meses_ok, meses_fora, meses_info = [], [], [], []
    for m, gr in tr.groupby("Month"):
        vc = gr["is_good"].value_counts()
        n_g, n_b = int(vc.get(1, 0)), int(vc.get(0, 0))
        if len(vc) < 2:
            meses_fora.append(m)
            meses_info.append({"caso": nome, "mes": m, "GOOD_antes": n_g, "BAD_antes": n_b,
                               "incluido": False, "por_classe_apos": 0})
            continue
        n = min(vc[0], vc[1])
        chunks.append(pd.concat([
            gr[gr["is_good"] == 0].sample(n=n, random_state=RANDOM_STATE),
            gr[gr["is_good"] == 1].sample(n=n, random_state=RANDOM_STATE)]))
        meses_ok.append(m)
        meses_info.append({"caso": nome, "mes": m, "GOOD_antes": n_g, "BAD_antes": n_b,
                           "incluido": True, "por_classe_apos": int(n)})
    tr = pd.concat(chunks).reset_index(drop=True)
    g, b = cls_counts(tr); lin.update({"treino_balanc_total": len(tr), "treino_balanc_GOOD": g,
                                       "treino_balanc_BAD": b})
    lin["meses_no_treino_balanceado"] = ",".join(map(str, meses_ok))
    lin["meses_excluidos_por_uma_so_classe"] = ",".join(map(str, meses_fora)) or "nenhum"
    lin["meses_com_dados_no_ano"] = ",".join(map(str, sorted(d["Date"].dt.month.unique())))

    feats1 = [x for x in tr.columns if x not in META_COLS + ["is_good", "Month"]
              and pd.api.types.is_numeric_dtype(tr[x])]
    corr = tr[feats1].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    cortadas = [x for x in upper.columns if any(upper[x] > CORR_THRESHOLD)]
    # pares que causaram o corte (para documentar)
    pares = []
    for x in upper.columns:
        for y in upper.index:
            v = upper.loc[y, x]
            if pd.notna(v) and v > CORR_THRESHOLD:
                pares.append(f"{y}~{x}:{v:.3f}")
    tr = tr.drop(columns=cortadas, errors="ignore")
    te = te_raw.drop(columns=[x for x in cortadas if x in te_raw.columns], errors="ignore")
    feats = [x for x in tr.columns if x not in META_COLS + ["is_good", "Month"]
             and pd.api.types.is_numeric_dtype(tr[x])]
    lin["variaveis_removidas_correlacao"] = "|".join(cortadas) or "nenhuma"
    lin["pares_correlacao_acima_limiar"] = "|".join(pares) or "nenhum"
    lin["n_variaveis_finais"] = len(feats)
    lin["variaveis_finais"] = "|".join(feats)

    sc = MinMaxScaler().fit(tr[feats])
    Xtr = pd.DataFrame(sc.transform(tr[feats]), columns=feats)
    Xte = pd.DataFrame(sc.transform(te[feats]), columns=feats)
    fora = ((Xte < 0) | (Xte > 1))
    lin["teste_linhas_fora_0_1_pct"] = 100 * float(fora.any(axis=1).mean())
    for f in feats:
        lin[f"teste_fora_0_1_pct__{f}"] = 100 * float(fora[f].mean())
    return lin, meses_info, Xtr, tr["is_good"].values, Xte, te["is_good"].values, te.index.values, feats, bounds


# ------------------------------------------------------------------
# SVM-TEST (comparar configurações no caso rápido)
# ------------------------------------------------------------------
if args.svm_test:
    lin, _, Xtr, ytr, Xte, yte, _, feats, _ = prepara_caso("BCXBC2-2024")
    print("Treino:", Xtr.shape, "Teste:", Xte.shape)
    linhas = []
    for cfg in ["A", "B", "C"]:
        warnings.simplefilter("ignore")
        t0 = time.time()
        m = SVC(**SVM_CONFIGS[cfg]).fit(Xtr, ytr)
        pr = m.predict(Xte)
        r = metricas(yte, pr)
        r.update({"config": cfg, "parametros": json.dumps(SVM_CONFIGS[cfg]),
                  "n_iter": str(getattr(m, "n_iter_", "")), "tempo_s": round(time.time() - t0, 1)})
        linhas.append(r)
        print(cfg, SVM_CONFIGS[cfg], f"acc={r['accuracy']:.4f} recall_BAD={r['recall_BAD']:.4f} "
              f"bal={r['accuracy_equilibrada']:.4f} n_iter={r['n_iter']} t={r['tempo_s']}s", flush=True)
    pd.DataFrame(linhas).to_csv(p("svm_teste_configuracoes_BCXBC2.csv"), index=False)
    print("Histórico (execução do utilizador, BCXBC2 SVM): acc=0.974 recall_BAD=0.998")
    sys.exit(0)

# ------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ------------------------------------------------------------------
cases = list(CASES) if args.cases == "all" else [x.strip() for x in args.cases.split(",")]
for nome in cases:
    assert nome in CASES, f"caso desconhecido: {nome}"

linhagem_csv = p("linhagem_casos.csv")
meses_csv = p("meses_treino_balanceado.csv")

for nome in cases:
    print(f"\n{'=' * 70}\n{nome}\n{'=' * 70}", flush=True)
    lin, meses_info, Xtr, ytr, Xte, yte, te_idx, feats, bounds = prepara_caso(nome)
    c = CASES[nome]
    print(f"GOOD={c['good']} BAD={c['bad']} | treino balanceado={len(ytr)} | teste={len(yte)} | "
          f"features: {feats}")
    print(f"IQR removeu GOOD {lin['IQR_removido_GOOD_pct']:.1f}% / BAD {lin['IQR_removido_BAD_pct']:.1f}% do treino | "
          f"baseline maioria no teste = {lin['baseline_maioria_accuracy']:.4f} | "
          f"teste fora de [0,1]: {lin['teste_linhas_fora_0_1_pct']:.2f}% das linhas")

    # linhagem e meses: só se este modo não for --only-svm (evita duplicar)
    if not args.only_svm:
        # formato longo (caso, item, valor): colunas fixas, por isso pode acrescentar-se caso a caso
        pd.DataFrame([{"caso": nome, "item": k, "valor": v} for k, v in lin.items() if k != "caso"]).to_csv(
            linhagem_csv, mode="a", index=False, header=not os.path.exists(linhagem_csv))
        pd.DataFrame(meses_info).to_csv(meses_csv, mode="a", index=False,
                                        header=not os.path.exists(meses_csv))
        pd.DataFrame([{"caso": nome, "variavel": k, "limite_inferior": v[0], "limite_superior": v[1]}
                      for k, v in bounds.items()]).to_csv(
            p(f"limites_IQR_{nome}.csv"), index=False)

    models = make_models(args.svm_config)
    if args.skip_svm:
        models.pop("SVM", None)
    if args.only_svm:
        models = {"SVM": models["SVM"]}

    if mlflow is not None:
        mlflow.set_experiment(f"{nome}_reproducao")

    for mname, model in models.items():
        warnings.simplefilter("ignore")
        t0 = time.time()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        dt = time.time() - t0
        r = metricas(yte, pred)
        r.update({
            "caso": nome, "modelo": mname, "n_treino": len(ytr), "n_teste": len(yte),
            "n_GOOD_teste": int((yte == 1).sum()), "n_BAD_teste": int((yte == 0).sum()),
            "baseline_maioria_accuracy": lin["baseline_maioria_accuracy"],
            "tempo_s": round(dt, 1),
            "hiperparametros_efetivos": json.dumps(model.get_params(), default=str, sort_keys=True),
            "svm_config": args.svm_config if mname == "SVM" else "",
        })
        pd.DataFrame([r]).to_csv(METRICS_CSV, mode="a", index=False,
                                 header=not os.path.exists(METRICS_CSV))
        np.savez_compressed(p(f"pred_{nome}_{mname}.npz"), y_true=yte, y_pred=pred, index_teste=te_idx)

        # importâncias com tipo explícito
        imp_rows = []
        if mname in ("GradientBoosting", "RandomForest", "DecisionTree"):
            for f, v in zip(feats, model.feature_importances_):
                imp_rows.append({"caso": nome, "modelo": mname, "variavel": f, "valor": v,
                                 "tipo_importancia": "reducao_media_de_impureza (feature_importances_)"})
        elif mname == "XGBoost":
            tipo = getattr(model, "importance_type", None)
            for f, v in zip(feats, model.feature_importances_):
                imp_rows.append({"caso": nome, "modelo": mname, "variavel": f, "valor": v,
                                 "tipo_importancia": f"feature_importances_ com importance_type={tipo}"})
            booster = model.get_booster()
            for it in ["weight", "gain", "cover", "total_gain"]:
                sc_ = booster.get_score(importance_type=it)
                for i, f in enumerate(feats):
                    imp_rows.append({"caso": nome, "modelo": mname, "variavel": f,
                                     "valor": sc_.get(f, sc_.get(f"f{i}", 0.0)),
                                     "tipo_importancia": f"booster.get_score({it}) (não normalizado)"})
        elif mname == "LogisticRegression":
            for f, v in zip(feats, model.coef_[0]):
                imp_rows.append({"caso": nome, "modelo": mname, "variavel": f, "valor": abs(v),
                                 "tipo_importancia": "|coeficiente| (variáveis em Min-Max)"})
                imp_rows.append({"caso": nome, "modelo": mname, "variavel": f, "valor": v,
                                 "tipo_importancia": "coeficiente com sinal"})
        if imp_rows:
            pd.DataFrame(imp_rows).to_csv(p("importancias.csv"), mode="a", index=False,
                                          header=not os.path.exists(p("importancias.csv")))

        if mlflow is not None:
            try:
                with mlflow.start_run(run_name=f"{mname}__reproducao"):
                    mlflow.log_param("caso", nome)
                    mlflow.log_param("good_asset", c["good"]); mlflow.log_param("bad_asset", c["bad"])
                    mlflow.log_param("classe_positiva", "GOOD=1")
                    for k, v in model.get_params().items():
                        mlflow.log_param(f"modelo__{k}", v)
                    for k in ["accuracy", "accuracy_equilibrada", "recall_GOOD", "recall_BAD",
                              "precision_GOOD", "precision_BAD", "F1_GOOD", "F1_BAD"]:
                        mlflow.log_metric(k, float(r[k]))
            except Exception as e:
                print("[AVISO] MLflow falhou:", e)

        print(f"  {mname:<18} acc={r['accuracy']:.3f}  bal={r['accuracy_equilibrada']:.3f}  "
              f"recall_GOOD={r['recall_GOOD']:.3f}  recall_BAD={r['recall_BAD']:.3f}  "
              f"F1(GOOD)={r['F1_GOOD']:.3f}  ({dt:.1f}s)", flush=True)

print(f"\nCONCLUÍDO em {(time.time() - t_start) / 60:.1f} min. Resultados em {os.path.abspath(args.out)}")
