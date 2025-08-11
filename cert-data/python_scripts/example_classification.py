
# Eigenentwicklung im Rahmen dieser Arbeit

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
from sklearn.calibration import calibration_curve





# -----------------------------
# Config
# -----------------------------
np.random.seed(42)
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
SAVEFIG_KW = dict(dpi=200, bbox_inches="tight")

def simulate_budget(df_in, alarms_per_day):
    df = df_in.copy()
    df["pred"] = 0
    def mark_top(g):
        k = min(alarms_per_day, len(g))
        if k > 0:
            top_idx = g["score"].nlargest(k).index
            g.loc[top_idx, "pred"] = 1
        return g
    # pro Tag die Top-K Scores alarmieren
    df = df.groupby(df["starttime"].dt.date, group_keys=False).apply(mark_top)
    tp = ((df.pred == 1) & (df.insider == 1)).sum()
    fp = ((df.pred == 1) & (df.insider == 0)).sum()
    fn = ((df.pred == 0) & (df.insider == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"alarms/day": alarms_per_day, "precision": precision, "recall": recall, "TP": tp, "FP": fp, "FN": fn}
def daily_recall_report(df_eval, K, save_prefix=f"{FIGDIR}/daily_recall"):
    df = df_eval.copy()
    df["day"] = pd.to_datetime(df["starttime"]).dt.normalize()
    df["pred"] = 0

    # Top-K je Tag markieren (ohne Schwelle, wie in deiner Budget-Variante)
    def mark_top(g):
        k = min(K, len(g))
        if k > 0:
            idx = g["score"].nlargest(k).index
            g.loc[idx, "pred"] = 1
        return g

    df = df.groupby("day", group_keys=False).apply(mark_top)

    # Tages-Stats
    day_stats = df.groupby("day").apply(lambda g: pd.Series({
        "positives": int((g.insider == 1).sum()),
        "tp":        int(((g.pred == 1) & (g.insider == 1)).sum()),
        "fp":        int(((g.pred == 1) & (g.insider == 0)).sum()),
        "alarms":    int(g.pred.sum())
    })).reset_index()

    # Recall je Tag (nur für Tage mit Positiven sinnvoll)
    day_stats["recall_day"] = np.where(
        day_stats["positives"] > 0,
        day_stats["tp"] / day_stats["positives"],
        np.nan
    )
    pos_days = day_stats[day_stats["positives"] > 0]

    share_all_caught = float((pos_days["recall_day"] == 1.0).mean()) if len(pos_days) else np.nan
    mean_recall_posdays = float(pos_days["recall_day"].mean()) if len(pos_days) else np.nan
    median_recall_posdays = float(pos_days["recall_day"].median()) if len(pos_days) else np.nan

    # Speichern & kurze Zusammenfassung
    out_csv = Path(save_prefix + f"_K{K}.csv")
    day_stats.to_csv(out_csv, index=False)

    print(f"\nTages-Recall für K={K}:")
    print(f"  Tage mit Positiven: {len(pos_days)} von {day_stats.shape[0]} Gesamt-Tagen")
    print(f"  Anteil Tage mit Recall=1.0 (alle Positiven erwischt): {share_all_caught:.3f}")
    print(f"  Ø Recall (nur Tage mit Positiven): {mean_recall_posdays:.3f} | Median: {median_recall_posdays:.3f}")
    print(f"  Datei gespeichert: {out_csv}")

    return day_stats




# 1) Daten einlesen & Label binär machen
SCRIPT = Path(__file__).resolve()
CERT = SCRIPT.parents[1]
BASE = CERT / "r5.2" / "ExtractedData"
CSV_DAY = BASE / "dayr5.2.csv"

print(f"Lade Daten: {CSV_DAY}")
df = pd.read_csv(CSV_DAY)
df["starttime"] = pd.to_datetime(df["starttime"], unit="s", errors="coerce")
df["date"] = df["starttime"].dt.date
# Label binär
df["insider"] = (df["insider"] > 0).astype(int)

# 2) Numerische Features auswählen (ohne Label und IDs)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "insider" in num_cols:
    num_cols.remove("insider")
X_num = df[num_cols]
y = df["insider"].values

# 3) Feature-Selection (Variance + Correlation)
vt = VarianceThreshold(threshold=0.01)
X_vt = pd.DataFrame(
    vt.fit_transform(X_num),
    columns=[c for c, keep in zip(num_cols, vt.get_support()) if keep],
    index=df.index
)
# Korrelationen berechnen auf Trainingsuniversum (wird unten erstellt); hier zunächst global -> Hinweis in Text
corr = X_vt.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
X_sel = X_vt.drop(columns=to_drop)
print(f"Features: original={len(num_cols)}, nach VarThresh={X_vt.shape[1]}, nach CorrDrop={X_sel.shape[1]}")

# 4) User-Holdout: 80% der User für Training, 20% für finalen Test
all_users = df["user"].unique()
train_users = np.random.choice(all_users, size=int(0.8 * len(all_users)), replace=False)
mask_train = df["user"].isin(train_users)

order = df.loc[mask_train, "starttime"].sort_values().index
X_train_full = X_sel[mask_train]
y_train_full = y[mask_train]
X_test = X_sel[~mask_train]
y_test = y[~mask_train]
df_test = df[~mask_train].copy()  # für SOC-Simulation / TTD

# 5) Sliding-Window-CV auf Training mit OOF-Probabilities
print("\nZeitliche CV (TimeSeriesSplit, n_splits=5):")
tscv = TimeSeriesSplit(n_splits=5)
params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",  # PR-AUC als Val-Metrik
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# OOF-Container
oof_pred = np.zeros(len(X_train_full))
cv_aupr = []
cv_roc = []
cv_best_iter = []

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_full)):
    X_tr, y_tr = X_train_full.iloc[tr_idx], y_train_full[tr_idx]
    X_va, y_va = X_train_full.iloc[va_idx], y_train_full[va_idx]

    # leichte Noise-Injektion
    X_tr_noisy = X_tr + np.random.normal(0, 0.01, size=X_tr.shape)
    # (optional) leichter Drift auf einem bekannten Zählfeature (falls vorhanden)
    if "day_n_logon" in X_tr_noisy.columns:
        X_tr_noisy.loc[:, "day_n_logon"] *= np.random.uniform(0.9, 1.1, size=len(X_tr_noisy))

    # Imbalance-Handling pro Fold
    neg, pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
    spw = float(neg / max(pos, 1))
    fold_params = params.copy()
    fold_params["scale_pos_weight"] = spw

    dtr = xgb.DMatrix(X_tr_noisy, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

    bst_cv = xgb.train(
        fold_params, dtr,
        num_boost_round=600,
        evals=[(dva, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    p_va = bst_cv.predict(dva)
    oof_pred[va_idx] = p_va

    aupr = average_precision_score(y_va, p_va)
    auc = roc_auc_score(y_va, p_va)
    cv_aupr.append(aupr)
    cv_roc.append(auc)
    cv_best_iter.append(bst_cv.best_iteration)
    print(f" Fold {fold + 1}: AUPRC={aupr:.4f} | ROC-AUC={auc:.4f} | best_iter={bst_cv.best_iteration}")

# CV-Zusammenfassung und OOF-Plots
print(f"\nCV Mittelwerte: AUPRC={np.mean(cv_aupr):.4f}±{np.std(cv_aupr):.4f} | ROC-AUC={np.mean(cv_roc):.4f}±{np.std(cv_roc):.4f}")
mean_best_iter = int(np.median(cv_best_iter))
print(f"Gewählte Anzahl Bäume (Median best_iteration): {mean_best_iter}")

# PR-Kurve & ROC aus OOF
fpr, tpr, _ = roc_curve(y_train_full, oof_pred)
prec, rec, _ = precision_recall_curve(y_train_full, oof_pred)
aupr_oof = average_precision_score(y_train_full, oof_pred)
auc_oof = roc_auc_score(y_train_full, oof_pred)

plt.figure()
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC (OOF) — AUC={auc_oof:.3f}")
plt.savefig(FIGDIR / "cv_roc_oof.png", **SAVEFIG_KW)
plt.close()

plt.figure()
plt.plot(rec, prec, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall (OOF) — AP={aupr_oof:.3f}")
plt.savefig(FIGDIR / "cv_pr_oof.png", **SAVEFIG_KW)
plt.close()

# Optimaler Threshold nach F2 auf OOF (Recall-fokussiert)
def best_threshold_fbeta(y_true, scores, beta=2.0):
    qs = np.linspace(0.0, 1.0, 501)
    thresh = np.quantile(scores, qs)
    best_f, best_t = -1.0, 0.5
    from sklearn.metrics import fbeta_score
    for t in thresh:
        y_hat = (scores >= t).astype(int)
        f = fbeta_score(y_true, y_hat, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f

thr_f2, f2_oof = best_threshold_fbeta(y_train_full, oof_pred, beta=2.0)
print(f"Gewählter Schwellenwert nach F2 (OOF): t={thr_f2:.4f} (F2={f2_oof:.4f})")

# 6) Finales Training auf gesamtem Training-Set
final_params = params.copy()
neg, pos = np.sum(y_train_full == 0), np.sum(y_train_full == 1)
final_params["scale_pos_weight"] = float(neg / max(pos, 1))

dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
dtest = xgb.DMatrix(X_test, label=y_test)

bst_final = xgb.train(
    final_params, dtrain_full,
    num_boost_round=mean_best_iter,
    verbose_eval=False
)

# 7) Test-Evaluation & Plots
y_pred_prob = bst_final.predict(dtest)
df_eval = df_test[["starttime", "insider"]].copy()
df_eval["score"] = y_pred_prob
y_pred_05 = (y_pred_prob >= 0.5).astype(int)
y_pred_f2 = (y_pred_prob >= thr_f2).astype(int)

print("\nFinal Test Evaluation (t=0.5):")
print(classification_report(y_test, y_pred_05, zero_division=0))
print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

# ROC/PR auf Test
fpr_t, tpr_t, _ = roc_curve(y_test, y_pred_prob)
prec_t, rec_t, _ = precision_recall_curve(y_test, y_pred_prob)
auc_t = roc_auc_score(y_test, y_pred_prob)
ap_t = average_precision_score(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr_t, tpr_t, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC (Test) — AUC={auc_t:.3f}")
plt.savefig(FIGDIR / "test_roc.png", **SAVEFIG_KW)
plt.close()

plt.figure()
plt.plot(rec_t, prec_t, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall (Test) — AP={ap_t:.3f}")
plt.savefig(FIGDIR / "test_pr.png", **SAVEFIG_KW)
plt.close()

# Confusion-Matrixen für t=0.5 und t=thr_f2
cm_05 = confusion_matrix(y_test, y_pred_05)
cm_f2 = confusion_matrix(y_test, y_pred_f2)

def plot_cm(cm, title, fname):
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["Normal", "Insider"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(FIGDIR / fname, **SAVEFIG_KW)
    plt.close()

plot_cm(cm_05, "Confusion Matrix (t=0.5)", "cm_t05.png")
plot_cm(cm_f2, f"Confusion Matrix (t={thr_f2:.2f})", "cm_tf2.png")




budgets = [25, 50, 100]
res_df = pd.DataFrame([simulate_budget(df_eval, b) for b in budgets])
print("\nSOC-Budget (Test):")
print(res_df.to_string(index=False))

res_df.to_csv(FIGDIR / "soc_budget_results.csv", index=False)

plt.figure()
plt.plot(res_df["alarms/day"], res_df["precision"], marker='o', label='Precision')
plt.plot(res_df["alarms/day"], res_df["recall"], marker='o', label='Recall')
plt.xlabel("Alarme pro Tag (Budget)")
plt.ylabel("Wert")
plt.title("SOC Budget-Simulation (Test)")
plt.legend()
plt.savefig(FIGDIR / "soc_budget_precision_recall.png", **SAVEFIG_KW)
plt.close()
# --- Ende SOC-Budget ---




# --- Tages-Recall-Report (nach SOC-Budget-Block einfügen) ---

# 7b) Regelbasierte Baseline (train-only Quantil-Regeln) + Metriken

# Idee: sehr einfache Heuristik ohne Lernen.
# - Für "Zähl"-Features (day_* mit 'n_' oder 'count') nutze 99%-Quantil (train)
# - Für "Bytes"-Features (day_* mit 'bytes') nutze 99.5%-Quantil (train)
# - Flag = 1, wenn mind. eine Regel greift; Score = Anteil überschrittener Regeln

# 1) Spalten finden (robust gegen fehlende Namen)
rb_cols_count = [c for c in df.columns if c.startswith("day_") and ("n_" in c or "count" in c)]
rb_cols_bytes = [c for c in df.columns if c.startswith("day_") and ("bytes" in c)]

# 2) Schwellen nur auf TRAIN-Usern fitten (keine Leckage)
q_count = df.loc[mask_train, rb_cols_count].quantile(0.99, numeric_only=True) if rb_cols_count else pd.Series(dtype=float)
q_bytes = df.loc[mask_train, rb_cols_bytes].quantile(0.995, numeric_only=True) if rb_cols_bytes else pd.Series(dtype=float)

def rules_score(frame: pd.DataFrame) -> pd.Series:
    # Anzahl aktiver Regeln je Zeile
    trig = pd.Series(0, index=frame.index, dtype=int)
    denom = 0
    if len(q_count) > 0:
        hit_count = (frame[q_count.index] >= q_count).sum(axis=1)
        trig = trig.add(hit_count, fill_value=0)
        denom += len(q_count)
    if len(q_bytes) > 0:
        hit_bytes = (frame[q_bytes.index] >= q_bytes).sum(axis=1)
        trig = trig.add(hit_bytes, fill_value=0)
        denom += len(q_bytes)
    # Score: Anteil getriggerter Regeln (0..1), Pred: mind. eine Regel
    score = (trig / max(denom, 1)).clip(0, 1)
    return score


# 3) Scores/Preds auf TEST
rule_score_test = rules_score(df.loc[~mask_train])
rule_pred_test = (rule_score_test > 0).astype(int)

# 4) Performance wie beim Modell
print("\nRegelbasierte Baseline — Testmetriken:")
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
print(classification_report(y_test, rule_pred_test, zero_division=0))

# Für ROC/PR nutzen wir den kontinuierlichen Rule-Score
try:
    auc_rb = roc_auc_score(y_test, rule_score_test)
    ap_rb = average_precision_score(y_test, rule_score_test)
    print(f"ROC-AUC (Baseline): {auc_rb:.6f}")
    print(f"PR-AUC  (Baseline): {ap_rb:.6f}")
except Exception as e:
    print(f"AUC/AP (Baseline) nicht berechenbar: {e}")

# Optional: ROC/PR-Plots für die Baseline
fpr_rb, tpr_rb, _ = roc_curve(y_test, rule_score_test)
prec_rb, rec_rb, _ = precision_recall_curve(y_test, rule_score_test)

plt.figure()
plt.plot(fpr_rb, tpr_rb, linewidth=2)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (Regel-Baseline)")
plt.savefig(FIGDIR / "baseline_roc.png", **SAVEFIG_KW)
plt.close()

plt.figure()
plt.plot(rec_rb, prec_rb, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall (Regel-Baseline)")
plt.savefig(FIGDIR / "baseline_pr.png", **SAVEFIG_KW)
plt.close()

# Konfusionsmatrix (t=0/1 auf Rule-Score)
cm_rb = confusion_matrix(y_test, rule_pred_test)
def plot_cm_rb(cm, title, fname):
    plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = ["Normal", "Insider"]; ticks = np.arange(len(classes))
    plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.savefig(FIGDIR / fname, **SAVEFIG_KW); plt.close()
plot_cm_rb(cm_rb, "Confusion Matrix — Regel-Baseline", "baseline_cm.png")

# 5) SOC-Budget auf Regel-Baseline (gleiche Funktion wie oben nutzen)
df_eval_rule = df_test[["starttime", "insider"]].copy()
df_eval_rule["score"] = rule_score_test.values  # wichtiger: kontinuierlicher Score
budgets_rb = [25, 50, 100]
res_rb = pd.DataFrame([simulate_budget(df_eval_rule, b) for b in budgets_rb])
print("\nSOC-Budget (Regel-Baseline, Test):")
print(res_rb.to_string(index=False))
res_rb.to_csv(FIGDIR / "soc_budget_baseline_results.csv", index=False)

# 6) Tages-Recall-Quote für Baseline (falls du den Report schon definiert hast)
try:
    for K in budgets_rb:
        _ = daily_recall_report(df_eval_rule, K, save_prefix=f"{FIGDIR}/daily_recall_baseline")
except NameError:
    # Falls daily_recall_report in deinem Skript nicht definiert ist, diesen Block ignorieren
    pass




# --- SOC-Budget-Simulation (25/50/100) ---
df_eval = df_test[["starttime", "insider"]].copy()
df_eval["score"] = y_pred_prob




# Für K in {25, 50, 100} ausgeben
for K in [25, 50, 100]:
    _ = daily_recall_report(df_eval, K)
# --- Ende Tages-Recall-Report ---

# 9) Kalibrierung (Test)
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
plt.figure()
plt.plot([0,1],[0,1], linestyle='--')
plt.plot(prob_pred, prob_true, marker='o')
plt.xlabel('Vorhergesagte Wahrscheinlichkeit')
plt.ylabel('Beobachtete Häufigkeit')
plt.title('Kalibrierungskurve (Test)')
plt.savefig(FIGDIR / "calibration_curve_test.png", **SAVEFIG_KW)
plt.close()

# 10) Feature-Importance (Gain)
imp_gain = bst_final.get_score(importance_type='gain')
if len(imp_gain) > 0:
    imp_series = pd.Series(imp_gain).sort_values(ascending=False).head(20)
    plt.figure(figsize=(8, max(4, len(imp_series)*0.3)))
    plt.barh(imp_series.index, imp_series.values)
    plt.gca().invert_yaxis()
    plt.xlabel('Gain')
    plt.title('Top-20 Feature-Importances (Gain)')
    plt.savefig(FIGDIR / "feature_importance_gain_top20.png", **SAVEFIG_KW)
    plt.close()
    imp_series.to_csv(FIGDIR / "feature_importance_gain_top20.csv")

# 11) Time-To-Detect (TTD) pro Insider-Fenster (Test)
# Annahme: Insider = 1 an zusammenhängenden Tagen bildet ein Fenster
# Wir werten die erste Detektion (per t=F2-Threshold) innerhalb jedes Fensters aus.

df_ttd = df_test[["user", "starttime", "insider"]].copy()
df_ttd["pred"] = y_pred_f2
# Fenster-ID pro User: immer wenn insider von 0->1 wechselt, erhöhen

df_ttd = df_ttd.sort_values(["user", "starttime"]).reset_index(drop=True)
df_ttd["insider_shift"] = df_ttd.groupby("user")["insider"].shift(1).fillna(0)
df_ttd["new_window"] = ((df_ttd["insider"] == 1) & (df_ttd["insider_shift"] == 0)).astype(int)
df_ttd["win_id"] = df_ttd.groupby("user")["new_window"].cumsum()

# Nur positive Fenster
pos_windows = df_ttd[df_ttd["insider"] == 1].groupby(["user", "win_id"])
rows = []
for (u, wid), g in pos_windows:
    if len(g) == 0:
        continue
    start = g["starttime"].min()
    hits = g[g["pred"] == 1]
    det_time = hits["starttime"].min() if len(hits) > 0 else pd.NaT
    ttd_days = (det_time - start).days if pd.notna(det_time) else np.nan
    rows.append({"user": u, "win_id": wid, "start": start, "detected_at": det_time, "ttd_days": ttd_days})

ttd_df = pd.DataFrame(rows)
ttd_df.to_csv(FIGDIR / "ttd_by_window.csv", index=False)

# Plot TTD
if not ttd_df.empty and ttd_df["ttd_days"].notna().any():
    plt.figure()
    plt.hist(ttd_df["ttd_days"].dropna(), bins=20)
    plt.xlabel("Time-to-Detect (Tage)")
    plt.ylabel("Anzahl Fenster")
    hit_rate = (ttd_df["ttd_days"].notna().mean()) * 100
    plt.title(f"TTD pro Insider-Fenster (Trefferquote={hit_rate:.1f}%)")
    plt.savefig(FIGDIR / "ttd_hist.png", **SAVEFIG_KW)
    plt.close()

# 12) (Optional) SHAP-Summary – wird übersprungen, wenn SHAP nicht installiert ist
try:
    import shap
    # Warnungen unterdrücken
    shap.logger.setLevel('ERROR')
    explainer = shap.TreeExplainer(bst_final)
    # Achtung: aus Speichergründen ggf. sampeln
    X_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(FIGDIR / "shap_summary_test.png", **SAVEFIG_KW)
    plt.close()
except Exception as e:
    print(f"SHAP-Summary übersprungen: {e}")

# 13) Parameter- & Metrics-Summary speichern
summary = {
    "n_train_users": int(len(train_users)),
    "n_test_users": int(len(all_users) - len(train_users)),
    "n_features_final": int(X_sel.shape[1]),
    "cv_aupr_mean": float(np.mean(cv_aupr)),
    "cv_roc_mean": float(np.mean(cv_roc)),
    "best_iter_median": int(mean_best_iter),
    "test_auc": float(auc_t),
    "test_ap": float(ap_t),
    "threshold_f2": float(thr_f2)
}
print("Test-Zeitraum:", df_test["starttime"].min(), "→", df_test["starttime"].max())
print("Einzigartige Tage im Test:", df_test["starttime"].dt.normalize().nunique())

pd.Series(summary).to_csv(FIGDIR / "training_summary.csv")
print("\nFinal Test Evaluation (t=F2):")
from sklearn.metrics import classification_report
print(classification_report(y_test, (y_pred_prob >= thr_f2).astype(int), zero_division=0))

print(f"\nFertige Grafiken & Tabellen gespeichert in: {FIGDIR}")