
# Eigenentwicklung im Rahmen dieser Arbeit

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd

# Configure plotting style
plt.style.use('ggplot')

# Base directories
dir_script = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.join(dir_script, '..'))
r5_dir = os.path.join(data_root, 'r5.2')

# Target CSV files
csv_names = [
    'decoy_file.csv', 'device.csv', 'email.csv', 'file.csv',
    'http.csv', 'logon.csv', 'psychometric.csv'
]
csv_paths = {name: os.path.join(r5_dir, name) for name in csv_names}
ldap_dir = os.path.join(r5_dir, 'LDAP')

# Verify existence
available = {n: p for n, p in csv_paths.items() if os.path.isfile(p)}
if not available:
    raise RuntimeError(f"Keine Haupt-CSV-Dateien gefunden in {r5_dir}")
for n, p in available.items():
    print(f"Gefunden: {n} @ {p}")
# Add LDAP files
ldap_files = []
if os.path.isdir(ldap_dir):
    for root, _, files in os.walk(ldap_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                ldap_files.append(os.path.join(root, f))
    print(f"LDAP-Dateien: {len(ldap_files)} gefunden")
all_paths = list(available.values()) + ldap_files

# --------------------------------------
# Helpers (robust against empty dtypes)
# --------------------------------------

def safe_numeric_describe(ddf: dd.DataFrame):
    num_cols = list(ddf.select_dtypes(include=['number']).columns)
    if not num_cols:
        return pd.DataFrame()  # no numeric columns
    num_dd = ddf[num_cols]
    # describe -> transpose so rows = features
    desc = num_dd.describe(percentiles=[0.5, 0.95]).compute().T
    # rename percentiles
    desc = desc.rename(columns={'50%': 'median', '95%': 'q95'})
    # skewness per column
    try:
        sk = num_dd.skew().compute()
        desc['skewness'] = sk
    except Exception as e:
        print(f"Warnung: Skewness konnte nicht berechnet werden: {e}")
        desc['skewness'] = np.nan
    return desc


def safe_missing(ddf: dd.DataFrame):
    try:
        miss = (ddf.isna().mean() * 100).compute()
    except Exception as e:
        print(f"Warnung: Missing-Rate konnte nicht berechnet werden: {e}")
        miss = pd.Series(dtype=float)
    return miss


def safe_categorical_summary(ddf: dd.DataFrame, top_n=5):
    cat_cols = list(ddf.select_dtypes(include=['object', 'category']).columns)
    if not cat_cols:
        return pd.DataFrame()
    out = {}
    # total rows once
    try:
        total = int(ddf.shape[0].compute())
    except Exception:
        total = None
    for col in cat_cols:
        try:
            vc = ddf[col].value_counts().nlargest(top_n).compute()
            card = int(ddf[col].nunique().compute())
            if total is None or total == 0:
                tail_pct = np.nan
            else:
                top_sum = int(vc.sum())
                tail = max(total - top_sum, 0)
                tail_pct = 100.0 * tail / total
            row = {'cardinality': card, 'long_tail_pct': tail_pct}
            for i, (k, v) in enumerate(vc.items(), start=1):
                row[f'top_{i}'] = f"{k} ({int(v)})"
            out[col] = row
        except Exception as e:
            print(f"Warnung: Kategorie-Zusammenfassung für '{col}' fehlgeschlagen: {e}")
    return pd.DataFrame(out).T


def safe_time_pattern(ddf: dd.DataFrame):
    # flexible Spaltennamen für Zeitstempel
    candidates = ['timestamp', 'date', 'time', 'datetime', 'logon_time', 'event_time']
    time_col = next((c for c in candidates if c in ddf.columns), None)
    if not time_col:
        return None
    try:
        ts = dd.to_datetime(ddf[time_col], errors='coerce')
        office = ts.dt.hour.between(8, 18).mean().compute() * 100
        weekend = ts.dt.weekday.isin([5, 6]).mean().compute() * 100
        return pd.Series({'office_hours_pct': office, 'weekend_pct': weekend})
    except Exception as e:
        print(f"Warnung: Zeitmuster konnten nicht berechnet werden: {e}")
        return None


# --------------------------------------
# Main: generate reports (streaming)
# --------------------------------------
reports_dir = os.path.join(data_root, 'reports')
os.makedirs(reports_dir, exist_ok=True)

numeric_reports = []
missing_reports = []
category_reports = []
time_reports = []

for p in all_paths:
    name = os.path.splitext(os.path.basename(p))[0]
    print(f"\n-- Bearbeite: {name}")
    ddf = dd.read_csv(p, assume_missing=True)

    # numeric
    num_desc = safe_numeric_describe(ddf)
    if not num_desc.empty:
        # add file name namespace to columns
        num_desc.columns = [f"{name}:{c}" for c in num_desc.columns]
        num_desc.index.name = 'feature'
        numeric_reports.append(num_desc)

    # missing
    miss = safe_missing(ddf)
    if not miss.empty:
        missing_reports.append(miss.rename(name))

    # categorical
    cat_df = safe_categorical_summary(ddf)
    if not cat_df.empty:
        cat_df.columns = [f"{name}:{c}" for c in cat_df.columns]
        cat_df.index.name = 'feature'
        category_reports.append(cat_df)

    # time
    tp = safe_time_pattern(ddf)
    if tp is not None:
        tp.name = name
        time_reports.append(tp)

# Save conditionally
if numeric_reports:
    pd.concat(numeric_reports, axis=1).to_csv(os.path.join(reports_dir, 'numeric_summary.csv'))
if missing_reports:
    pd.concat(missing_reports, axis=1).to_csv(os.path.join(reports_dir, 'missing_summary.csv'))
if category_reports:
    pd.concat(category_reports, axis=1).to_csv(os.path.join(reports_dir, 'categorical_summary.csv'))
if time_reports:
    pd.DataFrame(time_reports).to_csv(os.path.join(reports_dir, 'time_summary.csv'))

print("\nBerichte gespeichert in 'reports/' (nur erzeugt, wenn Daten vorlagen)")