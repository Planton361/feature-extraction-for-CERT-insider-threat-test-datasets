from pathlib import Path
import csv
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Pfad zum answers-Hauptordner anpassen
answers_root = Path(r'..\r5.2\answers')

# Regex für Release-Präfix
re_rel = re.compile(r'^(r\d+(?:\.\d+)?)', re.I)

# Speicher für eindeutige Events pro Release
unique_events = defaultdict(set)

# CSV-Dateien durchsuchen
for csv_file in answers_root.rglob('*.csv'):
    if csv_file.name.lower() == 'insiders.csv':
        continue

    match = re_rel.match(csv_file.name)
    if not match:
        continue
    release = match.group(1).lower()

    with csv_file.open('r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            event_type = row[0].strip().lower()
            date_val = row[2].strip()
            time_val = row[3].strip()
            user_val = row[4].strip().lower()
            timestamp = f"{date_val} {time_val}"
            event_id = (user_val, timestamp, event_type)
            unique_events[release].add(event_id)

# Daten in Listen umwandeln
releases = []
counts = []
for rel, events in sorted(unique_events.items(),
                          key=lambda kv: tuple(int(x) for x in kv[0][1:].split('.'))):
    releases.append(rel)
    counts.append(len(events))

# Konsole
print("Eindeutige Insider-Events pro Release (User+Datum/Zeit+Event-Typ):")
for rel, cnt in zip(releases, counts):
    print(f"{rel:5s} : {cnt}")

# Grafik erstellen
plt.figure(figsize=(8, 5))
bars = plt.bar(releases, counts, color='steelblue')
plt.title("Eindeutige Insider-Events pro CERT-Release")
plt.xlabel("Release-Version")
plt.ylabel("Anzahl eindeutiger Insider-Events")

# Werte direkt über die Balken schreiben
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(cnt), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
