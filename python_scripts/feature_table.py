# Create the CSV file the user requested based on the earlier table.
import pandas as pd
from pathlib import Path

rows = [
    {
        "CSV": "logon.csv",
        "Raw fields (r5.2)": "id, date, user, pc, activity",
        "Derived per-event features (r5.2)": "(keine direkten Event-Features; Nutzung via 'act' und spätere Aggregation, z. B. Logon-Counts nach Zeit & PC-Typ)",
        "Notes": "activity ∈ {Logon, Logoff}; Zeitklassifikation über 'time' (1=Workhour, 2=After‑hours, 3=Weekend, 4=Weekend+After‑hours); PC‑Bezug via from_pc() (0=own,1=shared,2=other,3=supervisor)."
    },
    {
        "CSV": "device.csv",
        "Raw fields (r5.2)": "id, date, user, pc, content (file_tree), activity",
        "Derived per-event features (r5.2)": "usb_dur, file_tree_len",
        "Notes": "usb_dur=Zeit bis zum nächsten Disconnect (Sek.); file_tree_len=Anzahl Verzeichnisse im semikolon‑getrennten file_tree."
    },
    {
        "CSV": "http.csv",
        "Raw fields (r5.2)": "id, date, user, pc, url/fname, content",
        "Derived per-event features (r5.2)": "http_type, url_len, url_depth, http_c_len, http_c_nwords",
        "Notes": "http_type∈{1=other,2=socnet,3=cloud,4=job,5=leak,6=hack}; url_depth=Anzahl '/'−2; content=Schlüsselwörter."
    },
    {
        "CSV": "email.csv",
        "Raw fields (r5.2)": "id, date, user, pc, to, cc, bcc, from, activity, size, att, content",
        "Derived per-event features (r5.2)": "send_mail, receive_mail, n_des, n_atts, Xemail, n_exdes, n_bccdes, exbccmail, email_size, email_text_slen, email_text_nwords, e_att_other/e_att_comp/e_att_pho/e_att_doc/e_att_txt/e_att_exe, e_att_sother/e_att_scomp/e_att_spho/e_att_sdoc/e_att_stxt/e_att_sexe",
        "Notes": "send_mail=1 bei activity='Send', receive_mail=1 bei 'Receive'/'View'; Xemail=empfängt/sendet an externe Domain (!= dtaa.com); e_att_* zählen Typen (other, comp, photo, doc, text, exe) und e_att_s* deren Größen."
    },
    {
        "CSV": "file.csv",
        "Raw fields (r5.2)": "id, date, user, pc, url/fname, activity, to, from, content",
        "Derived per-event features (r5.2)": "file_type, file_len, file_nwords, disk, file_depth, file_act, to_usb, from_usb",
        "Notes": "file_type∈{1=other,2=archive,3=image,4=document,5=text,6=executable}; disk (0=netz/sonst,1=C:\\,2=R:\\); file_act∈{1=open,2=copy,3=write,4=delete}; to_usb/from_usb aus den Booleans 'to'/'from'."
    },
]

df = pd.DataFrame(rows, columns=["CSV", "Raw fields (r5.2)", "Derived per-event features (r5.2)", "Notes"])
out_path = Path("cert_r52_event_features_from_code.csv")
df.to_csv(out_path, index=False)
out_path.as_posix()
