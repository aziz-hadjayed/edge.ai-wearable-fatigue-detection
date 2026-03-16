"""
integre_temp_hum2.py
--------------------
Intègre les colonnes `temperature_amb` et `Humid` du fichier
`pre_task_survey.xlsx` dans `datasets/data_selected1.csv`.

Logique de jointure :
- Le questionnaire pré-tâche est rempli UNE FOIS par participant et par session.
- Les conditions ambiantes (temp, humidité) sont stables pendant toute la session.
- La correspondance se fait sur : participant_id ↔ ID  et  session_id ↔ Session.
- Session "01" → session_id=1, "02" → session_id=2, "03" → session_id=3.
- Les nouvelles colonnes sont insérées juste après `session_id`.

Dépendances : stdlib uniquement (re, csv, zipfile) — pas de pandas requis.
"""

import csv
import re
import zipfile
from pathlib import Path

# ─── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
XLSX_PATH = BASE_DIR / "datasets/archive/fatigueset/pre_task_survey.xlsx"
CSV_IN    = BASE_DIR / "datasets/data_selected1.csv"
CSV_OUT   = BASE_DIR / "datasets/data_selected1.csv"


# ─── 1. Lecture du xlsx (regex, sans ET.parse) ─────────────────────────────────
def _read_text(zf, path):
    with zf.open(path) as f:
        return f.read().decode("utf-8")


def load_survey(xlsx_path: Path) -> dict:
    """
    Retourne un dict {(participant_id: str, session_id: str): (temp: str, humid: str)}
    où les clés sont normalisées : "1", "2", ..., "12" et "1", "2", "3".
    """
    with zipfile.ZipFile(xlsx_path) as zf:
        ss_xml   = _read_text(zf, "xl/sharedStrings.xml")
        ws_xml   = _read_text(zf, "xl/worksheets/sheet1.xml")

    # Extraire les chaînes partagées
    shared = re.findall(r'<t(?:\s[^>]*)?>([^<]*)</t>', ss_xml)

    def cell_val(cell_xml, col_letter):
        """Extrait la valeur de la cellule de colonne `col_letter` dans le XML d'une ligne."""
        pattern = rf'r="{re.escape(col_letter)}\d+"[^>]*>(?:<f[^/]*/?>)?(?:<v>([^<]*)</v>)?'
        m = re.search(pattern, cell_xml)
        if not m or m.group(1) is None:
            return None
        raw = m.group(1)
        # Déterminer si c'est une chaîne partagée (t="s")
        cell_m = re.search(rf'<c r="{re.escape(col_letter)}\d+"[^>]*>', cell_xml)
        if cell_m and 't="s"' in cell_m.group(0):
            return shared[int(raw)]
        return raw

    # Extraire toutes les lignes (sauf header row 1)
    rows_xml = re.findall(r'<row r="(\d+)"[^>]*>(.*?)</row>', ws_xml, re.DOTALL)

    # Correspondance colonnes : A=ID, B=Session, N=temperature_amb, O=Humid
    survey_map = {}
    for row_num, row_body in rows_xml:
        if row_num == "1":
            continue  # skip header

        pid     = cell_val(row_body, "A")
        session = cell_val(row_body, "B")
        temp    = cell_val(row_body, "N")
        humid   = cell_val(row_body, "O")

        if pid is None or session is None:
            continue

        try:
            pid_norm     = str(int(float(pid)))
            session_norm = str(int(float(session)))
        except (ValueError, TypeError):
            continue

        survey_map[(pid_norm, session_norm)] = (
            temp  if temp  is not None else "",
            humid if humid is not None else "",
        )

    return survey_map


# ─── 2. Intégration dans le CSV ────────────────────────────────────────────────
def integrate(csv_in: Path, csv_out: Path, survey_map: dict) -> None:
    with open(csv_in, newline="", encoding="utf-8") as f:
        reader   = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        print("⚠  Le fichier CSV d'entrée est vide.")
        return

    header = all_rows[0]

    # Vérifier si les colonnes sont déjà présentes
    if "temperature_amb" in header and "Humid" in header:
        print("ℹ  Les colonnes temperature_amb et Humid sont déjà présentes — aucune modification.")
        return

    try:
        idx_pid     = header.index("participant_id")
        idx_session = header.index("session_id")
    except ValueError as e:
        raise ValueError(f"Colonne manquante dans {csv_in}: {e}")

    # Insérer après session_id
    insert_pos = idx_session + 1
    new_header = header[:insert_pos] + ["temperature_amb", "Humid"] + header[insert_pos:]
    new_rows   = [new_header]

    matched         = 0
    not_found_keys  = set()

    for row in all_rows[1:]:
        # Assurer que la ligne a assez de colonnes
        while len(row) < len(header):
            row.append("")

        pid_raw     = row[idx_pid].strip()
        session_raw = row[idx_session].strip()

        try:
            pid_norm     = str(int(float(pid_raw)))     if pid_raw     else ""
            session_norm = str(int(float(session_raw))) if session_raw else ""
        except (ValueError, TypeError):
            pid_norm = session_raw = ""

        key = (pid_norm, session_norm)
        if key in survey_map:
            temp, humid = survey_map[key]
            matched += 1
        else:
            temp, humid = "", ""
            if pid_norm and session_norm:
                not_found_keys.add(key)

        new_rows.append(row[:insert_pos] + [temp, humid] + row[insert_pos:])

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    total_data = len(all_rows) - 1
    print(f"✅ Intégration terminée !")
    print(f"   Lignes de données     : {total_data:,}")
    print(f"   Lignes avec temp/humid: {matched:,}")
    if not_found_keys:
        print(f"   ⚠  Clés sans correspondance : {sorted(not_found_keys)}")
    print(f"   Fichier de sortie     : {csv_out}")


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📖 Lecture du questionnaire pre_task_survey.xlsx ...")
    survey_map = load_survey(XLSX_PATH)

    print(f"   {len(survey_map)} enregistrements trouvés :\n")
    print(f"   {'Participant':>12}  {'Session':>7}  {'Temp(°C)':>8}  {'Humid(%)':>8}")
    print("   " + "-" * 45)
    for (pid, ses), (temp, humid) in sorted(survey_map.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        print(f"   {pid:>12}  {ses:>7}  {temp:>8}  {humid:>8}")

    print()
    print("🔗 Intégration dans data_selected1.csv ...")
    integrate(CSV_IN, CSV_OUT, survey_map)
