"""
associator_v4.py — секторная ассоциация с объединением результатов.

Логика:
  1. Западный сектор (13 станций, lon ≤ 41.1°E + GRYR overlap) → association/west/
  2. Восточный сектор (8 станций, lon ≥ 41.1°E)                → association/east/
  3. Объединение с дедупликацией зоны перекрытия:
     если одно и то же событие найдено в обоих секторах (разница во времени
     <= DEDUP_TOLERANCE), оставляем запись с большим числом ассоциированных
     станций (количество picks в QuakeML объекте события).
  4. Итог → association/merged/associations.xml

Станции перекрытия: GRYR (41.1°E) — входит в оба сектора.
Западный: ANN SUKR GLDR TMNR SPGR SRGR GOYR SOC MRNR VSLR GUZR ERBR LABN GRYR
Восточный: GRYR DOMR SHA1 BEYR PYA1 GOFR NCK ZEI
"""

import os
import sys
import csv
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from obspy import Catalog
from obspy import read_events
from EQTransformer.utils.associator import run_associator_v2


# ─── Конфигурация ────────────────────────────────────────────────────────────

SOURCE_DIR  = os.path.join(_ROOT, 'data-in-memory', 'output_gpu')
INPUT_DIR   = os.path.join(_ROOT, 'data-in-memory', 'association_gpu')
OUTPUT_BASE = os.path.join(_ROOT, 'data-in-memory', 'association_gpu', 'sectors')

SECTORS = {
    'west': ['ANN', 'SUKR', 'GLDR', 'TMNR', 'SPGR', 'SRGR', 'GOYR',
             'SOC', 'MRNR', 'VSLR', 'GUZR', 'ERBR', 'LABN', 'GRYR'],
    'east': ['GRYR', 'DOMR', 'SHA1', 'BEYR', 'PYA1',  'ZEI'],
}

# 'NCK', 'GOFR'

DET_THR = 0.7
P_THR   = 0.3
S_THR   = 0.2
KEEP_PS = True

START_TIME = '2024-01-01 00:00:00.000'
END_TIME   = '2024-02-01 00:00:00.000'

# Допуск для дедупликации: события в пределах N секунд — одно событие
DEDUP_TOLERANCE = 30.0

# ─────────────────────────────────────────────────────────────────────────────


def _passes(row):
    try:
        det = float(row.get('detection_probability', 0) or 0)
    except ValueError:
        return False
    if det < DET_THR:
        return False

    p_time = row.get('p_arrival_time', '').strip()
    try:
        p_prob = float(row.get('p_probability', 0) or 0)
    except ValueError:
        p_prob = 0.0
    has_p = bool(p_time) and p_time.lower() != 'none' and p_prob >= P_THR

    s_time = row.get('s_arrival_time', '').strip()
    try:
        s_prob = float(row.get('s_probability', 0) or 0)
    except ValueError:
        s_prob = 0.0
    has_s = bool(s_time) and s_time.lower() != 'none' and s_prob >= S_THR

    return (has_p and has_s) if KEEP_PS else (has_p or has_s)


def _prepare_station(src, dst):
    kept = 0
    with open(src, newline='', encoding='utf-8') as fin, \
         open(dst, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if _passes(row):
                writer.writerow(row)
                kept += 1
    return kept


def run_sector(sector_name, stations):
    """Подготавливает staging CSV и запускает ассоциатор для одного сектора."""
    print(f"\n{'='*60}")
    print(f"Сектор: {sector_name.upper()}  ({', '.join(stations)})")
    print('='*60)

    sector_input  = os.path.join(INPUT_DIR,   sector_name)
    sector_output = os.path.join(OUTPUT_BASE, sector_name)

    if os.path.isdir(sector_input):
        shutil.rmtree(sector_input)

    found_any = False
    for st in stations:
        src = os.path.join(SOURCE_DIR, st, f"{st.lower()}.csv")
        dst_dir = os.path.join(sector_input, st)
        dst     = os.path.join(dst_dir, "X_prediction_results.csv")
        if os.path.exists(src):
            os.makedirs(dst_dir, exist_ok=True)
            kept = _prepare_station(src, dst)
            print(f"  {st}: {kept} детекций после фильтра")
            found_any = True
        else:
            print(f"  {st}: нет данных (пропускаем)")

    if not found_any:
        print(f"  Сектор {sector_name}: нет данных, пропускаем ассоциатор.")
        return None

    os.makedirs(sector_output, exist_ok=True)
    print(f"\nЗапуск run_associator_v2 для сектора {sector_name}...")
    run_associator_v2(
        input_dir=sector_input,
        output_dir=sector_output,
        start_time=START_TIME,
        end_time=END_TIME,
        moving_window=60,
        consider_combination=False,
        pair_n=3,
    )
    return sector_output


def _origin_time(event):
    orig = event.preferred_origin() or (event.origins[0] if event.origins else None)
    return float(orig.time) if orig else None


def merge_sector_catalogs(west_dir, east_dir, merged_dir):
    """
    Объединяет два секторных каталога. Для событий в зоне перекрытия
    (разница во времени <= DEDUP_TOLERANCE сек) оставляет запись с
    большим числом ассоциированных станций (len(event.picks)).
    """
    def _load(sector_dir):
        if sector_dir is None:
            return Catalog()
        xml = os.path.join(sector_dir, "associations.xml")
        if not os.path.exists(xml):
            print(f"  associations.xml не найден: {xml}")
            return Catalog()
        return read_events(xml)

    west_cat = _load(west_dir)
    east_cat = _load(east_dir)

    print(f"\n{'='*60}")
    print(f"Объединение каталогов")
    print(f"  Запад: {len(west_cat)} событий")
    print(f"  Восток: {len(east_cat)} событий")

    merged     = []
    east_times = [_origin_time(ev) for ev in east_cat]
    used_east  = set()
    n_west     = len(west_cat)
    dup_count  = 0
    PROGRESS_STEP = 500

    for idx_w, ev_w in enumerate(west_cat):
        t_w   = _origin_time(ev_w)
        n_w   = len(ev_w.picks)
        match = None

        for i, t_e in enumerate(east_times):
            if i in used_east:
                continue
            if t_w is not None and t_e is not None and abs(t_w - t_e) <= DEDUP_TOLERANCE:
                match = (i, east_cat[i])
                break

        if match is not None:
            i_e, ev_e = match
            n_e = len(ev_e.picks)
            kept = ev_w if n_w >= n_e else ev_e
            sector = 'west' if n_w >= n_e else 'east'
            dup_count += 1
            print(f"  Дубль ±{DEDUP_TOLERANCE:.0f}s: запад={n_w} ст., восток={n_e} ст. → оставляем {sector}")
            merged.append(kept)
            used_east.add(i_e)
        else:
            merged.append(ev_w)

        if (idx_w + 1) % PROGRESS_STEP == 0 or (idx_w + 1) == n_west:
            pct = (idx_w + 1) / n_west * 100
            print(f"  [прогресс] {idx_w+1}/{n_west} ({pct:.0f}%)  дублей найдено: {dup_count}", flush=True)

    for i, ev_e in enumerate(east_cat):
        if i not in used_east:
            merged.append(ev_e)

    merged_cat = Catalog(events=merged)
    os.makedirs(merged_dir, exist_ok=True)
    out_xml = os.path.join(merged_dir, "associations.xml")
    merged_cat.write(out_xml, format="QUAKEML")

    print(f"\nОбъединённый каталог: {len(merged_cat)} событий → {out_xml}")
    return merged_cat


if __name__ == "__main__":
    west_out  = run_sector('west', SECTORS['west'])
    east_out  = run_sector('east', SECTORS['east'])

    merged_dir = os.path.join(OUTPUT_BASE, 'merged')
    merge_sector_catalogs(west_out, east_out, merged_dir)
