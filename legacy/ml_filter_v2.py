"""
ml_filter_v2.py — фильтрация associations.xml по локальной магнитуде ML.

Ключевые отличия от ml_filter.py (v1):
  - R вычисляется по формуле S-P: R = Vp*Vs/(Vp-Vs) * ΔT(S-P) — для каждой
    станции независимо, grid search не нужен
  - Амплитуда измеряется в окне S-волны (S_time .. S_time+WIN_SEC), а не P-волны
  - Приоритет горизонтальных компонент: E > N > Z
  - Кэш амплитуд по умолчанию: amps_wa.csv (Wood-Anderson, требует WA simulation)
  - Формула 5б ожидает WA-амплитуды в нм; при запуске без --wood-anderson
    используется amps_disp.csv — тогда результаты ML будут занижены
  - События без ML удаляются (не сохраняются как в v1); --keep-no-ml — старое поведение

Формула (Дягилев et al. 2023, Терско-Каспийский прогиб, формула 5б):
    ML = lg(A_nm) + 1.024·lg(R) + 0.001648·R − 1.889 + S

Формула — для ОДНОЙ станции. ML события = медиана ML_sta по всем станциям.
A — максимальная амплитуда WA-сейсмограммы в нм (после remove_response + WA sim).
R — гипоцентральное расстояние, км: R = Vp·Vs/(Vp-Vs) · ΔT(S-P).
S — станционная поправка (0 по умолчанию).

Запуск:
    python core/ml_filter_v2.py --info
    python core/ml_filter_v2.py --ml-threshold 1.0 --validate
    python core/ml_filter_v2.py --ml-threshold 1.0 --wood-anderson --metadata-dir metadata
    python core/ml_filter_v2.py --ml-threshold 1.0 --sta-corrections sta_corr.csv --validate
"""

import argparse
import copy
import csv
import gc
import glob
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ── Пути по умолчанию ─────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(_ROOT, 'data-in-memory', 'gpu_splimit_45_march', 'assoc_output_lim', 'associations.xml')
DEFAULT_ASSOC_OUT = None   # None → рядом с assoc-in, имя associations_ml<thr>.xml
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'geofiles')
DEFAULT_STA_DIR   = os.path.join(_ROOT, 'json')
# v2: WA-амплитуды по умолчанию (формула 5б калибрована по WA)
DEFAULT_CACHE_AMP = os.path.join(_ROOT, 'amps_wa.csv')
DEFAULT_METADATA  = os.path.join(_ROOT, 'metadata')
DEFAULT_THRESHOLD = 1.0
DEFAULT_YEAR      = 2024
DEFAULT_MONTH     = 4

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'

ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

# ── Константы формулы (Терско-Каспийский прогиб, формула 5б) ─────────────────

ML_B_LOG     = 1.024     # коэффициент при lg(R)
ML_B_LIN     = 0.001648  # коэффициент при R (км)
ML_C         = -1.889    # калибровочная константа (для WA-амплитуд в нм)
AMP_SCALE_NM = 1e9       # перевод м→нм (при remove_response output='DISP' или WA)

# ── Скоростная модель (Wadati, S-P формула) ───────────────────────────────────

VP        = 6.0      # скорость P-волны, км/с  (= Vs * 1.72)
VS        = 3.4883   # скорость S-волны, км/с  (Vp/Vs = 1.72, регион Терско-Каспий)

WIN_SEC   = 5.0      # длина окна после S-прихода для измерения амплитуды, с

# PAZ Wood-Anderson (Richter 1935; T0=0.8s, h=0.8, V=2800)
_PAZ_WA = {
    'poles': [(-6.283185307 + 4.712388980j),
              (-6.283185307 - 4.712388980j)],
    'zeros': [0j, 0j],
    'gain': 1.0,
    'sensitivity': 2800.0,
}

# ── Приоритет компонент: горизонтальные первые (E/N), Z последним ─────────────
# ML измеряется на горизонтальных компонентах согласно NMSOP и статье
_COMP_PRIORITY = ('E', 'N', 'Z')


# ── Вспомогательные функции ───────────────────────────────────────────────────

def parse_time(s):
    if not s:
        return None
    s = s.strip().rstrip('Z').replace('T', ' ')
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def sp_distance_km(p_time, s_time):
    """Гипоцентральное расстояние из S-P времени: R = Vp*Vs/(Vp-Vs) * ΔT(S-P)."""
    dt = (s_time - p_time).total_seconds()
    if dt <= 0.0:
        return None
    return VP * VS / (VP - VS) * dt


# ── Парсинг XML ───────────────────────────────────────────────────────────────

def parse_xml(tree):
    """Возвращает list of (ev_elem, pub_id, picks_p, picks_s).
    picks_p/picks_s: {station: datetime}"""
    ns = BED_NS
    root = tree.getroot()
    result = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id = ev.get('publicID', '')
        picks_p, picks_s = {}, {}
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf = pick.find(f'{{{ns}}}waveformID')
            if wf is None:
                continue
            sta = wf.get('stationCode', '').strip()
            if not sta:
                continue
            t_el     = pick.find(f'{{{ns}}}time/{{{ns}}}value')
            phase_el = pick.find(f'{{{ns}}}phaseHint')
            if t_el is None:
                continue
            t = parse_time(t_el.text)
            if t is None:
                continue
            phase = phase_el.text.strip() if phase_el is not None else 'P'
            if phase == 'P':
                picks_p[sta] = t
            elif phase == 'S':
                picks_s[sta] = t
        result.append((ev, pub_id, picks_p, picks_s))
    return result


# ── Вычисление R по S-P формуле для всех станций события ─────────────────────

def compute_r_dict(picks_p, picks_s):
    """
    Возвращает {station: R_km} только для станций с обоими пиками P и S
    и корректным S-P интервалом.
    """
    r_dict = {}
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_distance_km(picks_p[sta], picks_s[sta])
        if r is not None and r > 0.0:
            r_dict[sta] = r
    return r_dict


# ── Станционные поправки ──────────────────────────────────────────────────────

def load_sta_corrections(path):
    """CSV с колонками station,S. Возвращает dict {station: float}."""
    corr = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sta = row.get('station', '').strip()
            val = row.get('S', row.get('correction', '')).strip()
            if sta and val:
                try:
                    corr[sta] = float(val)
                except ValueError:
                    pass
    return corr


# ── Кэш амплитуд ─────────────────────────────────────────────────────────────

def save_amp_cache(amplitudes, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ev_idx', 'sta', 'A'])
        for (ev_idx, sta), A in amplitudes.items():
            w.writerow([ev_idx, sta, '' if A is None else A])


def load_amp_cache(path):
    cache = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cache[(int(row['ev_idx']), row['sta'])] = float(row['A']) if row['A'] else None
    return cache


# ── Индекс и поиск форм волн ──────────────────────────────────────────────────

def _parse_file_times(fname):
    parts = fname.split('__')
    if len(parts) != 3:
        return None, None
    try:
        return (datetime.strptime(parts[1].rstrip('Z'), '%Y%m%dT%H%M%S'),
                datetime.strptime(parts[2].rstrip('Z'), '%Y%m%dT%H%M%S'))
    except ValueError:
        return None, None


def build_waveform_index(waveform_dir):
    """
    dict: station -> list of (t_start, t_end, fpath).
    Приоритет компонент: E > N > Z (горизонтальные для ML).
    """
    index = {}
    if not os.path.isdir(waveform_dir):
        return index
    for sta in os.listdir(waveform_dir):
        sta_path = os.path.join(waveform_dir, sta)
        if not os.path.isdir(sta_path):
            continue
        by_comp = defaultdict(list)
        for fname in os.listdir(sta_path):
            parts = fname.split('__')
            if len(parts) != 3:
                continue
            dot_parts = parts[0].split('.')
            if len(dot_parts) < 4:
                continue
            comp = dot_parts[3][-1].upper()
            if comp not in _COMP_PRIORITY:
                continue
            t_start, t_end = _parse_file_times(fname)
            if t_start is None:
                continue
            by_comp[comp].append((t_start, t_end, os.path.join(sta_path, fname)))
        for comp in _COMP_PRIORITY:   # E > N > Z
            if comp in by_comp:
                index[sta] = sorted(by_comp[comp], key=lambda x: x[0])
                break
    return index


def find_waveform_file(index, sta, t):
    for t_start, t_end, fpath in index.get(sta, []):
        if t_start <= t < t_end:
            return fpath
    return None


# ── Извлечение амплитуд S-волны ───────────────────────────────────────────────

def extract_amplitudes(events_with_r, waveform_index, inventory=None, wood_anderson=False):
    """
    Для каждого события и станции с r_dict:
      - Открывает файл по S-времени (горизонтальная компонента)
      - Берёт окно [S_time .. S_time + WIN_SEC]
      - Опционально симулирует WA
      - Возвращает dict: (ev_idx, sta) -> амплитуда (float или None) в метрах.

    events_with_r: list of (ev_idx, pub_id, picks_p, picks_s, r_dict)
    """
    from obspy import read as obspy_read, UTCDateTime

    # Группируем по файлу: {fpath: [(ev_idx, sta, s_time), ...]}
    groups = defaultdict(list)
    for ev_idx, _, picks_p, picks_s, r_dict in events_with_r:
        for sta in r_dict:
            s_time = picks_s.get(sta)
            if s_time is None:
                continue
            fpath = find_waveform_file(waveform_index, sta, s_time)
            if fpath is None:
                continue
            groups[fpath].append((ev_idx, sta, s_time))

    amplitudes = {}
    n = len(groups)
    for i, (fpath, triples) in enumerate(groups.items()):
        if i % 20 == 0:
            print(f"  Амплитуды: {i}/{n} файлов...", end='\r', flush=True)
        try:
            st = obspy_read(fpath)
            if not st:
                continue
            tr = st[0]
            actual_sr = tr.stats.sampling_rate
            if actual_sr not in (50.0, 100.0, 80.0, 200.0):
                actual_sr = 100.0 if actual_sr > 75 else 50.0
                tr.stats.sampling_rate = actual_sr

            if inventory is not None:
                nyq = actual_sr / 2.0
                pre_filt = (0.5, 1.0, nyq * 0.85, nyq * 0.95)
                try:
                    tr.remove_response(inventory=inventory, output='DISP',
                                       pre_filt=pre_filt, water_level=60)
                    if wood_anderson:
                        tr.simulate(paz_simulate=_PAZ_WA, paz_remove=None)
                except Exception:
                    tr.filter('highpass', freq=1.0)
            else:
                tr.filter('highpass', freq=1.0)

            for ev_idx, sta, s_time in triples:
                try:
                    s_utc   = UTCDateTime(s_time)
                    end_utc = s_utc + WIN_SEC
                    tr_sig  = tr.slice(s_utc, end_utc)
                    if tr_sig is None or len(tr_sig.data) == 0:
                        amplitudes[(ev_idx, sta)] = None
                        continue
                    A = float(np.max(np.abs(tr_sig.data)))
                    amplitudes[(ev_idx, sta)] = A if A > 0.0 else None
                except Exception:
                    amplitudes[(ev_idx, sta)] = None
            del tr, st
        except Exception:
            pass
        gc.collect()

    print(f"  Амплитуды: {n}/{n} — готово.         ")
    return amplitudes


# ── Вычисление ML по формуле 5б ──────────────────────────────────────────────

def compute_event_ml(ev_idx, r_dict, amplitudes, sta_corr, amp_scale=AMP_SCALE_NM):
    """
    Формула 5б (Дягилев et al. 2023) для одной станции:
        ML_sta = lg(A_nm) + 1.024·lg(R) + 0.001648·R − 1.889 + S

    R берётся из r_dict (из S-P формулы).
    A_nm = amplitudes[(ev_idx, sta)] * amp_scale.

    Итог: медиана ML_sta по всем станциям. Возвращает (ML, n_sta).
    """
    ml_vals = []
    for sta, r_km in r_dict.items():
        A = amplitudes.get((ev_idx, sta))
        if A is None or A <= 0.0 or r_km <= 0.0:
            continue
        A_nm = A * amp_scale
        if A_nm <= 0.0:
            continue
        S  = sta_corr.get(sta, 0.0)
        ml = (math.log10(A_nm)
              + ML_B_LOG * math.log10(r_km)
              + ML_B_LIN * r_km
              + ML_C
              + S)
        ml_vals.append(ml)
    if not ml_vals:
        return None, 0
    return float(np.median(ml_vals)), len(ml_vals)


# ── Фильтрация XML ────────────────────────────────────────────────────────────

def filter_xml(tree, ml_by_idx, threshold, out_path, keep_no_ml=False):
    """
    Удаляет события с ML < threshold.

    keep_no_ml=False (v2 по умолчанию): события без ML удаляются.
    keep_no_ml=True  (поведение v1):    события без ML сохраняются.
    """
    tree2     = copy.deepcopy(tree)
    ns        = BED_NS
    root2     = tree2.getroot()
    ev_params = root2.find(f'.//{{{ns}}}eventParameters') or root2
    events2   = ev_params.findall(f'{{{ns}}}event')

    total = len(events2)
    kept = removed = no_ml_kept = no_ml_removed = 0
    for i, ev_elem in enumerate(events2):
        ml = ml_by_idx.get(i)
        if ml is None:
            if keep_no_ml:
                no_ml_kept += 1
                kept += 1
            else:
                ev_params.remove(ev_elem)
                no_ml_removed += 1
                removed += 1
        elif ml >= threshold:
            kept += 1
        else:
            ev_params.remove(ev_elem)
            removed += 1

    tree2.write(out_path, encoding='unicode', xml_declaration=True)
    return total, kept, removed, no_ml_kept, no_ml_removed


# ── Статистика распределения ML ───────────────────────────────────────────────

def print_distribution(ml_values):
    vals   = sorted(v for v in ml_values if v is not None)
    n_none = sum(1 for v in ml_values if v is None)
    no_ml_fate = "УДАЛЕНЫ при фильтрации (--keep-no-ml чтобы сохранить)"
    print(f"\n  Событий с оценкой ML:  {len(vals)}")
    print(f"  Событий без оценки ML: {n_none}  ({no_ml_fate})")
    if not vals:
        return
    n    = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение ML (b_log={ML_B_LOG}, b_lin={ML_B_LIN}, c={ML_C}):")
    for p in pcts:
        print(f"    P{p:2d}: {vals[max(0, int(p / 100 * n) - 1)]:.2f}")
    print(f"    min: {vals[0]:.2f}   max: {vals[-1]:.2f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global VP, VS
    parser = argparse.ArgumentParser(
        description='Фильтрация associations.xml по ML v2 (S-P дистанция, S-волна, горизонталь)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--assoc-in',        default=DEFAULT_ASSOC_IN,
                        help='Входной XML ассоциатора')
    parser.add_argument('--assoc-out',       default=DEFAULT_ASSOC_OUT,
                        help='Выходной XML. По умолчанию рядом с assoc-in')
    parser.add_argument('--waveforms',       default=DEFAULT_WAVEFORMS,
                        help='Корень папки с формами волн')
    parser.add_argument('--stations-dir',    default=DEFAULT_STA_DIR,
                        help='Папка с json/station_*.json (не используется в v2, '
                             'оставлен для совместимости)')
    parser.add_argument('--sta-corrections', default=None,
                        help='CSV с колонками station,S — станционные поправки')
    parser.add_argument('--ml-threshold',    type=float, default=DEFAULT_THRESHOLD,
                        help='Порог ML: оставить события с ML >= X')
    parser.add_argument('--keep-no-ml',     action='store_true', default=False,
                        help='Сохранять события без ML (поведение v1). '
                             'По умолчанию такие события удаляются.')
    parser.add_argument('--info',            action='store_true',
                        help='Только показать распределение ML, без записи файлов')
    parser.add_argument('--cache-amp',       default=DEFAULT_CACHE_AMP,
                        help='CSV-кэш амплитуд. v2 по умолчанию: amps_wa.csv. '
                             'ВАЖНО: для правильных ML нужны WA-амплитуды. '
                             'Если кэш от v1 (amps_disp.csv) — ML будут занижены.')
    parser.add_argument('--metadata-dir',    default=DEFAULT_METADATA,
                        help='Папка с FDSNStationXML (.xml) для remove_response')
    parser.add_argument('--wood-anderson',   action='store_true', default=False,
                        help='Симулировать Wood-Anderson после remove_response(DISP). '
                             'Требуется для формулы 5б. Нужен --metadata-dir.')
    parser.add_argument('--vp',              type=float, default=VP,
                        help='Скорость P-волны, км/с (для S-P формулы)')
    parser.add_argument('--vs',              type=float, default=VS,
                        help='Скорость S-волны, км/с (для S-P формулы)')
    parser.add_argument('--validate',        action='store_true',
                        help='Запустить validate_associator.py на выходном XML')
    parser.add_argument('--year',            type=int, default=DEFAULT_YEAR)
    parser.add_argument('--month',           type=int, default=DEFAULT_MONTH)
    args = parser.parse_args()

    # ── Заголовок ─────────────────────────────────────────────────────────────
    k = args.vp * args.vs / (args.vp - args.vs)
    print(f"Входной XML:   {args.assoc_in}")
    print(f"Кэш амплитуд:  {args.cache_amp}")
    wa_s = "ДА" if args.wood_anderson else "нет"
    print(f"Wood-Anderson: {wa_s}")
    print(f"S-P формула:   R = {args.vp:.1f}·{args.vs:.1f}/({args.vp:.1f}-{args.vs:.1f}) · ΔT = {k:.2f} · ΔT(S-P) км")
    print(f"Окно амплитуды: S_time .. S_time + {WIN_SEC}s (S-волна, горизонталь E>N>Z)")
    no_ml_s = "сохранять (--keep-no-ml)" if args.keep_no_ml else "удалять"
    print(f"Без ML:         {no_ml_s}")
    print(f"Формула:        ML = lg(A_nm) + {ML_B_LOG}·lg(R) + {ML_B_LIN}·R + ({ML_C}) + S\n")

    # ── Парсинг XML ───────────────────────────────────────────────────────────
    print("Парсинг XML...")
    tree          = ET.parse(args.assoc_in)
    events_parsed = parse_xml(tree)
    print(f"  Событий: {len(events_parsed)}")

    sta_corr = {}
    if args.sta_corrections:
        sta_corr = load_sta_corrections(args.sta_corrections)
        print(f"  Станционных поправок: {len(sta_corr)}")

    # ── Вычисление R из S-P формулы для каждой станции ───────────────────────
    print("\nВычисление R по S-P формуле (Vp={:.1f}, Vs={:.1f})...".format(args.vp, args.vs))
    VP = args.vp
    VS = args.vs

    events_with_r = []
    n_has_r = 0
    for i, (_, pub_id, picks_p, picks_s) in enumerate(events_parsed):
        r_dict = compute_r_dict(picks_p, picks_s)
        events_with_r.append((i, pub_id, picks_p, picks_s, r_dict))
        if r_dict:
            n_has_r += 1
    print(f"  Событий с S-P расстоянием: {n_has_r} / {len(events_parsed)}")

    # Статистика распределения R
    all_r = [r for _, _, _, _, rd in events_with_r for r in rd.values()]
    if all_r:
        print(f"  R: min={min(all_r):.1f} км,  медиана={float(np.median(all_r)):.1f} км,  max={max(all_r):.1f} км")

    # ── StationXML / inventory ─────────────────────────────────────────────────
    inventory = None
    if args.wood_anderson or args.metadata_dir:
        meta_dir = args.metadata_dir
        if meta_dir and not os.path.isdir(meta_dir):
            meta_dir = os.path.join(_ROOT, meta_dir)
        if meta_dir and os.path.isdir(meta_dir):
            from obspy import read_inventory
            print(f"\nЗагрузка StationXML из {meta_dir}...")
            for fpath in sorted(glob.glob(os.path.join(meta_dir, '*.xml'))):
                try:
                    inv = read_inventory(fpath)
                    inventory = inv if inventory is None else inventory + inv
                except Exception:
                    pass
            if inventory is None:
                print("  Предупреждение: StationXML не найдены, используется highpass")
            else:
                n_sta = sum(len(net) for net in inventory.networks)
                print(f"  Загружено: {n_sta} станций")

    if args.wood_anderson and inventory is None:
        print("  Предупреждение: нет StationXML — WA simulation пропущена")

    # ── Амплитуды ─────────────────────────────────────────────────────────────
    if args.cache_amp and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Загружено пар: {len(amplitudes)}  с амплитудой: {n_amp}")
        if args.cache_amp == os.path.join(_ROOT, 'amps_disp.csv'):
            print("  ВНИМАНИЕ: используется amps_disp.csv (смещение грунта), "
                  "а не WA-амплитуды. ML будут систематически занижены ~1.5–2 ед.")
    else:
        print("\nПостроение индекса форм волн (компонент E > N > Z)...")
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        print("\nИзвлечение амплитуд S-волны (окно [S .. S+{:.0f}s])...".format(WIN_SEC))
        use_wa = args.wood_anderson and inventory is not None
        amplitudes = extract_amplitudes(events_with_r, waveform_index,
                                        inventory=inventory,
                                        wood_anderson=use_wa)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        if args.cache_amp:
            save_amp_cache(amplitudes, args.cache_amp)
            print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление ML ─────────────────────────────────────────────────────────
    print("\nВычисление ML по формуле 5б (медиана по станциям)...")
    ml_by_idx = {}
    for ev_idx, pub_id, picks_p, picks_s, r_dict in events_with_r:
        ml, n_sta = compute_event_ml(ev_idx, r_dict, amplitudes, sta_corr)
        ml_by_idx[ev_idx] = ml

    print_distribution(list(ml_by_idx.values()))

    if args.info or args.ml_threshold is None:
        return

    # ── Фильтрация и запись XML ───────────────────────────────────────────────
    thr = args.ml_threshold
    if args.assoc_out:
        out_path = args.assoc_out
    else:
        tag      = f"{thr:.1f}".replace('-', 'm').replace('.', 'p')
        out_path = os.path.join(os.path.dirname(args.assoc_in),
                                f'associations_ml{tag}.xml')

    total, kept, removed, no_ml_kept, no_ml_removed = filter_xml(
        tree, ml_by_idx, thr, out_path, keep_no_ml=args.keep_no_ml
    )

    print(f"\n{'=' * 60}")
    print(f"Порог ML >= {thr}")
    print(f"  Входных событий:  {total}")
    kept_str = f"ML >= {thr}"
    if no_ml_kept:
        kept_str += f"  +  {no_ml_kept} без оценки"
    print(f"  Сохранено:        {kept}  ({kept_str})")
    removed_str = str(removed)
    if no_ml_removed:
        removed_str += f"  (из них {no_ml_removed} без оценки ML)"
    print(f"  Удалено:          {removed_str}")
    print(f"  Выходной файл:    {out_path}")

    if args.validate:
        print(f"\n{'─' * 60}")
        print("Запуск validate_associator.py...")
        script = os.path.join(_ROOT, 'core', 'validate_associator.py')
        subprocess.run([sys.executable, script,
                        '--assoc', out_path,
                        '--year',  str(args.year),
                        '--month', str(args.month)])


if __name__ == '__main__':
    main()
