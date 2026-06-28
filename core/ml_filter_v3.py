"""
ml_filter_v3.py — быстрая настройка порогов ML на выборке событий.

Отличия от v2:
  - Работает на подмножестве: 10 каталожных событий (SAMPLE_CATALOG_EVENTS) +
    N_BETWEEN случайных событий между каждой соседней парой каталожных.
  - Отдельный кэш: amps_sample.csv (ключ = publicID события, не ev_idx).
  - Первый запуск: строит выборку + вычисляет амплитуды (несколько минут).
  - Повторные запуски (кэш есть): только меняет порог — мгновенно.
  - Recall считается только по тем каталожным событиям, которые нашлись в XML
    (в пределах MATCH_WIN секунд от времени каталога).

Формула (Дягилев et al. 2023, Терско-Каспийский прогиб, формула 5б):
    ML = lg(A_nm) + 1.024·lg(R) + 0.001648·R − 1.889 + S
A   = смещение грунта в нм (remove_response output='DISP' × 1e9).
R   = S-P формула: R = Vp·Vs/(Vp-Vs) · ΔT(S-P) для каждой станции.
S   = станционная поправка (0 по умолчанию).

Запуск:
    python core/ml_filter_v3.py --info                    # статистика
    python core/ml_filter_v3.py --ml-threshold 1.0        # порог + recall
    python core/ml_filter_v3.py --rebuild-cache           # пересобрать кэш
"""

import argparse
import csv
import gc
import glob
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC_IN  = os.path.join(
    _ROOT, 'data-in-memory', 'gpu_splimit_45_march',
    'assoc_output_lim', 'associations.xml')
DEFAULT_WAVEFORMS = os.path.join(_ROOT, 'geofiles')
DEFAULT_CACHE_AMP = os.path.join(_ROOT, 'amps_sample_new.csv')
DEFAULT_METADATA  = os.path.join(_ROOT, 'metadata')
DEFAULT_THRESHOLD = 1.0

# ── Параметры выборки ─────────────────────────────────────────────────────────

N_BETWEEN  = 100   # событий между каждой парой соседних каталожных
MATCH_WIN  = 120   # окно матчинга каталог→XML, сек (учитывает P-traveltime)
RANDOM_SEED = 42

# 10 каталожных событий апреля 2024 (origin_time каталога, Ms)
SAMPLE_CATALOG_EVENTS = [
    {'time': datetime(2024, 4, 10, 22, 31, 18), 'ms': 4.1},
    {'time': datetime(2024, 4, 12,  2,  8, 17), 'ms': 0.9},
    {'time': datetime(2024, 4, 12, 16,  3, 45), 'ms': 0.6},
    {'time': datetime(2024, 4, 14, 18, 48, 52), 'ms': 2.0},
    {'time': datetime(2024, 4, 16,  4, 13, 27), 'ms': 1.0},
]

# ── Пространства имён QuakeML ──────────────────────────────────────────────────

BED_NS   = 'http://quakeml.org/xmlns/bed/1.2'
QUAKE_NS = 'http://quakeml.org/xmlns/quakeml/1.2'
ET.register_namespace('',  BED_NS)
ET.register_namespace('q', QUAKE_NS)

# ── Константы ML ──────────────────────────────────────────────────────────────

ML_B_LOG      = 1.024
ML_B_LIN      = 0.001648
ML_C          = -1.889
AMP_SCALE_NM  = 1e9
A_MAX_NM      = 1e6    # физический предел ~1 мм смещения; выше = сбой удаления ответа
AMP_RATIO_MIN = 0.3    # отношение actual/expected; ниже — неверная калибровка станции
N_ITER_FILTER = 4      # итераций фильтрации по amplitude ratio
N_MIN_STA     = 3      # минимум станций после фильтрации

VP          = 6.0
VS          = 3.4883
WIN_SEC     = 5.0
R_MIN_KM    = 30.0    # формула 5б ненадёжна ближе 30 км (Hutton & Boore)

_PAZ_WA = {
    'poles': [(-6.283185307 + 4.712388980j),
              (-6.283185307 - 4.712388980j)],
    'zeros': [0j, 0j],
    'gain': 1.0,
    'sensitivity': 2800.0,
}
_COMP_PRIORITY = ('E', 'N', 'Z')


# ── Парсинг времени ───────────────────────────────────────────────────────────

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


# ── Парсинг XML: origin_time + пики ──────────────────────────────────────────

def parse_xml_full(tree):
    """
    Возвращает list of:
      (ev_elem, pub_id, origin_time, picks_p, picks_s)

    origin_time берётся из <origin>/<time>/<value>. Если отсутствует — None.
    picks_p/picks_s: {station: datetime}
    """
    ns   = BED_NS
    root = tree.getroot()
    result = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id = ev.get('publicID', '')

        # origin_time
        origin_time = None
        orig = ev.find(f'{{{ns}}}origin')
        if orig is not None:
            t_el = orig.find(f'{{{ns}}}time/{{{ns}}}value')
            if t_el is not None:
                origin_time = parse_time(t_el.text)

        # пики
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

        result.append((ev, pub_id, origin_time, picks_p, picks_s))
    return result


# ── Построение выборки ────────────────────────────────────────────────────────

def build_sample(events_full, catalog_events, n_between, match_win, seed):
    """
    1. Для каждого каталожного события ищет ближайшее XML событие
       в пределах ±match_win секунд (по origin_time).
    2. Между каждой парой найденных каталожных событий (сортировка по времени)
       берёт до n_between случайных некаталожных событий.

    Возвращает:
      sample_pub_ids : list[str] — pub_id в порядке (каталожные + фон)
      cat_hits       : list[dict] — каждый элемент:
                         {'cat_idx': int, 'cat_time': datetime, 'ms': float,
                          'pub_id': str, 'xml_origin': datetime, 'delta_sec': float}
      cat_missed     : list[dict] — каталожные события без совпадения в XML
    """
    rng = random.Random(seed)
    ns  = BED_NS

    # Индекс XML по времени (только с origin_time)
    timed = [(i, pub_id, origin_time)
             for i, (_, pub_id, origin_time, _, _) in enumerate(events_full)
             if origin_time is not None]
    timed.sort(key=lambda x: x[2])
    xml_times = [t for _, _, t in timed]   # отсортированные времена

    def find_nearest(cat_time):
        best, best_delta = None, None
        for i, pub_id, ot in timed:
            delta = abs((ot - cat_time).total_seconds())
            if delta <= match_win:
                if best_delta is None or delta < best_delta:
                    best, best_delta = (i, pub_id, ot), delta
        return best, best_delta

    cat_hits, cat_missed = [], []

    for ci, cev in enumerate(catalog_events):
        match, delta = find_nearest(cev['time'])
        if match is None:
            cat_missed.append({'cat_idx': ci, **cev})
        else:
            _, pub_id, xml_ot = match
            cat_hits.append({
                'cat_idx':    ci,
                'cat_time':   cev['time'],
                'ms':         cev['ms'],
                'pub_id':     pub_id,
                'xml_origin': xml_ot,
                'delta_sec':  delta,
            })

    # Сортируем найденные каталожные по времени XML
    cat_hits.sort(key=lambda x: x['xml_origin'])

    # Между каждой парой соседних каталожных берём случайные фоновые события
    sample_pub_ids = set(h['pub_id'] for h in cat_hits)
    bg_pub_ids = []

    for k in range(len(cat_hits) - 1):
        t_lo = cat_hits[k]['xml_origin']
        t_hi = cat_hits[k + 1]['xml_origin']

        # Все XML события в этом интервале (кроме каталожных)
        candidates = [
            pub_id for _, pub_id, ot in timed
            if t_lo < ot < t_hi and pub_id not in sample_pub_ids
        ]
        chosen = rng.sample(candidates, min(n_between, len(candidates)))
        bg_pub_ids.extend(chosen)
        sample_pub_ids.update(chosen)

    # Итоговый список: каталожные первыми (порядок по времени), затем фон
    result_pub_ids = [h['pub_id'] for h in cat_hits] + bg_pub_ids
    return result_pub_ids, cat_hits, cat_missed


# ── Кэш амплитуд (ключ = pub_id, sta) ────────────────────────────────────────

def save_amp_cache(amplitudes, path):
    """amplitudes: {(pub_id, sta): float|None}"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pub_id', 'sta', 'A'])
        for (pub_id, sta), A in amplitudes.items():
            w.writerow([pub_id, sta, '' if A is None else A])


def load_amp_cache(path):
    """Возвращает {(pub_id, sta): float|None}"""
    cache = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            A = float(row['A']) if row['A'] else None
            cache[(row['pub_id'], row['sta'])] = A
    return cache


# ── S-P расстояние ────────────────────────────────────────────────────────────

def sp_distance_km(p_time, s_time):
    dt = (s_time - p_time).total_seconds()
    if dt <= 0.0:
        return None
    return VP * VS / (VP - VS) * dt


def compute_r_dict(picks_p, picks_s):
    r_dict = {}
    for sta in picks_p:
        if sta not in picks_s:
            continue
        r = sp_distance_km(picks_p[sta], picks_s[sta])
        if r is not None and r > 0.0:
            r_dict[sta] = r
    return r_dict


# ── Индекс форм волн ──────────────────────────────────────────────────────────

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
        for comp in _COMP_PRIORITY:
            if comp in by_comp:
                index[sta] = sorted(by_comp[comp], key=lambda x: x[0])
                break
    return index


def find_waveform_file(index, sta, t):
    for t_start, t_end, fpath in index.get(sta, []):
        if t_start <= t < t_end:
            return fpath
    return None


READ_MARGIN_SEC    = 60.0    # контекст по краям для remove_response
MAX_GROUP_SPAN_SEC = 300.0   # макс. временно́й диапазон одного кластера (сек)
N_WORKERS          = 4       # параллельных потоков для извлечения амплитуд


def _split_triples_by_time(triples, max_span_sec):
    """
    Разбивает [(pub_id, sta, s_time), ...] на кластеры, где первое и
    последнее s_time в кластере отличаются не более чем на max_span_sec.
    """
    if not triples:
        return []
    srt = sorted(triples, key=lambda x: x[2])
    clusters, cur = [], [srt[0]]
    for item in srt[1:]:
        if (item[2] - cur[0][2]).total_seconds() > max_span_sec:
            clusters.append(cur)
            cur = []
        cur.append(item)
    clusters.append(cur)
    return clusters


def _process_file_cluster(fpath, triples, inventory, wood_anderson):
    """
    Читает один временно́й кластер из MSEED-файла, снимает инструментальный
    ответ и возвращает {(pub_id, sta): float|None} — амплитуды в метрах.
    """
    from obspy import read as obspy_read, UTCDateTime
    result = {}
    try:
        s_utcs  = [UTCDateTime(s_time) for _, _, s_time in triples]
        t_start = min(s_utcs) - READ_MARGIN_SEC
        t_end   = max(s_utcs) + WIN_SEC + READ_MARGIN_SEC

        st = obspy_read(fpath, starttime=t_start, endtime=t_end)
        if not st:
            return result
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
                for pub_id, sta, _ in triples:
                    result[(pub_id, sta)] = None
                return result
        else:
            for pub_id, sta, _ in triples:
                result[(pub_id, sta)] = None
            return result

        for pub_id, sta, s_time in triples:
            try:
                s_utc  = UTCDateTime(s_time)
                tr_sig = tr.slice(s_utc, s_utc + WIN_SEC)
                if tr_sig is None or len(tr_sig.data) == 0:
                    result[(pub_id, sta)] = None
                    continue
                A = float(np.max(np.abs(tr_sig.data)))
                result[(pub_id, sta)] = A if A > 0.0 else None
            except Exception:
                result[(pub_id, sta)] = None
        del tr, st
    except Exception:
        pass
    return result


# ── Извлечение амплитуд S-волны ───────────────────────────────────────────────

def extract_amplitudes(sample_events, waveform_index, inventory=None,
                       wood_anderson=False):
    """
    sample_events: list of (pub_id, picks_p, picks_s, r_dict)
    Возвращает {(pub_id, sta): float|None} — амплитуда в метрах.

    Каждый MSEED-файл разбивается на временны́е кластеры ≤ MAX_GROUP_SPAN_SEC,
    кластеры обрабатываются параллельно (N_WORKERS потоков).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Группируем по файлу
    groups = defaultdict(list)
    for pub_id, _, picks_s, r_dict in sample_events:
        for sta in r_dict:
            s_time = picks_s.get(sta)
            if s_time is None:
                continue
            fpath = find_waveform_file(waveform_index, sta, s_time)
            if fpath is None:
                continue
            groups[fpath].append((pub_id, sta, s_time))

    # Разбиваем каждый файл на временны́е кластеры
    tasks = []
    for fpath, triples in groups.items():
        for cluster in _split_triples_by_time(triples, MAX_GROUP_SPAN_SEC):
            tasks.append((fpath, cluster))

    n = len(tasks)
    print(f"  Файлов: {len(groups)}, кластеров: {n}, потоков: {N_WORKERS}")

    amplitudes = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futs = {
            executor.submit(_process_file_cluster, fpath, cluster,
                            inventory, wood_anderson): i
            for i, (fpath, cluster) in enumerate(tasks)
        }
        for done, fut in enumerate(as_completed(futs), 1):
            amplitudes.update(fut.result())
            if done % 50 == 0 or done == n:
                print(f"  Кластеров: {done}/{n}...", end='\r', flush=True)

    gc.collect()
    print(f"  Кластеров: {n}/{n} — готово.         ")
    return amplitudes


# ── Вычисление ML по формуле 5б ──────────────────────────────────────────────

def _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                amp_scale, r_min, a_max_nm):
    """
    Возвращает [(sta, r_km, A_nm, ml_sta), ...] — первичные кандидаты
    после R_MIN и A_MAX фильтров.
    """
    entries = []
    for sta, r_km in r_dict.items():
        if r_km < r_min:
            continue
        A = amplitudes.get((pub_id, sta))
        if A is None or A <= 0.0:
            continue
        A_nm = A * amp_scale
        if A_nm <= 0.0 or A_nm > a_max_nm:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ML_B_LOG * math.log10(r_km)
                + ML_B_LIN * r_km
                + ML_C + S)
        entries.append((sta, r_km, A_nm, ml_s))
    return entries


def _filter_by_amp_ratio(entries, sta_corr, n_iter, ratio_min, n_min):
    """
    Итеративно отбрасывает станции, у которых амплитуда < ratio_min × ожидаемой
    при текущем медианном ML. Неверная калибровка → actual << expected.
    Возвращает отфильтрованный список entries.
    """
    cur = entries
    for _ in range(n_iter):
        if len(cur) < n_min:
            break
        ml_med = float(np.median([e[3] for e in cur]))
        survivors = []
        for sta, r_km, A_nm, ml_s in cur:
            S = sta_corr.get(sta, 0.0)
            A_exp = 10 ** (ml_med
                           - ML_B_LOG * math.log10(r_km)
                           - ML_B_LIN * r_km
                           - ML_C - S)
            if A_nm / A_exp >= ratio_min:
                survivors.append((sta, r_km, A_nm, ml_s))
        if len(survivors) < n_min or len(survivors) == len(cur):
            break
        cur = survivors
    return cur


def compute_event_ml(pub_id, r_dict, amplitudes, sta_corr,
                     amp_scale=AMP_SCALE_NM, r_min=R_MIN_KM, a_max_nm=A_MAX_NM):
    entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                          amp_scale, r_min, a_max_nm)
    entries = _filter_by_amp_ratio(entries, sta_corr,
                                   N_ITER_FILTER, AMP_RATIO_MIN, N_MIN_STA)
    if not entries:
        return None, 0
    return float(np.median([e[3] for e in entries])), len(entries)


def print_diag_catalog(cat_hits, sample_events_dict, amplitudes, sta_corr,
                       amp_scale=AMP_SCALE_NM):
    """
    Для каждого каталожного события — подробная таблица по станциям.
    Показывает ML_sta и причину пропуска (R, A_MAX, ratio-filter).
    """
    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА: детали по каталожным событиям")
    print("=" * 70)
    for h in cat_hits:
        pub_id = h['pub_id']
        *_, r_dict = sample_events_dict.get(pub_id, (None, {}, {}, {}))

        # Первичные кандидаты (R и A фильтры)
        raw_entries = _ml_entries(pub_id, r_dict, amplitudes, sta_corr,
                                  amp_scale, R_MIN_KM, A_MAX_NM)
        # После amplitude-ratio фильтрации
        filt_entries = _filter_by_amp_ratio(raw_entries, sta_corr,
                                            N_ITER_FILTER, AMP_RATIO_MIN, N_MIN_STA)
        used_stas = {e[0] for e in filt_entries}
        ml_final = (float(np.median([e[3] for e in filt_entries]))
                    if filt_entries else None)
        ml_str = f"{ml_final:.2f}" if ml_final is not None else "нет ML"

        print(f"\n  {h['cat_time'].strftime('%Y-%m-%d %H:%M:%S')}  "
              f"Ms={h['ms']:+.1f}  →  ML={ml_str}  "
              f"(Δt до XML: {h['delta_sec']:.0f}с)")
        if not r_dict:
            print("    нет S-P пар")
            continue

        print(f"  {'Станция':8s}  {'R,км':>7s}  {'A,нм':>12s}  "
              f"{'ML_sta':>7s}  {'Статус'}")
        print("  " + "-" * 62)

        # Вычислим expected A при финальном ML для отображения ratio
        def expected_a(r_km, sta):
            if ml_final is None:
                return None
            S = sta_corr.get(sta, 0.0)
            return 10 ** (ml_final
                          - ML_B_LOG * math.log10(r_km)
                          - ML_B_LIN * r_km
                          - ML_C - S)

        for sta in sorted(r_dict):
            r_km = r_dict[sta]
            A    = amplitudes.get((pub_id, sta))
            if r_km < R_MIN_KM:
                A_nm_s = f"{A * amp_scale:.2f}" if (A and A > 0) else "—"
                print(f"  {sta:8s}  {r_km:7.1f}  {A_nm_s:>12s}  "
                      f"{'—':>7s}  R<{R_MIN_KM:.0f}км")
            elif A is None:
                print(f"  {sta:8s}  {r_km:7.1f}  {'—':>12s}  "
                      f"{'—':>7s}  нет ответа")
            elif A <= 0.0:
                print(f"  {sta:8s}  {r_km:7.1f}  {'0':>12s}  "
                      f"{'—':>7s}  A=0")
            else:
                A_nm = A * amp_scale
                if A_nm > A_MAX_NM:
                    print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2e}  "
                          f"{'—':>7s}  A>A_MAX")
                else:
                    S    = sta_corr.get(sta, 0.0)
                    ml_s = (math.log10(A_nm)
                            + ML_B_LOG * math.log10(r_km)
                            + ML_B_LIN * r_km
                            + ML_C + S)
                    if sta in used_stas:
                        status = "OK"
                    else:
                        exp = expected_a(r_km, sta)
                        ratio = A_nm / exp if exp else 0
                        status = f"ratio={ratio:.2f}<{AMP_RATIO_MIN}"
                    print(f"  {sta:8s}  {r_km:7.1f}  {A_nm:12.2f}  "
                          f"{ml_s:7.2f}  {status}")
    print("\n" + "=" * 70)


# ── Статистика и recall ────────────────────────────────────────────────────────

def print_distribution(ml_values):
    vals   = sorted(v for v in ml_values if v is not None)
    n_none = sum(1 for v in ml_values if v is None)
    print(f"\n  Событий с оценкой ML:  {len(vals)}")
    print(f"  Событий без оценки ML: {n_none}  (удаляются при фильтрации)")
    if not vals:
        return
    n    = len(vals)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Распределение ML (b_log={ML_B_LOG}, b_lin={ML_B_LIN}, c={ML_C}):")
    for p in pcts:
        print(f"    P{p:2d}: {vals[max(0, int(p / 100 * n) - 1)]:.2f}")
    print(f"    min: {vals[0]:.2f}   max: {vals[-1]:.2f}")


def print_recall(cat_hits, ml_by_pubid, threshold):
    """Показывает, какие из каталожных событий прошли порог ML."""
    print(f"\n  Recall по {len(cat_hits)} каталожным событиям (порог ML >= {threshold}):")
    survived = 0
    for h in cat_hits:
        ml = ml_by_pubid.get(h['pub_id'])
        ok = ml is not None and ml >= threshold
        if ok:
            survived += 1
        status = f"ML={ml:.2f}" if ml is not None else "нет ML"
        flag = "OK" if ok else "пропущено"
        print(f"    {h['cat_time'].strftime('%Y-%m-%d %H:%M:%S')}  "
              f"Ms={h['ms']:+.1f}  {flag}  ({status})")
    print(f"\n  Recall: {survived}/{len(cat_hits)} = "
          f"{100 * survived / max(1, len(cat_hits)):.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global VP, VS
    parser = argparse.ArgumentParser(
        description='Быстрая настройка порогов ML на выборке событий (v3)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--assoc-in',      default=DEFAULT_ASSOC_IN,
                        help='Входной XML ассоциатора (полный)')
    parser.add_argument('--waveforms',     default=DEFAULT_WAVEFORMS,
                        help='Папка с формами волн')
    parser.add_argument('--cache-amp',     default=DEFAULT_CACHE_AMP,
                        help='CSV-кэш амплитуд выборки')
    parser.add_argument('--metadata-dir',  default=DEFAULT_METADATA,
                        help='Папка с FDSNStationXML для remove_response')
    parser.add_argument('--wood-anderson', action='store_true', default=False,
                        help='WA симуляция (для формулы 5б не нужна — A=DISP)')
    parser.add_argument('--ml-threshold',  type=float, default=DEFAULT_THRESHOLD,
                        help='Порог ML: показать recall при ML >= X')
    parser.add_argument('--info',          action='store_true',
                        help='Только показать статистику, не применять порог')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Пересобрать кэш амплитуд (игнорировать существующий)')
    parser.add_argument('--n-between',     type=int, default=N_BETWEEN,
                        help='Событий между каждой парой каталожных в выборке')
    parser.add_argument('--match-win',     type=float, default=MATCH_WIN,
                        help='Окно матчинга каталог→XML, сек')
    parser.add_argument('--seed',          type=int, default=RANDOM_SEED,
                        help='random seed для выборки')
    parser.add_argument('--vp',            type=float, default=VP)
    parser.add_argument('--vs',            type=float, default=VS)
    parser.add_argument('--sta-corrections', default=None,
                        help='CSV станционных поправок: station,S')
    parser.add_argument('--diag',           action='store_true',
                        help='Подробная диагностика по каталожным событиям: '
                             'R, A_нм, ML_sta по каждой станции')
    args = parser.parse_args()

    VP = args.vp
    VS = args.vs
    k = VP * VS / (VP - VS)

    print(f"Входной XML:   {args.assoc_in}")
    print(f"Кэш выборки:   {args.cache_amp}")
    print(f"S-P формула:   R = {k:.3f} · ΔT(S-P) км")
    print(f"Выборка:       10 каталожных + {args.n_between} фоновых на интервал\n")

    # ── Парсинг полного XML ───────────────────────────────────────────────────
    print("Парсинг полного XML...")
    tree         = ET.parse(args.assoc_in)
    events_full  = parse_xml_full(tree)
    print(f"  Событий в XML: {len(events_full)}")

    # ── Построение выборки ────────────────────────────────────────────────────
    print(f"\nПоиск {len(SAMPLE_CATALOG_EVENTS)} каталожных событий в XML "
          f"(окно ±{args.match_win:.0f}s)...")
    sample_pub_ids, cat_hits, cat_missed = build_sample(
        events_full, SAMPLE_CATALOG_EVENTS,
        n_between=args.n_between,
        match_win=args.match_win,
        seed=args.seed,
    )

    print(f"  Найдено каталожных: {len(cat_hits)}/{len(SAMPLE_CATALOG_EVENTS)}")
    if cat_missed:
        print(f"  Не найдено ({len(cat_missed)}):")
        for m in cat_missed:
            print(f"    {m['time'].strftime('%Y-%m-%d %H:%M:%S')}  Ms={m['ms']:+.1f}")
    print(f"  Фоновых событий:   {len(sample_pub_ids) - len(cat_hits)}")
    print(f"  Итого в выборке:   {len(sample_pub_ids)}")

    # Строим r_dict для событий выборки
    pub_id_set = set(sample_pub_ids)
    sample_events = []   # (pub_id, picks_p, picks_s, r_dict)
    for _, pub_id, origin_time, picks_p, picks_s in events_full:
        if pub_id not in pub_id_set:
            continue
        r_dict = compute_r_dict(picks_p, picks_s)
        sample_events.append((pub_id, picks_p, picks_s, r_dict))

    # Словарь для быстрого доступа: {pub_id: (pub_id, picks_p, picks_s, r_dict)}
    sample_events_dict = {pub_id: ev for ev in sample_events
                          for pub_id in [ev[0]]}

    # ── Станционные поправки ──────────────────────────────────────────────────
    sta_corr = {}
    if args.sta_corrections:
        import csv as _csv
        with open(args.sta_corrections, newline='', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                sta = row.get('station', '').strip()
                val = row.get('S', row.get('correction', '')).strip()
                if sta and val:
                    try:
                        sta_corr[sta] = float(val)
                    except ValueError:
                        pass
        print(f"  Станционных поправок: {len(sta_corr)}")

    # ── Кэш амплитуд ─────────────────────────────────────────────────────────
    if not args.rebuild_cache and os.path.isfile(args.cache_amp):
        print(f"\nЗагрузка кэша амплитуд: {args.cache_amp}")
        amplitudes = load_amp_cache(args.cache_amp)
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Загружено пар: {len(amplitudes)}  с амплитудой: {n_amp}")
    else:
        if args.rebuild_cache:
            print("\nПересборка кэша (--rebuild-cache).")

        # Загрузка StationXML
        inventory = None
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

        # Индекс форм волн
        print(f"\nПостроение индекса форм волн...")
        waveform_index = build_waveform_index(args.waveforms)
        n_files = sum(len(v) for v in waveform_index.values())
        print(f"  Файлов: {n_files}  станций: {len(waveform_index)}")

        # Извлечение амплитуд только для выборки
        print(f"\nИзвлечение амплитуд S-волны для {len(sample_events)} событий "
              f"(окно S..S+{WIN_SEC:.0f}s, горизонталь E>N>Z)...")
        use_wa = args.wood_anderson and inventory is not None
        amplitudes = extract_amplitudes(
            sample_events, waveform_index,
            inventory=inventory,
            wood_anderson=use_wa,
        )
        n_amp = sum(1 for v in amplitudes.values() if v is not None)
        print(f"  Пар с амплитудой: {n_amp}")

        save_amp_cache(amplitudes, args.cache_amp)
        print(f"  Кэш сохранён: {args.cache_amp}")

    # ── Вычисление ML ─────────────────────────────────────────────────────────
    print("\nВычисление ML...")
    ml_by_pubid = {}
    for pub_id, _, _, r_dict in sample_events:
        ml, _ = compute_event_ml(pub_id, r_dict, amplitudes, sta_corr)
        ml_by_pubid[pub_id] = ml

    print_distribution(list(ml_by_pubid.values()))

    if args.diag:
        print_diag_catalog(cat_hits, sample_events_dict, amplitudes, sta_corr)

    if args.info:
        print_recall(cat_hits, ml_by_pubid, args.ml_threshold)
        return

    # ── Фильтрация и recall ───────────────────────────────────────────────────
    thr = args.ml_threshold
    total  = len(sample_events)
    kept   = sum(1 for ml in ml_by_pubid.values()
                 if ml is not None and ml >= thr)
    no_ml  = sum(1 for ml in ml_by_pubid.values() if ml is None)

    print(f"\n{'=' * 60}")
    print(f"Порог ML >= {thr}")
    print(f"  Всего в выборке:  {total}")
    print(f"  Сохранено:        {kept}  (ML >= {thr})")
    print(f"  Удалено:          {total - kept}  "
          f"(из них {no_ml} без оценки ML)")

    print_recall(cat_hits, ml_by_pubid, thr)


if __name__ == '__main__':
    main()
