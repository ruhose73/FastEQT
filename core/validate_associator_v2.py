"""
validate_associator_v2.py — оценка origin time через S-P пики без координат.

Для каждой станции события (при наличии P и S пиков):
    ΔT  = Ts − Tp
    R   = Vp·Vs / (Vp − Vs) · ΔT    (гипоцентральное расстояние)
    Tp0 = Tp − R / Vp
    Ts0 = Ts − R / Vs               (математически ≡ Tp0 — тождество)

Фильтрация выбросов внутри события:
    1. Медиана T0 по всем станциям
    2. σ = std(T0 − median)
    3. Отброс станций где |T0 − median| > SIGMA_MULT · σ  → флаг [!STA]
    4. origin_time = mean(T0_remaining)

Флаги событий:

    [!N]   — осталось < MIN_STATIONS станций после фильтрации

Использование:
    python core/validate_associator_v2.py --year 2024 --month 5
    python core/validate_associator_v2.py --window 25 --sigma 2.0
    python core/validate_associator_v2.py --assoc data-in-memory/.../associations.xml
"""

import argparse
import csv
import math
import os
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import openpyxl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATALOG_PATH   = os.path.join(_ROOT, "catalog.xlsx")
DEFAULT_ASSOC  = os.path.join(
    _ROOT, "data-in-memory", "gpu_splimit_45_may_v2",
    "assoc_output_lim_weight_v2", "associations_ml1p5.xml")
DEFAULT_AMPS   = os.path.join(_ROOT, "amps_wa_v3_weight_v2.csv")
DEFAULT_WINDOW = 60       # сек — допуск сравнения с каталогом
DEFAULT_VP     = 6.0      # км/с
DEFAULT_VS     = 3.4883   # км/с
DEFAULT_SIGMA  = 1.0      # множитель σ для отброса выбросов

MIN_STATIONS   = 3        # минимум станций после фильтрации для оценки T0

BED_NS = "http://quakeml.org/xmlns/bed/1.2"

# ── Двухскоростная модель (Pg/Sg ↔ Pn/Sn) ─────────────────────────────────────
VP_PN    = 8.0    # км/с — скорость Pn (мантия)
VS_SN    = 4.6    # км/с — скорость Sn (мантия)
R_PN_THR = 150.0  # км  — порог перехода Pg → Pn

# ── Константы ML (Дягилев et al. 2023, формула 5б) ────────────────────────────
ML_B_LOG      = 1.024
ML_B_LIN      = 0.001648
ML_C          = -1.889
AMP_SCALE_NM  = 1e9
A_MAX_NM      = 1e6
A_NM_MIN      = 0.005
ML_R_MIN      = 20.0
ML_N_MIN      = 4
ML_SIGMA      = 1


# ── Загрузка данных ────────────────────────────────────────────────────────────

def load_catalog(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    events = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        origin_time, lat, lon, depth_km = row[0], row[1], row[2], row[3]
        ms = row[4] if len(row) > 4 else None
        if origin_time is None or lat is None or lon is None:
            continue
        ot = origin_time
        if hasattr(ot, 'tzinfo') and ot.tzinfo is not None:
            ot = ot.replace(tzinfo=None)
        events.append({
            'origin_time': ot,
            'lat': lat, 'lon': lon,
            'depth_km': depth_km if depth_km is not None else 10.0,
            'ms': float(ms) if ms is not None else None,
        })
    wb.close()
    return events


def load_associations(path):
    """
    Возвращает список событий. Каждое событие:
        pub_id   — publicID
        picks    — {station: {'p': datetime, 's': datetime}}
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns = BED_NS
    events = []
    for ev in root.findall(f'.//{{{ns}}}event'):
        pub_id = ev.get('publicID', '')
        picks_by_sta = {}
        for pick in ev.findall(f'{{{ns}}}pick'):
            wf       = pick.find(f'{{{ns}}}waveformID')
            t_el     = pick.find(f'{{{ns}}}time/{{{ns}}}value')
            phase_el = pick.find(f'{{{ns}}}phaseHint')
            if wf is None or t_el is None or phase_el is None:
                continue
            sta   = wf.get('stationCode', '').strip()
            net   = wf.get('networkCode', '').strip()
            phase = phase_el.text.strip()
            try:
                t = datetime.fromisoformat(t_el.text.rstrip('Z'))
            except ValueError:
                continue
            if sta not in picks_by_sta:
                picks_by_sta[sta] = {'net': net}
            if phase == 'P':
                picks_by_sta[sta]['p'] = t
            elif phase == 'S':
                picks_by_sta[sta]['s'] = t
        if picks_by_sta:
            events.append({'pub_id': pub_id, 'picks': picks_by_sta})
    return events


# ── ML ────────────────────────────────────────────────────────────────────────

def load_pick_probabilities(assoc_input_dir):
    """
    Загружает p_probability и p_snr для каждого P-пика из assoc_input/{STA}/X_prediction_results.csv.
    Возвращает {(station, round(p_arrival_time.timestamp(), 3)): {'prob': float, 'snr': float|None}}.
    """
    result = {}
    if not os.path.isdir(assoc_input_dir):
        return result
    for sta_name in os.listdir(assoc_input_dir):
        csv_path = os.path.join(assoc_input_dir, sta_name, 'X_prediction_results.csv')
        if not os.path.isfile(csv_path):
            continue
        sta = sta_name.strip()
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                p_t    = row.get('p_arrival_time', '').strip()
                p_prob = row.get('p_probability',  '').strip()
                if not p_t or not p_prob:
                    continue
                try:
                    pt_dt  = datetime.fromisoformat(p_t)
                    prob_v = float(p_prob)
                except (ValueError, TypeError):
                    continue
                snr_s = row.get('p_snr', '').strip()
                try:
                    snr_v = float(snr_s) if snr_s and snr_s.lower() != 'nan' else None
                except (ValueError, TypeError):
                    snr_v = None
                result[(sta, round(pt_dt.timestamp(), 3))] = {'prob': prob_v, 'snr': snr_v}
    return result


def load_amplitudes(path):
    """Загружает amps_wa.csv → {(pub_id, sta): A_meters}."""
    amps = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                amps[(row['pub_id'], row['sta'])] = float(row['A'])
            except (ValueError, KeyError):
                pass
    return amps


def _r_dict_all(picks_by_sta, vp, vs):
    """R (км) для всех станций с P+S пиком, без фильтра по расстоянию."""
    k = vp * vs / (vp - vs)
    result = {}
    for sta, picks in picks_by_sta.items():
        if 'p' not in picks or 's' not in picks:
            continue
        dt_sp = (picks['s'] - picks['p']).total_seconds()
        if dt_sp > 0:
            result[sta] = round(k * dt_sp, 1)
    return result


def compute_ml(pub_id, r_dict, amplitudes, sta_corr=None):
    """
    ML события: медиана → удаление выбросов (1.5σ) → среднее.
    Формула 5б Дягилев et al. 2023:
        ML = log10(A_nm) + ML_B_LOG*log10(R) + ML_B_LIN*R + ML_C + S
    Возвращает (ml, n_stations) или (None, 0).
    """
    if sta_corr is None:
        sta_corr = {}
    entries    = []
    ml_per_sta = {}   # {station: ml_value} — до удаления выбросов
    sta_order  = []   # чтобы сохранить связь entries[i] ↔ sta
    for sta, r_km in r_dict.items():
        if r_km < ML_R_MIN:
            continue
        A = amplitudes.get((pub_id, sta))
        if A is None or A <= 0:
            continue
        A_nm = A * AMP_SCALE_NM
        if A_nm < A_NM_MIN or A_nm > A_MAX_NM:
            continue
        S = sta_corr.get(sta, 0.0)
        ml_s = (math.log10(A_nm)
                + ML_B_LOG * math.log10(r_km)
                + ML_B_LIN * r_km
                + ML_C + S)
        entries.append(ml_s)
        sta_order.append(sta)
        ml_per_sta[sta] = round(ml_s, 2)

    if len(entries) < ML_N_MIN:
        return None, len(entries), ml_per_sta

    med   = statistics.median(entries)
    sigma = statistics.pstdev(entries)
    if sigma > 0:
        inliers = [v for v in entries if abs(v - med) <= ML_SIGMA * sigma]
        if len(inliers) >= ML_N_MIN:
            entries = inliers

    return round(statistics.mean(entries), 2), len(entries), ml_per_sta


# ── Оценка origin time ─────────────────────────────────────────────────────────

def estimate_origin_times(picks_by_sta, vp, vs, r_min=20.0, r_max=None, prob_dict=None):
    """
    Для каждой станции с P и S: вычисляет T0, Tp0, Ts0, R, ΔT.
    Tp0 = Ts0 аналитически — это тождество при данной формуле R.
    r_min: станции с R < r_min км исключаются (слишком близко — неустойчивая оценка).
    r_max: станции с R > r_max км исключаются (Pn-эффект искажает T0).
    prob_dict: {(station, round(tp.timestamp(),3)): p_probability} — для взвешенного среднего.
    """
    k = vp * vs / (vp - vs)   # коэффициент: R = k · ΔT
    results = []
    for sta, picks in picks_by_sta.items():
        if 'p' not in picks or 's' not in picks:
            continue
        tp, ts = picks['p'], picks['s']
        dt_sp = (ts - tp).total_seconds()
        if dt_sp <= 0:
            continue
        R   = k * dt_sp
        if R < r_min:
            continue
        if r_max is not None and R > r_max:
            continue
        tp0 = tp - timedelta(seconds=R / vp)
        ts0 = ts - timedelta(seconds=R / vs)
        p_prob, p_snr = None, None
        if prob_dict is not None:
            entry  = prob_dict.get((sta, round(tp.timestamp(), 3)))
            if entry is not None:
                p_prob = entry['prob']
                p_snr  = entry['snr']
        results.append({
            'station': sta,
            'tp':      tp,
            'ts':      ts,
            'dt_sp':   round(dt_sp, 2),
            'R_km':    round(R, 1),
            'tp0':     tp0,
            'ts0':     ts0,
            'ps_diff': round((ts0 - tp0).total_seconds(), 4),
            't0':      tp0,
            'p_prob':  p_prob,
            'p_snr':   p_snr,
        })
    return results


def estimate_origin_times_v2(picks_by_sta, vp, vs, r_min=20.0, r_max=None,
                              vp_pn=VP_PN, vs_sn=VS_SN, r_pn_thr=R_PN_THR, prob_dict=None):
    """
    Двухскоростная модель: Pg/Sg при R < r_pn_thr км, Pn/Sn при R >= r_pn_thr.
    Координаты события не нужны — R оценивается из S-P напрямую.

    Алгоритм для каждой станции:
      1. R_pg = k_pg · ΔT  (коровая модель)
      2. R_pg < r_pn_thr  → режим Pg/Sg, T0 = Tp − R_pg / Vp
         R_pg >= r_pn_thr → режим Pn/Sn, R_pn = k_pn · ΔT, T0 = Tp − R_pn / Vp_pn

    Ps_diff: Tp0 = Ts0 — тождество выполняется в каждом режиме отдельно.
    """
    k_pg = vp * vs / (vp - vs)
    k_pn = vp_pn * vs_sn / (vp_pn - vs_sn)
    results = []
    for sta, picks in picks_by_sta.items():
        if 'p' not in picks or 's' not in picks:
            continue
        tp, ts = picks['p'], picks['s']
        dt_sp = (ts - tp).total_seconds()
        if dt_sp <= 0:
            continue

        R_pg = k_pg * dt_sp
        if R_pg < r_pn_thr:
            R, vp_eff, vs_eff, regime = R_pg, vp, vs, 'Pg'
        else:
            R, vp_eff, vs_eff, regime = k_pn * dt_sp, vp_pn, vs_sn, 'Pn'

        if R < r_min:
            continue
        if r_max is not None and R > r_max:
            continue

        tp0 = tp - timedelta(seconds=R / vp_eff)
        ts0 = ts - timedelta(seconds=R / vs_eff)
        p_prob, p_snr = None, None
        if prob_dict is not None:
            entry  = prob_dict.get((sta, round(tp.timestamp(), 3)))
            if entry is not None:
                p_prob = entry['prob']
                p_snr  = entry['snr']
        results.append({
            'station': sta,
            'tp':      tp,
            'ts':      ts,
            'dt_sp':   round(dt_sp, 2),
            'R_km':    round(R, 1),
            'regime':  regime,
            'tp0':     tp0,
            'ts0':     ts0,
            'ps_diff': round((ts0 - tp0).total_seconds(), 4),
            't0':      tp0,
            'p_prob':  p_prob,
            'p_snr':   p_snr,
        })
    return results


def sigma_filter(station_ests, sigma_mult):
    """
    Медиана → σ → отброс |T0 − median| > sigma_mult·σ.
    Возвращает (inliers, outliers).
    """
    if len(station_ests) < 2:
        return station_ests, []
    t0_secs = [e['t0'].timestamp() for e in station_ests]
    med     = statistics.median(t0_secs)
    sigma   = statistics.pstdev(t0_secs)   # population std
    if sigma == 0.0:
        return station_ests, []
    thr = sigma_mult * sigma
    inliers  = [e for e in station_ests if abs(e['t0'].timestamp() - med) <= thr]
    outliers = [e for e in station_ests if abs(e['t0'].timestamp() - med) > thr]
    if len(inliers) < MIN_STATIONS:
        return station_ests, []   # не хватает — возвращаем всё
    return inliers, outliers


def event_origin_time(station_ests):
    """T0 станции с минимальным ΔT(S-P) — ближайшая станция даёт наиболее точную оценку."""
    nearest = min(station_ests, key=lambda e: e['dt_sp'])
    return nearest['t0']


# ── Валидация ──────────────────────────────────────────────────────────────────

def validate(catalog, assoc_events, window_sec, vp, vs, sigma_mult,
             r_min=20.0, r_max=None, amplitudes=None, sta_corr=None,
             use_dual_vel=False, prob_dict=None,
             match_by='min', min_stations_match=1, min_stations_mode='xml'):
    matched, missed = [], []
    est_func = estimate_origin_times_v2 if use_dual_vel else estimate_origin_times

    # Предвычислить origin times для всех ассоциированных событий
    processed = []
    for ae in assoc_events:
        raw     = est_func(ae['picks'], vp, vs, r_min, r_max, prob_dict=prob_dict)
        if not raw:
            continue
        spread  = max(e['t0'] for e in raw) - min(e['t0'] for e in raw)

        inliers, outliers = sigma_filter(raw, sigma_mult)
        n_flag  = len(inliers) < MIN_STATIONS
        nearest = min(inliers, key=lambda e: e['dt_sp']) if not n_flag else None
        ot      = nearest['t0'] if nearest else None

        # среднее T0 по inliers (после σ-фильтра)
        if not n_flag:
            ref      = inliers[0]['t0']
            mean_off = sum((e['t0'] - ref).total_seconds() for e in inliers) / len(inliers)
            ot_mean  = ref + timedelta(seconds=mean_off)

            # взвешенное среднее T0 (вес = 1/dt_sp²: ближние станции важнее)
            weights   = [1.0 / (e['dt_sp'] ** 2) for e in inliers]
            w_total   = sum(weights)
            wmean_off = sum(w * (e['t0'] - ref).total_seconds()
                            for w, e in zip(weights, inliers)) / w_total
            ot_wmean  = ref + timedelta(seconds=wmean_off)

            # взвешенное среднее T0 (вес = p_probability из EQT)
            prob_weights = [e['p_prob'] for e in inliers
                            if e.get('p_prob') is not None and e['p_prob'] > 0]
            if len(prob_weights) >= MIN_STATIONS:
                pw_entries = [(e['p_prob'], e) for e in inliers
                              if e.get('p_prob') is not None and e['p_prob'] > 0]
                pw_total   = sum(pw for pw, _ in pw_entries)
                probw_off  = sum(pw * (e['t0'] - ref).total_seconds()
                                 for pw, e in pw_entries) / pw_total
                ot_probw   = ref + timedelta(seconds=probw_off)
            else:
                ot_probw   = None

            # взвешенное T0 (вес = p_probability × p_snr)
            psnr_entries = [(e['p_prob'] * e['p_snr'], e) for e in inliers
                            if e.get('p_prob') is not None and e['p_prob'] > 0
                            and e.get('p_snr') is not None and e['p_snr'] > 0]
            if len(psnr_entries) >= MIN_STATIONS:
                ps_total    = sum(pw for pw, _ in psnr_entries)
                psnr_off    = sum(pw * (e['t0'] - ref).total_seconds()
                                  for pw, e in psnr_entries) / ps_total
                ot_probsnr  = ref + timedelta(seconds=psnr_off)
            else:
                ot_probsnr  = None
        else:
            ot_mean    = None
            ot_wmean   = None
            ot_probw   = None
            ot_probsnr = None

        # среднее T0 по всем raw станциям (без σ-фильтра) — для сравнения
        ref_raw  = raw[0]['t0']
        raw_off  = sum((e['t0'] - ref_raw).total_seconds() for e in raw) / len(raw)
        ot_raw   = ref_raw + timedelta(seconds=raw_off)

        ml, ml_n, ml_per_sta = None, 0, {}
        if amplitudes is not None:
            r_dict = _r_dict_all(ae['picks'], vp, vs)
            ml, ml_n, ml_per_sta = compute_ml(ae['pub_id'], r_dict, amplitudes, sta_corr)

        processed.append({
            'pub_id':      ae['pub_id'],
            'raw':         raw,
            'inliers':     inliers,
            'outliers':    outliers,
            'n_xml':       len(ae['picks']),
            'ot':          ot,
            'ot_mean':     ot_mean,
            'ot_wmean':    ot_wmean,
            'ot_probw':    ot_probw,
            'ot_probsnr':  ot_probsnr,
            'ot_raw':      ot_raw,
            'nearest_sta': nearest['station'] if nearest else None,
            'nearest_r':   nearest['R_km']    if nearest else None,
            'n_flag':      n_flag,
            'spread_s':    round(spread.total_seconds(), 1),
            'ml':          ml,
            'ml_n':        ml_n,
            'ml_per_sta':  ml_per_sta,
            'picks_raw':   ae['picks'],   # {sta: {'net', 'p', 's'}}
        })

    _match_key = {'min': 'ot', 'mean': 'ot_mean', 'psnr': 'ot_probsnr'}.get(match_by, 'ot')

    for ev in catalog:
        cat_t = ev['origin_time']

        best, min_diff, best_signed = None, None, None
        nearest, nearest_diff, nearest_signed = None, None, None
        for ae in processed:
            # выбираем T0 для матчинга; если недоступен — откат по цепочке
            ot_match = ae.get(_match_key) or ae.get('ot_mean') or ae.get('ot')
            if ot_match is None:
                continue
            signed = (ot_match - cat_t).total_seconds()   # + = T0 позже каталога
            diff   = abs(signed)
            if nearest_diff is None or diff < nearest_diff:
                nearest_diff, nearest, nearest_signed = diff, ae, signed
            n_check = ae['n_xml'] if min_stations_mode == 'xml' else len(ae['inliers'])
            if diff <= window_sec and n_check >= min_stations_match:
                if min_diff is None or diff < min_diff:
                    min_diff, best, best_signed = diff, ae, signed

        record = {**ev}
        if best:
            record['assoc']    = best
            record['dt_sec']   = round(best_signed, 1)
            dt_mean  = ((best['ot_mean']  - cat_t).total_seconds()
                        if best['ot_mean']  is not None else None)
            dt_wmean = ((best['ot_wmean'] - cat_t).total_seconds()
                        if best['ot_wmean'] is not None else None)
            dt_probw   = ((best['ot_probw']   - cat_t).total_seconds()
                          if best['ot_probw']   is not None else None)
            dt_probsnr = ((best['ot_probsnr'] - cat_t).total_seconds()
                          if best['ot_probsnr'] is not None else None)
            record['dt_mean']    = round(dt_mean,    1) if dt_mean    is not None else None
            record['dt_wmean']   = round(dt_wmean,   1) if dt_wmean   is not None else None
            record['dt_probw']   = round(dt_probw,   1) if dt_probw   is not None else None
            record['dt_probsnr'] = round(dt_probsnr, 1) if dt_probsnr is not None else None
            record['dt_raw']     = round((best['ot_raw'] - cat_t).total_seconds(), 1)
            matched.append(record)
        else:
            record['nearest']      = nearest
            record['nearest_dt']   = round(nearest_signed, 1) if nearest_signed is not None else None
            missed.append(record)

    return matched, missed, processed


# ── Отчёт ──────────────────────────────────────────────────────────────────────

def print_report(matched, missed, processed, window_sec, sigma_mult, show_picks,
                 min_stations=1, min_stations_mode='xml'):
    total  = len(matched) + len(missed)
    recall = len(matched) / total * 100 if total > 0 else 0

    n_n_flag  = sum(1 for ae in processed if ae['n_flag'])

    if min_stations > 1:
        n_pass = sum(1 for ae in processed if
                     (ae['n_xml'] if min_stations_mode == 'xml' else len(ae['inliers'])) >= min_stations)
        mode_label = 'N' if min_stations_mode == 'xml' else 'σN'
        print(f"После фильтра {mode_label} >= {min_stations}: {n_pass} (из {len(processed)})")

    print("=" * 80)
    print(f"Recall: {len(matched)}/{total} ({recall:.1f}%)  "
          f"окно ±{window_sec}с  σ-множитель={sigma_mult}")
    print(f"Ассоц. событий: {len(processed)}  [!N]: {n_n_flag}")
    print("=" * 80)

    if matched:
        print(f"\nОбнаружено ({len(matched)}):")
        hdr = (f"  {'Cat origin time':<25} {'T0_psnr':<25} {'MLH':>4} {'ML':>5}"
               f" {'ΔCat,с':>7}"
               f" {'Sta':>6} {'R,км':>6} {'N':>4} {'σN':>4}")
        print(hdr)
        for ev in sorted(matched, key=lambda x: x['origin_time']):
            ae     = ev['assoc']
            ms_s   = f"{ev['ms']:.1f}"  if ev['ms']  is not None else "  -"
            ml_s   = f"{ae['ml']:.2f}"  if ae['ml']  is not None else "  —"
            dcat_s = f"{ev['dt_sec']:>+7.1f}"
            t0psnr = ae.get('ot_probsnr') or ae['ot'] or ae['ot_mean']
            t0s    = str(t0psnr)[:25] if t0psnr else "—"
            sta_s  = ae['nearest_sta'] or "—"
            r_s    = f"{ae['nearest_r']:.0f}" if ae['nearest_r'] else "—"
            print(f"  {str(ev['origin_time']):<25} {t0s:<25} {ms_s:>4} {ml_s:>5}"
                  f" {dcat_s}"
                  f" {sta_s:>6} {r_s:>6} {ae['n_xml']:>4} {len(ae['inliers']):>4}")
            if show_picks:
                _print_station_table(ae)

    if missed:
        print(f"\nПропущено ({len(missed)}):")
        print(f"  {'Origin time':<25} {'Ms':>4} {'Depth':>6}")
        for ev in sorted(missed, key=lambda x: x['origin_time']):
            ms_s = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
            print(f"  {str(ev['origin_time']):<25} {ms_s:>4} {ev['depth_km']:>6.1f}")

    print_summary_table(matched, missed, min_stations=min_stations,
                        min_stations_mode=min_stations_mode, window_sec=window_sec)


def print_summary_table(matched, missed, min_stations=1, min_stations_mode='xml',
                        window_sec=DEFAULT_WINDOW):
    all_events = (
        [(ev, True)  for ev in matched] +
        [(ev, False) for ev in missed]
    )
    all_events.sort(key=lambda x: x[0]['origin_time'])

    n_col = 'σN' if min_stations_mode == 'sigma' else 'N'

    def _n(ae):
        val = len(ae['inliers']) if min_stations_mode == 'sigma' else ae['n_xml']
        return f"{val:>5}"

    print(f"\n{'─' * 80}")
    print(f"  {'#':<4} {'Origin time':<22} {'Ms':>4} {'Совп.':>6} {'dt,с':>7} {n_col:>5} {'spread':>7} {'reason'}")
    print(f"  {'':52} * = ближайший кандидат")
    print(f"{'─' * 80}")
    for i, (ev, is_matched) in enumerate(all_events, 1):
        ms_s   = f"{ev['ms']:.1f}" if ev['ms'] is not None else "  -"
        status = "ДА" if is_matched else "НЕТ"
        reason = ""
        if is_matched:
            ae    = ev['assoc']
            dt_s  = f"{ev['dt_sec']:>+7.1f}"
            n_s   = _n(ae)
            spr_s = f"{ae['spread_s']:>6.1f}с"
            if ae['n_flag']:
                reason = "[!N]"
        else:
            ae = ev.get('nearest')
            if ae is not None and ev['nearest_dt'] is not None:
                dt_s  = f"{ev['nearest_dt']:>6.1f}*"
                n_s   = _n(ae)
                spr_s = f"{ae['spread_s']:>6.1f}с"
                n_val = len(ae['inliers']) if min_stations_mode == 'sigma' else ae['n_xml']
                abs_dt = abs(ev['nearest_dt'])
                if abs_dt > window_sec and n_val < min_stations:
                    reason = f"окно+{n_col}"
                elif abs_dt > window_sec:
                    reason = "окно"
                elif n_val < min_stations:
                    reason = f"{n_col}<{min_stations}"
            else:
                dt_s, n_s, spr_s = "     —", "    —", "      —"
        print(f"  {i:<4} {str(ev['origin_time']):<22} {ms_s:>4} {status:>6}"
              f" {dt_s} {n_s} {spr_s}  {reason}")
    print(f"{'─' * 80}")


def print_all_events(processed, matched, out_csv=None):
    """
    Все ассоциированные события в одной таблице.
    Для событий, совпавших с каталогом, показывает MLH, T0min, T0mean.
    """
    # pub_id → запись из matched (для быстрого поиска)
    matched_by_pubid = {ev['assoc']['pub_id']: ev for ev in matched}

    HDR = (f"  {'T0 (nearest sta)':<25} {'ML':>5}"
           f" {'MLH':>4} {'T0min,с':>8} {'T0mean,с':>9} {'T0prb,с':>8} {'T0psnr,с':>9}"
           f" {'Sta':>6} {'R,км':>6} {'N_sta':>5} {'spread':>7} {'flags'}")

    # сортируем по ot (если есть) или ot_mean
    def sort_key(ae):
        t = ae['ot'] or ae['ot_mean']
        return t if t is not None else datetime(2099, 1, 1)

    rows = []
    for ae in sorted(processed, key=sort_key):
        ot_s   = str(ae['ot'])[:23]      if ae['ot']      else "—"
        ml_s   = f"{ae['ml']:.2f}"       if ae['ml']      is not None else "—"
        sta_s  = ae['nearest_sta'] or "—"
        r_s    = f"{ae['nearest_r']:.0f}" if ae['nearest_r'] else "—"
        flags  = ("[!N]" if ae['n_flag'] else "")

        cat_match = matched_by_pubid.get(ae['pub_id'])
        if cat_match:
            mlh_s    = f"{cat_match['ms']:.1f}" if cat_match['ms'] is not None else "—"
            dtmin_s  = f"{cat_match['dt_sec']:>+8.1f}"
            dtmean_s = f"{cat_match['dt_mean']:>+9.1f}"    if cat_match['dt_mean']    is not None else "        —"
            dtprb_s  = f"{cat_match['dt_probw']:>+8.1f}"   if cat_match.get('dt_probw')   is not None else "       —"
            dtpsnr_s = f"{cat_match['dt_probsnr']:>+9.1f}" if cat_match.get('dt_probsnr') is not None else "        —"
        else:
            mlh_s, dtmin_s, dtmean_s, dtprb_s, dtpsnr_s = "—", "       —", "        —", "       —", "        —"

        rows.append((ot_s, ml_s, mlh_s, dtmin_s, dtmean_s, dtprb_s, dtpsnr_s,
                     sta_s, r_s, ae['n_xml'], ae['spread_s'], flags))

    if out_csv:
        import csv as _csv
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow(['T0', 'ML', 'MLH', 'T0min_s', 'T0mean_s', 'T0prb_s', 'T0psnr_s',
                        'sta', 'R_km', 'N_sta', 'spread_s', 'flags'])
            for r in rows:
                w.writerow(r)
        print(f"\nВсе события записаны в {out_csv}  ({len(rows)} строк)")
    else:
        print(f"\nВсе ассоциированные события ({len(rows)}):")
        print(HDR)
        for r in rows:
            ot_s, ml_s, mlh_s, dtmin_s, dtmean_s, dtprb_s, dtpsnr_s, sta_s, r_s, n, spr, flags = r
            print(f"  {ot_s:<25} {ml_s:>5}"
                  f" {mlh_s:>4} {dtmin_s} {dtmean_s} {dtprb_s} {dtpsnr_s}"
                  f" {sta_s:>6} {r_s:>6} {n:>5} {spr:>6.1f}с  {flags}")


def _sta_table_lines(ae):
    """Строки таблицы пиков по станциям для события."""
    outlier_stas = {e['station'] for e in ae['outliers']}
    ml_per_sta   = ae.get('ml_per_sta', {})
    has_regime   = any('regime' in e for e in ae['raw'])
    has_prob     = any(e.get('p_prob') is not None for e in ae['raw'])
    has_ml_sta   = bool(ml_per_sta)

    reg_hdr  = f"  {'Mod':>3}" if has_regime else ""
    prob_hdr = f"  {'p_prob':>6}"  if has_prob   else ""
    ml_hdr   = f"  {'ML':>5}"  if has_ml_sta else ""

    header = (f"    {'Sta':<5} {'ΔT,с':>7} {'R,км':>7}"
              f"{reg_hdr}{prob_hdr}{ml_hdr}  {'Tp0':<26}  status")
    lines  = [header]

    for e in sorted(ae['raw'], key=lambda x: x['tp']):
        flag   = "[!STA]" if e['station'] in outlier_stas else "OK"
        reg_s  = f"  {e['regime']:>3}"  if has_regime else ""
        if has_prob:
            prob_s = f"  {e['p_prob']:>6.3f}" if e.get('p_prob') is not None else f"  {'—':>6}"
        else:
            prob_s = ""
        if has_ml_sta:
            ml_v  = ml_per_sta.get(e['station'])
            ml_s  = f"  {ml_v:>+5.2f}" if ml_v is not None else f"  {'—':>5}"
        else:
            ml_s  = ""
        tp0_s = str(e['tp0'])[:26]
        lines.append(
            f"    {e['station']:<5} {e['dt_sp']:>7.2f} {e['R_km']:>7.1f}"
            f"{reg_s}{prob_s}{ml_s}  {tp0_s:<26}  {flag}"
        )
    return lines


def _print_station_table(ae):
    for line in _sta_table_lines(ae):
        print(line)


def _format_station_table_lines(ae):
    return _sta_table_lines(ae)


def print_or_save_all_picks(processed, matched, year=None, month=None, out_csv=None,
                            min_stations=1, min_stations_mode='xml'):
    """
    Вывод таблицы всех ассоциированных событий с пиками по станциям.
    Формат идентичен --show-picks для matched, но для всех событий.
    При out_csv — пишет в файл, иначе в консоль.
    Фильтрует по year/month по оценённому T0.
    """
    matched_by_pubid = {ev['assoc']['pub_id']: ev for ev in matched}

    def get_t0(ae):
        return ae.get('ot_probsnr') or ae['ot'] or ae['ot_mean']

    def in_period(ae):
        t = get_t0(ae)
        if t is None:
            return True
        if year is not None and t.year != year:
            return False
        if month is not None and t.month != month:
            return False
        return True

    def passes_min_sta(ae):
        n = len(ae['inliers']) if min_stations_mode == 'sigma' else ae['n_xml']
        return n >= min_stations

    filtered = sorted(
        (ae for ae in processed if in_period(ae) and passes_min_sta(ae)),
        key=lambda ae: get_t0(ae) or datetime(2099, 1, 1)
    )

    hdr = (f"  {'T0_psnr':<25} {'Cat origin time':<25} {'MLH':>4} {'ML':>5}"
           f" {'ΔCat,с':>7}"
           f" {'Sta':>6} {'R,км':>6} {'N':>4} {'σN':>4}")
    lines = [hdr]
    for ae in filtered:
        ot_s  = str(get_t0(ae))[:25] if get_t0(ae) else "—"
        ml_s  = f"{ae['ml']:.2f}" if ae['ml'] is not None else "  —"
        sta_s = ae['nearest_sta'] or "—"
        r_s   = f"{ae['nearest_r']:.0f}" if ae['nearest_r'] else "—"

        cat_match = matched_by_pubid.get(ae['pub_id'])
        if cat_match:
            mlh_s   = f"{cat_match['ms']:.1f}" if cat_match['ms'] is not None else "  -"
            cat_s   = str(cat_match['origin_time'])[:25]
            dcat_s  = f"{cat_match['dt_sec']:>+7.1f}"
        else:
            mlh_s  = "  -"
            cat_s  = "—"
            dcat_s = "      —"

        lines.append(
            f"  {ot_s:<25} {cat_s:<25} {mlh_s:>4} {ml_s:>5}"
            f" {dcat_s}"
            f" {sta_s:>6} {r_s:>6} {ae['n_xml']:>4} {len(ae['inliers']):>4}"
        )
        lines.extend(_format_station_table_lines(ae))

    text = '\n'.join(lines)
    if out_csv:
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f"\nВсе события с пиками записаны в {out_csv}  ({len(filtered)} событий)")
    else:
        print(f"\nВсе ассоциированные события с пиками ({len(filtered)}):")
        print(text)


def write_quakeml(filtered_events, output_path):
    """
    Записывает отфильтрованные ассоциированные события в QuakeML (BED 1.2).
    Origin time = T0_psnr (или fallback на ot/ot_mean).
    Координаты не вычисляются — origin содержит только время.
    """
    import uuid as _uuid

    BED  = BED_NS
    Q_NS = "http://quakeml.org/xmlns/quakeml/1.2"

    root = ET.Element(f'{{{Q_NS}}}quakeml',
                      {f'xmlns': BED, f'xmlns:q': Q_NS})
    ep = ET.SubElement(root, 'eventParameters',
                       publicID=f'smi:local/{_uuid.uuid4()}')

    for ae in filtered_events:
        t0 = ae.get('ot_probsnr') or ae.get('ot') or ae.get('ot_mean')
        ev_el = ET.SubElement(ep, 'event', publicID=ae['pub_id'])

        orig_id = f'smi:local/{_uuid.uuid4()}'
        ET.SubElement(ev_el, 'preferredOriginID').text = orig_id

        orig_el = ET.SubElement(ev_el, 'origin', publicID=orig_id)
        t_el = ET.SubElement(orig_el, 'time')
        ET.SubElement(t_el, 'value').text = (
            t0.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z' if t0 else '')

        if ae.get('ml') is not None:
            mag_el = ET.SubElement(ev_el, 'magnitude',
                                   publicID=f'smi:local/{_uuid.uuid4()}')
            mv_el = ET.SubElement(mag_el, 'mag')
            ET.SubElement(mv_el, 'value').text = str(ae['ml'])
            ET.SubElement(mag_el, 'type').text = 'ML'

        picks_raw = ae.get('picks_raw', {})
        for sta, info in picks_raw.items():
            net = info.get('net', '')
            for phase in ('p', 's'):
                t_pick = info.get(phase)
                if t_pick is None:
                    continue
                pick_el = ET.SubElement(ev_el, 'pick',
                                        publicID=f'smi:local/{_uuid.uuid4()}')
                pt_el = ET.SubElement(pick_el, 'time')
                ET.SubElement(pt_el, 'value').text = (
                    t_pick.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z')
                ET.SubElement(pick_el, 'waveformID',
                              networkCode=net, stationCode=sta)
                ET.SubElement(pick_el, 'methodID').text = 'smi:local/EqTransformer'
                ET.SubElement(pick_el, 'phaseHint').text = phase.upper()

    ET.indent(root, space='  ')
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"QuakeML записан: {output_path}  ({len(filtered_events)} событий)")


def print_flagged_events(processed):
    flagged = [ae for ae in processed if ae['n_flag']]
    if not flagged:
        print("\nФлагов нет.")
        return
    print(f"\nФлагованные ассоциированные события ({len(flagged)}):")
    for ae in flagged:
        ot_s = str(ae['ot']) if ae['ot'] else "—"
        print(f"  {ot_s}  [!N n={len(ae['inliers'])}]")
        _print_station_table(ae)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Валидация ассоциатора v2 (S-P origin time)")
    parser.add_argument('--assoc',    default=DEFAULT_ASSOC)
    parser.add_argument('--catalog',  default=CATALOG_PATH)
    parser.add_argument('--window',   type=float, default=DEFAULT_WINDOW,
                        help="Допуск сравнения с каталогом, сек (default 30)")
    parser.add_argument('--vp',       type=float, default=DEFAULT_VP)
    parser.add_argument('--vs',       type=float, default=DEFAULT_VS)
    parser.add_argument('--sigma',    type=float, default=DEFAULT_SIGMA,
                        help="Множитель σ для отброса выбросов (default 2.0)")
    parser.add_argument('--year',     type=int,   default=None)
    parser.add_argument('--month',    type=int,   default=None)
    parser.add_argument('--min-mag',  type=float, default=None)
    parser.add_argument('--show-picks', action='store_true',
                        help="Показывать таблицу пиков по станциям для каждого совпадения")
    parser.add_argument('--show-all-picks', action='store_true',
                        help="Показывать таблицу пиков по станциям для ВСЕХ ассоц. событий (с учётом фильтров по году/месяцу)")
    parser.add_argument('--show-flagged', action='store_true',
                        help="Показывать все флагованные ассоц. события (вне зависимости от совпадения с каталогом)")
    parser.add_argument('--show-all', action='store_true',
                        help="Вывести все ассоциированные события в консоль")
    parser.add_argument('--out-csv', default=None,
                        help="С --show-all-picks: текстовый файл с пиками всех событий; иначе — CSV-таблица всех событий")
    parser.add_argument('--out-xml', default=None,
                        help="Записать отфильтрованные ассоц. события в QuakeML (те же фильтры что и --show-all-picks)")
    parser.add_argument('--r-min', type=float, default=20.0,
                        help="Минимальный R (км) станции для вычисления T0 (default: 20)")
    parser.add_argument('--r-max', type=float, default=None,
                        help="Максимальный R (км) станции для вычисления T0; дальние исключаются (default: без ограничения)")
    parser.add_argument('--dual-vel', action='store_true',
                        help=f"Двухскоростная модель: Pg/Sg < {R_PN_THR:.0f} км (Vp={DEFAULT_VP}, Vs={DEFAULT_VS}), "
                             f"Pn/Sn >= {R_PN_THR:.0f} км (Vp={VP_PN}, Vs={VS_SN})")
    parser.add_argument('--amps', default=DEFAULT_AMPS,
                        help=f"CSV с амплитудами (pub_id,sta,A) для расчёта ML (default: {DEFAULT_AMPS})")
    parser.add_argument('--no-amps', action='store_true',
                        help="Не загружать амплитуды, не считать ML")
    parser.add_argument('--sta-corrections', default=None,
                        help="CSV станционных поправок (station,S) — добавляются к ML каждой станции")
    parser.add_argument('--prob-dir', default=None,
                        help="Папка assoc_input с X_prediction_results.csv для взвешивания T0 по p_probability")
    parser.add_argument('--match-by', default='min', choices=['min', 'mean', 'psnr'],
                        help="T0 для матчинга с каталогом: min=ближайшая станция (default), mean=σ-среднее, psnr=p_prob×p_snr")
    parser.add_argument('--min-stations', type=int, default=1,
                        help="Минимальное число станций для матчинга (default: 1)")
    parser.add_argument('--min-stations-mode', default='xml', choices=['xml', 'sigma'],
                        help="Столбец для --min-stations: xml=N (все станции из XML), sigma=σN (inliers после σ-фильтра)")
    args = parser.parse_args()

    k = args.vp * args.vs / (args.vp - args.vs)
    print(f"Каталог:   {args.catalog}")
    print(f"Ассоциатор:{args.assoc}")
    if args.dual_vel:
        k_pn = VP_PN * VS_SN / (VP_PN - VS_SN)
        print(f"Скорости:  Pg/Sg < {R_PN_THR:.0f} км → Vp={args.vp}, Vs={args.vs}, k={k:.4f}")
        print(f"           Pn/Sn >= {R_PN_THR:.0f} км → Vp={VP_PN}, Vs={VS_SN}, k={k_pn:.4f}  [--dual-vel]")
    else:
        print(f"Vp={args.vp} км/с  Vs={args.vs} км/с  k={k:.4f}")
    r_max_s = f"{args.r_max} км" if args.r_max is not None else "без ограничения"
    print(f"Окно:      ±{args.window} сек  σ-множитель={args.sigma}  R=[{args.r_min}–{r_max_s}]\n")

    catalog = load_catalog(args.catalog)
    print(f"Событий в каталоге: {len(catalog)}")

    if args.year is not None or args.month is not None:
        before  = len(catalog)
        catalog = [
            ev for ev in catalog
            if (args.year  is None or ev['origin_time'].year  == args.year)
            and (args.month is None or ev['origin_time'].month == args.month)
        ]
        print(f"После фильтра год={args.year} месяц={args.month}: {len(catalog)} (из {before})")

    if args.min_mag is not None:
        before  = len(catalog)
        catalog = [ev for ev in catalog if ev['ms'] is not None and ev['ms'] >= args.min_mag]
        print(f"После фильтра Ms >= {args.min_mag}: {len(catalog)} (из {before})")

    assoc_events = load_associations(args.assoc)
    print(f"Ассоциированных событий: {len(assoc_events)}")

    amplitudes = None
    if not args.no_amps and os.path.isfile(args.amps):
        amplitudes = load_amplitudes(args.amps)
        print(f"Амплитуд загружено: {len(amplitudes)}  ({args.amps})")
    elif not args.no_amps:
        print(f"Файл амплитуд не найден: {args.amps}  (ML не будет)")

    prob_dict = None
    if args.prob_dir:
        prob_dict = load_pick_probabilities(args.prob_dir)
        print(f"Вероятностей пиков загружено: {len(prob_dict)}  ({args.prob_dir})")

    sta_corr = {}
    if args.sta_corrections:
        with open(args.sta_corrections, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                sta = row.get('station', '').strip()
                val = row.get('S', row.get('correction', '')).strip()
                if sta and val:
                    try:
                        sta_corr[sta] = float(val)
                    except ValueError:
                        pass
        print(f"Станционных поправок: {len(sta_corr)}  ({args.sta_corrections})")
    print()

    matched, missed, processed = validate(
        catalog, assoc_events, args.window, args.vp, args.vs, args.sigma,
        args.r_min, args.r_max, amplitudes, sta_corr,
        use_dual_vel=args.dual_vel, prob_dict=prob_dict,
        match_by=args.match_by, min_stations_match=args.min_stations,
        min_stations_mode=args.min_stations_mode)
    print_report(matched, missed, processed, args.window, args.sigma, args.show_picks,
                 min_stations=args.min_stations, min_stations_mode=args.min_stations_mode)

    if args.show_flagged:
        print_flagged_events(processed)

    if args.show_all_picks:
        print_or_save_all_picks(
            processed, matched,
            year=args.year, month=args.month,
            out_csv=args.out_csv if args.out_csv else None,
            min_stations=args.min_stations,
            min_stations_mode=args.min_stations_mode,
        )
    elif args.show_all or args.out_csv:
        print_all_events(processed, matched, out_csv=args.out_csv)

    if args.out_xml:
        def _get_t0(ae):
            return ae.get('ot_probsnr') or ae.get('ot') or ae.get('ot_mean')
        def _in_period(ae):
            t = _get_t0(ae)
            if t is None:
                return True
            if args.year  is not None and t.year  != args.year:  return False
            if args.month is not None and t.month != args.month: return False
            return True
        def _passes(ae):
            n = len(ae['inliers']) if args.min_stations_mode == 'sigma' else ae['n_xml']
            return n >= args.min_stations
        xml_events = sorted(
            (ae for ae in processed if _in_period(ae) and _passes(ae)),
            key=lambda ae: _get_t0(ae) or datetime(2099, 1, 1)
        )
        write_quakeml(xml_events, args.out_xml)


if __name__ == "__main__":
    main()
