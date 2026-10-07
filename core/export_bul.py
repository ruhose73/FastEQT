"""
export_bul.py — экспорт ассоциированных событий в бюллетень формата IMS1.0:SHORT
(визуально похожий на bul/*.BUL от ГС РАН), без сравнения с catalog.xlsx.

Использует ту же логику фильтрации, что validate_associator_v2.py: S-P origin
time по двухскоростной модели Pg/Sg-Pn/Sn (estimate_origin_times_v2), σ-фильтр
выбросов (sigma_filter), ML по Дягилеву et al. 2023 (compute_ml). Отличие —
нет сравнения с каталогом, на выходе ВСЕ события, прошедшие фильтры.

Гипоцентр по умолчанию не заполняется (`--locator-mode off`, старое поведение) —
Lat/Lon/Depth и производные от них поля (Smaj/Smin/Az/RMS) остаются пустыми,
`fmt_origin_line()` используется как раньше. mdist/Mdist — не производные от
эпицентра: это min/max R_km (S-P расстояние до станции), переведённое в градусы —
их вычислить можно и без координат, поэтому они заполняются всегда, независимо
от `--locator-mode`.

С `--locator-mode enrich|filter` (`context/locsat-plan.md`, Шаг 5) подключается
`core/locator.py` (LOCSAT через SeisComP) — `--hypocenters` указывает на его выход
(`workspace/locator/output/hypocenters.csv`). Для событий, у которых там есть
сошедшееся решение, `fmt_origin_line_v2()` заполняет Lat/Lon/Depth/Err(depth)/RMS
и меняет код метода локации с `'o'` (other, S-P) на `'i'` (inversion) — по
спецификации IMS1.0:SHORT (см. комментарий у `ORIGIN_HEADER`). `filter`
дополнительно отбрасывает события, чьё (сошедшееся) решение не проходит
`_passes_locator_quality()` — события без решения локатора (или не найденные в
`hypocenters.csv`) НЕ считаются провалившими фильтр, для них — старое поведение
(`fmt_origin_line`, только T0). Раздел 1.7b плана: `locator.py` порогов не знает,
их считает и применяет только этот модуль.

Регион в шапке EVENT определяется по административному региону ближайшей
станции (metadata-25/*.xml, StationXML Site/Name, поле после первой запятой).

Станции дальше --max-station-dist-km от опорной точки (--ref-lat/--ref-lon, по
умолчанию Сочи 43.6/40.0; по умолчанию 800км) исключаются
из расчёта T0/ML целиком — в output_cpu/assoc_input_cpu_2 вперемешку с
кавказской сетью (макс. ~700км друг от друга) попали опорные станции по всей
России и даже Антарктиде (см. load_station_coords/filter_far_stations).

ВАЖНО: раскладка столбцов — приближение к визуальному виду bul/*.BUL, а не
побайтовая копия официальной спецификации ISF/IMS1.0 целиком (часть подписей в
шапке не выровнена по правому краю поля, напр. Date/Time). Поля, которые
реально заполняются (Date, Time, Ndef, Nsta, Author, OrigID, Sta, Phase, Time,
SNR, Amp, Magnitude, ArrID — всегда; Latitude, Longitude, Depth, Err depth, RMS
— только с `--locator-mode enrich|filter`), выровнены по позициям, выверенным
побайтово по реальным официальным бюллетеням (см. комментарий у
`fmt_origin_line_v2()`); остальные — просто резерв места под пробелами.
Перед отправкой в НИИ стоит сверить пару событий с примером глазами.

Использование:
    python core/export_bul.py --year 2025 --month 1 --out workspace/bulletin/output/2025_01.BUL
    python core/export_bul.py --min-stations 4 --out workspace/bulletin/output/all.BUL
"""

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_associator_v2 as v2  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ASSOC     = v2.DEFAULT_ASSOC
DEFAULT_AMPS      = v2.DEFAULT_AMPS
DEFAULT_PROB_DIR  = os.path.join(_ROOT, "workspace", "associator", "input")
DEFAULT_METADATA_DIR   = os.path.join(_ROOT, "workspace", "data_processors", "input", "metadata")
DEFAULT_OUTPUT_CPU_DIR = os.path.join(_ROOT, "workspace", "detector", "output")
DEFAULT_HYPOCENTERS    = os.path.join(_ROOT, "workspace", "locator", "output", "hypocenters.csv")

# depth_uncertainty_km == 0.0 (в пределах эпсилон) — надёжный признак того, что LOCSAT упёрся
# в границу таблицы годографов (0 или ~750 км), а не решил глубину как свободный параметр
# (context/locsat-plan.md, раздел 5, "Результат... 2026-09-09" — проверено на 20 событиях,
# не гипотеза). НЕ порог качества по RMS — тот пользователь выбирает сам через
# --locator-max-rms, откалиброванного значения по умолчанию пока нет (Шаг 6 плана не сделан).
DEFAULT_LOCATOR_MIN_DEPTH_UNCERTAINTY_KM = 1e-3

AUTHOR = "EQT"
KM_PER_DEG = 111.195

# Легитимная кавказская сеть — максимум ~700км друг от друга (Сочи-Дербент).
# В output_cpu/assoc_input_cpu_2 вперемешку с ней попали опорные/референсные
# станции по всей России (Москва, Пермский край, Ленобласть, Калининград,
# Архангельск, Заполярье) и даже Владивосток/Антарктида — все они на 1300+ км
# дальше, чем самая дальняя реальная кавказская станция. Отсекаем по расстоянию
# от координат станции (не по названию региона — иначе легитимные, но не
# описанные в metadata-25 станции типа KGUR/MRMR отсеклись бы тоже).
CAUCASUS_REF_LAT = 43.6   # Сочи
CAUCASUS_REF_LON = 40.0
DEFAULT_MAX_STATION_DIST_KM = 800.0

# ── Регион по ближайшей станции (StationXML Site/Name) ────────────────────────

REGION_ALIASES = {
    'dagestan rep.':               'DAGESTAN REGION',
    'north osetia rep.':           'NORTH OSETIA REGION',
    'krasnodar reg.':              'KRASNODAR REGION',
    'stavropol reg.':               'STAVROPOL REGION',
    'chechen rep.':                'CHECHEN REGION',
    'chechen republic':            'CHECHEN REGION',
    'kabardino-balkar rep.':       'KABARDINO-BALKAR REGION',
    'kabardino-balkaria rep':      'KABARDINO-BALKAR REGION',
    'karachay-cherkess republic':  'KARACHAY-CHERKESS REGION',
    'rostov reg.':                 'ROSTOV REGION',
    'arkhangelsk reg.':            'ARKHANGELSK REGION',
    'primorsky reg.':              'PRIMORSKY REGION',
}
DEFAULT_REGION = 'NORTH CAUCASUS REGION'

# Опечатка в metadata-25/RU_TMNR_*.xml: "Tamanskiy. Krasnodar reg., Russia"
# (точка вместо запятой после города) -> парсер иначе берёт "Russia" как регион.
STATION_REGION_OVERRIDES = {
    'TMNR': 'KRASNODAR REGION',
}

_FDSN_NS = {'f': 'http://www.fdsn.org/xml/station/1'}


def load_station_regions(metadata_dir):
    """station -> 'XXX REGION', разбирая Site/Name из metadata-25/*.xml."""
    regions = {}
    if not os.path.isdir(metadata_dir):
        return regions
    for fname in os.listdir(metadata_dir):
        if not fname.endswith('.xml'):
            continue
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        sta = parts[1].strip()
        try:
            tree = ET.parse(os.path.join(metadata_dir, fname))
        except ET.ParseError:
            continue
        name_el = tree.getroot().find('.//f:Station/f:Site/f:Name', _FDSN_NS)
        if name_el is None or not name_el.text:
            continue
        subj = name_el.text.split(',')
        if len(subj) < 2:
            continue
        key = subj[1].strip().lower()
        regions[sta] = REGION_ALIASES.get(key, key.upper() + ' REGION')

    regions.update(STATION_REGION_OVERRIDES)
    return regions


# ── Координаты станций и отсев дальних (не-кавказских) станций ─────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_station_coords(metadata_dir, output_cpu_dir):
    """
    station -> (lat, lon). Приоритет metadata-25/*.xml (StationXML); для станций,
    которых там нет, — координаты из первой строки их CSV в output_cpu/{STA}/.
    """
    coords = {}
    if os.path.isdir(metadata_dir):
        for fname in os.listdir(metadata_dir):
            if not fname.endswith('.xml'):
                continue
            parts = fname.split('_')
            if len(parts) < 2:
                continue
            sta = parts[1].strip()
            try:
                tree = ET.parse(os.path.join(metadata_dir, fname))
            except ET.ParseError:
                continue
            root  = tree.getroot()
            lat_el = root.find('.//f:Station/f:Latitude', _FDSN_NS)
            lon_el = root.find('.//f:Station/f:Longitude', _FDSN_NS)
            if lat_el is not None and lon_el is not None:
                coords[sta] = (float(lat_el.text), float(lon_el.text))

    if os.path.isdir(output_cpu_dir):
        for sta in os.listdir(output_cpu_dir):
            if sta in coords:
                continue
            sta_dir = os.path.join(output_cpu_dir, sta)
            if not os.path.isdir(sta_dir):
                continue
            csv_files = [f for f in os.listdir(sta_dir) if f.endswith('.csv')]
            if not csv_files:
                continue
            with open(os.path.join(sta_dir, csv_files[0]), newline='', encoding='utf-8') as f:
                row = next(csv.DictReader(f), None)
            if row and row.get('station_lat') and row.get('station_lon'):
                try:
                    coords[sta] = (float(row['station_lat']), float(row['station_lon']))
                except ValueError:
                    pass
    return coords


def filter_far_stations(picks_by_sta, station_coords, ref_lat, ref_lon, max_km, excluded_out=None):
    """Убирает станции дальше max_km от (ref_lat, ref_lon). Станции без известных
    координат НЕ отбрасываются вслепую — только те, для которых точно посчитано."""
    if max_km is None or station_coords is None:
        return picks_by_sta
    kept = {}
    for sta, picks in picks_by_sta.items():
        coords = station_coords.get(sta)
        if coords is None:
            kept[sta] = picks
            continue
        d = _haversine_km(ref_lat, ref_lon, coords[0], coords[1])
        if d <= max_km:
            kept[sta] = picks
        elif excluded_out is not None:
            excluded_out.add(sta)
    return kept


# ── S-SNR (validate_associator_v2.load_pick_probabilities грузит только P) ────

def load_s_snr(assoc_input_dir):
    """{(station, round(s_arrival_time.timestamp(),3)): s_snr}."""
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
                s_t = row.get('s_arrival_time', '').strip()
                if not s_t:
                    continue
                try:
                    st_dt = datetime.fromisoformat(s_t)
                except ValueError:
                    continue
                snr_s = row.get('s_snr', '').strip()
                try:
                    snr_v = float(snr_s) if snr_s and snr_s.lower() != 'nan' else None
                except ValueError:
                    snr_v = None
                result[(sta, round(st_dt.timestamp(), 3))] = snr_v
    return result


# ── Гипоцентры core/locator.py (--locator-mode enrich|filter) ─────────────────

def load_hypocenters(path):
    """
    event_id (publicID, тот же ключ, что ae['pub_id']) -> строка core/locator.py::CSV_FIELDS,
    с числовыми полями уже приведёнными к float/bool (сырой CSV пишет их как текст/'True'/'False').
    Событие без сошедшегося решения (converged=False) тоже попадает в словарь — по нему
    вызывающий код узнаёт "решения нет" и откатывается к fmt_origin_line() (раздел 7 плана),
    а не молча считает событие отсутствующим.
    """
    hyp = {}
    if not path or not os.path.isfile(path):
        return hyp
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            event_id = row.get('event_id', '').strip()
            if not event_id:
                continue
            row['converged'] = row.get('converged', '').strip() == 'True'
            for key in ('lat', 'lon', 'depth_km', 'depth_uncertainty_km', 'rms'):
                val = (row.get(key) or '').strip()
                row[key] = float(val) if val else None
            hyp[event_id] = row
    return hyp


def _passes_locator_quality(row, max_rms=None,
                             min_depth_uncertainty_km=DEFAULT_LOCATOR_MIN_DEPTH_UNCERTAINTY_KM):
    """
    Решение "проходит/не проходит" по метрикам из hypocenters.csv (раздел 1.7b плана —
    locator.py таких решений не принимает, это ответственность потребителя). Вызывать только
    для строк с converged=True — несошедшееся решение обрабатывается отдельно (раздел 7:
    отсутствие решения — не провал фильтра, а откат к старому поведению).

    - depth_uncertainty_km ниже min_depth_uncertainty_km (по умолчанию, условно, "== 0") —
      признак упора в границу таблицы годографов, не настоящее решение (см. комментарий у
      DEFAULT_LOCATOR_MIN_DEPTH_UNCERTAINTY_KM).
    - max_rms — опциональный порог невязки (сек); без него (None) не проверяется вообще, т.к.
      откалиброванного значения по умолчанию нет (Шаг 6 плана "калибровка порога" не сделан).
    """
    if row['depth_uncertainty_km'] is None or row['depth_uncertainty_km'] < min_depth_uncertainty_km:
        return False
    if max_rms is not None and (row['rms'] is None or row['rms'] > max_rms):
        return False
    return True


# ── Фильтрация событий (без сравнения с каталогом) ────────────────────────────

def process_events(assoc_events, vp, vs, sigma_mult, r_min, r_max,
                    amplitudes, sta_corr, prob_dict,
                    station_coords=None, ref_lat=CAUCASUS_REF_LAT,
                    ref_lon=CAUCASUS_REF_LON, max_station_dist_km=None):
    """
    Тот же расчёт T0/ML, что в validate_associator_v2.validate(), но без
    привязки к catalog.xlsx. Всегда двухскоростная модель (Pg/Sg-Pn/Sn) —
    без неё Phase в пикинге не имел бы смысла.

    Если задан station_coords/max_station_dist_km — станции дальше этого
    расстояния от (ref_lat, ref_lon) убираются из пикинга события ДО расчёта
    T0/ML (см. filter_far_stations) — это про не-кавказские опорные станции,
    случайно попавшие в assoc-input вместе с региональной сетью.
    """
    processed = []
    excluded_stations = set()
    for ae in assoc_events:
        picks = filter_far_stations(ae['picks'], station_coords, ref_lat, ref_lon,
                                     max_station_dist_km, excluded_out=excluded_stations)
        if not picks:
            continue
        raw = v2.estimate_origin_times_v2(picks, vp, vs, r_min, r_max,
                                           prob_dict=prob_dict)
        if not raw:
            continue

        inliers, outliers = v2.sigma_filter(raw, sigma_mult)
        n_flag  = len(inliers) < v2.MIN_STATIONS
        nearest = min(inliers, key=lambda e: e['dt_sp']) if not n_flag else None
        ot      = nearest['t0'] if nearest else None

        ot_probsnr = None
        if not n_flag:
            ref = inliers[0]['t0']
            psnr_entries = [(e['p_prob'] * e['p_snr'], e) for e in inliers
                            if e.get('p_prob') is not None and e['p_prob'] > 0
                            and e.get('p_snr') is not None and e['p_snr'] > 0]
            if len(psnr_entries) >= v2.MIN_STATIONS:
                ps_total = sum(pw for pw, _ in psnr_entries)
                off = sum(pw * (e['t0'] - ref).total_seconds()
                          for pw, e in psnr_entries) / ps_total
                ot_probsnr = ref + timedelta(seconds=off)

        t0 = ot_probsnr or ot
        if t0 is None:
            continue

        ml, ml_n, ml_per_sta = None, 0, {}
        if amplitudes is not None:
            r_dict = v2._r_dict_all(picks, vp, vs)
            ml, ml_n, ml_per_sta = v2.compute_ml(ae['pub_id'], r_dict, amplitudes, sta_corr)

        n_def = sum(1 for pk in picks.values() for ph in ('p', 's') if ph in pk)

        processed.append({
            'pub_id':      ae['pub_id'],
            'picks_raw':   picks,
            'raw':         raw,
            'inliers':     inliers,
            't0':          t0,
            'n_xml':       len(picks),
            'n_def':       n_def,
            'n_sta':       len(inliers),
            'nearest_sta': nearest['station'] if nearest else raw[0]['station'],
            'ml':          ml,
            'ml_n':        ml_n,
            'ml_per_sta':  ml_per_sta,
        })
    return processed, excluded_stations


# ── Форматирование BUL ─────────────────────────────────────────────────────────
#
# Точная побайтовая раскладка по официальной спецификации IDC-3.4.1Rev1
# "Formats and Protocols for Messages — IMS1.0" (Table 41: Origin Block Format,
# Table 42: Phase Block Format; https://www.isc.ac.uk/standards/isf/download/ims1_0.pdf).
# Колонки Phase Block проверены побайтово на примерах из приложения A документа
# (стр. A16-A17) — совпадают буква в букву. Origin Block (поля, заполнявшиеся до
# Шага 5 плана: Date/Time/Ndef/Nsta/mdist/Mdist/analysis type/location
# method/event type/Author/OrigID) сверен по номерам колонок из той же таблицы.
#
# ⚠️ Поля, добавленные на Шаге 5 (Latitude/Longitude/Depth/Err depth/RMS,
# `fmt_origin_line_v2()`) — НЕ по номерам колонок из этого PDF: при извлечении
# текста из его двухколоночной вёрстки (`pdftotext -layout`) Position/Format
# перемешиваются с описаниями полей не в том порядке (проверено конкретно на
# этом PDF, 2026-09-13) — доверять числам оттуда для ЭТИХ полей нельзя. Вместо
# этого позиции выверены побайтово по двум независимым реальным источникам,
# давшим один и тот же результат: официальный бюллетень ГС РАН
# (`workspace/bulletin/input/2024_jan-dec_NCAU.BUL`) и пример с сайта ISC
# (событие 11691339, `www.isc.ac.uk/cgi-bin/web-db-v4?...out_format=IMS1.0`).
#
# Важная деталь, из-за которой Def "наезжал" на SRes: Def — это НЕ строка "T__"
# одним куском, а три однобайтовых флага в колонках 74/75/76 (T/_, A/_, S/_ —
# time/azimuth/slowness defining). "T__" получается только потому, что у нас
# определено время прихода, но не азимут и не медленность.

ORIGIN_HEADER = ("   Date       Time        Err   RMS Latitude Longitude  Smaj  Smin"
                 "  Az Depth   Err Ndef Nsta Gap  mdist  Mdist Qual   Author      OrigID")
MAG_HEADER    = "Magnitude  Err Nsta Author      OrigID"
PICK_HEADER   = ("Sta     Dist  EvAz Phase        Time      TRes  Azim AzRes   Slow"
                 "   SRes Def   SNR       Amp   Per Qual Magnitude    ArrID")

ORIGIN_LEN = 136
MAG_LEN    = 38
PHASE_LEN  = 122


def _put(chars, start, end, value, align='right'):
    """Вставляет value в chars на позиции [start,end] (1-индексация, включительно)."""
    width = end - start + 1
    s = str(value)
    s = s[-width:].rjust(width) if align == 'right' else s[:width].ljust(width)
    chars[start - 1:end] = list(s)


def fmt_origin_line(t0, n_def, n_sta, mdist, Mdist, orig_id):
    c = [' '] * ORIGIN_LEN
    _put(c, 1, 10, t0.strftime('%Y/%m/%d'), 'left')          # date
    _put(c, 12, 22, t0.strftime('%H:%M:%S.%f')[:-4], 'left')  # time (hh:mm:ss.ss)
    _put(c, 84, 87, n_def, 'right')                            # Ndef
    _put(c, 89, 92, n_sta, 'right')                            # Nsta
    if mdist is not None:
        _put(c, 98, 103, f"{mdist:.2f}")                       # mdist (degrees)
    if Mdist is not None:
        _put(c, 105, 110, f"{Mdist:.2f}")                      # Mdist (degrees)
    _put(c, 112, 112, 'a')                                     # analysis type: automatic
    _put(c, 114, 114, 'o')                                     # location method: other (S-P, без инверсии)
    _put(c, 116, 117, 'uk')                                    # event type: unknown (не верифицировано)
    _put(c, 119, 127, AUTHOR, 'left')                          # author
    _put(c, 129, 136, orig_id)                                 # origid
    return ''.join(c).rstrip()


def _fmt_fit(value, width, max_decimals):
    """
    Форматирует float так, чтобы не превысить width символов, снижая число знаков после
    запятой при необходимости (переменная точность — так же ведут себя реальные бюллетени
    ISC/ГС РАН, где, например, RMS в одном и том же 5-символьном поле — то "1.140", то
    "119.4" в зависимости от величины). Нужно, чтобы _put() не обрезал старшие разряды
    молча (_put() усекает строку СЛЕВА, если она не влезает — для числа это испортило бы
    значение, напр. RMS 119.41 стало бы "19.41").
    """
    for dec in range(max_decimals, -1, -1):
        s = f"{value:.{dec}f}"
        if len(s) <= width:
            return s
    return f"{value:.0f}"  # даже целая часть не влезает — не наша ответственность обрезать


def fmt_origin_line_v2(t0, n_def, n_sta, mdist, Mdist, orig_id,
                        lat=None, lon=None, depth_km=None, depth_uncertainty_km=None, rms=None):
    """
    fmt_origin_line() + Latitude/Longitude/Depth/Err(depth)/RMS/location method реальные
    (не пустые) — заполняются только когда есть решение локатора (Rule 13: fmt_origin_line()
    не трогаем, эта функция рядом, для событий с core/locator.py-решением).

    Позиции столбцов (Latitude/Longitude/Smaj/Smin/Az/Depth/Err/RMS), которых не было в
    fmt_origin_line(), НЕ взяты из спецификации IDC-3.4.1Rev1 на глаз (PDF Table 41 при
    извлечении текста из двухколоночной вёрстки перемешивает Position/Format с описаниями
    полей не в том порядке — проверено, доверять нельзя) — вместо этого побайтово выверены по
    реальному официальному бюллетеню ГС РАН (`workspace/bulletin/input/2024_jan-dec_NCAU.BUL`)
    и по примеру ISC IMS1.0 (событие 11691339, `www.isc.ac.uk`) — оба независимо дают одни и
    те же границы полей, что и подтверждает их корректность.
    """
    c = list(fmt_origin_line(t0, n_def, n_sta, mdist, Mdist, orig_id).ljust(ORIGIN_LEN))
    if rms is not None:
        _put(c, 31, 35, _fmt_fit(rms, 5, 2))
    if lat is not None:
        _put(c, 37, 44, f"{lat:.4f}")   # +-90, всегда влезает в 8 символов с 4 знаками
    if lon is not None:
        _put(c, 46, 54, f"{lon:.4f}")   # +-180, всегда влезает в 9 символов с 4 знаками
    if depth_km is not None:
        _put(c, 72, 76, _fmt_fit(depth_km, 5, 1))
    if depth_uncertainty_km is not None:
        _put(c, 78, 82, _fmt_fit(depth_uncertainty_km, 5, 1))
    if lat is not None or lon is not None or depth_km is not None:
        _put(c, 114, 114, 'i')  # location method: inversion (LOCSAT/NonLinLoc), не 'o' (S-P)
    return ''.join(c).rstrip()


def fmt_magnitude_line(ml, ml_n, orig_id):
    c = [' '] * MAG_LEN
    _put(c, 1, 5, 'ML', 'left')
    _put(c, 7, 10, f"{ml:.1f}")
    _put(c, 16, 19, ml_n)
    _put(c, 21, 29, AUTHOR, 'left')
    _put(c, 31, 38, orig_id)
    return ''.join(c).rstrip()


def fmt_pick_line(sta, phase, t, snr, amp_nm, mag_val, arr_id):
    c = [' '] * PHASE_LEN
    _put(c, 1, 5, sta, 'left')                                  # station code
    _put(c, 20, 27, phase, 'left')                              # phase code
    _put(c, 29, 40, t.strftime('%H:%M:%S.%f')[:-3], 'left')     # time (hh:mm:ss.sss)
    _put(c, 74, 74, 'T')                                        # time defining
    _put(c, 75, 75, '_')                                        # azimuth defining (нет данных)
    _put(c, 76, 76, '_')                                        # slowness defining (нет данных)
    _put(c, 100, 100, 'a')                                      # pick type: automatic
    _put(c, 101, 101, '_')                                      # short-period motion: null
    _put(c, 102, 102, '_')                                      # onset quality: null
    if snr is not None:
        _put(c, 78, 82, f"{snr:.1f}")
    if amp_nm is not None:
        _put(c, 84, 92, f"{amp_nm:.1f}")
    if mag_val is not None:
        _put(c, 104, 108, 'ML', 'left')
        _put(c, 110, 113, f"{mag_val:.1f}")
    _put(c, 115, 122, arr_id)
    return ''.join(c).rstrip()


def write_bul(processed, region_map, amplitudes, p_snr_dict, s_snr_dict, out_path):
    orig_id = 0
    arr_id  = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("DATA_TYPE BULLETIN IMS1.0:SHORT\n")
        f.write("NORTH CAUCASUS REGION Bulletin (EQTransformer, automatic, for review)\n\n")

        for ae in sorted(processed, key=lambda x: x['t0']):
            orig_id += 1
            region = region_map.get(ae['nearest_sta'], DEFAULT_REGION)

            r_kms = [e['R_km'] for e in (ae['inliers'] or ae['raw'])]
            mdist = min(r_kms) / KM_PER_DEG if r_kms else None
            Mdist = max(r_kms) / KM_PER_DEG if r_kms else None

            f.write(f"EVENT {orig_id} {region}\n")
            f.write(ORIGIN_HEADER + "\n")
            loc = ae.get('locator')
            if loc and loc['converged']:
                origin_line = fmt_origin_line_v2(
                    ae['t0'], ae['n_def'], ae['n_sta'], mdist, Mdist, orig_id,
                    lat=loc['lat'], lon=loc['lon'], depth_km=loc['depth_km'],
                    depth_uncertainty_km=loc['depth_uncertainty_km'], rms=loc['rms'])
            else:
                origin_line = fmt_origin_line(ae['t0'], ae['n_def'], ae['n_sta'],
                                               mdist, Mdist, orig_id)
            f.write(origin_line + "\n\n")

            if ae['ml'] is not None:
                f.write(MAG_HEADER + "\n")
                f.write(fmt_magnitude_line(ae['ml'], ae['ml_n'], orig_id) + "\n\n")

            f.write(PICK_HEADER + "\n")
            regime_by_sta = {e['station']: e['regime'] for e in ae['raw']}
            for sta, picks in sorted(ae['picks_raw'].items(),
                                      key=lambda kv: kv[1].get('p') or kv[1].get('s')):
                regime = regime_by_sta.get(sta, 'Pg')  # Pg или Pn
                if 'p' in picks:
                    arr_id += 1
                    snr = p_snr_dict.get((sta, round(picks['p'].timestamp(), 3)))
                    f.write(fmt_pick_line(sta, regime, picks['p'], snr, None, None, arr_id) + "\n")
                if 's' in picks:
                    arr_id += 1
                    s_phase = 'Sg' if regime == 'Pg' else 'Sn'
                    snr    = s_snr_dict.get((sta, round(picks['s'].timestamp(), 3)))
                    amp    = amplitudes.get((ae['pub_id'], sta)) if amplitudes else None
                    amp_nm = amp * 1e9 if amp else None
                    ml_val = ae['ml_per_sta'].get(sta)
                    f.write(fmt_pick_line(sta, s_phase, picks['s'], snr, amp_nm, ml_val, arr_id) + "\n")
            f.write("\n")

        f.write("STOP\n")
    print(f"BUL записан: {out_path}  ({orig_id} событий, {arr_id} пиков)")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Экспорт ассоциированных событий в бюллетень IMS1.0:SHORT (без сверки с каталогом)")
    parser.add_argument('--assoc',   default=DEFAULT_ASSOC)
    parser.add_argument('--amps',    default=DEFAULT_AMPS)
    parser.add_argument('--no-amps', action='store_true')
    parser.add_argument('--sta-corrections', default=None)
    parser.add_argument('--prob-dir', default=DEFAULT_PROB_DIR)
    parser.add_argument('--metadata-dir', default=DEFAULT_METADATA_DIR)
    parser.add_argument('--output-cpu-dir', default=DEFAULT_OUTPUT_CPU_DIR,
                        help="Фолбэк для координат станций, отсутствующих в metadata-25")
    parser.add_argument('--max-station-dist-km', type=float, default=DEFAULT_MAX_STATION_DIST_KM,
                        help=f"Станции дальше этого расстояния от Сочи (км) исключаются из расчёта "
                             f"T0/ML (default: {DEFAULT_MAX_STATION_DIST_KM:.0f}); 0/отрицательное — не фильтровать")
    parser.add_argument('--ref-lat', type=float, default=CAUCASUS_REF_LAT,
                        help=f"Опорная точка для --max-station-dist-km, широта (default: {CAUCASUS_REF_LAT}, Сочи)")
    parser.add_argument('--ref-lon', type=float, default=CAUCASUS_REF_LON,
                        help=f"Опорная точка для --max-station-dist-km, долгота (default: {CAUCASUS_REF_LON}, Сочи)")
    parser.add_argument('--vp', type=float, default=v2.DEFAULT_VP)
    parser.add_argument('--vs', type=float, default=v2.DEFAULT_VS)
    parser.add_argument('--sigma', type=float, default=v2.DEFAULT_SIGMA)
    parser.add_argument('--r-min', type=float, default=20.0)
    parser.add_argument('--r-max', type=float, default=None)
    parser.add_argument('--year',  type=int, default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--min-stations', type=int, default=1)
    parser.add_argument('--min-stations-mode', default='xml', choices=['xml', 'sigma'])
    parser.add_argument('--locator-mode', default='off', choices=['off', 'enrich', 'filter'],
                        help="off (default) — старое поведение, Lat/Long/Depth пустые; "
                             "enrich — заполнять их из --hypocenters там, где есть решение "
                             "core/locator.py; filter — enrich + отбрасывать события, чьё "
                             "решение не проходит --locator-max-rms/--locator-min-depth-uncertainty-km "
                             "(_passes_locator_quality(), context/locsat-plan.md раздел 7). "
                             "Событие БЕЗ решения локатора (не сошлось/нет в файле) не "
                             "отбрасывается в filter — для него всегда старое поведение (T0-only)")
    parser.add_argument('--hypocenters', default=DEFAULT_HYPOCENTERS,
                        help=f"CSV — выход core/locator.py, используется только при "
                             f"--locator-mode enrich|filter (default: {DEFAULT_HYPOCENTERS})")
    parser.add_argument('--locator-max-rms', type=float, default=None,
                        help="--locator-mode filter: отбрасывать события с rms выше этого "
                             "порога (сек). Без значения (по умолчанию) RMS не проверяется — "
                             "откалиброванного порога по умолчанию нет, Шаг 6 плана "
                             "(валидация против каталога) ещё не сделан, см. context/locsat-plan.md")
    parser.add_argument('--locator-min-depth-uncertainty-km', type=float,
                        default=DEFAULT_LOCATOR_MIN_DEPTH_UNCERTAINTY_KM,
                        help="--locator-mode filter: отбрасывать сошедшиеся события с "
                             "depth_uncertainty_km ниже этого значения — признак упора в "
                             f"границу таблицы годографов, не настоящее решение (default: "
                             f"{DEFAULT_LOCATOR_MIN_DEPTH_UNCERTAINTY_KM}, см. "
                             "context/locsat-plan.md раздел 5)")
    parser.add_argument('--out', required=True, help="Путь к выходному .BUL файлу")
    args = parser.parse_args()

    print(f"Ассоциатор: {args.assoc}")
    assoc_events = v2.load_associations(args.assoc)
    print(f"Ассоциированных событий: {len(assoc_events)}")

    amplitudes = None
    if not args.no_amps and os.path.isfile(args.amps):
        amplitudes = v2.load_amplitudes(args.amps)
        print(f"Амплитуд загружено: {len(amplitudes)}  ({args.amps})")

    sta_corr = {}
    if args.sta_corrections:
        with open(args.sta_corrections, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                sta = row.get('station', '').strip()
                val = row.get('S', row.get('correction', '')).strip()
                if sta and val:
                    try:
                        sta_corr[sta] = float(val)
                    except ValueError:
                        pass

    prob_dict = v2.load_pick_probabilities(args.prob_dir)
    s_snr_dict = load_s_snr(args.prob_dir)
    print(f"P-вероятностей/SNR: {len(prob_dict)}  S-SNR: {len(s_snr_dict)}  ({args.prob_dir})")

    region_map = load_station_regions(args.metadata_dir)
    print(f"Регионов станций: {len(region_map)}  ({args.metadata_dir})")

    max_dist = args.max_station_dist_km if args.max_station_dist_km and args.max_station_dist_km > 0 else None
    station_coords = None
    if max_dist is not None:
        station_coords = load_station_coords(args.metadata_dir, args.output_cpu_dir)
        print(f"Координат станций: {len(station_coords)}  "
              f"(отсев дальше {max_dist:.0f} км от ({args.ref_lat}, {args.ref_lon}))")

    processed, excluded_stations = process_events(
        assoc_events, args.vp, args.vs, args.sigma, args.r_min, args.r_max,
        amplitudes, sta_corr, prob_dict,
        station_coords=station_coords, ref_lat=args.ref_lat, ref_lon=args.ref_lon,
        max_station_dist_km=max_dist)
    if excluded_stations:
        print(f"Исключено станций (дальше {max_dist:.0f} км): {len(excluded_stations)}  "
              f"[{', '.join(sorted(excluded_stations))}]")
    print(f"Событий после T0/sigma-фильтра: {len(processed)}")

    if args.locator_mode != 'off':
        hypocenters = load_hypocenters(args.hypocenters)
        print(f"Гипоцентров загружено: {len(hypocenters)}  ({args.hypocenters})")
        n_enriched = 0
        for ae in processed:
            ae['locator'] = hypocenters.get(ae['pub_id'])
            if ae['locator'] and ae['locator']['converged']:
                n_enriched += 1
        print(f"Событий с решением локатора: {n_enriched} (из {len(processed)})")

        if args.locator_mode == 'filter':
            before = len(processed)
            processed = [ae for ae in processed
                         if not (ae['locator'] and ae['locator']['converged']
                                 and not _passes_locator_quality(
                                     ae['locator'], args.locator_max_rms,
                                     args.locator_min_depth_uncertainty_km))]
            print(f"После фильтра по качеству локации: {len(processed)} (из {before}, "
                  f"отброшено {before - len(processed)} с решением, не прошедшим "
                  f"_passes_locator_quality — события без решения локатора не отбрасываются)")

    if args.year is not None or args.month is not None:
        before = len(processed)
        processed = [ae for ae in processed
                     if (args.year  is None or ae['t0'].year  == args.year)
                     and (args.month is None or ae['t0'].month == args.month)]
        print(f"После фильтра год={args.year} месяц={args.month}: {len(processed)} (из {before})")

    if args.min_stations > 1:
        before = len(processed)
        col = 'n_sta' if args.min_stations_mode == 'sigma' else 'n_xml'
        processed = [ae for ae in processed if ae[col] >= args.min_stations]
        print(f"После фильтра N >= {args.min_stations} ({args.min_stations_mode}): "
              f"{len(processed)} (из {before})")

    p_snr_dict = {k: v['snr'] for k, v in prob_dict.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    write_bul(processed, region_map, amplitudes, p_snr_dict, s_snr_dict, args.out)


if __name__ == "__main__":
    main()
