"""
core/locator.py — гипоцентры (LOCSAT/NonLinLoc через SeisComP) для событий ассоциатора.

Точка врезки в пайплайн (`CLAUDE.md`, `context/locsat-plan.md` раздел 2):
    core/associator.py -> core/locator.py (этот модуль) -> core/ml_filter_v5.py -> ...

Архитектура (раздел 1.7b плана, согласовано с пользователем 2026-09-13): модуль считает
и пишет ТОЛЬКО сырые метрики на событие (depth_km, depth_uncertainty_km, rms,
n_stations_used, n_arrivals_used, признак сходимости/ошибки) — здесь нет ни одного порога
и ни одного решения "проходит/не проходит". Это ответственность потребителя
(`core/export_bul.py`) — по аналогии с тем, как `core/detector.py` пишет сырые
probability/SNR, а порог применяет `core/associator.py`
(`prepare_station()`/`_passes_v2()`, `core/associator.py:200-223`). Единственное
исключение — `--min-arrivals` (см. ниже, это не порог качества результата).

ВАЖНО — известное ограничение запуска (см. `core/locator.md`, "Известные ограничения"):
реальный вызов LOCSAT требует Python-модуль `seiscomp`, которого нет в conda `eqt3` —
он ставится только вместе с отдельной установкой SeisComP и запускается через
`seiscomp exec seiscomp-python`. Импорт `seiscomp.*` в этом файле отложен внутрь функций,
которые его реально используют — `--help` и разбор `associations.xml` работают под любым
python, реальный прогон (не `--help`) — только под `seiscomp-python`.

Использование (в WSL, из каталога ~/seiscomp-install/seiscomp):
    ./bin/seiscomp exec seiscomp-python /mnt/d/.../core/locator.py --help
    ./bin/seiscomp exec seiscomp-python /mnt/d/.../core/locator.py \
        --assoc /mnt/d/.../workspace/associator/output/associations.xml \
        --inventory /mnt/d/.../workspace/locator/input/inventory.scml \
        --out /mnt/d/.../workspace/locator/output/hypocenters.csv

Как собрать `--inventory` (не делается этим скриптом, см. core/locator.md): смерджить
per-station StationXML (`workspace/data_processors/input/metadata/`) в один файл
(`context/scratch/merge_stationxml.py`), затем `seiscomp exec fdsnxml2inv in.xml out.scml`.
"""

import sys

# Форсируем UTF-8 на stdout/stderr — в help-строках есть не-ASCII (кириллица); без этого
# argparse.print_help() может упасть UnicodeEncodeError в консоли по умолчанию (cp1251) на
# Windows (тот же фикс, что в validate_associator_v2.py).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import argparse
import csv
import datetime
import os
import xml.etree.ElementTree as ET

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BED_NS = "http://quakeml.org/xmlns/bed/1.2"

DEFAULT_ASSOC = os.path.join(_ROOT, "workspace", "associator", "output", "associations.xml")
DEFAULT_INVENTORY = os.path.join(_ROOT, "workspace", "locator", "input", "inventory.scml")
DEFAULT_OUT = os.path.join(_ROOT, "workspace", "locator", "output", "hypocenters.csv")
DEFAULT_ENGINE = "locsat"
DEFAULT_DEPTH_KM = 5.0  # конвенция Y2000.phs / EQTransformer/utils/associator.py, раздел 12 плана
# Мат. минимум arrival'ов, чтобы вообще было что инвертировать (4 неизвестных:
# lat/lon/depth/T0) — НЕ порог качества решения, просто не тратим попытку туда, где решения
# заведомо не будет (раздел 1.7b плана: это не то же самое, что фильтр по RMS/сходимости,
# который сюда сознательно не добавлен).
DEFAULT_MIN_ARRIVALS = 4

ENGINE_NAMES = {"locsat": "LOCSAT", "nonlinloc": "NonLinLoc"}

CSV_FIELDS = [
    "event_id", "engine", "converged", "error",
    "t0", "lat", "lon", "depth_km", "depth_uncertainty_km", "rms",
    "n_stations_used", "n_arrivals_used",
]


def parse_iso(text):
    text = text.rstrip("Z")
    if "." in text:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f")
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")


def load_events(assoc_path):
    """
    Разбирает associations.xml (QuakeML, namespace bed/1.2) целиком, все события разом.
    Ключ события — publicID, как в validate_associator_v2.py::load_associations()
    (core/validate_associator_v2.py:103). Дополнительно (в отличие от той функции) читает
    placeholder-origin (lat/lon/time) события — нужен как начальное приближение для LOCSAT.
    Формат тегов сверен с реальным райтером `EQTransformer/utils/associator.py`
    (ObsPy Catalog.write(..., format="QUAKEML")), не только с тестовым сэмплом.
    Не требует `seiscomp` — работает под любым python.
    """
    ns = BED_NS
    tree = ET.parse(assoc_path)
    root = tree.getroot()
    events = []
    for ev in root.findall(f".//{{{ns}}}event"):
        pub_id = ev.get("publicID", "")
        origin_el = ev.find(f"{{{ns}}}origin")
        if origin_el is None:
            continue
        try:
            origin_time = parse_iso(origin_el.find(f"{{{ns}}}time/{{{ns}}}value").text)
            origin_lat = float(origin_el.find(f"{{{ns}}}latitude/{{{ns}}}value").text)
            origin_lon = float(origin_el.find(f"{{{ns}}}longitude/{{{ns}}}value").text)
        except (AttributeError, ValueError):
            continue

        picks = []
        for pick_el in ev.findall(f"{{{ns}}}pick"):
            wf = pick_el.find(f"{{{ns}}}waveformID")
            t_el = pick_el.find(f"{{{ns}}}time/{{{ns}}}value")
            phase_el = pick_el.find(f"{{{ns}}}phaseHint")
            if wf is None or t_el is None or phase_el is None:
                continue
            try:
                t = parse_iso(t_el.text)
            except ValueError:
                continue
            picks.append({
                "net": wf.get("networkCode", "").strip(),
                "sta": wf.get("stationCode", "").strip(),
                "phase": (phase_el.text or "").strip(),
                "time": t,
            })

        if not picks:
            continue
        events.append({
            "pub_id": pub_id,
            "origin_time": origin_time,
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "picks": picks,
        })
    return events


def _import_seiscomp():
    """Отложенный импорт seiscomp.* — только когда реально нужен вызов локатора. Позволяет
    --help и load_events() работать под обычным python/conda eqt3 (см. докстринг модуля)."""
    try:
        import seiscomp.client
        import seiscomp.core
        import seiscomp.datamodel
        import seiscomp.seismology
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Модуль 'seiscomp' недоступен в этом интерпретаторе. core/locator.py нужно "
            "запускать через seiscomp-python ('seiscomp exec seiscomp-python core/locator.py "
            "...'), не обычный python/conda eqt3 — см. core/locator.md, раздел "
            "'Известные ограничения запуска'."
        ) from e
    return seiscomp.client, seiscomp.core, seiscomp.datamodel, seiscomp.seismology


def _to_sc_time(sc_core, dt):
    return sc_core.Time(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)


def build_origin(sc_core, sc_datamodel, event, depth_km):
    """
    Собирает seiscomp.datamodel.Origin с прикреплёнными Pick/Arrival по каждому пику события.
    Координаты станций здесь НЕ резолвятся вручную — этим занимается
    seiscomp.client.Inventory (должен быть загружен заранее, см. main()) по
    Pick.waveformID() внутри relocate() (раздел 12 плана, "Способ загрузки Inventory без
    БД — решено"). Пик со станцией, которую LOCSAT не может резолвить, отбрасывается молча
    самим relocate() — второй раз этот же фильтр здесь не дублируем.
    """
    sc_datamodel.PublicObject.SetRegistrationEnabled(True)

    origin = sc_datamodel.Origin.Create()
    origin.setTime(sc_datamodel.TimeQuantity(_to_sc_time(sc_core, event["origin_time"])))
    origin.setLatitude(sc_datamodel.RealQuantity(event["origin_lat"]))
    origin.setLongitude(sc_datamodel.RealQuantity(event["origin_lon"]))
    origin.setDepth(sc_datamodel.RealQuantity(depth_km))

    picks_keepalive = []  # держим ссылки — Python GC иначе убивает Pick до relocate()
    for pk in event["picks"]:
        pick = sc_datamodel.Pick.Create()
        pick.setTime(sc_datamodel.TimeQuantity(_to_sc_time(sc_core, pk["time"])))
        pick.setWaveformID(sc_datamodel.WaveformStreamID(pk["net"], pk["sta"], "", "", ""))
        pick.setPhaseHint(sc_datamodel.Phase(pk["phase"]))
        pick.setEvaluationMode(sc_datamodel.AUTOMATIC)
        picks_keepalive.append(pick)

        arrival = sc_datamodel.Arrival()
        arrival.setPickID(pick.publicID())
        arrival.setPhase(sc_datamodel.Phase(pk["phase"]))
        arrival.setTimeUsed(True)
        arrival.setWeight(1.0)
        origin.add(arrival)

    return origin, picks_keepalive


def _value(rq):
    try:
        return rq.value()
    except Exception:
        return None


def _uncertainty(rq):
    try:
        return rq.uncertainty()
    except Exception:
        return None


def relocate_event(sc_core, sc_datamodel, sc_seismology, event, engine_name, depth_km):
    """
    Считает сырые метрики для одного события. Не содержит порогов и не решает
    "проходит/не проходит" (раздел 1.7b плана) — только вычисление. При отсутствии решения
    converged=False, error заполнен, остальные числовые поля — None/0.
    """
    row = {f: None for f in CSV_FIELDS}
    row["event_id"] = event["pub_id"]
    row["engine"] = engine_name
    row["converged"] = False
    row["n_stations_used"] = 0
    row["n_arrivals_used"] = 0

    origin, _keepalive = build_origin(sc_core, sc_datamodel, event, depth_km)

    loc = sc_seismology.LocatorInterface.Create(engine_name)
    if loc is None:
        row["error"] = f"LocatorInterface.Create({engine_name!r}) returned None"
        return row

    try:
        relocated = loc.relocate(origin)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    if relocated is None:
        row["error"] = "relocate() returned None (no solution)"
        return row

    relocated = sc_datamodel.Origin.Cast(relocated)
    row["converged"] = True
    row["t0"] = str(relocated.time().value())
    row["lat"] = _value(relocated.latitude())
    row["lon"] = _value(relocated.longitude())
    row["depth_km"] = _value(relocated.depth())
    row["depth_uncertainty_km"] = _uncertainty(relocated.depth())
    try:
        row["rms"] = relocated.quality().standardError()
    except Exception:
        row["rms"] = None

    n_arrivals = relocated.arrivalCount()
    row["n_arrivals_used"] = n_arrivals
    stations = set()
    for i in range(n_arrivals):
        arr = relocated.arrival(i)
        try:
            pick = sc_datamodel.Pick.Find(arr.pickID())
        except Exception:
            pick = None
        if pick is not None:
            wfid = pick.waveformID()
            stations.add((wfid.networkCode(), wfid.stationCode()))
    row["n_stations_used"] = len(stations) if stations else n_arrivals
    return row


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--assoc", default=DEFAULT_ASSOC,
                    help=f"associations.xml, выход core/associator.py (default: {DEFAULT_ASSOC})")
    ap.add_argument("--inventory", default=DEFAULT_INVENTORY,
                    help="SC3ML-инвентарь станций — собирается заранее вручную, не этим "
                         f"скриптом, см. докстринг модуля (default: {DEFAULT_INVENTORY})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"выходной CSV с гипоцентрами (default: {DEFAULT_OUT})")
    ap.add_argument("--engine", choices=sorted(ENGINE_NAMES), default=DEFAULT_ENGINE,
                    help="локатор SeisComP. --engine nonlinloc требует заранее посчитанной "
                         "сетки времён пробега (раздел 8 плана) — не реализовано, "
                         "Create('NonLinLoc') в этой установке возвращает None (раздел 12 плана)")
    ap.add_argument("--depth-km", type=float, default=DEFAULT_DEPTH_KM,
                    help=f"начальная глубина, км (default: {DEFAULT_DEPTH_KM}, конвенция Y2000.phs)")
    ap.add_argument("--min-arrivals", type=int, default=DEFAULT_MIN_ARRIVALS,
                    help="события с меньшим числом arrival'ов (P+S по всем станциям, до "
                         f"резолва инвентарём) не пытаемся локировать вообще (default: "
                         f"{DEFAULT_MIN_ARRIVALS}) — НЕ фильтр качества результата, только "
                         "пропуск заведомо недоинвертируемой попытки, см. докстринг модуля")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    events = load_events(args.assoc)
    print(f"{len(events)} events with picks in {args.assoc}")

    try:
        sc_client, sc_core, sc_datamodel, sc_seismology = _import_seiscomp()
    except ModuleNotFoundError as e:
        raise SystemExit(str(e))

    load_result = sc_client.Inventory.Instance().load(args.inventory)
    # .load() возвращает None даже при успехе — не индикатор ошибки, см. раздел 12 плана.
    print(f"Inventory.load({args.inventory!r}) -> {load_result!r}")

    engine_name = ENGINE_NAMES[args.engine]

    rows = []
    for ev in events:
        if len(ev["picks"]) < args.min_arrivals:
            row = {f: None for f in CSV_FIELDS}
            row.update(
                event_id=ev["pub_id"], engine=engine_name, converged=False,
                error=f"skipped: {len(ev['picks'])} arrivals < --min-arrivals={args.min_arrivals}",
                n_stations_used=0, n_arrivals_used=0,
            )
        else:
            row = relocate_event(sc_core, sc_datamodel, sc_seismology, ev, engine_name, args.depth_km)
        rows.append(row)
        status = "OK" if row["converged"] else f"NO SOLUTION ({row['error']})"
        print(f"  {row['event_id']}: {status}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["converged"])
    print(f"\nWrote {len(rows)} rows to {args.out} "
          f"({n_ok} converged, {len(rows) - n_ok} no solution/skipped)")


if __name__ == "__main__":
    main()
