"""
detector.py — детекция EQTransformer по станциям (CLI).

Все параметры теперь задаются флагами командной строки (см. --help);
значения по умолчанию соответствуют прежним хардкод-константам, поведение
без флагов не меняется (кроме удаления V6-цепочки, см. ниже — она была
небезопасна для этой сети, а не просто альтернативным путём).

Файл используется двумя способами:
  1. Напрямую (python core/detector.py) — обрабатывает список станций
     последовательно, одним процессом, одной уже загруженной моделью
     (process_station_v3 → preproc_sequential_v5 → worker_v4, с MC Dropout,
     опционально).
  2. Как библиотека — core/cpu_pipeline.py и core/gpu_pipeline.py загружают
     этот файл через importlib и вызывают process_station_v3() из нескольких
     процессов параллельно. Пулом процессов сам этот файл не управляет —
     это забота вызывающих pipeline-файлов.

Правки этой ревизии (production-plan.md, Трек 1, п.1.1/1.3, 2026-09-06),
внесены как новый файл (не правка на месте — пользователь пока не может
закоммитить репозиторий), предыдущая версия сохранена как legacy/detector.py:
  - удалена process_station_v2 (мёртвый код, нигде не вызывалась);
  - удалена неиспользуемая константа MAX_WORKERS — докстринг предыдущей
    версии обещал параллельный режим «по умолчанию», которого в этом файле
    никогда не было (реальный параллелизм — только в cpu_pipeline.py /
    gpu_pipeline.py);
  - detection_threshold/P_threshold/S_threshold/keepPS/allowonlyS/spLimit/
    batch_size, ранее зашитые в телах worker_v3/worker_v4, подняты как
    параметры функций по всей цепочке до CLI-флагов;
  - список станций для прямого запуска — не список кортежей в коде, а
    флаг --stations (коды через запятую), путь к данным и station_*.json
    достраивается из --input-dir/--json-dir по тому же шаблону, что и
    раньше в коде.

Отдельная, более поздняя правка той же ревизии (2026-09-06, по прямому
запросу пользователя) — удалена V6-цепочка целиком. Реальный, используемый
в проде путь (cpu_pipeline.py/gpu_pipeline.py → process_station_v3 →
preproc_sequential_v5 → worker_v4 → preprocessorV7_mem/predictor_v6) НЕ
менялся вообще:
  - geofile_splitter_multi_chanels_v2, worker_v3, preproc_sequential_v4,
    process_station удалены из этого файла — их и так не вызывали ни
    cpu_pipeline.py, ни gpu_pipeline.py, только __main__ этого файла и
    вспомогательный plot_event.py в корне репозитория. EQTransformer/ не
    трогаем (правило №1 CLAUDE.md — только код проекта, не форк/апдейты
    EQTransformer): preprocessorV6_mem (EQTransformer/utils/hdf5_maker.py) и
    predictor_mem_non_hdf_load_model_v4 (EQTransformer/core/predictor.py)
    остаются в EQTransformer/ как есть — просто больше не импортируются и
    не вызываются отсюда;
  - причина удаления — не просто «мёртвый код»: для станций с частотой
    дискретизации <90 Гц (8 из 10 проверенных станций сети) порядок
    filter→resample в preprocessorV6_mem приводил к тому, что ObsPy молча
    подставлял high-pass 1 Гц без верхней границы вместо полосового фильтра
    1–45 Гц — небезопасно для этой сети (см. EQTransformer/eqt_internals.md,
    разд. 3);
  - __main__ пересобран поверх той же самой V7-цепочки, что и в
    cpu_pipeline.py/gpu_pipeline.py (process_station_v3) — диапазон дат для
    прямого запуска теперь --date-from/--date-to вместо --month/--year
    (унификация с cpu_pipeline.py/gpu_pipeline.py — сигнатура
    process_station_v3 принимает даты, а не месяц/год); добавлены
    --estimate-uncertainty/--number-of-sampling;
  - plot_event.py — единственный внешний потребитель
    geofile_splitter_multi_chanels_v2/preproc_sequential_v4 — переведён на
    geofile_splitter_multi_chanels_v3/preproc_sequential_v5 отдельным шагом.
"""

import argparse
import gc
import os
import sys
import csv
import re
from datetime import datetime
from traceback import format_exc

# Форсируем UTF-8 на stdout/stderr — в help-строках/докстринге есть не-ASCII
# (стрелки →, кириллица); без этого argparse.print_help() может упасть с
# UnicodeEncodeError в консоли по умолчанию (cp1251) на Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from EQTransformer.core.EqT_utils import FeedForward, LayerNormalization, SeqSelfAttention, f1
from obspy import read, Stream, UTCDateTime
from EQTransformer.utils.hdf5_maker import preprocessorV7_mem
from EQTransformer.core.predictor import predictor_mem_non_hdf_load_model_v6


# ─── Конфигурация (значения по умолчанию для CLI-флагов) ─────────────────────

MODEL_PATH      = os.path.join(_ROOT, "ModelsAndSampleData", "EqT_original_model.h5")
OUTPUT_BASE_DIR = os.path.join(_ROOT, "workspace", "detector", "output")
LOG_FILE        = os.path.join(_ROOT, "workspace", "detector", "output", "processing_log.csv")

DATE_FROM = "2024-01-01"
DATE_TO   = "2024-02-01"   # не включается

# По умолчанию — выход data_processors/main.py (workspace/, production-plan.md
# Трек 1); workspace/detector/input/ существует отдельно для случая, когда
# волновые файлы/station_*.json кладутся туда напрямую, минуя data_processors.
_IN = os.path.join(_ROOT, "workspace", "data_processors", "output", "geofiles")
_JS = os.path.join(_ROOT, "workspace", "data_processors", "output")

# Список станций для прямого запуска (python core/detector.py).
# Оптимальный состав: recall 33/49 (без NCK/SRGR/GOFR — они снижают до 32/49).
STATIONS = "SOC,VSLR,GUZR,BEYR,SHA1,MRNR,SPGR,DOMR,ZEI,LABN,GOYR,PYA1"

ESTIMATE_UNCERTAINTY = False
NUMBER_OF_SAMPLING   = 10

# Пороги предиктора — были зашиты в теле worker_v4, теперь CLI-флаги.
DETECTION_THRESHOLD = 0.75
P_THRESHOLD         = 0.3
S_THRESHOLD         = 0.2
KEEP_PS             = True
ALLOW_ONLY_S        = False
SP_LIMIT            = 45
BATCH_SIZE          = 32

# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADER = [
    'file_name', 'network', 'station', 'instrument_type',
    'station_lat', 'station_lon', 'station_elv',
    'event_start_time', 'event_end_time',
    'detection_probability', 'detection_uncertainty',
    'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
    's_arrival_time', 's_probability', 's_uncertainty', 's_snr'
]


# ─── Функции обработки ───────────────────────────────────────────────────────

def geofile_splitter_multi_chanels_v3(base_directory, date_from, date_to):
    """
    Генератор: читает один день за раз, отдаёт сегменты по одному, затем освобождает RAM.
    Фильтрует файлы по диапазону дат date_from..date_to (UTCDateTime, правая граница не включается).

    Максимальное потребление: 1 день сырых данных + 1 сегмент в обработке.
    """
    pattern = re.compile(
        r'^(?P<net>[A-Z0-9]+)\.'
        r'(?P<sta>[A-Z0-9]+)\.'
        r'(?P<loc>[A-Z0-9]{0,2})\.'
        r'(?P<cha>[A-Z0-9]+)\.[A-Z_]*__'
        r'(?P<start>\d{8}T\d{6}Z)__'
        r'(?P<end>\d{8}T\d{6}Z)$'
    )

    files = [f for f in os.listdir(base_directory)
             if os.path.isfile(os.path.join(base_directory, f))]
    station_groups = {}

    for file_name in files:
        match = pattern.match(file_name)
        if not match:
            print(f"⚠️ Пропускаем файл: {file_name} — не подходит под шаблон")
            continue
        sta = match.group("sta")
        start = match.group("start")
        key = f"{sta}_{start}"
        station_groups.setdefault(key, []).append(
            os.path.join(base_directory, file_name)
        )

    total_yielded = 0

    for key, file_list in sorted(station_groups.items()):
        try:
            st_all = Stream()
            for fpath in file_list:
                try:
                    st_all += read(fpath)
                except Exception as e:
                    print(f"Ошибка чтения {fpath}: {e}")

            if len(st_all) < 2:
                print(f"⚠️ Пропускаем {key}: найдено только {len(st_all)} канал(ов)")
                del st_all
                continue

            tr0 = st_all[0]
            start_time = tr0.stats.starttime
            end_time = tr0.stats.endtime

            if not (date_from <= start_time < date_to):
                del st_all
                continue

            window_length = 10 * 60
            step = 5 * 60
            t = start_time

            while t + window_length <= end_time:
                segment = st_all.slice(t, t + window_length)
                if len(segment) >= 2 and all(tr.stats.npts > 0 for tr in segment):
                    seg_name = (
                        f"{tr0.stats.network}.{tr0.stats.station}__"
                        f"{t.strftime('%Y%m%dT%H%M%SZ')}__"
                        f"{(t + window_length).strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    total_yielded += 1
                    yield (segment, seg_name)
                t += step

            if t < end_time:
                segment = st_all.slice(end_time - window_length, end_time)
                if len(segment) >= 2 and all(tr.stats.npts > 0 for tr in segment):
                    seg_name = (
                        f"{tr0.stats.network}.{tr0.stats.station}__"
                        f"{(end_time - window_length).strftime('%Y%m%dT%H%M%SZ')}__"
                        f"{end_time.strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    total_yielded += 1
                    yield (segment, seg_name)

        except Exception as e:
            print(f"❌ Ошибка обработки группы {key}: {e}")
        finally:
            if 'st_all' in dir():
                del st_all

    print(f"✅ Всего сегментов обработано генератором: {total_yielded}")


def worker_v4(segment_item, model, save_figs=None, number_of_plots=None,
              estimate_uncertainty=False, number_of_sampling=10,
              detection_threshold=DETECTION_THRESHOLD, P_threshold=P_THRESHOLD,
              S_threshold=S_THRESHOLD, keep_ps=KEEP_PS, allow_only_s=ALLOW_ONLY_S,
              sp_limit=SP_LIMIT, batch_size=BATCH_SIZE):
    """
    preprocessorV7_mem (st.resample, корректный FFT-ресемплинг с антиалиасингом)
    + predictor_v6 (реальный MC Dropout через model(X, training=True), опционально).
    """
    segment, seg_name, stations_json = segment_item
    status = "success"
    message = ""
    rows = []

    try:
        print(f"▶️ Обработка сегмента {seg_name}: {len(segment)} каналов → {[tr.stats.channel for tr in segment]}")
        start_time = datetime.now()

        csv_data, hdf5_data = preprocessorV7_mem(
            stream_list=[(segment, seg_name)],
            stations_json=stations_json,
            overlap=0.3,
            n_processor=1,
            estimate_uncertainty=estimate_uncertainty,
        )
        t_preproc = (datetime.now() - start_time).total_seconds()

        seg_csv = csv_data[seg_name][1:]
        seg_hdf = hdf5_data[seg_name]

        t_pred_start = datetime.now()
        rows = predictor_mem_non_hdf_load_model_v6(
            csv_segment=seg_csv,
            hdf_segment=seg_hdf,
            model=model,
            save_figs=save_figs,
            detection_threshold=detection_threshold,
            P_threshold=P_threshold,
            S_threshold=S_threshold,
            number_of_plots=number_of_plots,
            plot_mode='time',
            estimate_uncertainty=estimate_uncertainty,
            number_of_sampling=number_of_sampling,
            gpuid=0,
            keepPS=keep_ps,
            allowonlyS=allow_only_s,
            spLimit=sp_limit,
            batch_size=batch_size,
        )
        t_pred = (datetime.now() - t_pred_start).total_seconds()

        total = (datetime.now() - start_time).total_seconds()
        print(f"  preproc={t_preproc:.2f}s  predict={t_pred:.2f}s  total={total:.2f}s")
        print(f"✅ Сегмент {seg_name}: {len(rows)} событий")

    except Exception:
        status = "error"
        message = format_exc()
        print(f"❌ Ошибка обработки {seg_name}: {message}")

    return (seg_name, status, message, rows)


def preproc_sequential_v5(segment_gen, stations_json, model, output_csv,
                           save_figs=None, number_of_plots=10,
                           estimate_uncertainty=False, number_of_sampling=10,
                           log_file=LOG_FILE,
                           detection_threshold=DETECTION_THRESHOLD, P_threshold=P_THRESHOLD,
                           S_threshold=S_THRESHOLD, keep_ps=KEEP_PS, allow_only_s=ALLOW_ONLY_S,
                           sp_limit=SP_LIMIT, batch_size=BATCH_SIZE):
    """
    Обрабатывает сегменты из генератора последовательно, пишет все события в один CSV.
    Каждый сегмент явно удаляется из памяти после обработки.
    gc.collect() каждые 50 сегментов.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    log_results = []
    total_events = 0
    processed = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(CSV_HEADER)
        csv_out.flush()

        for segment, seg_name in segment_gen:
            seg_name_out, status, message, rows = worker_v4(
                (segment, seg_name, stations_json),
                model,
                save_figs=save_figs,
                number_of_plots=number_of_plots,
                estimate_uncertainty=estimate_uncertainty,
                number_of_sampling=number_of_sampling,
                detection_threshold=detection_threshold,
                P_threshold=P_threshold,
                S_threshold=S_threshold,
                keep_ps=keep_ps,
                allow_only_s=allow_only_s,
                sp_limit=sp_limit,
                batch_size=batch_size,
            )
            del segment

            if rows:
                writer.writerows(rows)
                csv_out.flush()
                total_events += len(rows)
            log_results.append((seg_name_out, status, message, len(rows)))

            processed += 1
            if processed % 50 == 0:
                gc.collect()
                print(f"  [gc] собрано после {processed} сегментов, событий: {total_events}")

    gc.collect()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["file_name", "status", "message", "events_found"])
        for r in log_results:
            log_writer.writerow(r)

    print(f"Лог: {log_file}")
    print(f"CSV: {output_csv} ({total_events} событий)")


def _patch_lstm_inplace(model):
    """
    Патчит LSTM cell напрямую, без JSON rebuild.

    recurrent_dropout и implementation — read-only @property на LSTM слое,
    они читаются из lstm.cell. Патчим cell напрямую: LSTMCell хранит их
    как обычные instance-атрибуты, поэтому присвоение работает.

    Warning при load_model (трассировка графа) неизбежен.
    После патча cell — при model.predict() cuDNN-проверка проходит,
    cuDNN kernel используется, новый warning не появляется.

    Bidirectional хранит два LSTM (forward_layer, backward_layer) — патчим оба.
    """
    patched = 0
    for layer in model.layers:
        lstm_layers = []
        if isinstance(layer, tf.keras.layers.LSTM):
            lstm_layers = [layer]
        elif isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_layers = [layer.forward_layer, layer.backward_layer]

        for lstm in lstm_layers:
            if not isinstance(lstm, tf.keras.layers.LSTM):
                continue
            if hasattr(lstm, 'cell'):
                lstm.cell.recurrent_dropout = 0.0
                lstm.cell.implementation = 2
                patched += 1

    print(f"  [cuDNN patch] патчировано {patched} LSTM cell объектов")
    return model


def load_model_cudnn_v2(model_path):
    """
    Загружает модель и применяет in-place cuDNN LSTM патч.

    v1 пересоздавал модель через model_from_json — TF 2.10 игнорировал patched config
    при десериализации Bidirectional. v2 патчит объекты напрямую после load_model,
    до первого вызова predict().
    """
    custom_objects = {
        'SeqSelfAttention': SeqSelfAttention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'f1': f1
    }
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    _patch_lstm_inplace(model)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=['binary_crossentropy'] * 3,
        metrics=[f1]
    )
    return model


def process_station_v3(base_directory, stations_json, model,
                        date_from, date_to, output_base_dir,
                        estimate_uncertainty=False, number_of_sampling=10,
                        log_file=LOG_FILE,
                        detection_threshold=DETECTION_THRESHOLD, P_threshold=P_THRESHOLD,
                        S_threshold=S_THRESHOLD, keep_ps=KEEP_PS, allow_only_s=ALLOW_ONLY_S,
                        sp_limit=SP_LIMIT, batch_size=BATCH_SIZE):
    """
    Обрабатывает одну станцию с preproc_sequential_v5 (worker_v4, preprocessorV7, predictor_v6).
    Принимает date_from/date_to (UTCDateTime).
    """
    station_name = os.path.basename(os.path.normpath(base_directory))
    output_dir = os.path.join(output_base_dir, station_name)
    output_csv = os.path.join(output_dir, f"{station_name.lower()}.csv")

    segment_gen = geofile_splitter_multi_chanels_v3(base_directory, date_from, date_to)
    preproc_sequential_v5(
        segment_gen, stations_json, model,
        output_csv=output_csv,
        save_figs=None,
        number_of_plots=10,
        estimate_uncertainty=estimate_uncertainty,
        number_of_sampling=number_of_sampling,
        log_file=log_file,
        detection_threshold=detection_threshold,
        P_threshold=P_threshold,
        S_threshold=S_threshold,
        keep_ps=keep_ps,
        allow_only_s=allow_only_s,
        sp_limit=sp_limit,
        batch_size=batch_size,
    )


def _build_stations(codes, input_dir, json_dir):
    """Достраивает пары (входная_директория, station_*.json) по кодам станций."""
    return [
        (os.path.join(input_dir, code), os.path.join(json_dir, f"station_{code}.json"))
        for code in codes
    ]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Последовательная обработка станций EQTransformer "
                    "(process_station_v3 -> preproc_sequential_v5 -> worker_v4, "
                    "MC Dropout опционально). Для параллельной обработки нескольких "
                    "станций используйте cpu_pipeline.py/gpu_pipeline.py."
    )
    parser.add_argument('--model-path', default=MODEL_PATH,
                        help="Путь к весам модели EQT (.h5)")
    parser.add_argument('--output-base-dir', default=OUTPUT_BASE_DIR,
                        help="Куда писать CSV детекций по станциям")
    parser.add_argument('--log-file', default=LOG_FILE,
                        help="Куда писать общий лог по сегментам (перезаписывается на каждый вызов)")
    parser.add_argument('--input-dir', default=_IN,
                        help="Корневая директория с входными данными станций ({input-dir}/{код})")
    parser.add_argument('--json-dir', default=_JS,
                        help="Директория с station_*.json ({json-dir}/station_{код}.json)")
    parser.add_argument('--stations', default=STATIONS,
                        help="Коды станций через запятую")
    parser.add_argument('--date-from', default=DATE_FROM, help="UTCDateTime-совместимая строка")
    parser.add_argument('--date-to', default=DATE_TO, help="Не включается")
    parser.add_argument('--estimate-uncertainty', dest='estimate_uncertainty',
                        action='store_true', default=ESTIMATE_UNCERTAINTY,
                        help="MC Dropout неопределённость (медленнее — number-of-sampling проходов на сегмент)")
    parser.add_argument('--number-of-sampling', type=int, default=NUMBER_OF_SAMPLING)
    parser.add_argument('--detection-threshold', type=float, default=DETECTION_THRESHOLD)
    parser.add_argument('--p-threshold', type=float, default=P_THRESHOLD)
    parser.add_argument('--s-threshold', type=float, default=S_THRESHOLD)
    parser.add_argument('--keep-ps', dest='keep_ps', action='store_true', default=KEEP_PS,
                        help="Событие засчитывается только если есть и P, и S (по умолчанию включено)")
    parser.add_argument('--no-keep-ps', dest='keep_ps', action='store_false',
                        help="Достаточно одного из P/S")
    parser.add_argument('--allow-only-s', action='store_true', default=ALLOW_ONLY_S)
    parser.add_argument('--sp-limit', type=float, default=SP_LIMIT,
                        help="Максимальное расстояние S-P (км) для допустимого пика")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    start = datetime.now()

    date_from = UTCDateTime(args.date_from)
    date_to   = UTCDateTime(args.date_to)

    station_codes = [s.strip() for s in args.stations.split(',') if s.strip()]
    stations = _build_stations(station_codes, args.input_dir, args.json_dir)

    print(f"Режим:   GPU последовательный")
    print(f"Станций: {len(stations)}")
    print("-" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    model = load_model_cudnn_v2(args.model_path)

    ok = err = 0
    for bd, sj in stations:
        station_name = os.path.basename(os.path.normpath(bd))
        try:
            process_station_v3(
                bd, sj, model, date_from, date_to, args.output_base_dir,
                estimate_uncertainty=args.estimate_uncertainty,
                number_of_sampling=args.number_of_sampling,
                log_file=args.log_file,
                detection_threshold=args.detection_threshold,
                P_threshold=args.p_threshold,
                S_threshold=args.s_threshold,
                keep_ps=args.keep_ps,
                allow_only_s=args.allow_only_s,
                sp_limit=args.sp_limit,
                batch_size=args.batch_size,
            )
            elapsed = (datetime.now() - start).total_seconds() / 60
            print(f"[{elapsed:5.1f} мин] [SUCCESS] {station_name}")
            ok += 1
        except Exception:
            elapsed = (datetime.now() - start).total_seconds() / 60
            print(f"[{elapsed:5.1f} мин] [ERROR  ] {station_name}")
            print(format_exc()[:800])
            err += 1

    total = (datetime.now() - start).total_seconds() / 60
    print("-" * 60)
    print(f"Успешно: {ok}  Ошибок: {err}  Время: {total:.1f} мин")
