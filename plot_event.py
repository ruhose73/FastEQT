"""
Временный скрипт: запускает детектор на сегменте вокруг события
2024-01-28 ~16:01:47 UTC для всех 11 станций.

Выход: data-in-memory/output_event/{STATION}/
  - X_prediction_results.csv
  - figures/  (до 10 графиков EQTransformer на станцию)

Использует geofile_splitter_multi_chanels_v3/preproc_sequential_v5 — ту же
цепочку предобработки, что и продакшн-детектор (core/detector.py).
"""

import os
import sys

import tensorflow as tf
from obspy import UTCDateTime

# ── Пути ──────────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.abspath(__file__))
_IN        = os.path.join(ROOT, 'data-in-memory', 'input')
_JS        = os.path.join(ROOT, 'json')
MODEL_PATH = os.path.join(ROOT, 'ModelsAndSampleData', 'EqT_original_model.h5')
OUT_BASE   = os.path.join(ROOT, 'data-in-memory', 'output_event')

sys.path.insert(0, ROOT)
from core.detector import (
    load_model_cudnn_v2,
    geofile_splitter_multi_chanels_v3,
    preproc_sequential_v5,
)

# ── Параметры события ─────────────────────────────────────────────────────────

TARGET_YEAR  = 2024
TARGET_MONTH = 1
TARGET_DAY   = 28

# Диапазон суток для geofile_splitter_multi_chanels_v3 (правая граница не включена).
DAY_FROM = UTCDateTime(TARGET_YEAR, TARGET_MONTH, TARGET_DAY)
DAY_TO   = DAY_FROM + 86400

# Временно́е окно захвата: захватываем оба 10-минутных сегмента,
# перекрывающих событие (16:00–16:10 и 16:05–16:15)
WINDOW_FROM = UTCDateTime('2024-01-28T15:50:00')
WINDOW_TO   = UTCDateTime('2024-01-28T16:15:00')

NUMBER_OF_PLOTS = 10

# ── Станции ───────────────────────────────────────────────────────────────────

STATIONS = [
    (os.path.join(_IN, 'GUZR'), os.path.join(_JS, 'station_GUZR.json')),
    (os.path.join(_IN, 'LABN'), os.path.join(_JS, 'station_LABN.json')),
    (os.path.join(_IN, 'VSLR'), os.path.join(_JS, 'station_VSLR.json')),
    (os.path.join(_IN, 'MRNR'), os.path.join(_JS, 'station_MRNR.json')),
    (os.path.join(_IN, 'SOC'),  os.path.join(_JS, 'station_SOC.json')),
    (os.path.join(_IN, 'GOYR'), os.path.join(_JS, 'station_GOYR.json')),
    (os.path.join(_IN, 'DOMR'), os.path.join(_JS, 'station_DOMR.json')),
    (os.path.join(_IN, 'SHA1'), os.path.join(_JS, 'station_SHA1.json')),
    (os.path.join(_IN, 'ZEI'),  os.path.join(_JS, 'station_ZEI.json')),
    (os.path.join(_IN, 'PYA1'), os.path.join(_JS, 'station_PYA1.json')),
    (os.path.join(_IN, 'SPGR'), os.path.join(_JS, 'station_SPGR.json')),
]

# ── Фильтрующий генератор ─────────────────────────────────────────────────────

def _filtered_gen(base_dir, t_from, t_to):
    """Пропускает только сегменты, перекрывающиеся с [t_from, t_to]."""
    for segment, seg_name in geofile_splitter_multi_chanels_v3(
        base_dir, DAY_FROM, DAY_TO
    ):
        seg_start = segment[0].stats.starttime
        seg_end   = segment[0].stats.endtime
        if seg_start < t_to and seg_end > t_from:
            print(f'  сегмент: {seg_name}')
            yield segment, seg_name

# ── Основной цикл ─────────────────────────────────────────────────────────────

def main():
    # GPU — разрешаем динамический рост памяти
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    print('Загрузка модели...')
    model = load_model_cudnn_v2(MODEL_PATH)
    print('Модель загружена.\n')

    for base_dir, json_path in STATIONS:
        station_name = os.path.basename(base_dir)
        output_dir   = os.path.join(OUT_BASE, station_name)
        output_csv   = os.path.join(output_dir, 'X_prediction_results.csv')
        save_figs    = os.path.join(output_dir, 'figures')
        os.makedirs(save_figs, exist_ok=True)

        print(f'═══ {station_name} ══════════════════════════════')
        gen = _filtered_gen(base_dir, WINDOW_FROM, WINDOW_TO)
        preproc_sequential_v5(
            gen,
            json_path,
            model,
            output_csv=output_csv,
            save_figs=save_figs,
            number_of_plots=NUMBER_OF_PLOTS,
        )
        print(f'  → {output_csv}\n')

    print(f'Готово. Результаты в: {OUT_BASE}')


if __name__ == '__main__':
    main()
