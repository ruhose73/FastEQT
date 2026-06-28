# Обновлённый пайплайн детекции — описание изменений

## Обзор

В рамках сессии обновлены четыре компонента пайплайна:

```
gpu_pipeline.py / cpu_pipeline.py
    └── detector.py: process_station_v3
            └── preproc_sequential_v5
                    └── worker_v4
                            ├── hdf5_maker.py: preprocessorV7_mem   (новый)
                            └── predictor.py:  predictor_mem_non_hdf_load_model_v6  (новый)
```

---

## 1. preprocessorV7_mem (EQTransformer/utils/hdf5_maker.py)

### Что изменилось относительно v6

**Метод приведения частоты:**

| | v6 (старый) | v7 (новый) |
|---|---|---|
| Метод | `np.interp` — растяжение массива | `scipy.signal.resample` — FFT |
| АЧХ | −6 dB на Найквисте (25 Hz для 50 Hz станций) | Плоская до Найквиста |
| Физика | Линейная интерполяция значений | Sinc-интерполяция (теоретически оптимальная) |

**Порядок операций:**

```
v6: detrend → filter(native Hz) → taper → np.interp per window
v7: detrend → taper → resample(100 Hz) → filter(100 Hz) → trim → windowing
```

Ключевая проблема v6: `bandpass(freqmax=45 Hz)` применялся к 50 Hz данным,
где Найквист = 25 Hz. Параметр freqmax=45 нарушал границу допустимого диапазона.
В v7 фильтр всегда видит 100 Hz данные — freqmax=45 корректен для любой станции.

**Новый параметр `estimate_uncertainty`:**

```python
preprocessorV7_mem(stream_list, stations_json, overlap=0.3,
                   n_processor=1, estimate_uncertainty=False)
```

Сохраняется в attrs каждой трассы. Позволяет передать флаг через цепочку
вызовов без изменения сигнатур промежуточных функций.

**Важное замечание:** `st.resample(100.0)` падал с "Unknown window type"
в окружении eqt3 — ObsPy передавал тип окна в scipy, который его не распознавал.
Решение: `scipy.signal.resample(tr.data, new_npts)` — прямой вызов без ObsPy-обёртки,
без параметра window (window=None по умолчанию = чистое FFT zero-padding).

---

## 2. predictor_mem_non_hdf_load_model_v6 (EQTransformer/core/predictor.py)

### Что изменилось относительно v5

**Реальный MC Dropout:**

v4 и v5 при `estimate_uncertainty=True` вызывали `model.predict()` в цикле.
`model.predict()` использует `training=False` → dropout выключен →
все проходы детерминированы → std всегда ~0.

v6 использует `model(X, training=True)` → dropout активен →
каждый проход даёт разный выход → std > 0 → реальная оценка неопределённости.

**Батчинг MC-проходов (ключевая оптимизация):**

```
Наивный подход (v6 изначально):
  for _ in range(n_samples):        # 5 последовательных вызовов
      model(X[13], training=True)   # каждый ~0.7s → итого ~3.5s

Оптимизированный (v6 финальный):
  X_tiled = tile(X, n_samples)      # (65, 6000, 3)
  _mc_forward(model, X_tiled)       # один вызов → ~0.2s
```

Dropout генерирует независимые маски для каждого сэмпла в батче,
поэтому 5 копий одного X дают 5 разных выходов — это валидный MC Dropout.

**`@tf.function` на уровне модуля:**

```python
# predictor.py, уровень модуля (НЕ внутри функции)
@tf.function
def _mc_forward(model, batch):
    return model(batch, training=True)
```

Компилируется в граф TF при первом вызове, кэшируется на всё время работы.
Если определить внутри функции — перекомпилируется на каждом сегменте (×14 медленнее).

**Нормализация:** `X / std` без вычитания среднего — как в оригинальном EQT (v5).

**Производительность:**

| Режим | Время/сегмент |
|---|---|
| v4/v5 (model.predict, без uncertainty) | ~0.25 с |
| v6 (model.predict, estimate_uncertainty=False) | ~0.2 с |
| v6 (MC Dropout, n_samples=5, наивный) | ~3.5 с |
| v6 (MC Dropout, n_samples=5, батчинг + @tf.function) | ~0.2 с |
| v6 (MC Dropout, n_samples=5, батчинг, многосегментный прогон) | ~1.0 с |

Память: 18–20 GB RAM при MAX_WORKERS=6 — норма для TF с multiprocessing
(~2.5 GB на воркер × 6 = 15 GB + main process + OS).

---

## 3. worker_v4, preproc_sequential_v5, process_station_v3 (core/detector.py)

**Новые импорты:**
```python
from EQTransformer.utils.hdf5_maker import preprocessorV6_mem, preprocessorV7_mem
from EQTransformer.core.predictor import (predictor_mem_non_hdf_load_model_v4,
                                           predictor_mem_non_hdf_load_model_v6)
```

**worker_v4** — аналог worker_v3 с новым препроцессором и предиктором:
```python
def worker_v4(segment_item, model, save_figs=None, number_of_plots=None,
              estimate_uncertainty=False, number_of_sampling=10)
```

**preproc_sequential_v5** — аналог v4, вызывает worker_v4.
Пробрасывает `estimate_uncertainty` и `number_of_sampling`.

**process_station_v3** — новая точка входа:
```python
def process_station_v3(base_directory, stations_json, model,
                        date_from, date_to, output_base_dir,
                        estimate_uncertainty=False, number_of_sampling=10)
```

Принимает `date_from/date_to` (UTCDateTime) вместо `target_month/target_year`.

---

## 4. gpu_pipeline.py и cpu_pipeline.py

**gpu_pipeline.py:** вызов изменён с `process_station_v2` на `process_station_v3`.
Параметры добавлены напрямую в вызов `run_station`:
```python
_detector_mod.process_station_v3(
    base_directory=base_directory,
    stations_json=stations_json,
    model=_model,
    date_from=date_from,
    date_to=date_to,
    output_base_dir=OUTPUT_BASE_DIR,
    estimate_uncertainty=True,
    number_of_sampling=5,
)
```

**cpu_pipeline.py:** `run_station` конвертирует `target_month/target_year` в
`date_from/date_to` через `calendar.monthrange`, затем вызывает `process_station_v3`.

---

## Как откатиться на предыдущую версию

Все старые функции сохранены без изменений. Для отката достаточно поменять
вызовы в двух файлах.

### gpu_pipeline.py

```python
# НОВАЯ версия (сейчас):
_detector_mod.process_station_v3(
    base_directory=base_directory,
    stations_json=stations_json,
    model=_model,
    date_from=date_from,
    date_to=date_to,
    output_base_dir=OUTPUT_BASE_DIR,
    estimate_uncertainty=True,
    number_of_sampling=5,
)

# ОТКАТ на старую:
_detector_mod.process_station_v2(
    base_directory=base_directory,
    stations_json=stations_json,
    model=_model,
    date_from=date_from,
    date_to=date_to,
    output_base_dir=OUTPUT_BASE_DIR,
)
```

### cpu_pipeline.py

```python
# НОВАЯ версия (сейчас) — в run_station:
from obspy import UTCDateTime
import calendar
last_day = calendar.monthrange(target_year, target_month)[1]
date_from = UTCDateTime(target_year, target_month, 1)
date_to   = UTCDateTime(target_year, target_month, last_day) + 86400
_detector_mod.process_station_v3(
    base_directory=base_directory,
    stations_json=stations_json,
    model=_model,
    date_from=date_from,
    date_to=date_to,
    output_base_dir=OUTPUT_BASE_DIR,
)

# ОТКАТ на старую:
_detector_mod.process_station(
    base_directory=base_directory,
    stations_json=stations_json,
    model=_model,
    target_month=target_month,
    target_year=target_year,
    output_base_dir=OUTPUT_BASE_DIR,
)
```

### Что не нужно трогать при откате

- `hdf5_maker.py` и `predictor.py` — новые функции добавлены рядом,
  старые v6/v4 не изменены и продолжают работать.
- `detector.py` — старые `worker_v3`, `preproc_sequential_v4`,
  `process_station`, `process_station_v2` не изменены.

---

## Сравнение старого и нового пайплайна

| Компонент | Старый | Новый |
|---|---|---|
| Препроцессор | preprocessorV6_mem | preprocessorV7_mem |
| Ресемплинг | np.interp (растяжение массива) | scipy.signal.resample (FFT sinc) |
| Порядок фильтрации | filter → taper → np.interp | taper → resample → filter |
| freqmax=45 Hz для 50 Hz станций | Нарушает Найквист | Корректен (данные уже 100 Hz) |
| Предиктор | predictor_v4 | predictor_v6 |
| MC Dropout | model.predict() — std=0 | model(X, training=True) — реальный |
| MC батчинг | 5 вызовов по N сэмплов | 1 вызов × 5N сэмплов |
| @tf.function | Нет | Да, уровень модуля |
| estimate_uncertainty | Параметр есть, не работал | Работает корректно |
| Точка входа | process_station / process_station_v2 | process_station_v3 |
| Сигнатура даты | target_month / target_year | date_from / date_to (UTCDateTime) |
