# План дообучения EQTransformer на региональных данных

## Контекст

- **Данные:** ~1 год непрерывных записей, 20 станций, формат MSEED по суткам
- **Именование MSEED:** `{NET}.{STAT}..{INST}{comp}.D__{YYYYMMDD}T000000Z__{YYYYMMDD+1}T000000Z`
- **Geofiles:** `geofiles/{STATION}/`
- **Детекции:** `data-in-memory/assoc_input_gpu_100_150/{STATION}/X_prediction_results.csv`
- **Ассоциации:** `data-in-memory/association_gpu_100_150/associations.xml`
- **Масштаб:** ~20 000 ассоциированных событий на станцию → суммарно потенциально сотни тысяч трасс
- **Цель:** fine-tuning предобученной модели EQT под конкретный регион для улучшения точности и полноты детекций

---

## Фаза 1. Подготовка данных (dataset_builder.py)

### 1.1 Парсинг associations.xml

Для каждого события извлечь:

- `origin_time` — время события
- `src_lat`, `src_lon` — координаты источника (из `<origin>`)
- Для каждой станции: `P_time`, `S_time` (из `<pick>` с `<phaseHint>`)

Структура на выходе: список словарей `{origin_time, src_lat, src_lon, picks: {STATION: {P, S}}}`.

### 1.2 Сбор метаданных станций

Из `assoc_input_gpu_100_150/{STATION}/X_prediction_results.csv` взять первую строку и вытащить:

- `station_lat`, `station_lon`, `station_elv`
- `network` (поле `network`)
- `instrument_type` (поле `instrument_type` — EH, SH, HH...)

Сделать словарь `station_meta[STATION] = {lat, lon, elv, network, instrument}`.

### 1.3 Обнаружение каналов из geofiles

Для каждой станции сканируем `geofiles/{STATION}/`:

- Парсим имя файла: `NET.STAT..CHN.D__...`
- Извлекаем список доступных компонент (`E`, `N`, `Z`, `1`, `2`)
- Определяем порядок: `[E или 1, N или 2, Z]`

Если нет вертикальной компоненты `Z` → станция пропускается.

Сохранить словарь `station_channels[STATION] = {comp: channel_code}`, например:

```text
{'E': 'EHE', 'N': 'EHN', 'Z': 'EHZ'}
```

### 1.4 Параметры окна

```text
WINDOW_LEN  = 60.0 с  → 6000 сэмплов @ 100 Hz
P_OFFSET    = 5.0 с   → P пик на сэмпле 500
S_OFFSET    = вычисляется из S_time - P_time
TARGET_SR   = 100.0 Гц
```

Минимальные требования к окну:

- P_time доступно (обязательно)
- S_time доступно (желательно; если нет — трасса всё равно попадает в датасет, s_sample = -1)
- `s_sample - p_sample > 30` (реальное расстояние, а не шум)
- `s_sample < 5500` (S-пик не уходит за конец окна)

### 1.5 Загрузка MSEED и нарезка

Для каждого пика:

1. `win_start = P_time - P_OFFSET`
2. `win_end = win_start + 60.0`
3. Определить суточный файл по дате `win_start`. Если окно пересекает полночь — загрузить два суточных файла.
4. Загрузить 3 канала через ObsPy `read()`.
5. `st.merge(fill_value=0)` → `st.slice(win_start, win_end)`
6. Проверить: длина трасс ≥ 5500 сэмплов после ресемплинга (иначе пропустить).

### 1.6 Препроцессинг

```python
st.detrend('demean')
st.taper(max_percentage=0.05, type='cosine')
st.filter('bandpass', freqmin=1.0, freqmax=45.0, corners=4, zerophase=True)
st.resample(100.0)
```

Нормализация: `arr /= np.max(np.abs(arr))` (если max == 0 → пропустить трассу).

### 1.7 Фильтрация качества

Применять до записи в HDF5:

| Условие | Порог | Комментарий |
| --- | --- | --- |
| `detection_probability` | ≥ 0.85 | только уверенные детекции |
| `p_snr` | ≥ 2.0 | исключить шумовые P |
| `s_snr` | ≥ 1.0 | мягче, S всегда сложнее |
| Длина трассы | ≥ 5500 сэмплов | защита от неполных записей |
| `s_sample - p_sample` | > 30 | исключить S ≈ P |

Для соответствия пика и CSV: искать строку в `X_prediction_results.csv` с `p_arrival_time` наиболее близким к P_time события (допуск ±0.5 с). Это нужно для получения p_snr, s_snr, detection_probability.

### 1.8 Генерация шумовых окон

На каждую станцию:

1. Собрать все P-времена (из CSV детекций станции).
2. Отсортировать; найти промежутки между событиями > 120 с.
3. В каждом промежутке случайно выбрать начало окна так, чтобы окно не пересекалось с событиями.
4. Загрузить и препроцессить аналогично событийным окнам.
5. `p_arrival_sample = -1`, `s_arrival_sample = -1`, `trace_category = 'noise'`.

Целевое соотношение: **1 шумовое окно на 3 события** (можно настраивать флагом `--noise-ratio`).

### 1.9 Формат HDF5

```text
/data/{trace_name}    dtype=float32    shape=(6000, 3)
```

Атрибуты каждой трассы:

```text
trace_name             str   # уникальный ключ
network                str
station                str
instrument_type        str   # EH, SH, ...
station_lat            float
station_lon            float
station_elv            float
p_arrival_sample       int   # ≈500 для событий, -1 для шума
s_arrival_sample       int   # или -1
snr_db                 float # p_snr из CSV
trace_start_time       str   # ISO8601
trace_end_time         str
trace_category         str   # 'earthquake' | 'noise'
source_id              str   # publicID события из XML
source_latitude        float
source_longitude       float
source_magnitude       float # -1 если неизвестно
coda_end_sample        int   # s_sample + 200 (эвристика)
```

### 1.10 Companion CSV

Одна строка на трассу, те же поля что в attrs плюс `trace_category`. Нужен для EQTransformer `trainer()`.

---

## Фаза 2. Разбивка на train/dev/test

Разбивать **по времени**, не случайно — чтобы не было утечки данных:

```text
Jan–Sep (9 мес.)  → train  (~80%)
Oct               → dev    (~8%)
Nov–Dec           → test   (~12%)
```

Скрипт разбивки: читает companion CSV, фильтрует по `trace_start_time`, пишет три CSV + три HDF5 (или три индексных файла с ключами).

Важно: шумовые окна разбивать по тому же временному принципу (не перемешивать шум из test-периода в train).

---

## Фаза 3. Fine-tuning модели

### 3.1 Базовая модель

Использовать предобученную `EqT_original_model.h5` из `pre_trained_models/`.
Это модель, обученная на глобальном датасете STEAD (1.2M трасс).

### 3.2 Стратегия дообучения

**Вариант A — полное дообучение (рекомендуется при >50k трасс):**

- Размораживаем все слои
- Небольшой learning rate: `lr=1e-4` (в 10× меньше, чем при обучении с нуля)
- Градиентный клиппинг: `clipnorm=0.1`

**Вариант B — только верхние слои (при <10k трасс):**

- Замораживаем encoder (первые ~60% слоёв)
- Дообучаем только декодеры P, S, детектора

При 20k событий × 20 станций → потенциально 400k трасс-событий → **Вариант A**.

### 3.3 Параметры обучения

```python
from EQTransformer.core.trainer import trainer

trainer(
    input_hdf5        = 'training_data/train.hdf5',
    input_csv         = 'training_data/train.csv',
    input_testset     = 'training_data/test.hdf5',
    input_model       = 'pre_trained_models/EqT_original_model.h5',
    output_name       = 'regional_model_v1',
    epochs            = 20,
    batch_size        = 64,
    loss_weights      = [0.05, 0.40, 0.55],  # [detector, P, S]
    loss_types        = ['binary_crossentropy']*3,
    patience          = 5,                   # early stopping
    gpuid             = 0,
    gpu_limit         = 0.9,
    augmentation      = True,
)
```

`loss_weights`: усилить вес P и S пиков относительно детектора — региональные данные обычно имеют более узкое распределение амплитуд, поэтому пики важнее общей вероятности.

### 3.4 Аугментация (встроенная в trainer)

Стандартная аугментация EQT:

- Случайный сдвиг окна (±2 с)
- Добавление гауссового шума
- Амплитудное масштабирование

### 3.5 Мониторинг обучения

Следить за `val_loss` для P и S пикеров отдельно. Если `val_P_loss` растёт — уменьшить `lr` или заморозить encoder.

---

## Фаза 4. Валидация новой модели

### 4.1 Детекция по тестовому периоду (Nov–Dec)

Запустить `detector.py` с новой моделью на тестовом периоде:

```python
# В detector.py: load_model_cudnn_v2('regional_model_v1/...')
```

Сравнить с каталогом: `python core/validate_catalog.py --year 2024 --month 11`.

### 4.2 Метрики сравнения

| Метрика | Базовая модель | Региональная модель |
| --- | --- | --- |
| Recall (catalog match) | 85.7% (42/49) | ? |
| Precision (false events) | ? | ? |
| P-pick error (median, с) | ? | ? |
| S-pick error (median, с) | ? | ? |
| Число ассоциаций | ~21 502 | ? |

P/S pick error можно посчитать по каталожным событиям с ручными фазами.

### 4.3 Порог-тюнинг

После обучения: запустить `threshold_tuner.py` по тестовым CSV новой модели.
Возможно, оптимальные пороги сместятся относительно текущих (det=0.7, P=0.3, S=0.2).

---

## Фаза 5. Итерация (опционально)

Если recall улучшился, но precision упал:

- Повысить `detection_threshold` → 0.75–0.80
- Пересобрать ассоциации

Если recall не улучшился:

- Проверить качество обучающих данных (snr-фильтр слишком мягкий?)
- Попробовать Вариант B (заморозить encoder) — возможно, глобальные признаки важнее региональных

---

## Файлы и пути

```text
core/
  dataset_builder.py        ← написать (Фаза 1)
  dataset_splitter.py       ← написать (Фаза 2)

training_data/
  train.hdf5 / train.csv
  dev.hdf5   / dev.csv
  test.hdf5  / test.csv

pre_trained_models/
  EqT_original_model.h5     ← скачать если нет

regional_model_v1/          ← выход trainer()
```

---

## Чеклист

- [ ] Скачать `EqT_original_model.h5` (если нет локально)
- [ ] Написать `dataset_builder.py` (Фаза 1)
- [ ] Запустить на 1 станции (BEYR), проверить HDF5 вручную
- [ ] Масштабировать на все 20 станций
- [ ] Написать `dataset_splitter.py` (Фаза 2)
- [ ] Запустить `trainer()` (Фаза 3)
- [ ] Сравнить метрики с базовой моделью (Фаза 4)
- [ ] Порог-тюнинг при необходимости (Фаза 5)
