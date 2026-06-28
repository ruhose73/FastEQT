# sta_correction_estimator.py — вычисление станционных поправок ML

## Назначение

Вычисляет станционные поправки `S` для формулы ML (Дягилев et al. 2023, формула 5б)
на основе каталожных событий с известной магнитудой Ms.

Поправка компенсирует систематические отличия конкретной станции от модели:
особенности грунта, азимутальная анизотропия, погрешности в StationXML.

---

## Как это работает

### Шаг 1 — Матчинг каталога с ассоциатором

Для каждого события из `catalog.xlsx` ищется ближайшее по времени событие
в `associations.xml`, с поправкой на время пробега P-волны до ближайшей станции:

```
expected_assoc_time = catalog_origin_time + P_travel_time_to_nearest_station
совпадение если |assoc_time − expected| ≤ match_win (по умолчанию 15 с)
```

Алгоритм совпадает с `validate_associator.py`.

### Шаг 2 — Измерение амплитуды S-волны

Для каждого совпавшего события на каждой станции:
- Берётся горизонтальная компонента (E > N > Z)
- Читается окно `[S_time .. S_time + 5с]` из MSEED
- Снимается инструментальный ответ: `remove_response(output='DISP')` → смещение грунта в метрах
- Максимум |данных| в окне = амплитуда A (м)

Чтение оконное (`starttime/endtime`) — не весь файл. Обработка параллельная:
станции делятся по воркерам, каждый воркер работает только со своими файлами.

### Шаг 3 — Вычисление поправки на событие

Для каждой пары (событие, станция):

```
ML_raw  = lg(A_нм) + 1.024·lg(R) + 0.001648·R − 1.889
S_ev    = Ms_каталог − ML_raw
```

`ML_raw` — магнитуда без поправки. `S_ev` — поправка, необходимая чтобы
формула дала каталожную магнитуду. Расстояние R — из S-P времени:
`R = Vp·Vs / (Vp−Vs) · ΔT(S-P)`. Станции с `R < 30 км` отбрасываются.

### Шаг 4 — Усреднение и детектор выбросов

По каждой станции собирается вектор `[S_ev_1, S_ev_2, ..., S_ev_N]`.

Выбросы определяются методом IQR:
```
Q1, Q3 = 25-й и 75-й перцентили
IQR    = Q3 − Q1
выброс если S < Q1 − 1.5·IQR  или  S > Q3 + 1.5·IQR
```

Итоговая поправка = **среднее без выбросов** (`mean_clean`).
В терминал выводится и `mean_all` (с выбросами) для сравнения.

Станция включается в результат только если N ≥ `--min-events` (по умолчанию 10).

---

## Выходные файлы

### sta_corrections_computed.csv — итоговые поправки
Формат совпадает с `sta_corrections_dyagilev2023.csv`. Используется напрямую
в `ml_filter_v4.py` через `--sta-corrections`.

```
station,S
BEYR,0.231
SOC,-0.042
...
```

### sta_corrections_detail.csv — детальные поправки по событиям

```
station, cat_time,           pub_id,  Ms,   R_km,  A_nm,      ML_raw, S,     is_outlier
BEYR,    2024-01-05T14:23:11, quak…, 1.20, 45.3,  1234.56,   0.971,  0.229,
BEYR,    2024-01-07T08:41:05, quak…, 2.10, 38.1,  8901.23,   1.843,  0.257, yes
```

---

## Вывод в терминал

```
Станция    N  Выбросы  S (чистое)  S (с выбросами)
BEYR       14        1       0.231            0.253  *** ВЫБРОСЫ (1) ***
GLDR       11        0       0.187            0.187
SOC        18        0      -0.042           -0.042

ДЕТАЛИ ВЫБРОСОВ:
  BEYR: IQR=[0.210, 0.250]  iqr=0.040  границы=[0.150, 0.310]
    2024-01-07 08:41:05  S=+0.380  (отклонение от чистого: +0.149)

Исключены (менее 10 событий):
  GRYR  N=3  S_mean=0.142
```

---

## Запуск

```powershell
conda activate eqt3

# Посмотреть только матчинг (без чтения форм волн)
python core/sta_correction_estimator.py --info --year 2024 --month 1

# Вычислить поправки (январь 2024, мин. 10 событий на станцию)
python core/sta_correction_estimator.py --year 2024 --month 1

# Мягче: мин. 5 событий на станцию
python core/sta_correction_estimator.py --year 2024 --month 1 --min-events 5

# Только по сильным событиям (Ms >= 1.0)
python core/sta_correction_estimator.py --year 2024 --month 1 --min-ms 1.0

# Пересобрать кэш амплитуд
python core/sta_correction_estimator.py --rebuild-cache --year 2024 --month 1

# Кастомные пути
python core/sta_correction_estimator.py `
  --assoc-in data-in-memory/association_gpu_100_150/associations.xml `
  --out-summary sta_corrections_new.csv `
  --year 2024 --month 1

# Ослабить IQR-фильтр (больше выбросов оставить)
python core/sta_correction_estimator.py --year 2024 --month 1 --iqr-k 2.0

# Больше воркеров
python core/sta_correction_estimator.py --year 2024 --month 1 --workers 8
```

---

## Применение поправок в ml_filter_v4

```powershell
python core/ml_filter_v4.py `
  --ml-threshold 1.0 `
  --sta-corrections sta_corrections_computed.csv `
  --validate --year 2024 --month 1
```

---

## Параметры по умолчанию

| Параметр | Значение | Аргумент |
|---|---|---|
| XML ассоциатора | `data-in-memory/association_gpu_100_150/associations.xml` | `--assoc-in` |
| Каталог | `catalog.xlsx` | `--catalog` |
| Формы волн | `geofiles/` | `--waveforms` |
| StationXML | `metadata/` | `--metadata-dir` |
| Кэш амплитуд | `amps_sta_corr.csv` | `--cache-amp` |
| Итоговый CSV | `sta_corrections_computed.csv` | `--out-summary` |
| Детальный CSV | `sta_corrections_detail.csv` | `--out-detail` |
| Окно матчинга | 15 с | `--match-win` |
| Мин. событий | 10 | `--min-events` |
| IQR множитель | 1.5 | `--iqr-k` |
| Воркеров | 4 | `--workers` |
| Vp / Vs | 6.0 / 3.4883 км/с | `--vp` / `--vs` |
| Окно S-волны | 5.0 с | `--win-sec` |
| Amp ratio max | 5.0 | `--amp-ratio-max` |
| Amp ratio min | 0.0 (выкл.) | `--amp-ratio-min` |

---

## Важные замечания

- `--wood-anderson` **не применяется** — формула 5б ожидает смещение грунта, не WA-амплитуду.
- Кэш `amps_sta_corr.csv` отдельный от кэша `ml_filter_v4` (`amps_wa.csv`).
  Это разные наборы событий: здесь только каталожные, там все ассоциации.
- При смене `--assoc-in` или `--year/--month` нужен `--rebuild-cache`.
  При смене `--amp-ratio-max` или `--win-sec` — **не нужен** (кэш хранит сырые амплитуды,
  фильтр применяется позже).
- **Amplitude ratio filter** (`--amp-ratio-max 5.0`): удаляет измерения где A_nm > 5× ожидаемой
  амплитуды по каталожной Ms и R. Ловит шум и сбои remove_response, но **не** затрагивает
  станции с аномально низкой амплитудой (те получают крупную положительную поправку).
- Выбросы чаще всего означают: плохой пик S у конкретного события на станции,
  или сбой `remove_response` (шум в окне S-волны).
