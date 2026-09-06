# gpu_pipeline.py — описание алгоритма

## Цель

Параллельно обрабатывает список станций (SEED → CSV детекций) **на GPU**, несколькими процессами одновременно (`ProcessPoolExecutor`) — используется для точечного/дополнительного прогона станций отдельно от основной конфигурации `core/detector.py::__main__` (например, когда нужно посчитать несколько станций с другими датами или добавить станции, не трогая основной запуск). Собственной логики детекции не содержит — загружает `core/detector.py` через `importlib` и вызывает `process_station_v3()` для каждой станции; сам `gpu_pipeline.py` отвечает только за пул процессов, настройку GPU-памяти и список станций/дат для конкретного прогона.

Модель, список станций, диапазон дат, число воркеров задаются флагами (`--help`); константы в начале файла остались как значения по умолчанию для этих флагов. Более старая версия без CLI — `legacy/gpu_pipeline.py`.

---

## Отличие от `cpu_pipeline.py`

Единственная содержательная разница — способ работы с вычислительным устройством: здесь GPU не скрывается, а настраивается через `set_memory_growth` (растёт по необходимости, не резервирует сразу всю VRAM), а число воркеров подбирается под объём видеопамяти, а не под число ядер CPU. Ограничения TF-потоков (`inter_op`/`intra_op`), которые нужны CPU-версии, здесь не требуются. Подробное сравнение — в `cpu_pipeline.md`.

---

## Конфигурация — CLI-флаги (`--help`)

Константы с теми же именами остались в файле как значения по умолчанию.

| Флаг | Константа-дефолт | Назначение |
|---|---|---|
| `--model-path` | `MODEL_PATH` | путь к весам EQT (`.h5`), передаётся в `load_model_cudnn_v2` |
| `--output-base-dir` | `OUTPUT_BASE_DIR` | корневая директория, куда `process_station_v3` пишет CSV по каждой станции (`{OUTPUT_BASE_DIR}/{STATION}/{station}.csv`) — под какое поколение данных считать, определяется этим путём |
| `--max-workers` | `MAX_WORKERS` | число процессов `ProcessPoolExecutor` — сколько станций обрабатывается одновременно; для GPU-режима ограничено объёмом VRAM (несколько процессов держат в памяти каждый свою копию модели и активаций), а не числом ядер CPU |
| `--input-dir` / `--json-dir` | `_IN` / `_JS` | из них по кодам из `--stations` достраиваются пары `(входная_директория, station_*.json)` |
| `--stations` | `STATIONS` | коды станций через запятую — какие обрабатывать в этом прогоне |
| `--date-from` / `--date-to` | `DATE_FROM` / `DATE_TO` | диапазон дат (строка, парсится через `UTCDateTime`), правая граница не включается — тот же флаг, что и в `cpu_pipeline.py` |
| `--estimate-uncertainty`(`--no-estimate-uncertainty`) / `--number-of-sampling` | `ESTIMATE_UNCERTAINTY` / `NUMBER_OF_SAMPLING` | раньше зашиты прямо в вызове `process_station_v3` внутри `run_station` (`estimate_uncertainty=True, number_of_sampling=5`) — теперь CLI-флаги |
| `--detection-threshold` / `--p-threshold` / `--s-threshold` / `--keep-ps`(`--no-keep-ps`) / `--allow-only-s` / `--sp-limit` / `--batch-size` | одноимённые константы | пороги предиктора, раньше зашитые в `worker_v4` внутри `detector.py` (см. `detector.md`) |

---

## Как работает: initializer pattern

Загрузка модели — самая дорогая по времени операция, поэтому её выполняют один раз на процесс, а не один раз на станцию:

```python
ProcessPoolExecutor(max_workers=MAX_WORKERS,
                     initializer=_init_worker,
                     initargs=(MODEL_PATH,))
```

1. **`_init_worker(model_path)`** — выполняется один раз, сразу при старте каждого процесса-воркера:
   - добавляет `_ROOT` (корень репозитория) в `sys.path` — процесс-воркер запускается как отдельный интерпретатор Python и не наследует изменения `sys.path`, сделанные родителем после старта;
   - `gpus = tf.config.list_physical_devices('GPU')`; если GPU найден — `tf.config.experimental.set_memory_growth(gpus[0], True)`; `RuntimeError` (устройство уже было инициализировано до этого вызова) молча проглатывается — эта настройка должна выполняться до первого использования GPU в процессе, повторный вызов недопустим, отсюда `try/except`;
   - загружает `core/detector.py` через `importlib.util.spec_from_file_location` (`_load_detector()`) — так каждый процесс получает собственный экземпляр модуля;
   - загружает модель один раз через `detector.load_model_cudnn_v2(model_path)`, сохраняет в глобальную `_model` — переиспользуется для всех станций, назначенных этому процессу.
2. **`run_station(args)`** — функция-задача, отправляется в пул через `executor.submit()` один раз на каждую станцию из списка, построенного по `--stations`:
   - вызывает `_detector_mod.process_station_v3(base_directory, stations_json, model=_model, date_from=..., date_to=..., output_base_dir=..., estimate_uncertainty=..., number_of_sampling=..., ...)` — все значения приходят из CLI-флагов, разобранных до запуска пула;
   - любое исключение перехватывается целиком, возвращается как `(station_name, "error", traceback.format_exc())` — ошибка одной станции не прерывает обработку остальных.

---

## Что делает `process_station_v3` (внутри `core/detector.py`)

Идентично `cpu_pipeline.py` — оба файла вызывают одну и ту же функцию, вся специфика детекции живёт в `detector.py`, не здесь. Полное описание цепочки — в `cpu_pipeline.md` (раздел «Что делает `process_station_v3`»):

`geofile_splitter_multi_chanels_v3` (нарезка на 10-минутные сегменты, шаг 5 минут) → `preproc_sequential_v5` (последовательное потребление генератора, запись CSV по мере готовности, `gc.collect()` каждые 50 сегментов) → `worker_v4` на каждый сегмент (`preprocessorV7_mem` + `predictor_mem_non_hdf_load_model_v6`, с порогами `detection_threshold/P_threshold/S_threshold/keepPS/allowonlyS/spLimit/batch_size` — настоящими параметрами по всей цепочке, задаются флагами `--detection-threshold`/`--p-threshold`/`--s-threshold`/`--keep-ps`(`--no-keep-ps`)/`--allow-only-s`/`--sp-limit`/`--batch-size`).

При `estimate_uncertainty=True` модель вызывается `number_of_sampling` раз в режиме `model(X, training=True)` (реальный MC Dropout — см. правило №8 в `CLAUDE.md`) вместо одного `model.predict()`, и в CSV дополнительно попадают `detection_uncertainty`/`p_uncertainty`/`s_uncertainty`, которые затем использует `core/associator.py` в режимах `UNCERTAINTY_MODE='weight'`/`'filter'`.

**Несколько процессов на одном физическом GPU:** `--max-workers` процессов означает столько же отдельных копий модели и активаций в видеопамяти одновременно. `set_memory_growth` позволяет каждому процессу занимать VRAM по мере надобности, а не резервировать всё сразу, но не гарантирует, что все процессы уместятся без взаимного вытеснения при больших сегментах — явного деления/лимита VRAM между воркерами в коде нет.

---

## Вывод и логи

Идентично `cpu_pipeline.py` (см. соответствующий раздел там):
- построчный вывод по завершении каждой станции (`[elapsed мин] [STATUS] station_name`, при ошибке — первые 800 символов traceback);
- сводка в конце (успехи/ошибки, общее время);
- `preproc_sequential_v5` дополнительно пишет лог по сегментам в общий лог (`--log-file` в `detector.py`, по умолчанию `workspace/detector/output/processing_log.csv`) — та же гонка при `--max-workers > 1`, что и в CPU-версии: файл в режиме перезаписи, общий на все процессы, сохраняется лог только последней завершившейся станции.

---

## Мёртвый код

`get_threads_to_use(percent)` была определена в файле, но нигде не вызывалась — дублировала `core/threads.py::get_threads_to_use`. Убрана; в `legacy/gpu_pipeline.py` ещё присутствует.
