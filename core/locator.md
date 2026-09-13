# locator.py — гипоцентры (LOCSAT/NonLinLoc через SeisComP)

## Назначение

Точка врезки в пайплайн (`CLAUDE.md`, `context/locsat-plan.md` раздел 2):

```
core/associator.py → core/locator.py (этот модуль) → core/ml_filter_v5.py → ...
```

Читает `associations.xml` (выход `core/associator.py`) — QuakeML с событиями и их
P/S-пиками, но **без гипоцентра** (в пайплайне до этого шага координаты события вообще не
вычислялись, `CLAUDE.md` Rule 20). Для каждого события строит `seiscomp.datamodel.Origin` с
прикреплёнными `Pick`/`Arrival` и вызывает `LocatorInterface.Create("LOCSAT").relocate()` —
классический локатор годографов (iasp91/ak135), поставляемый вместе с SeisComP. Пишет один
CSV, одна строка на событие — вход для `core/export_bul.py` (Lat/Long/Depth в бюллетене) и,
опционально, для `core/ml_filter_v5.py`/`core/validate_associator_v2.py` (реальная
эпицентральная дистанция вместо S-P-оценки — отдельный режим, не сделан на момент написания).

**Архитектурное решение (раздел 1.7b плана, согласовано с пользователем 2026-09-13):** этот
модуль считает и пишет **только сырые метрики** — `depth_km`, `depth_uncertainty_km`, `rms`,
`n_stations_used`, `n_arrivals_used`, признак сходимости/ошибки. Здесь **нет ни одного
порога и ни одного решения "проходит/не проходит"** — ни по RMS, ни по погрешности глубины,
ни по чему-либо ещё. Это осознанно зеркалит существующий паттерн `core/detector.py` (пишет
сырые `probability`/`SNR`) → `core/associator.py` (`prepare_station()`/`_passes_v2()`,
`core/associator.py:200-223`, применяет порог через свои CLI-флаги). Порог/фильтр по
качеству локации живёт в `core/export_bul.py` (Шаг 5 плана), не здесь.

Единственное исключение — `--min-arrivals`: это **не** фильтр качества результата, это
пропуск попытки локации там, где она математически не может дать решение (меньше 4
arrival'ов на 4 неизвестных — lat/lon/depth/T0). Событие всё равно попадает в CSV строкой с
`converged=False` и объясняющим `error`, как и любое другое несошедшееся событие — просто
без реальной попытки вызвать `relocate()`.

---

## Известные ограничения запуска

**Модуль `seiscomp` недоступен в conda `eqt3`.** Он ставится только вместе с отдельной
установкой SeisComP (см. `context/locsat-plan.md`, раздел 3 "Шаг 1" и раздел 12 — версия,
пути, обходы) и запускается через `seiscomp exec seiscomp-python`, не через обычный `python`.
Импорт `seiscomp.*` в этом файле **отложен** внутрь функций, которые его реально используют
(`_import_seiscomp()`, вызывается только из `main()`) — поэтому:

- `python core/locator.py --help` работает под **любым** интерпретатором, включая conda `eqt3`;
- `load_events(assoc_path)` (разбор `associations.xml`) тоже работает под любым python — не
  использует `seiscomp` вообще;
- реальный прогон (без `--help`, доходящий до вызова локатора) требует `seiscomp-python` —
  под conda `eqt3` упадёт с понятным `SystemExit`, объясняющим, что запускать нужно иначе
  (не голым `ModuleNotFoundError` без контекста).

Это не воркэраунд/маскировка — граница процессов (conda `eqt3` python 3.10 vs
`seiscomp-python`, другая версия) реальна и не устранена, отражена явно, как и предполагал
план (раздел 6, риски). Проверено на практике: разделение на "подготовка входа / вызов
locsat" (которое обсуждалось в плане как вероятно необходимое) **не понадобилось** — весь
модуль работает одним `seiscomp-python`-процессом, парсинг XML использует только stdlib
`xml.etree`.

**`--engine nonlinloc` не реализован до конца.** `LocatorInterface.Create('NonLinLoc')` в
текущей установке SeisComP (раздел 12 плана) возвращает `None` — плагин не подключён в
дефолтном конфиге. Флаг принимается CLI (не отвергается), но каждое событие получит строку
с `error="LocatorInterface.Create('NonLinLoc') returned None"` — проверено практикой, не
падает, просто ни одно событие не сойдётся. NonLinLoc — отдельная, опциональная работа
(раздел 10 плана, Шаг 8): нужна региональная скоростная модель + предпосчитанная сетка
времён пробега (`Vel2Grid`/`Grid2Time`), которых в проекте пока нет.

**`--inventory` не собирается этим скриптом.** Нужен заранее подготовленный SC3ML-файл:

1. Слить per-station StationXML (`workspace/data_processors/input/metadata/`) в один файл —
   `context/scratch/merge_stationxml.py` (учитывает узкие/конфликтующие epoch-окна у части
   станций, раздел 12 плана, "Реальная проблема метаданных").
2. Сконвертировать: `seiscomp exec fdsnxml2inv in.xml out.scml`.

Готовый файл на 68 станций (RU/N0/DA) уже лежит в `workspace/locator/input/inventory.scml`
(копия `context/scratch/combined_inventory.scml`) — пересобирать заново нужно только если
изменится состав `workspace/data_processors/input/metadata/`.

**Кастомные таблицы годографов не реализованы.** Локатор использует только то, что уже
установлено вместе с SeisComP (`~/seiscomp-install/seiscomp/share/locsat/tables/iasp91*`,
включая региональные `Pg`/`Pn`/`Sg`, раздел 12 плана) — CLI-флага для указания своего набора
таблиц (`--tables-path`/`--profile`) нет. Не добавлено намеренно: единственная доступная на
момент написания модель — встроенная iasp91/ak135 (раздел 1.4 плана — запрос региональной
скоростной модели у НИИ снят с текущего хода плана), делать непроверенный API вслепую не
стали. При появлении кастомных таблиц (`jsaul/locsat-tables`, раздел 1.4) — добавить флаг,
предварительно проверив `loc.setProfile()`/`loc.setParameter()` на практике, не по догадке.

---

## Алгоритм

1. `load_events(--assoc)` — разбирает `associations.xml` (namespace QuakeML `bed/1.2`)
   целиком, все события разом. Ключ события — `publicID` (тот же, что использует
   `validate_associator_v2.py::load_associations()`), плюс placeholder-`origin`
   (lat/lon/time) события — validate_associator_v2 его не читает (не нужен для S-P-оценки),
   а здесь это начальное приближение для LOCSAT. По коду реального райтера
   (`EQTransformer/utils/associator.py`, `Origin(..., longitude=..., latitude=...,
   method="EqTransformer")` — координаты первой сработавшей станции) — не предположение,
   а то, что реально пишется в продакшене.
2. Инвентарь (`--inventory`) грузится один раз в `seiscomp.client.Inventory.Instance()` —
   синглтон без БД/мессджинга (раздел 12 плана). **Важно**: `.load()` возвращает `None` даже
   при полностью успешной загрузке — это не индикатор ошибки, не проверяется как булево.
3. Для каждого события (кроме отфильтрованных `--min-arrivals`) — `build_origin()` собирает
   `Origin`+`Pick`+`Arrival` (глубина — `--depth-km`, по умолчанию 5 км, конвенция
   `Y2000.phs`), `relocate_event()` вызывает `LocatorInterface.Create(--engine).relocate()` и
   считывает результат. Координаты станций резолвятся самим `relocate()` через
   `Pick.waveformID()` + загруженный инвентарь — вручную не проверяются и не фильтруются
   (раньше в `context/scratch/step2_smoke_test.py` такая ручная проверка была и оказалась
   избыточной/неточной, см. раздел 12 плана, находка про эпохи станций).
4. Строка CSV пишется всегда — сошедшееся событие получает полные метрики, несошедшееся
   (`RuntimeError`, `None` от `relocate()`, `Create()` вернул `None`, либо пропуск по
   `--min-arrivals`) — `converged=False` и текст ошибки в `error`, остальные числовые поля
   `None`/`0`. `core/export_bul.py` (Шаг 5 плана) откатывается к прежнему поведению (только
   T0 по S-P) для строк без решения.

---

## Формат `hypocenters.csv`

| Колонка | Тип | Значение |
|---|---|---|
| `event_id` | str | `publicID` события — тот же ключ, что в `associations.xml` |
| `engine` | str | `"LOCSAT"` / `"NonLinLoc"` |
| `converged` | bool | `True`, если `relocate()` вернул решение |
| `error` | str/пусто | причина отсутствия решения (пусто при `converged=True`) |
| `t0` | str/пусто | время очага после локации (не placeholder) |
| `lat`, `lon` | float/пусто | гипоцентр, градусы |
| `depth_km` | float/пусто | глубина, км |
| `depth_uncertainty_km` | float/пусто | **важно** — `0.0` почти всегда означает упор в границу таблицы годографов (0 или ~750 км), а не настоящее решение; см. `context/locsat-plan.md` раздел 5, находка Части A. Это сырое число, не решение — интерпретация и порог остаются за потребителем |
| `rms` | float/пусто | `quality().standardError()`, секунды |
| `n_stations_used`, `n_arrivals_used` | int | после `relocate()` — сколько станций/arrival'ов реально вошло в решение (может быть меньше входных, если инвентарь не резолвит часть станций) |

---

## Полный список аргументов CLI

| Флаг | По умолчанию | Назначение |
|---|---|---|
| `--assoc` | `workspace/associator/output/associations.xml` | входной XML ассоциатора |
| `--inventory` | `workspace/locator/input/inventory.scml` | SC3ML-инвентарь станций (собирается вручную, см. выше) |
| `--out` | `workspace/locator/output/hypocenters.csv` | выходной CSV |
| `--engine` | `locsat` | `locsat` \| `nonlinloc` (nonlinloc не реализован до конца, см. выше) |
| `--depth-km` | `5.0` | начальная глубина, км |
| `--min-arrivals` | `4` | не пытаться локировать события с меньшим числом arrival'ов — не порог качества, см. выше |

---

## Запуск

В WSL, из каталога установки SeisComP (`~/seiscomp-install/seiscomp`):

```bash
# CLI-справка — работает и без seiscomp (например, из Windows/conda eqt3 для проверки флагов)
python core/locator.py --help

# Реальный прогон — обязательно seiscomp-python
./bin/seiscomp exec seiscomp-python /mnt/d/codding/aspirantura/EQTransformer/core/locator.py \
    --assoc /mnt/d/codding/aspirantura/EQTransformer/workspace/associator/output/associations.xml \
    --out   /mnt/d/codding/aspirantura/EQTransformer/workspace/locator/output/hypocenters.csv

# Другой движок / другая начальная глубина
./bin/seiscomp exec seiscomp-python /mnt/d/.../core/locator.py --engine nonlinloc
./bin/seiscomp exec seiscomp-python /mnt/d/.../core/locator.py --depth-km 10.0
```

---

## Проверено практикой (2026-09-13)

Прогнан на `workspace/bulletin/input/sample_associations_20ev.xml` (тот же сэмпл, что и
Часть A/B плана — на момент написания единственный `associations.xml`, реально
существующий в рабочем каталоге; полного прогона детектор+ассоциатор здесь нет).
Результат — **12/20 сошлось, 8/20 нет**, значения по каждому событию (depth/RMS/погрешность)
**совпадают число-в-число** с прогоном `context/scratch/locator_convert.py` (Шаг 3 Часть B) и
с результатами Части A (`context/locsat-plan.md`, раздел 5). `--engine nonlinloc` проверен
отдельно — не падает, каждая строка получает явный `error`, `0 converged`.

На другом/большем реальном файле не проверялось — такого локально нет. Прогнать повторно на
`workspace/associator/output/associations.xml`, когда он появится, прежде чем считать формат
и путь окончательно проверенными на "реальных данных" в буквальном смысле.
