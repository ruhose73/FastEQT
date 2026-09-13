# workspace/ — рабочая директория пайплайна

Единая точка для всех входных/выходных данных пайплайна. Структура (папки)
закоммичена в git, **содержимое — нет** (см. корневой `.gitignore`:
`workspace/*/input/*`, `workspace/*/output/*`). Скачавший проект получает
готовые папки под свои файлы и пути, не создавая мусора в `git diff`.

Одна папка на стадию пайплайна (см. `core/README.md` и порядок в `CLAUDE.md`),
внутри каждой — `input/` и `output/`:

```
workspace/
  data_processors/   input/   — StationXML (метаданные), сырые SDS-файлы волн
                      output/  — переименованные волновые файлы, station_*.json
  detector/          input/   — волновые файлы + station_*.json для детектора
                      output/  — CSV детекций по станциям, processing_log.csv
  associator/        input/   — стейджинг детекций по станциям (assoc_input)
                      output/  — associations.xml, Y2000.phs, traceNmae_dic.json
  locator/           input/   — SC3ML-инвентарь станций для SeisComP (inventory.scml)
                      output/  — hypocenters.csv (гипоцентры LOCSAT/NonLinLoc)
  magnitude/         input/   — (обычно не используется — см. ниже)
                      output/  — sta_corrections_*.csv, amps_*.csv, associations_ml*.xml
  validation/        input/   — (обычно не используется — см. ниже)
                      output/  — отчёты валидации/recall
  bulletin/          input/   — (обычно не используется — см. ниже)
                      output/  — экспортированные .BUL бюллетени
```

**Между стадиями данные не дублируются вручную.** Выход одной стадии — это
вход следующей по факту (так и настроены CLI-дефолты в `core/*.py`), поэтому
`input/` заполнен на диске не у каждой стадии — только там, где реально нужен
independent от предыдущей стадии ввод (`data_processors/input`,
`detector/input`, `associator/input`, `locator/input` — SC3ML-инвентарь
собирается один раз вручную, не производится предыдущей стадией пайплайна, см.
`core/locator.md`). `magnitude/`, `validation/`, `bulletin/` по умолчанию читают
`associator/output`/`magnitude/output` напрямую — их `input/` существует как
явное место, если хотите передать данные из другого источника (не по
дефолтному пути), не трогая код.

Все дефолты — только дефолты: любой путь переопределяется явным CLI-флагом
(`--input-dir`, `--output-base-dir`, `--assoc`, `--sta-corrections` и т.д. —
см. `--help` каждого скрипта). Никто не обязан класть файлы именно сюда —
это просто готовое место, если не хочется думать о путях самому.

Каталог событий (`catalog.xlsx`) и модель (`ModelsAndSampleData/`) — вне
`workspace/`, в корне репозитория (они не привязаны к конкретной стадии).
