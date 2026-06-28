# Установка FastEQT на Linux

Два варианта установки: с GPU (для обучения/инференса) и CPU-only (для сервера без GPU или со слабой видеокартой для вывода изображения).

## Требования

- Linux x86_64
- Miniconda / Anaconda
- **GPU-вариант:** NVIDIA GPU с драйвером >= 450.x (проверено на 570.x, CUDA 12.8)

---

## Вариант A: CPU-only (сервер без GPU)

### A1 — создать окружение

```bash
conda env create -f environment_linux_cpu.yml
conda activate eqt3
```

### A2 — установить TensorFlow (CPU)

```bash
pip install --no-build-isolation tensorflow==2.8.0 keras==2.8.0 keras-preprocessing==1.1.2
```

`tensorflow==2.8.0` (без суффикса `-gpu`) — CPU-only пакет, не требует CUDA.

### A3, A4, A5 — остальные шаги те же, что и для GPU варианта (шаги 3–5 ниже)

Шаг 6 (LD_LIBRARY_PATH) **не нужен**.

### Проверка (CPU)

```bash
python -c "import EQTransformer; import tensorflow as tf; print('TF:', tf.__version__)"
```

---

## Вариант B: GPU

### B1 — создать conda-окружение

```bash
conda env create -f environment_linux.yml
conda activate eqt3
```

`environment_linux.yml` устанавливает через conda: Python 3.10, CUDA 11.2, cuDNN 8.1, numpy, scipy, obspy, matplotlib, pandas, h5py, jupyterlab и прочие научные пакеты.

**Почему не использовать экспортированные `.yml` из Windows?**
Файлы `fasteqt_env.yml` / `fasteqt_env_no_build.yml` содержат Windows-специфичные пакеты (`pywin32`, `pywinpty`, `win_inet_pton`, MSVC runtime) и не работают на Linux. Использовать только `environment_linux.yml`.

### B2 — установить TensorFlow (GPU)

```bash
pip install --no-build-isolation tensorflow-gpu==2.8.0 keras==2.8.0 keras-preprocessing==1.1.2
```

Флаг `--no-build-isolation` обязателен — без него pip создаёт изолированную среду сборки и скачивает несовместимые старые версии numpy/scipy, которые не собираются с современным setuptools.

---

## Общие шаги (варианты A и B)

### Шаг 3 — установить локальный пакет

```bash
pip install -e . --no-deps
```

`--no-deps` нужен потому что `setup.py` указывает точные версии зависимостей, уже установленных на шагах 1–2. Без флага pip попытается пересобрать `scipy==1.4.1` из исходников, что не работает на Python 3.10 с numpy 1.23.

### Шаг 4 — исправить protobuf

TensorFlow 2.8.0 несовместим с protobuf >= 3.20:

```bash
pip install "protobuf<3.20"
```

### Шаг 5 — исправить setuptools

setuptools >= 70 не экспортирует `pkg_resources`, который нужен obspy:

```bash
pip install "setuptools<70"
```

### Шаг 6 — прописать путь к CUDA библиотекам (только GPU-вариант)

conda устанавливает CUDA в директорию окружения, но TF ищет их в системных путях. Добавляем автоматическую установку при активации:

```bash
mkdir -p /home/ruhose73/miniconda3/envs/eqt3/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=/home/ruhose73/miniconda3/envs/eqt3/lib:$LD_LIBRARY_PATH' \
    > /home/ruhose73/miniconda3/envs/eqt3/etc/conda/activate.d/cuda.sh
```

После этого переменная выставляется автоматически при каждом `conda activate eqt3`.

## Проверка (GPU-вариант)

```bash
conda activate eqt3
python -c "import EQTransformer; import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Ожидаемый вывод:
```
TF: 2.8.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Известные проблемы

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `Could not find eqtransformer==0.1.61` | На PyPI нет такой версии, это локальный пакет | `pip install -e . --no-deps` |
| `scipy==1.4.1` не собирается | `distutils.msvccompiler` — Windows-only модуль | `pip install -e . --no-deps` |
| `TypeError: Descriptors cannot be created directly` | protobuf >= 3.20 несовместим с TF 2.8 | `pip install "protobuf<3.20"` |
| `No module named 'pkg_resources'` | setuptools >= 70 убрал `pkg_resources` | `pip install "setuptools<70"` |
| `GPU: []` — GPU не видна | CUDA libs conda не в `LD_LIBRARY_PATH` | Шаг 6 выше |
| `python_requires '==3.10.5'` | Жёсткий пин версии в `setup.py` | Изменено на `~=3.10` |
