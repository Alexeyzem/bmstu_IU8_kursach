# Статическое обнаружение вредоносных Windows PE

Проект классифицирует PE по статическим признакам EMBER. Для экспериментов используются готовые векторы SOREL-20M и отдельная внешняя коллекция gold. Решение фиксировано заранее — класс 1 при оценке ≥ 0,5.

## Быстрый запуск

Проверенное окружение — macOS на Apple Silicon, Python 3.14.6, OpenMP для LightGBM (на macOS устанавливается командой `brew install libomp`). Запускать из корня репозитория. Для демонстрации достаточно приложенных в `examples/` модели и вектора одного легитимного файла. Переменные окружения, ключи API и Docker не требуются.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detector.txt
python -m pe_static_analyzer.prediction.demo
```

Ожидаемый конец строки — `score=0.000261 class=0`. Демонстрация использует модель LightGBM, обученную на стратифицированных 100 тыс. объектов, с 600 деревьями и 2 155 признаками без группы `header`. Порядок входных колонок для проверки новых PE хранится в `examples/lightgbm.indices.json`. Вектор уже прошёл signed-log и отбор признаков. Сборка включает только его числовые признаки и SHA-256. Для просмотра выводов можно сразу открыть отчёт.

## Структура пакета pe_static_analyzer

| Каталог | Назначение |
|---|---|
| `features/` | Схема и отбор признаков. В `groups/` определены группы EMBER, в `ember.py` они собираются в вектор PE. |
| `sorel/`, `gold/` | Подготовка выборок SOREL и внешней коллекции gold. В `gold/` также находятся извлечение внешних векторов и пакетная проверка. |
| `training/` | Обучение моделей, сравнение параметров и переобучение с разными наборами признаков. |
| `evaluation/` | Калибровка, расчёт SHAP и графики результатов. Модуль `gold_shap` сравнивает вклады признаков у пропущенных и обнаруженных файлов панели. |
| `prediction/`, `gui/` | Загрузка готового детектора, прогноз для одного файла, демо и графический интерфейс. |
| `core/` | Чтение входных файлов и общие метрики классификации. |

## Где менять настройки

Вызов `BoostingModel.lightgbm()` создаёт итоговую конфигурацию для демонстрации. Это LightGBM на 100 000 стратифицированных файлах, 600 деревьев, шаг 0,08, до 63 листьев и минимум 100 объектов в листе. Тот же класс используется для контрольных опытов и CatBoost.

В [settings.py](pe_static_analyzer/settings.py) константа `ACTIVE_EXPERIMENT` выбирает объём выборки, seed и каталоги моделей и отчёта. Для основного опыта оставьте `BASELINE` (100 000 файлов, seed 42). Имена и пути специальных задач находятся в классах `…Settings` соответствующих модулей. Для проверки одного файла задайте `INPUT_FILE` в [single_file.py](pe_static_analyzer/prediction/single_file.py) или выберите файл в GUI. Порог решения зафиксирован на 0,5.

## Повторить исследование

Для обучения, пакетной оценки и воспроизведения сравнений установите полный набор зависимостей командой `python -m pip install -r requirements.txt`. Он включает зависимости детектора.

1. Получить `meta.db` и LMDB с обработанными EMBER-признаками по инструкции [авторов SOREL-20M](https://github.com/sophos/SOREL-20M). Разместить базу в `datasets/sorel20m/meta.db`, а `data.mdb` — в `datasets/sorel20m/ember_features/`. Локальная копия SOREL занимает около 75 ГиБ. Для расчётов нужны дополнительное место под манифесты и несколько ГиБ оперативной памяти.
2. Подготовить манифесты и список признаков. При наличии исходных файлов эксперимента в `outputs/matrices/` и `outputs/features/selected_features.json` можно перейти к обучению.

```sh
python -m pe_static_analyzer.sorel.manifests
python -m pe_static_analyzer.features.analysis
python -m pe_static_analyzer.sorel.sampling
python -m pe_static_analyzer.training.comparison
python -m pe_static_analyzer.training.validation
python -m pe_static_analyzer.evaluation.reliability
```

3. Команда `training.comparison` последовательно создаёт бустинг, вызывает `train()`, сохраняет модель и вызывает `validate_on_sorel()`. Если в каталоге опыта подготовлена папка `gold_panel/` с признаками, этот же запуск вызывает `validate_on_external()`. Модуль записывает модели в `outputs/research/`, а метрики, таблицы глобальной важности и локальных объяснений — в `reports/comparison/`. Детектор без заголовка сохраняется в `outputs/research/without_header_600.txt` вместе с `without_header_600.indices.json`, его результаты — в `reports/comparison/without_header_600/`. Для сравнения сохраняется и модель со всеми отобранными признаками (`final_lightgbm_600.txt`). Исходное сравнение LightGBM и CatBoost с равным числом деревьев (100) воспроизводится той же командой. Время в поле `train_seconds` измеряет только обучение после чтения матрицы. Команда validation проверяет отдельные параметры на другом блоке валидационного манифеста.
4. Для вложенной выборки сначала подготовьте основной манифест, затем установите `ACTIVE_EXPERIMENT = SMALL_SAMPLE`. Для независимых повторов выберите `REPEAT_43` или `REPEAT_44`. После смены профиля выполните те же две команды.

```sh
python -m pe_static_analyzer.sorel.sampling
python -m pe_static_analyzer.training.comparison
```

5. Внешняя проверка использует локальную коллекцию с метаданными `outputs/gold/gold_final_metadata.csv`. Для воспроизведения нужны те же файлы или ранее извлечённые признаки. Идентификаторы оценённых объектов опубликованы в `reports/comparison/stratified_lightgbm/gold/predictions.csv`. Если внешняя панель подготовлена после обучения, команда `gold.comparison` загружает сохранённые модели и вызывает тот же метод внешней проверки. Для воспроизведения основного опыта вернуть профиль `BASELINE`. При наличии локальной коллекции подготовить панель и графики следующими командами.

```sh
python -m pe_static_analyzer.gold.panel
python -m pe_static_analyzer.gold.extraction
python -m pe_static_analyzer.gold.comparison
python -m pe_static_analyzer.training.feature_ablation
python -m pe_static_analyzer.evaluation.figures
python -m pe_static_analyzer.evaluation.gold_shap
```

Для проверки всей очищенной коллекции gold запустите `python -m pe_static_analyzer.gold.lightgbm_fast_inference`. Команда использует приложенный детектор и записывает прогнозы, метрики и ошибки обработки в `outputs/metrics/` (имена содержат `without_header_600`).

GUI запускается через `python -m pe_static_analyzer.gui.desktop_app` и требует Tk. Для проверки используется приложенный LightGBM на 600 деревьях без группы заголовка.
