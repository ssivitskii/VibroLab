# VibroLab — Vibration Classification & Generalization

AI-диагностика зубчатых редукторов и подшипников по вибросигналам.
Обучено на **SEU Gearbox Dataset** — реальные данные с DDS-стенда (SpectraQuest).

## Проверка качества: что измерено

**Ниже приведены исторические метрики случайного разбиения окон.** Сигналы
сначала нарезаются с 50% перекрытием, затем окна случайно разделяются на train/test.
Связанные фрагменты одной записи могут оказаться по обе стороны split. Эти числа
не подтверждают обобщение на новые записи, подшипники или режимы.

В [ранее сохранённом LOSO-аудите](docs/model-audit-2026-05.md) перенос между
20 и 30 Hz дал лишь **20% accuracy для gear-only и 11.1–11.4% для combined**.
Это исторические результаты из аудита, не новый запуск. Разрыв между random-window
и cross-speed оценками — главный исследовательский результат проекта.

Новый [benchmark протоколов](docs/validation-protocol.md) сравнивает случайные
окна, source-file holdout и leave-one-speed-out, сохраняет provenance и не меняет
экспортированную browser-модель.

### Исторические random-window результаты

| Модель | Данные | Признаков | Accuracy | F1 |
|--------|--------|-----------|----------|----|
| Combined full dataset | 20 файлов, 7200 seg | 53 | **98.4%** | **98.41%** |
| Gear single-channel | 10 файлов, 4000 seg | 53 | **98.9%** | 98.9% |
| Bearing single-channel | 10 файлов, 4000 seg | 53 | **100.0%** | 100.0% |
| Gear multi-channel (8ch) | 10 файлов, 4000 seg | 402 | **100.0%** | 100.0% |

5 классов шестерён: норма, скол зуба, отсутствие зуба, трещина корня, износ поверхности.
5 классов подшипников: норма, дефект шарика, внутренняя обойма, наружная обойма, комбинированный.

По умолчанию в [`web/model/`](./web/model/) лежит **экспортированная full-dataset SEU модель** для браузера.
Текущий frontend синхронизируется с `meta.json` и подстраивает классы/метрики под экспортированную модель.

## Быстрый старт

```bash
# 1. Установка
make install

# 2. Запуск полного стека
make run
# → http://localhost:8080                 (Platform + Dashboard)
# → http://localhost:8080/simulator.html  (3D Симулятор)
# → http://localhost:8080/docs            (FastAPI)
```

По умолчанию backend использует SQLite-базу в `runtime/vibrolab.db` и раздаёт `web/` вместе с API.
Теперь платформа поддерживает:

- серверную авторизацию по `email/password`
- `users`, `assets`, `inspections`, `snapshots`, `reports`
- перенос старого журнала из `localStorage` в БД
- cookie-сессии и серверную историю записей

Если нужен только базовый ML/runtime-стек без dev-инструментов:

```bash
pip install -r requirements.txt
```

## BI: сохранённые проверки

Экспорт из рабочей SQLite-базы создаёт UTF-8 CSV без учётных данных, идентификаторов,
названий объектов и свободных заметок:

```bash
python3 bi/export.py --source runtime/vibrolab.db --output /tmp/vibrolab-bi.csv
```

Путь к базе может отличаться при настройке `VIBROLAB_DATABASE_URL`. База создаётся
backend при запуске и заполняется по мере сохранения проверок. Экспорт не создаёт
данные, если база отсутствует. Одна строка соответствует одной сохранённой
проверке; в том числе учитываются проверки с демонстрационным входом (`input_type`).
Диагноз — предсказание модели, а не подтверждённая неисправность. SQL-витрины и
описание графиков находятся в `bi/schema.sql` и `bi/dashboard.json`.

Неизвестные значения категориальных полей заменяются на `unknown`; графики диагнозов
и статусов разделены по типу входа. Для локального демо можно запустить backend
с отдельной базой, зарегистрировать тестового пользователя и сохранить проверки
с демосигналом через интерфейс или API. Например, для Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-web.txt 'sqlmodel==0.0.34'
mkdir -p bi/demo-runtime
VIBROLAB_DATABASE_URL="sqlite:///$(pwd)/bi/demo-runtime/vibrolab.db" \
  .venv/bin/uvicorn backend.main:app --app-dir python --host 127.0.0.1 --port 18100
```

Созданная база демонстрационная, не является реальным потоком и не попадает в git.
На момент проверки установка без ограничения версии SQLModel давала ошибку
регистрации из-за обработки дат; версия 0.0.34 использована для локального демо.

Локальный Superset (Docker): `python3 bi/start.py --source <путь к источнику> --port 18088`. Снимок обновляется повторным запуском. Вход: `admin`, пароль хранится только в игнорируемом `bi/.env`. Стек слушает localhost; это локальная разработческая установка.
Остановить локальный BI без удаления данных: `BI_IMAGE=local/vibrolab-bi:6.1.0 BI_PORT=18088 docker compose --env-file bi/.env -p vibrolab-bi -f bi/compose.yaml down`.

## Обучение

```bash
# Структура данных:
# python/data/gear/    — Health_20_0.csv, Chipped_30_2.csv, ...
# python/data/bearing/ — Ball_20_0.csv, Inner_30_2.csv, ...

make train-gear        # Шестерни (5 классов)
make train-bearing     # Подшипники (5 классов)
make train-full        # Единая full-dataset модель (gear + bearing)
make train-gear-mc     # Мультиканал 8ch, 402 признака
make export            # Экспорт для браузера
make pipeline-full     # Full-dataset обучение → экспорт
```

Для расширенного DL/ONNX/SHAP-стека доступен optional extra:

```bash
pip install -e ".[advanced,dev]"
```

Автодеплой на сервер через GitHub Actions описан в [`docs/server-deploy.md`](./docs/server-deploy.md).
Операционные процедуры для backup / restore / rollback описаны в [`docs/operations.md`](./docs/operations.md).
Ручной server-side backup можно запускать и через GitHub Actions workflow `Backup Server`.

## Структура проекта

```
vibrolab/
├── python/              ML pipeline
│   ├── config.py        Конфигурация
│   ├── load_seu.py      Загрузка SEU Dataset
│   ├── features.py      53/402 признака
│   ├── train.py         Обучение Random Forest
│   ├── export_model.py  Экспорт в JSON
│   └── data/            Данные (gear/ + bearing/)
├── web/                 Frontend
│   ├── index.html       Dashboard
│   ├── simulator.html   3D Симулятор
│   ├── js/              Features + Model + FFT + Viz
│   └── model/           rf_model.json + meta.json
├── python/backend/      FastAPI + SQLite backend
├── tests/               Pytest
├── runtime/             SQLite database (gitignored)
├── .github/workflows/   CI/CD
├── Dockerfile
└── Makefile
```

## 3D Симулятор

- 3 типа: параллельный, планетарный, червячный
- Конструктор ступеней (до 5 пар, Z₁/Z₂ настраиваемые)
- 8 дефектов на каждую ступень (5 gear + 3 bearing)
- Эвольвентные зубья, подшипники, корпус-разрез, масло, акселерометры
- Звук, осциллограмма, FFT в реальном времени
- Upload CSV → inference через обученную RF модель

## Технологии

Python, FastAPI, SQLModel, scikit-learn, NumPy, Three.js, Web Audio API, Docker, GitHub Actions

## Лицензия

MIT
