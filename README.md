# Система тестирования плотностных алгоритмов для консенсус кластеризации

Проект для сравнения плотностных методов кластеризации, оценки качества на датасетах и запуска встроенных методов консенсуса с возможностью подключать свои алгоритмы и свои методы консенсуса через веб-интерфейс.

## Возможности

- Готовые и синтетические датасеты.
- Встроенные алгоритмы: DBSCAN, HDBSCAN, DPC, RD-DAC, CKDPC.
- Разделы: датасеты, сравнение алгоритмов, консенсус-анализ, загрузка пользовательских алгоритмов и методов консенсуса (`.py`).
- AI-ассистент через локальный Ollama (раздел «Сравнение алгоритмов» и «Консенсус-анализ»).

## Запуск веб-приложения

Из корня репозитория (нужен Python 3.10+ и зависимости ниже):

```bash
pip install numpy scipy scikit-learn pandas matplotlib streamlit hdbscan
streamlit run webapp/app.py
```

Браузер откроется по адресу, который покажет Streamlit (обычно `http://localhost:8501`). Рабочая директория должна быть корнем проекта, чтобы импорты `algorithms`, `consensus`, `data_generator`, `evaluation` разрешались.

При первом обращении к датасетам через OpenML данные кэшируются (каталог по умолчанию задаётся в коде `data_generator`).

## AI-ассистент (опционально)

В разделах «Сравнение алгоритмов» и «Консенсус-анализ» доступен встроенный чат-агент, который видит текущие данные (датасет, метрики, параметры) и отвечает на вопросы про результаты, теорию алгоритмов или диагностику проблем (высокий шум, k=1 и т.п.).

Работает локально через [Ollama](https://ollama.com) — без отправки данных в облако и без API-ключей.

### Установка Ollama

macOS (без Homebrew):

```bash
curl -L -o ~/Downloads/Ollama.dmg https://ollama.com/download/Ollama.dmg
hdiutil attach ~/Downloads/Ollama.dmg
cp -R "/Volumes/Ollama/Ollama.app" /Applications/
hdiutil detach "/Volumes/Ollama"
open /Applications/Ollama.app
```

Если приложение требует macOS 14+ — установить CLI-бинарник:

```bash
curl -L -o ~/bin/ollama https://github.com/ollama/ollama/releases/download/v0.3.14/ollama-darwin
chmod +x ~/bin/ollama
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Запуск сервера и модели

В отдельном окне терминала (его не закрывать):

```bash
ollama serve
```

Скачать модель (один раз):

```bash
ollama pull qwen2.5:7b   # рекомендуется (~4.7 ГБ, хороший русский)
# или
ollama pull qwen2.5:3b   # быстрее на CPU (~2 ГБ)
ollama pull llama3.2     # самая лёгкая (~2 ГБ), хуже на русском
```

Затем в боковой панели приложения в разделе **AI-агент (Ollama)** указать модель и хост (по умолчанию `http://localhost:11434`).

Первый запрос занимает 1–2 минуты — модель прогружается в RAM. Последующие — 10–30 секунд на CPU.

## Запуск тестов

С установленным pytest:

```bash
python3 -m pytest tests/ -v
```

Без pytest (сокращённый прогон):

```bash
python3 tests/run_tests.py
```

## Структура каталогов

| Каталог | Назначение |
|---------|------------|
| `algorithms/` | Обёртки алгоритмов кластеризации и реестр имён |
| `consensus/` | Методы консенсуса и `ConsensusRunner` |
| `data_generator/` | Загрузка и генерация датасетов |
| `evaluation/` | Метрики, бенчмарк, загрузка пользовательских `.py` |
| `visualization/` | Графики для UI и скриптов |
| `webapp/` | `app.py` — интерфейс Streamlit |
| `tests/` | Модульные тесты |
| `experiments/` | Ноутбуки и `paper_batch.py` |
