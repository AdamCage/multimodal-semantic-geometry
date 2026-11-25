# Данные и воспроизведение

Используйте [отчёт исследования](../RESEARCH.md) для интерпретации чисел. Эта папка хранит замороженные массивы признаков, манифесты, результаты расчётов и журнал решений.

| Путь | Содержание |
| --- | --- |
| `commons/roster.json` | Источники, лицензии, начало четырёхсекундного отрезка, хэши входов и группы разбиения |
| `commons/manifest.json`, `commons/selection-audit.json` | Данные до проверки меток и исход каждой обработанной записи |
| `commons/features.npz`, `commons/feature-meta.json` | Векторы в порядке `source_ids`, модели из ответов API и пропуски по представлениям |
| `commons/geometry.json`, `commons/reanalysis.json` | Основной расчёт и независимая реализация общего RSA на тех же входах |
| `commons/observability.json`, `commons/robustness.json` | Канальный наблюдатель, интервенции и проверки чувствительности |
| `commons/metadata-control.json`, `commons/caption-partial.json` | Контроль контейнера и повторный расчёт после учёта геометрии описаний кадров |
| `commons/provider-audit.json` | Идентификаторы BGE из ответов API на уровне источников и анализ при ограничении маршрута |
| `clotho/` | Отдельная выборка Clotho с человеческими описаниями, признаками и журналом технического восстановления |
| `routerai/` | Наблюдавшийся каталог, пробы форматов и сводка локально записанных API-запросов |
| `roles/` | Ответы агентных ролей, включая прерванные ответы и возражения |
| `episteme/` | Экспорт текущего состояния EpistemeOS, цепочки событий и графа зависимостей. Полный локальный Store находится в `.research/episteme` |
| `native-reopen/` | Повторный допуск восьми семейств, проверка источников для нативного эксперимента, синтетический тест вычислений и отдельный журнал EpistemeOS |
| `replication/` | Выбор провайдера, повтор пилота Commons, происхождение записей AVCaps, подбор EPIC-SOUNDS и аудит точности перестановок |
| `SHA256SUMS` | Контрольные суммы файлов этой папки |

У источников Wikimedia Commons и Freesound действуют собственные лицензии. URL, авторство и версия указаны в ростерах. В пакет входят признаки и описания, а медиафайлы хранятся в локальном рабочем каталоге.

## Проверка без API

В окружении Python 3.13 с зависимостями из [requirements.txt](../requirements.txt) выполните из корня проекта следующие команды.

```powershell
python -m study.bundle --verify
python -m unittest discover -s tests -v
New-Item -ItemType Directory -Path .research -Force | Out-Null
$replayRoot = Join-Path .research ("replay-" + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath evidence -Destination $replayRoot -Recurse
python -m study.geometry --root "$replayRoot/commons"
python -m study.check_geometry --root "$replayRoot/commons"
python -m study.robustness --root "$replayRoot/commons"
python -m study.caption_partial --root "$replayRoot/commons"
python -m study.clotho --root "$replayRoot/clotho" --offline
```

Для исходной проверки векторов через API потребуются FFmpeg, учётные данные RouterAI в переменной `ROUTERAI_API_KEY` и доступ к источникам. Текущая выдача Wikimedia Commons, набор API-моделей и их ответы могут меняться. Подробности повторного запуска приведены в [README проекта](../README.md).

В `native-reopen/discovery-candidate.json` и `replication/epic-candidate.json` указаны пути к локально подготовленным входам. Первые воспроизводятся командой `study.native_discovery`, вторые командой `study.epic_replication`. В пакет входят метаданные выбранных окон EPIC-SOUNDS, время фрагментов, идентификаторы источников, наблюдаемые контрольные суммы и текстовые описания. Локальные фрагменты WAV, JPEG и MP4 подрезаны и перекодированы из некоммерческих материалов Университета Бристоля. Авторы разметки событий Jaesung Huh, Jacob Chalk, Evangelos Kazakos, Dima Damen и Andrew Zisserman. Источник видеозаписей и человеческих описаний действий — команда EPIC-KITCHENS. [EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/site) · [EPIC-KITCHENS 2018](https://doi.org/10.5523/bris.3h91syskeag572hl6tvuovwv4d) · [EPIC-KITCHENS-100](https://doi.org/10.5523/bris.2g1n6qdydwa9u22shpxqzp0t8m) · [Условия для описаний](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/license.txt).
