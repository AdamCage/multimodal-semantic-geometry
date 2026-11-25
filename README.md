# Multimodal Semantic Geometry

Исследование сравнивает связи между понятиями в пространствах текста, изображений, экологического звука и видео без звука. Исходная задача также предусматривает проверку того, предсказывает ли эта геометрия качество выравнивания и необходимое число пар.

Результаты разведочного пилота, исходная постановка и границы проверки изложены в [RESEARCH.md](RESEARCH.md). Повторная проверка с фиксированным провайдером прошла для обеих текстовых семей. Для четырёхмодального теста и оценки бюджета пар пока отсутствуют подтверждённые маршруты независимо обученных энкодеров изображения, звука и видео. Пилот измеряет доступные текстовые и визуальные представления, описания отдельных каналов и фиксированные признаки, с отдельной проверкой аудио–текст на Clotho.

## Материалы

- [RESEARCH.md](RESEARCH.md) содержит выводы, исходные гипотезы и ссылки на первоисточники.
- [Протокол](study/protocol.json) задаёт пары моделей, перестановки, критерии включения и условие перехода к выравниванию.
- [Нативный протокол](study/native_protocol.json) задаёт две отдельные семьи на модальность и разбиения для последующей проверки.
- [Пакет данных](evidence/README.md) позволяет перепроверить результаты без API.
- [Журнал EpistemeOS](evidence/episteme/summary.json) показывает принятые решения, вычислительные проверки и открытые возражения.

## Повторный запуск

Подготовьте Python 3.13, FFmpeg и FFprobe. Для онлайн-запуска передайте ключ только через переменную среды `ROUTERAI_API_KEY`. Команды для PowerShell из корня проекта приведены ниже.

```powershell
uv venv --python 3.13
uv pip install -r requirements.txt
uv pip install "epistemeos @ git+https://github.com/AdamCage/EpistemeOS.git@7f5a1a0e43d7c35196cc8b3c9c74a618aa2559c3"
$studyPython = ".venv/Scripts/python.exe"
& $studyPython -m study.preflight
& $studyPython -m study.research_roles feasibility study/protocol.json
& $studyPython -m study.workflow audit --preflight .research/transport-audit/observed-transport.json
& $studyPython -m study.acquire
& $studyPython -m study.roster
& $studyPython -m study.workflow freeze
& $studyPython -m study.workflow run
& $studyPython -m study.clotho
& $studyPython -m study.review_packet results .research/roles/results-packet.json
& $studyPython -m study.research_roles results-review .research/roles/results-packet.json
& $studyPython -m study.workflow followup
& $studyPython -m study.workflow adapt
& $studyPython -m study.review_packet final .research/roles/final-packet.json
& $studyPython -m study.research_roles adaptive-closure .research/roles/final-packet.json
& $studyPython -m study.provider_audit
& $studyPython -m study.workflow postflight
```

В этой последовательности EpistemeOS использует отдельный рабочий каталог `.research/episteme`. Для повторения опубликованных расчётов по зафиксированным входам достаточно команд из [evidence/README.md](evidence/README.md).

## Нативный эксперимент

Команды проверяют RouterAI, готовят записи Commons и материализуют независимые окна EPIC-SOUNDS с человеческими описаниями из EPIC-KITCHENS. Для подготовки файлов нужно несколько минут и доступ к исходным записям Университета Бристоля.

```powershell
$nativeDir = ".research/native-reopen"
$replicationDir = ".research/replication-lab"
& $studyPython -m study.native_routes --root "$nativeDir/new-live"
& $studyPython -m study.native_provider_control --root "$nativeDir/new-provider-control"
& $studyPython -m study.native_discovery --output "$nativeDir/discovery"
& $studyPython -m study.native_data_audit --output "$nativeDir/avcaps.json"
& $studyPython -m study.epic_replication --root $replicationDir --acquire
& $studyPython -m study.replication_check --discovery "$nativeDir/discovery/discovery-candidate.json" --replication "$replicationDir/epic-candidate.json"
```

В общем манифесте указываются роли `discovery` и `replication`, происхождение, лицензия и оценка возможного пересечения с предобучением. Для источника фиксируются авторство, идентификаторы зависимых записей, объём материала, к которому относится SHA-256 записи, и относительный путь к каждому входу вместе с контрольной суммой. Связанные записи целиком остаются в одной части разбиения. Отдельный аудит показывает разрешение таксономических перестановок в текущих выборках. [Commons](evidence/native-reopen/discovery-candidate.json) · [EPIC-SOUNDS](evidence/replication/epic-candidate.json) · [Аудит разбиения](evidence/replication/replication-check.json).

```powershell
& $studyPython -m study.native_measure freeze --root "$nativeDir/measure" --manifest ".research/native-combined-candidate.json" --routes "$nativeDir/new-live/native-route-audit.json"
& $studyPython -m study.native_measure extract --root "$nativeDir/measure" --manifest ".research/native-combined-candidate.json"
& $studyPython -m study.native_measure analyze --root "$nativeDir/measure"
```

Заморозка проверяет допуск маршрутов до чтения медиа. Числа нативного расчёта появляются после извлечения четырёх типов признаков через RouterAI. Проверка бюджета пар зависит от результата независимой репликации.

### Расширенные выборки

Дополнительный поиск Commons и пересборка выборки EPIC используют отдельные каталоги. При нехватке подтверждённых голосов животных дополнительный проход сначала оценивает метаданные, затем скачивает подходящие медиа. Расширенное разбиение подбирает понятия и группы связанных источников совместно.

```powershell
$expandedCommons = ".research/commons-expansion"
$expandedEpic = ".research/replication-expanded"
$expandedDiscovery = ".research/native-discovery-expanded/supplemented"
& $studyPython -m study.commons_expansion --root $expandedCommons
& $studyPython -m study.acquire --root $expandedCommons --target 10 --extension study/commons_extension.json --reuse .research/commons-v3
& $studyPython -m study.roster --root $expandedCommons --screen-only
& $studyPython -m study.commons_expansion --root $expandedCommons --supplement-animals
& $studyPython -m study.native_discovery_expanded --source $expandedCommons --output $expandedDiscovery --augmented
& $studyPython -m study.epic_replication --root $expandedEpic --expanded --acquire
& $studyPython -m study.replication_check --discovery "$expandedDiscovery/discovery-candidate.json" --replication "$expandedEpic/epic-candidate.json" --manifest-output .research/native-expanded-combined-candidate.json --report-output "$expandedEpic/full-expanded-check.json"
& $studyPython -m study.native_smoke --manifest .research/native-expanded-combined-candidate.json --output "$expandedEpic/full-expanded-synthetic.json"
```

Проверка `native_smoke` использует синтетические векторы при фактическом разбиении источников. Такой прогон проверяет расчёты без запросов к энкодерам.
