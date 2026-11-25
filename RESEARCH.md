# Multimodal Semantic Geometry

## Экспериментальный отчёт

**Дата исследования** 25 ноября 2026 года. **Тип исследования** разведочный пилот с замороженным планом анализа и отдельной проверкой аудио–текст на Clotho.

Исходный вопрос касается геометрии **независимо обученных** текстовых, визуальных, звуковых и видеоэнкодеров. Через разрешённый API подтвердились две подходящие текстовые семьи с фиксированным провайдером. Доступные энкодеры изображений обучались с межмодальным выравниванием, а проверенные маршруты эмбеддингов WAV и MP4 не прошли проверку доставки входа. Исходный четырёхмодальный тест и зависящий от него эксперимент с бюджетами пар остаются недоступны. Выводы пилота относятся к доступным представлениям в выборках Commons и Clotho. [Снимок каталога и запросы](evidence/routerai/preflight.json) · [Проверка нативных маршрутов](evidence/replication/native-route-audit-pinned.json).

### Проверка RouterAI

| Канал | Что реально измерялось | Статус для исходного теста |
| --- | --- | --- |
| Текст | BGE-M3 и Qwen3-Embedding-8B на описаниях источников, Gemini Embedding 2 для совместно обученного сравнения | Две независимо обученные текстовые семьи доступны |
| Изображение | Векторы отдельных кадров из Titan, Voyage Multimodal 3.5 и Gemini Embedding 2 | Прямые визуальные входы доступны у моделей с совместным обучением на разных модальностях |
| Экологический звук | Описания Gemini 2.5 Flash с последующим BGE или Qwen3, а также фиксированные признаки аудиосигнала | BEATs и AudioMAE отсутствовали в снимке каталога |
| Видео без звука | Описания Gemini 2.5 Flash с последующим BGE или Qwen3, а также усреднённые яркости и изменения кадров | VideoMAE и V-JEPA 2 отсутствовали в снимке каталога |

Авторизованный снимок содержал 613 идентификаторов. Тестовые WAV и MP4 в запросах к обоим Gemini Embedding 2 получили HTTP 400. Два WAV, один MP4 и контрольный текст дали один и тот же SHA-256 вектора и одинаковый учёт трёх входных токенов, когда медиа прикладывалось к тексту. Два различных тестовых изображения дали разные векторы. Форматы inline и URL для Voyage 3.5 также получили HTTP 400 на проверенных маршрутах RouterAI. У Voyage заявлена поддержка видео через собственный интерфейс. Ещё шесть запросов с идентификаторами кандидатов DINOv2, MAE, BEATs, AudioMAE, VideoMAE и V-JEPA 2 завершились HTTP 400. Ограничение зафиксировано для наблюдавшегося состояния RouterAI и проверенных форматов. [RouterAI](https://routerai.ru/models/google/gemini-embedding-2) · [Voyage](https://docs.voyageai.com/reference/multimodal-embeddings-api) · [Повторные пробы](evidence/routerai/repeat-transport.json).

### Данные и расчёт

- **Wikimedia Commons.** Для 24 плановых событий выполнили зафиксированный поиск по четырём группам. Поисковые результаты ограничили файлами с указанным авторством и лицензией CC BY, CC BY-SA, CC0 либо Public Domain. Из каждой записи брали средние четыре секунды, отделяли WAV, удаляли звуковую дорожку MP4 и извлекали средний кадр. Отдельная модель проверяла соответствие целевому событию по исходным метаданным. Эта проверка оценивает запись целиком, поэтому видимость или слышимость события в среднем отрезке измерялась отдельно. [Манифест](evidence/commons/manifest.json) · [Аудит](evidence/commons/selection-audit.json) · [Исходные правила поиска](study/concepts.json).
- **Разбиение.** Нормализованные авторы, загрузившие пользователи, точные совпадения WAV или MP4 и отмеченные аудиовизуальные дубликаты образуют компоненты. Компоненты целиком назначены либо в A, либо в B. Каждому использованному понятию требовались минимум две полные записи в каждой половине. У пар есть собственные маски пропусков. Внутри пары число записей сбалансировано по понятию и половине. [Утверждённый ростер](evidence/commons/roster.json) · [Покрытие и ошибки](evidence/commons/feature-meta.json).
- **Геометрия доступных представлений.** Расстояние между понятиями равно единице минус косинус между центроидами нормированных векторов. Обучаемых проекций в этом расчёте нет. Сравнение на тех же источниках усредняет RSA для A–A и B–B, на разных источниках для A–B и B–A. Каждая перестановка совместно меняет строки и столбцы целевой матрицы в обеих половинах. Использованы глобальная перестановка, перестановка в пределах групп и отдельный анализ расстояний внутри групп. После 4 999 перестановок учитывается одна семья из 126 сравнений по Холму. [Протокол](study/protocol.json).
- **Дополнительные метрики.** Локальные соседи измерялись при `k = 1, 3, 5`. Проверялись порядок троек, CKA относительно перестановочного фона и численная ошибка расстояний. Для последней масштаб подбирался на одних понятиях и оценивался на других. Подбор масштаба повторялся в перестановках. Интервалы показывают чувствительность к весам компонентов авторов на фиксированном наборе понятий. Для изменения набора понятий отдельно выполнено исключение каждой крупной группы. [Полные метрики](evidence/commons/geometry.json).
- **Проверка наблюдаемости.** Отдельная сессия Gemini 3.5 Flash получала один канал за вызов, без названия файла, целевой метки и результатов геометрии. Наблюдателю предлагался список событий из соответствующей группы с вариантом «другое или неясно». Для аудио менялись фазы при сохранении амплитудного спектра, для видео менялся порядок одного и того же набора кадров. Оценка относится к этому наблюдателю и включает подсказку в виде списка вариантов. [Наблюдения и контроль сигналов](evidence/commons/observability.json).

### Геометрия на Wikimedia Commons

В обработанный список попали 258 записей. В замороженный ростер вошли **120 записей, 19 понятий, 4 группы**. В A и B по 60 записей. Пять понятий выбыли из-за недостатка подтверждённых исходных записей для двух половин. У каждого итогового понятия от четырёх до восьми записей. Все 21 сравнение сохранили 19 понятий. Внутригрупповой расчёт охватывает инструменты, транспорт и механические или природные события. Для голосов животных в ростере осталось два понятия, чего недостаточно для внутригруппового RSA. [Ростер](evidence/commons/roster.json) · [Результаты по всем парам](evidence/commons/geometry.json).

В сравнении на разных источниках **14 из 21** пар превысили глобальный перестановочный фон при скорректированном p < 0,05. Для перестановки с сохранением групп таких пар было **5 из 21**. При расчёте только на отношениях внутри групп их было **7 из 21**. Эти два таксономических теста используют разные статистики. Первый сохраняет крупные группы при перестановке полной матрицы. Второй отдельно ранжирует расстояния внутри каждой группы.

| Представления | RSA, те же источники | RSA, разные источники | RSA внутри групп, разные источники | p Холма, перестановка в группах |
| --- | ---: | ---: | ---: | ---: |
| BGE и Qwen3 на текстах источников | 0,677 | 0,487 | 0,533 | 0,080 |
| BGE на тексте и Titan на кадре | 0,522 | 0,592 | 0,615 | 0,025 |
| Qwen3 на тексте и Voyage на кадре | 0,496 | 0,503 | 0,604 | 0,025 |
| Gemini Embedding 2 на тексте и кадре | 0,501 | 0,430 | 0,416 | 0,047 |
| BGE на описаниях кадра и видео из Gemini | 0,877 | 0,699 | 0,605 | 0,025 |
| BGE на описаниях звука и видео из Gemini | 0,203 | 0,320 | 0,218 | 1,000 |
| Фиксированные признаки звука и движения | 0,285 | 0,044 | 0,152 | 1,000 |

У BGE × Titan и Qwen3 × Voyage дизельные, электрические и паровые поезда стабильно входили в тройку ближайших соседей друг друга. То же касалось винтового самолёта и реактивного самолёта. Эти связи встречались у обеих моделей и в обеих половинах. Среднее перекрытие трёх ближайших соседей для этих пар составило 0,500 и 0,544 при глобальном перестановочном фоне 0,167. Пары включают визуальные модели с выравниванием по тексту. Для BGE × Titan и Qwen3 × Voyage исключение инструментов снижало общий RSA до 0,480 и 0,305 соответственно. Здесь устойчивость зависит от состава понятий. [Таблицы соседей и исключения групп](evidence/commons/geometry.json).

Порядковое согласие и численная согласованность расстояний расходились. На разных источниках масштабирование обошло константный прогноз ошибки у 16 из 21 пар. Для BGE × Titan относительная ошибка составила 0,135 против 0,216 у константы, а для единого Gemini на тексте и кадре 0,231 против 0,195. Для описаний аудио и видео в BGE ошибка достигла 0,455 против 0,223. У BGE × Titan CKA составил 0,932 при медиане перестановочного фона 0,749. В сравнении условий медиана разности RSA «те же источники минус разные» по 21 паре составила −0,015. Разность заметно варьирует по парам. [График](evidence/commons/geometry.png) · [Расстояния, CKA и тройки](evidence/commons/geometry.json).

### Наблюдаемость и устойчивость

По одному источнику из каждой половины для каждого понятия дали 38 записей. Наблюдатель выбрал целевое событие в 19 из 38 звуковых отрезков, в 17 из 37 кадров и в 24 из 38 видео. Один ответ по кадру оборвался. Наблюдатель отметил цвет у 31 из 37 кадров, движение у 33 из 38 видео и различимый источник звука в 32 из 38 аудиофрагментов. Связь между совместным распознаванием и согласием в порядке расстояний от понятия до остальных составила ρ = 0,277 для описаний звука и видео и 0,230 для Titan и описаний видео. На понятие приходилось не более двух наблюдений, поэтому это описательные оценки. [Оценки по каналам и понятиям](evidence/commons/observability.json).

| Изменение входа | Целевое событие до | Целевое событие после | Необработанное p по компонентам авторов |
| --- | ---: | ---: | ---: |
| Случайные фазы в аудио | 20/38 | 6/38 | 0,0013 |
| Перестановка четырёх кадров видео | 21/38 | 21/38 | 1,000 |

Для аудио максимальное относительное отличие амплитудного спектра составило 0,163%. При перестановке видео декодированные кадры сохранились побайтно, а доля ответов с различимым порядком событий изменилась с 5/38 до 4/38. В этой выборке узнаваемость экологического звука зависела от информации, нарушенной фазовой рандомизацией. Грубая метка видео часто оставалась распознаваемой по содержанию кадров после изменения порядка. В обоих случаях измерялось поведение конкретного наблюдателя. [Интервенции и проверки сигналов](evidence/commons/observability.json).

**Чувствительность к отбору.** Из 120 записей у 88 метаданные получили оценку «явное соответствие». После ограничения этим уровнем и повторной балансировки осталось 12 понятий. Для BGE × Titan общий RSA на разных источниках снизился с 0,592 до 0,450. У аудиосигнала и признаков движения он изменился с 0,044 до −0,088. Фильтр по речи оставил 111 исходных записей до балансировки и 16 понятий после неё. Для описаний звука и видео в BGE этот фильтр снизил ρ с 0,320 до 0,250. Состав понятий менялся вместе с фильтрами. [Проверки отбора](evidence/commons/robustness.json).

**Языковой и таксономический контекст.** Для описаний кадра и видео совпадения слов дали ρ = 0,464 при представлении текста бинарным вектором слов. Для эмбеддингов BGE на тех же каналах ρ = 0,699. После основного расчёта для пяти пар с положительным результатом при сохранении широких групп применили более узкие подгруппы, например рельсовый и воздушный транспорт. Нижние необработанные перестановочные p были 0,008 у пары описаний кадра и видео и 0,010 у BGE × Titan. Этот расчёт использует пары, отобранные по итогам пилота, и служит описательной проверкой. [Словарные признаки и вложенная таксономия](evidence/commons/robustness.json).

**Служебные данные медиа.** Проверка FFprobe обнаружила два MP4 с дополнительным потоком данных. Во всех 120 MP4 отсутствовал аудиопоток. Десять входов с тегами или дополнительными потоками повторно обработали после удаления этих данных и проверки побайтового равенства декодированного сигнала. Изменились два текста описаний. Наибольшая абсолютная разница общего RSA на разных источниках после подстановки новых описаний составила 0,0016. Проверка относится к тегам контейнера. Видимый текст в кадрах остаётся частью визуального входа. [Контроль контейнера](evidence/commons/metadata-control.json).

**Проверка после критики.** Для трёх прямых сравнений текста и изображения пересчитали RSA после учёта типа пары категорий и геометрии описаний кадров. У BGE × Titan оценка снизилась с 0,592 до 0,323, у Qwen3 × Voyage с 0,503 до 0,140, у единого Gemini с 0,430 до 0,131. Признаки описаний и категории учитывались заново в той же схеме контролируемых перестановок. Остаточные корреляции зависят от выбранных контрольных признаков. Расчёт выполнен на том же корпусе после просмотра основных результатов и служит описательной проверкой. [Частный RSA](evidence/commons/caption-partial.json).

**Маршрутизация BGE.** При одном запрошенном идентификаторе ответа API были помечены как `BAAI/bge-m3` и `parasail-bge-m3`. Для текстов Commons это 24 и 96 векторов. Точная версия весов по этим строкам ответа не устанавливается. На подвыборке из 14 понятий, где для BGE × Titan использован только `parasail-bge-m3`, RSA составил 0,531. Для сравнения описаний кадра и видео одна метка BGE давала недостаточное покрытие обеих половин. Дополнительное сравнение тех же описаний через Qwen3 с одним наблюдавшимся идентификатором дало ρ = 0,705 на 19 понятиях. [Аудит маршрутов и повторный расчёт](evidence/commons/provider-audit.json).

### Отдельная проверка на Clotho

На тестовой части Clotho v2.1 зафиксировали 64 записи с лицензиями CC0, CC BY или CC BY-SA, по одной на автора Freesound. Для каждого звука сравнивались пять человеческих описаний, описание Gemini и акустические признаки. Единица анализа здесь — запись. Идентичность строк и всех текстов сверена с официальной таблицей. Скачивание использовало зеркало конкретной ревизии, а байты входов закреплены хэшами. [Описание набора](https://zenodo.org/records/4783391) · [Статья Clotho](https://doi.org/10.1109/ICASSP40776.2020.9052990) · [Ростер](evidence/clotho/clotho-roster.json).

| Отношения между записями | Spearman ρ | Перестановочное p после Холма |
| --- | ---: | ---: |
| Человеческие описания BGE × описание Gemini → BGE | 0,103 | 0,066 |
| Человеческие описания BGE × описание Gemini → Qwen3 | 0,183 | 0,008 |
| Человеческие описания BGE × фиксированные акустические признаки | 0,141 | 0,037 |
| Первые три против последних двух человеческих описаний в BGE | 0,493 | 0,002 |

Первый проход охватил 41 из 64 записей из-за ошибок загрузки. Повторная загрузка восстановила те же недостающие записи по исходному списку. Оценка для признаков аудиосигнала изменилась с 0,045 до 0,141, а скорректированное p с 0,298 до 0,037. Это показывает чувствительность вывода к покрытию источников. Для 33 записей с единым идентификатором `BAAI/bge-m3` у всех пяти человеческих описаний оценки для описаний Gemini → Qwen3 и акустических признаков составили 0,166 и 0,246 соответственно. Дополнительный описательный контроль BGE × Qwen3 на одних и тех же человеческих описаниях дал ρ = 0,876. Clotho проверяет аудио–текст на другом корпусе и на уровне записей. Полная четырёхмодальная репликация потребует другого дизайна и доступных энкодеров. [Первый проход](evidence/clotho/first-pass-geometry.json) · [Полный расчёт](evidence/clotho/clotho-geometry.json) · [Аудит маршрутов](evidence/commons/provider-audit.json).

### Воспроизведение

В протоколе и артефактах сохранены входы, решения, размеры выборок, ошибки, модели из ответов сервиса и хэши их результатов. Без API можно повторить статистику по сохранённым массивам и независимо пересчитать общий RSA через SciPy. Повторный онлайн-запуск зависит от текущей выдачи Wikimedia Commons, доступности маршрутов и версий удалённых моделей. [Пакет данных и команды](evidence/README.md).

### Как менялся план

1. **Брейншторм и методическая критика.** Роли RouterAI получили исходную постановку в отдельных запросах. После проверки каталога добавили различение доступных прямых входов, описаний и фиксированных признаков. Структуру поиска Commons изменили после нулевой выдачи на составные запросы. От предложенных универсальных порогов величины эффекта отказались. Число перестановок подняли до 4 999 до расчёта, чтобы минимально возможное скорректированное p могло пройти уровень 0,05 при 126 тестах. [Первая критика](evidence/roles/initial.json) · [Проверка реализуемости](evidence/roles/feasibility.json).
2. **Заморозка и вычисления.** EpistemeOS закрепил конкурирующие объяснения, ростер, конфигурацию и исходники расчёта до исполнения. Другой расчёт на SciPy воспроизвёл все 42 значения общего RSA с максимальным отличием 5,6 × 10⁻¹⁶. Это проверка вычисления на зафиксированных векторах. Отдельный проход восстановил пропущенные записи Clotho по исходному ростеру. [Пересчёт](evidence/commons/reanalysis.json) · [Журнал](evidence/episteme/events.json).
3. **Критика результата и доработка.** Три самостоятельных запроса RouterAI проверили выводы. После критики EpistemeOS запустил канальный контроль, более узкую таксономию, повтор после очистки контейнеров и дополнительный расчёт с учётом геометрии описаний. Числа в отчёте взяты из вычислительных результатов. Некоторые замечания моделей смешивали оценки для тех же и для разных источников. [Замечания](evidence/roles/results-review.json) · [Итог петли](evidence/episteme/summary.json).
4. **Закрытие вычислительных стадий.** Два рецензирующих запроса предложили учесть геометрию описаний, и EpistemeOS выбрал четвёртую стадию. Она завершилась на том же наборе данных. После остановки по лимиту стадий в журнал внесли заключительную критику и аудит ответов BGE. Все агентные вызовы прошли через RouterAI, с отдельными входными контекстами. Научный вердикт требует отдельной проверки. [Заключительная критика](evidence/roles/adaptive-closure.json) · [Журнал](evidence/episteme/summary.json) · [Граф зависимостей](evidence/episteme/graph.json).

### Статус исходных гипотез

| Проверка | Что позволяет утверждать пилот |
| --- | --- |
| Порядковое согласие | Часть контекстных пар сохраняет порядок расстояний на новых записях в пределах набора Commons. Результат зависит от крупной и более мелкой таксономии |
| Локальное согласие | У нескольких пар воспроизводятся отдельные соседи внутри транспорта и инструментов |
| Численное согласие | Величина ошибки после фиксированного масштабирования меняется между парами. Сходный порядок расстояний допускает большой остаток численной ошибки |
| Информация в каналах | Поддержка целевых событий и чувствительность к изменению входов измерены у отдельного модельного наблюдателя |
| Геометрия независимо обученных четырёх модальностей | Требуются подходящие визуальные, звуковые и видеоэнкодеры в разрешённом API |
| Предсказание качества и бюджета пар | Условие перехода требует воспроизводимой нативной геометрии и проверки на новых семьях и корпусах. Эксперименты с головами и бюджетами 16, 64, 256 и 1 024 пар остаются открытыми |

Основные ограничения пилота связаны с 19 отобранными понятиями, слабыми метками Commons, общим семейством моделей для описаний и распознавания, неизвестным пересечением с данными обучения и ограниченным числом источников на понятие. Масштаб 100 × 30 из постановки был предварительной целью. Роль научного рецензента в журнале остаётся открытой. Механический допуск EpistemeOS относится к целостности артефактов и повтору вычислений. [Ограничения EpistemeOS](https://github.com/AdamCage/EpistemeOS/tree/7f5a1a0e43d7c35196cc8b3c9c74a618aa2559c3) · [Статус](evidence/episteme/summary.json).

### Повторная проверка исходного эксперимента

В авторизованном каталоге RouterAI проверены восемь зарегистрированных семейств. Запросы к спискам точек доступа всех шести семейств изображения, звука и видео получили HTTP 404. Объявленные аудио- и видеовходы среди эмбеддингов принадлежат двум вариантам совместно обученного Gemini Embedding 2. В первоначальном повторе BGE-M3 менял идентификатор между DeepInfra и Parasail. Явный выбор провайдера в теле запроса дал четыре ответа от указанного провайдера для каждой из пяти проверенных комбинаций модели и провайдера. Во всех восьми запросах с заведомо неподдерживаемым для этих двух моделей провайдером API вернул HTTP 404. Новый протокол фиксирует DeepInfra для BGE-M3 и Qwen3-Embedding-8B. Повторные запросы подтвердили для обоих стабильные идентификаторы, размерность и чувствительность к двум разным текстам. [Ранний снимок](evidence/native-reopen/native-route-audit.json) · [Контроль выбора провайдера](evidence/replication/provider-pin.json) · [Новый допуск](evidence/replication/native-route-audit-pinned.json) · [Документация RouterAI](https://routerai.ru/docs/guides/overview/provider-selection).

На тех же 120 записях Commons заново получены текстовые векторы BGE-M3 и Qwen3-Embedding-8B и векторы BGE-M3 для трёх наборов описаний каналов. Во всех пяти наборах сохранены 120 входов и один идентификатор модели для каждого набора. После повторной поправки Холма p < 0,05 по-прежнему наблюдалось у 14/21 пар при глобальной перестановке, у 5/21 при сохранении групп и у 7/21 для внутригрупповых расстояний. Наблюдаемые изменения глобального RSA для затронутых пар не превышали 0,006. Это проверка чувствительности пилота на прежних источниках. [Повторный расчёт с фиксированным провайдером](evidence/replication/pinned-text-sensitivity.json).

Три роли RouterAI отдельно проверили транспорт, независимость разбиений и условия запуска эксперимента с бюджетом пар. В нативном протоколе зафиксированы две разные семьи на модальность. До извлечения векторов программа проверяет исходные форматы и SHA-256, затем удерживает связанные записи в одних частях. Геометрия рассчитывается отдельно по корпусам, понятиям и семействам. Масштаб расстояний подбирается по понятиям для разработки. Затем ошибка измеряется на отложенных понятиях. Запросы нативных энкодеров звука и видео передают типизированные WAV и MP4. На контролируемых синтетических данных рассчитаны 64 сравнения и проверено, что у двух калибровочных видео первые шесть декодированных кадров совпадают, а последующее изображение различается. [Протокол](study/native_protocol.json) · [Отзывы](evidence/native-reopen/method-critique.json) · [Проверка вычислений](evidence/native-reopen/mechanical-verification.json).

Для разработки подготовлены четыре входа каждой из 120 записей Commons. По связям авторов и загрузивших их пользователей 19 понятий разделены на 8 для разработки и 11 отложенных, в каждом наборе сохранены исходные половины A и B. В половине A понятий для разработки осталось 26 уникальных записей. Даже до отбора по понятиям верхняя граница на этой половине составляла 60 уникальных пар одного события на каждую пару модальностей. Одного этого корпуса недостаточно для уровней 64, 256 и 1 024 пар при независимой проверке. Записи Commons ранее использовались в разведочном пилоте. [Манифест и счётчики Commons](evidence/native-reopen/discovery-summary.json) · [Расчёт границы по числу источников](evidence/native-reopen/readiness.json).

В тестовой части AVCaps идентификаторы всех 200 записей совпали между индексом видеоархива, человеческими описаниями звука и изображения и исходной визуальной разметкой VidOR. По текущим метаданным Flickr у 101 записи из 91 авторской группы подтверждена исходная лицензия, разрешающая производные материалы в условиях этого исследования. У 34 записей условия не подходят для выбранной обработки, у 65 исходная страница ролика не подтверждена. Проверка метаданных оставляет открытыми совпадение байтов архива с оригиналом и наблюдаемость событий по каналам. [Аудит AVCaps](evidence/native-reopen/avcaps.json) · [Проверка источников Flickr](evidence/replication/avcaps-provenance.json) · [Архив AVCaps](https://zenodo.org/records/14536325) · [VidOR](https://xdshang.github.io/docs/vidor.html).

Для отдельной проверки подготовлены 64 трёхсекундных окна EPIC-SOUNDS из 64 исходных записей 26 участников. В выборке 16 классов событий. На каждое понятие приходится по две записи разных участников в каждой половине A и B. Участники и исходные записи целиком принадлежат одной из четырёх частей, образованных разбиением по понятиям и половинам A/B. Звуковая метка взята из человеческой разметки EPIC-SOUNDS, текст — из описания пересекающегося по времени действия EPIC-KITCHENS. Для всех 64 окон такое же описание получилось бы при выборе только по временному пересечению без фильтра по глаголу. В двух окнах уровень звука в пределах 100 мс около извлечённого кадра был ниже −60 dBFS. Визуальная наблюдаемость названного действия или материала в одном кадре требует отдельной оценки. Авторы EPIC-SOUNDS использовали визуальные данные для проверки материалов при столкновениях. [Подбор записей](evidence/replication/epic-selection.json) · [Проверки каналов и разбиения](evidence/replication/replication-check.json) · [EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/site) · [Разметка EPIC-KITCHENS](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

У 184 записей Commons и EPIC проверены форматы и контрольные суммы всех 736 входов. Разрешение точных таксономических перестановок ограничено размером групп. В EPIC возможны 16 перестановок в каждой части, поэтому минимальное точное p равно 0,0625. В двух частях Commons возможны 36 и 1 440 перестановок. Ни одна часть не достигает первого порога Холма 0,05/384 ≈ 0,000130 по этому нулю. Программа перебирает все варианты для малых пространств перестановок. Текущая выборка пригодна для проверки подготовки входов и оценки эффектов, а для запланированного подтверждающего теста нужна более широкая подборка понятий и независимых записей. [Расчёт разрешения](evidence/replication/replication-check.json) · [Критика дизайна](evidence/replication/independent-replication-review.json) · [Журнал EpistemeOS](evidence/replication/readiness.json).

После аудита разрешения выборка EPIC расширена до 112 записей 34 участников. В расширенной выборке действуют те же правила разбиения участников и исходных записей. В каждой части теперь по 14 понятий, распределённых по группам размера 5, 4, 3 и 2. Это даёт 34 560 точных таксономических перестановок на часть и минимальное p ≈ 0,000029. Форматы и контрольные суммы проверены для всех 448 входов расширенной выборки. Без фильтра по глаголу изменились бы описания четырёх окон. У двух окон звук около кадра остался ниже −60 dBFS. Выборки на 64 и 112 записей относятся к одному корпусу. Для увеличения выборки Commons проверены 56 запросов по понятиям, включая 32 новых. Для 44 понятий индекс показал как минимум четыре разных автора, загрузчика и страницы, для десяти — как минимум 30. Это предварительные потолки по метаданным до проверки звука, дублей, содержательного соответствия и разбиения. [Расширенный EPIC](evidence/replication/epic-expanded-check.json) · [Поиск Commons](evidence/replication/commons-inventory-extension.json).

В повторной рецензии остались два вопроса о данных. Нужно отдельно проверить наблюдаемость события и материала в выбранном кадре и оценить зависимость результатов от того, как в исходном корпусе проверяли метки столкновений. Предел точного p вычислен из числа перестановок. Мощность запланированного теста будет зависеть от разброса наблюдаемых эффектов и числа независимых участников. Поправка Холма допускает зависимые тесты при валидных индивидуальных p. Перебор всех 34 560 перестановок проверен на синтетических векторах при реальном разбиении EPIC. [Повторная рецензия](evidence/replication/expanded-design-review.json) · [Механическая проверка](evidence/replication/epic-expanded-synthetic-verification.json) · [Холм, 1979](https://www.jstor.org/stable/4615733).

Для Commons собрана 411 запись по 53 понятиям с прохождением порога уровня звука. После проверки метаданных только два понятия о голосах животных имели по четыре независимых группы источников. Дополнительная проверка метаданных и загрузка подходящих записей увеличили это число до пяти. Совместное разбиение по понятиям и связанным источникам оставило 170 записей Commons, по 14 понятий для разработки и отложенной проверки. Размеры групп 4, 2, 4 и 4 дают 27 648 точных перестановок в каждой части Commons. Для объединённого кандидата Commons и EPIC проверены 1 128 файлов, относящихся к 282 записям. Все четыре части достигают необходимого разрешения для таксономического теста. На синтетических векторах программа завершила все 64 сравнения с точными таксономическими перестановками. [Подбор Commons](evidence/replication/commons-expanded-summary.json) · [Совместная проверка](evidence/replication/full-expanded-check.json) · [Синтетический прогон](evidence/replication/full-expanded-synthetic.json).

Итоговый повтор запроса к RouterAI подтвердил обе текстовые семьи и HTTP 404 для всех шести зарегистрированных маршрутов изображения, звука и видео. С этой проверкой программа остановилась до извлечения векторов из расширенного манифеста. Нативная гипотеза и качество выравнивания остаются непроверенными. В половине разработки, предназначенной для обучения на парах, доступно 43 независимых записи Commons. По числу пар возможен уровень 16, а уровни 64, 256 и 1 024 требуют новых источников. Переход к этому эксперименту зависит от воспроизводимого нативного результата. [Итоговая проверка API](evidence/replication/native-route-audit-expanded.json) · [Проверка допуска](evidence/replication/full-expanded-gate-check.json) · [Запись в EpistemeOS](evidence/replication/full-expanded-readiness.json).

---

## Исходная постановка от 20 ноября 2026 года

**Status at the time of the proposal:** Research proposal / planning. No experiments or results yet.

**Last updated:** 2026-11-20.

This document describes a working research design, not a preregistration. Dataset selections, checkpoints, budgets, and statistical decisions remain provisional until the pilot protocol is frozen.

### 1. Research question

Do independently trained text, image, environmental-audio, and silent-video encoders preserve similar relationships between the same semantic concepts?

The primary objective is to measure **which relationships agree, at which semantic granularity, and under which information constraints**. The motivating example is whether the relationship between concepts such as cars, motorcycles, trains, and sand dunes is similar across representation spaces. Similarity of one distant pair is insufficient evidence; the study will compare structured sets of relationships.

A secondary objective is to investigate whether the measured geometry can predict the performance and paired-data requirements of a shared multimodal embedding system. This applied stage follows the geometry analysis; it does not replace it.

The project does not assume that all modalities contain the same information or that every concept has a recognizable realization in every modality.

### 2. Position relative to existing work

The Platonic Representation Hypothesis motivates convergence between independently learned representations [1]. Representational Similarity Analysis (RSA) provides an established way to compare spaces through relationships among matched conditions [2], while CKA offers a complementary representation-similarity measure [3].

Recent work makes stronger claims of universal convergence difficult to justify. Null-calibrated analyses distinguish robust neighborhood agreement from weaker distance agreement [4]. Dataset scale, semantic granularity, and many-to-many correspondences also affect observed alignment [5]. Indra representations already use relational profiles across vision, language, and audio [6].

GeoWarP connects shared geometry to unpaired and few-pair alignment [7]. Research on multimodal information structure shows that stronger alignment is not universally beneficial [8]. ImageBind and Omni-Embed-Nemotron already provide broad multimodal embedding capabilities [9, 10].

Consequently, adding modalities, computing distance correlations, or attaching projection heads is not sufficient novelty. Possible contributions must be established through a new reproducible finding, a more informative evaluation protocol, or a predictive method that improves a concrete decision beyond existing baselines.

### 3. Testable hypotheses

| Hypothesis | Question | Planned evidence |
|---|---|---|
| **Metric agreement** | Are numerical inter-concept dissimilarities similar after a prespecified scale treatment? | Held-out metric distortion relative to calibrated nulls and within-model reliability |
| **Ordinal agreement** | Do models agree about which concept pairs are closer? | RSA rank correlations and semantic triplet agreement |
| **Local agreement** | Do concepts have similar neighbors? | Neighborhood overlap across prespecified neighborhood sizes and dataset scales |
| **Information-conditioned agreement** | Does agreement depend on semantic factors observable in both channels? | Independent observability annotations and controlled removal or alteration of information |
| **Predictive utility** | Does initial geometry predict alignment quality or the required pair budget? | Prospective predictions evaluated on held-out concepts, datasets, and encoder families |

These hypotheses are distinct. Neighborhood agreement will not be reported as evidence of equal distances. A positive correlation will not establish a universal representation of reality. Supported null results and systematic failures are valid outcomes.

### 4. Data design

### Pilot scope

The provisional pilot targets **100 concepts with approximately 30 independent source recordings per concept**, yielding about 3,000 sources before exclusions. The initial scope emphasizes recognizable objects and events: barking dogs, musical instruments, moving trains, running engines, pouring water, bells, and applause.

Each usable source may yield audio, silent video, selected image frames, and independently written descriptions. A separate concept-matched condition will draw examples from different sources. The final sample size depends on source availability and annotation quality; this is not a claim that a complete dataset has already been assembled.

Two complementary conditions are required:

1. **Instance-paired:** channels describe the same recorded event. This supports correspondence at the event level.
2. **Concept-matched:** channels use independently sampled instances of the same concept. This tests whether agreement survives removal of shared recording context.

Multiple frames or excerpts from one recording do not count as independent sources. Related recordings and duplicates must remain within one split. Cross-source concept matching must preserve enough information to determine relevance without pretending that every example has one uniquely correct counterpart.

### Observability and semantic granularity

Annotators will record which factors are available in each channel: object, action, material, color, motion, temporal order, and sound source. They should judge a channel without being shown hidden information from the other channels.

Sand dunes paired with generic wind are an ambiguity control, not an ordinary fully observable concept. A spoken category name is a speech/language condition, not an independent environmental-sound observation. Similarly, a still image may identify a guitar without establishing that it is being played.

The dataset should include broad categories, finer distinctions within categories, and selected relational or temporal distinctions. Controlled conditions may remove audio, shuffle frame order, or restrict descriptions to visible or audible content. These interventions require explicit checks that the intended semantic factor changes while relevant alternatives remain controlled.

Candidate resources include VGGSound and AudioSet/AudioCaps for audiovisual material and Clotho for an audio–text replication. Availability, redistribution rights, training overlap, and actual observability will be checked before fixing a manifest. No single candidate is assumed to provide complete four-way supervision.

### 5. Encoders and extraction

The initial comparison will use two families per modality, subject to checkpoint and feasibility checks:

| Modality | Candidate families |
|---|---|
| Text | BGE-M3; Qwen3-Embedding |
| Image | Original DINOv2; pretrained MAE |
| Environmental audio | Pretrained BEATs; AudioMAE |
| Silent video | Pretrained VideoMAE; V-JEPA 2 |

The intended main condition is **no explicit cross-modal alignment training in the selected checkpoints**. This does not imply completely disjoint pretraining data or absence of shared labels, teachers, or sources. Pretrained and supervised-fine-tuned checkpoints must be distinguished. Co-aligned models, including ImageBind and an available modern multimodal retriever, form a separate comparison group [9, 10].

Extraction choices—layer, pooling, input duration, frame sampling, and normalization—will be fixed before test evaluation. Alternative choices are robustness checks, not an unrestricted search for the strongest agreement. Frozen encoders run separately and their features are cached; the pilot does not require joint backbone pretraining.

### 6. Measuring native geometry

For modality `m`, encoder `e`, concept `c`, and example `i`, let `z[m,e,c,i]` be the normalized native embedding. Let `mu[m,e,c]` be its concept mean. The initial representational dissimilarity matrix is:

```text
D[m,e,a,b] = 1 - cosine(mu[m,e,a], mu[m,e,b])
```

Every matrix has the same concept axes even when encoders have different feature dimensions. No shared coordinate basis is needed. Direct cosine comparisons between unrelated native encoders are not meaningful merely because their dimensions match.

The primary analysis will use RSA rank correlation; complementary analyses cover neighborhood overlap, metric distortion, and calibrated CKA [2–4]. Scale fitting, when needed, uses designated development concepts and is evaluated on held-out concepts. Learned projections, CCA, and metric learning are excluded from the native-geometry claim.

Centroids are a starting point. Split-half estimates assess stability; within-concept dispersion, multiple prototypes, and instance-level analyses test whether averaging hides inconsistent fine structure. Instance-level comparisons require valid matching or graded relevance annotations. Independent clustering is a secondary analysis rather than the definition of semantic agreement.

UMAP and t-SNE may illustrate findings. Conclusions will use original-space relationships, not visual resemblance of two-dimensional plots.

### 7. Statistical controls and generalization

The study will include the following controls:

- Split-half reliability within each encoder and comparisons between different encoders of the same modality.
- Concept-identity permutations, applied jointly to rows and columns of one dissimilarity matrix.
- Permutations and evaluation within broad semantic groups, separating fine agreement from coarse taxonomy.
- Dimension-matched random features and, where useful, untrained encoders.
- Explicitly co-aligned multimodal models as contextual baselines.
- Paired-instance versus independently sampled concept-matched data.
- Single-frame, frame-averaged, and temporal-video representations.

Pairwise distances are dependent observations. Uncertainty estimation will respect concepts, recordings, and shared sources; individual distance cells will not be treated as independent samples. If layer or metric aggregation is used, null calibration must reproduce the final selection procedure [4]. Primary outcomes and multiple-comparison treatment will be recorded before confirmatory evaluation.

Source-held-out evaluation tests new recordings; concept-held-out evaluation tests new meanings; family-held-out evaluation tests transfer beyond previously studied encoders. Dataset-held-out replication will be required for broader claims. Model pairs sharing an encoder will not be counted as independent evidence of predictive generalization.

There is no universal correlation threshold for success. Effect sizes must be interpreted against null distributions, reliability, semantic granularity, and the sampling scope.

### 8. Secondary stage: a shared multimodal embedding system

If native-geometry findings are reproducible, the next stage will connect frozen encoders through linear or small nonlinear projection heads. The provisional pair budgets are **16, 64, 256, and 1,024**, with repeated sampling and reporting of both pair count and concept coverage.

Candidate objectives include multi-positive contrastive learning and preservation of reliable relational structure. Missing or uninformative channels must be masked appropriately. Modality-specific information should be evaluated explicitly; a shared/private decomposition is a candidate design, not a claimed new contribution.

Required comparisons include a simple linear alignment, a small MLP, GeoWarP where applicable [7], and an available pretrained multimodal model. A brief, fixed-budget trial alignment is also a baseline for the proposed diagnostic: it may be cheaper and more accurate than an elaborate geometry analysis.

Evaluation will cover all feasible directed retrieval pairs, using Recall@K and a graded-relevance metric such as nDCG where appropriate. Multiple valid matches must be represented. Within-modality tasks will check whether alignment loses useful distinctions. Success after learned alignment will not be presented as evidence that native spaces were already identical.

Prospective quality or budget predictions must be fixed before observing the corresponding alignment outcomes. Diagnostic labels, matched concepts, inference, annotation, and fitting costs all count toward the budget. RSA and ordinary cross-model CKA require matched observations; their use cannot be described as fully unpaired prediction. Pair-free diagnostics remain a separate, more demanding question [7].

### 9. Milestones and decision criteria

1. **Freeze the pilot protocol:** confirm sources, checkpoints, observability labels, splits, metrics, and baselines.
2. **Establish reliable geometry measurements:** determine which agreement survives source changes, null calibration, and within-category analysis.
3. **Attempt independent replication:** test held-out concepts, encoder families, and another dataset before generalizing.
4. **Evaluate applied utility:** determine whether the findings predict alignment behavior or improve a concrete quality–cost tradeoff.

Expansion is justified by an effect that either explains a previously unresolved discrepancy or supports useful predictions beyond simple alternatives. A finding of coarse agreement alone should be reported as a bounded replication. A negative result is informative when it identifies the failed claim and rules out measurement instability or unavailable input information.

At this stage, the repository contains documentation only. Code, data manifests, trained heads, and results will be added only in later work.

### References

1. Huh et al. **The Platonic Representation Hypothesis.** ICML, 2024. [Paper](https://proceedings.mlr.press/v235/huh24a.html).
2. Kriegeskorte, Mur, and Bandettini. **Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience.** Frontiers in Systems Neuroscience, 2008. [Paper](https://doi.org/10.3389/neuro.06.004.2008).
3. Kornblith et al. **Similarity of Neural Network Representations Revisited.** ICML, 2019. [Paper](https://proceedings.mlr.press/v97/kornblith19a.html).
4. Gröger, Wen, and Brbic. **Revisiting the Platonic Representation Hypothesis: An Aristotelian View.** ICML, 2026. [Paper](https://proceedings.mlr.press/v306/groger26a.html).
5. Koepke et al. **Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale.** arXiv preprint, 2026. [Paper](https://arxiv.org/abs/2604.18572).
6. Lu et al. **The Indra Representation Hypothesis for Multimodal Alignment.** NeurIPS, 2025. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e63e91b27d16867cdd09c4722f3b0942-Abstract-Conference.html).
7. Schnaus et al. **Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data.** arXiv preprint, 2026. [Paper](https://arxiv.org/abs/2610.09411).
8. Tjandrasuwita et al. **Understanding the Emergence of Multimodal Representation Alignment.** ICML, 2025. [Paper](https://proceedings.mlr.press/v267/tjandrasuwita25a.html).
9. Girdhar et al. **ImageBind: One Embedding Space To Bind Them All.** CVPR, 2023. [Paper](https://arxiv.org/abs/2305.05665).
10. Xu et al. **Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video.** arXiv preprint, 2025. [Paper](https://arxiv.org/abs/2510.03458).
