# Qorğan — детекция телефонного мошенничества (GovTech Camp 2026)

> **Демо:** **https://govtech-production.up.railway.app**
> · [проверка звонка](https://govtech-production.up.railway.app/live.html)
> · [кабинет аналитика](https://govtech-production.up.railway.app/admin.html) (на демо открыт без ключа)

**Qorğan распознаёт приёмы социальной инженерии в телефонных разговорах, объясняет *почему*
звонок выглядит мошенническим, и собирает согласованные обращения граждан в «организации»
для аналитика. Решение всегда принимает человек — система ничего не блокирует и никого не
отключает.**

**Три языка — казахский, русский и английский:** интерфейс, объяснения и советы, а также
анализ звонка на всех трёх. Голосовой режим распознаёт казахскую и русскую речь, включая
смешанную (переключение между ними посреди фразы), и английскую — в тестовом режиме (ADR D62),
прямо на устройстве.

Анализ Уровня 1 выполняется **на устройстве гражданина**: страница загружает
`multilingual-e5-base` (int8 ONNX) и веса модели прямо в браузер. Ни один маршрут сервера
не принимает аудио; содержимое звонка покидает устройство только по явному согласию, когда
гражданин сам отправляет обращение.

---

## 1. Задача

Телефонное мошенничество в Казахстане — массовая проблема. По официальному сообщению
Нацбанка РК от 04.02.2026, на 1 января 2026 года Антифрод-центр зафиксировал **80 871
инцидент** с признаками мошенничества (плюс около 19 810 по наркообороту, игорному бизнесу
и пирамидам); заблокировано 2,8 млрд ₸
([nationalbank.kz/.../18571](https://www.nationalbank.kz/ru/news/informacionnye-soobshcheniya/18571)).

Масштаб подтверждается и уголовной статистикой: за январь–ноябрь 2025 года в Казахстане
зарегистрировано **26,3 тыс. уголовных правонарушений**, связанных с интернет-мошенничеством —
рекорд, при официально зафиксированном ущербе около 12,2 млрд ₸
([informburo.kz](https://informburo.kz/novosti/rekordnoe-kolicestvo-skolko-slucaev-mosennicestva-v-seti-zaregistrirovali-v-kazaxstane-v-2025-godu)).

Существующие меры работают с транзакциями и номерами — **до или после разговора, но не во
время него**: блокировка номера срабатывает на входе, антифрод по транзакции — на выходе.
Сам разговор, в котором человека убеждают, не видит никто. Размеченного корпуса
мошеннических разговоров в официальном контуре тоже нет.

Qorğan закрывает именно этот разрыв: он работает с **содержанием разговора** и даёт
гражданину объяснённое предупреждение в момент звонка, а государству — картину
организованных схем, собранную только из добровольных обращений.

**Два пользователя.** Гражданин (Уровень 1) — проверка звонка на своём устройстве.
Аналитик/правоохранитель (Уровень 2) — приоритетная очередь схем, построенная
исключительно на согласованных обращениях.

---

## 2. Что работает сегодня

Измеримое состояние на 28.09.2026. Все числа — вывод тестового харнесса, с доверительными
интервалами; основная метрика проекта — **FPR (доля ложных тревог)**, а не recall: ложная
тревога на настоящем звонке из банка разрушает доверие к продукту.

| Набор | FPR [95% ДИ] | Recall [95% ДИ] | N |
|---|---|---|---|
| `test` | 0.000 [0.000, 0.060] | 1.000 [0.957, 1.000] | 143¹ |
| `authored_heldout` (ручной) | 0.000 [0.000, 0.142] | 0.889 [0.653, 0.986] | 42 |
| `ood` (вне распределения) | 0.000 [0.000, 0.049] | 0.932 [0.813, 0.986] | 118 |
| `shift` (**другой генератор**) | 0.030 [0.001, 0.158] | 0.455 [0.281, 0.636] | 66 |

¹ Было 115 до сентября; рост — за счёт английских диалогов, добавленных на неделе 4.
По языкам на `test`: kk, ru, mixed, en — везде FPR 0.000 / recall 1.000.
Потоковый режим (звонок по репликам): ложная фиксация 3/60, срабатывание 0.988,
медиана — 3 реплики до тревоги.

**Откуда взялся recall 1.000 на `test`.** Не подгонкой под ошибки теста. Дефект («SMS»
латиницей в словаре против «СМС» кириллицей в живой речи) найден при разборе внешнего
корпуса реальных сообщений о мошенничестве и подтверждён на 272 реальных выходах
распознавателя в `data/asr_capture/` — то есть вне тестовой выборки. Исправление затронуло
53 строки корпуса, из них 3 в `test` (ADR D51).

**Честная оговорка, которую мы держим на виду.** `shift` — это 66 звонков, написанных
*другим* генератором, который не видел ни нашего корпуса, ни промптов, ни словарей. Recall
0.455 — и это настоящая цифра качества, а не 1.000 на `test`. Все остальные наборы делят
генератор с обучающей выборкой. Настоящих записей звонков у нас пока нет (см. §6).

**Что реализовано:**

- **Уровень 1, на устройстве** — PWA: распознавание речи в браузере (два движка Vosk,
  KK + RU, голосование по репликам), скользящее окно, калиброванная шкала подозрительности
  0–100, подсвеченные дословные триггер-фразы, теги тактик, объяснение и советы на
  KK / RU / EN, итог звонка и **редактируемое обращение с явным согласием**.
- **Уровень 2, для аналитика** — очередь организаций по номерному графу, флаг новой схемы,
  детализация; доступ к полной расшифровке требует роли следователя, кода цели и
  попадает в **HMAC-цепочку аудита**.
- **Контроль доступа к кабинету аналитика есть, но на демо-сервере он выключен**, чтобы
  жюри могло открыть `admin.html` без ключа. В рабочем режиме каждый аналитик входит по
  персональному ключу (`QORGAN_ANALYST_KEYS`, роли analyst / investigator), без ключей
  кабинет закрыт. На демо включён `QORGAN_ADMIN_OPEN_ACCESS=investigator`: посетитель без
  ключа работает как `public-demo`, и всё по-прежнему пишется в журнал аудита, а для
  полной расшифровки по-прежнему нужен код цели. Данные на демо — синтетические. Чтобы
  вернуть вход по ключу, достаточно удалить переменную (ADR D59).
- **Приватность, закреплённая тестами** (`tests/test_architecture.py`): аудио не принимает
  ни один маршрут; `/api/analyze` ничего не сохраняет; у хранилища аналитика ровно два
  входа; номера хранятся только как HMAC-дайджест + префикс `+7 700 ***`; расшифровки
  очищены от ПДн; у каждого обращения есть квитанция, его можно удалить, и оно истекает.
- **Правовая оценка** — `docs/LEGAL_ASSESSMENT.md`: разбор ЗРК «О персональных данных»
  (ст. 7, 9, 12, 16, 17, 19-1), Закона об ИИ № 230-VIII и Цифрового кодекса применительно
  к нашим потокам данных, со списком разрывов до пилота.
- **Качество**: 1329 тестов Python + 78 JS (включая пословную сверку браузерного и
  серверного расчёта на золотых фикстурах), 59 решений, номера D1–D62 (D11, D15, D16 пропущены) в `docs/DECISIONS.md`.

---

## 3. Команда и роли

| Участник | Зона ответственности |
|---|---|
| **Имангали** | Продукт и рынок: исследование рынка и конкурентов, позиционирование и value proposition, product narrative и ключевые сообщения. Сборка pitch deck и презентация концепта — проблема, ценность решения, почему это интересно пользователю и рынку. |
| **Сатжан** | Аудит проекта и данных, исследование рынка, законодательства и конкурентов, поиск технических проблем, подготовка к защите (Q&A, позиционирование, питч), ревью заявки на ресурсы. |
| **Санжар** | Разработка: модель и корпус, on-device пайплайн (PWA, ASR, инференс в браузере), оценка качества и харнесс, приватность как архитектура. |
| **Ақылжан** | Разработка: контроль доступа аналитика и аудит, приватность обращений, локализация интерфейса, правовая оценка, сборка и рантайм. |

---

## 4. Ход работы по неделям

Даты и содержание восстановлены по истории коммитов и журналу решений
(`docs/DECISIONS.md`), где у каждого решения стоит дата.

### Фаза 0 — отборочный этап (10–17 июля)

Спринт отбора: скелет проекта, таксономия из 15 тактик, синтетический корпус (Gemini),
первый классификатор и работающее демо «расшифровка → риск → объяснение». Уже тогда был
заложен принцип, который мы дальше не нарушали: **объяснение строится на дословных
фрагментах разговора, а не на свободном тексте модели**. 19 коммитов.

### Неделя 1 (4–10 сентября) — рынок, позиционирование, аудит

Коммитов на этой неделе нет — работа шла над тем, *что* именно строить.
Имангали собрал анализ рынка и конкурентов, сформулировал позиционирование и value
proposition, продумал product narrative. Сатжан провёл аудит проекта и данных, изучил
законодательство и конкурентов, собрал список технических проблем. Параллельно
готовился бриф для стейкхолдеров и запрашивались интервью с аналитиками
(`docs/PLAN_2026-09.md`, §7 и C7).

### Неделя 2 (11–17 сентября) — внешний разбор, разворот и новая архитектура

Ключевая неделя проекта. Независимый совет из трёх экспертных мандатов (Инженер,
Экономист, Red Team) вынес вердикт: **«принять с условиями, но не в текущем виде»** —
прослушивание звонков на сервере и государственный дашборд признаны преждевременными и
рискованными (`qorgan-council-verdict.md`).

Мы приняли это и развернули архитектуру:

- **Решения D12–D13**: сервер больше не принимает аудио; перехват оператором убран из
  дорожной карты навсегда; у Уровня 2 остаётся **единственный вход — согласованное
  обращение гражданина**.
- **D14, D17–D23**: минимизация хранения (хэшированные номера, очищенные расшифровки),
  выбор эмбеддера, партнёрский API с квотой и аудитом, агрегаты по умолчанию для аналитика,
  «новая схема» требует подтверждения.
- 11 сентября финализирован план `docs/PLAN_2026-09.md`; к 17 сентября закрыты стадии 1–3.

### Неделя 3 (18–24 сентября) — on-device, честные метрики, речь

- **D25–D26**: распознавание речи перенесено в браузер (Vosklet, два движка KK + RU);
  Whisper отклонён по измерениям; телефоны закрыты до прохождения бенчмарка.
- **D32–D33**: обнаружено, что WebGPU ломает int8-граф, а «остаточная погрешность рантайма»
  была расхождением версий ONNX Runtime. Модель теперь **обучается и оценивается на
  эмбеддингах самого браузера** — числа описывают то, что видит пользователь.
- **D35**: введён `shift` — 66 звонков от другого генератора. Recall упал с 0.95 до 0.242.
  Это неприятная цифра, и мы сделали её главной.
- **D39**: сопоставление триггер-фраз перестало ломаться об ошибки распознавания
  (измерено на 272 реальных выходах распознавателя); восстановление подсказок 67 % → 76 %.
- **D42–D43**: расширение регистра обучающих данных подняло `shift` 0.242 → 0.364;
  цена зафиксирована в журнале, а не спрятана.

### Неделя 4 (25–28 сентября) — доступ, право, третий язык, надёжность

- **D44–D46**: гражданин просматривает, редактирует и подтверждает обращение до отправки;
  консоль аналитика — персональные ключи, роли, коды цели и аудит в виде HMAC-цепочки.
- **D47, D52**: интерфейс и **содержание** (советы, названия тактик, объяснения) на
  KK / RU / EN — раньше английский интерфейс показывал русские тексты.
- **D49**: закрыты маршруты, которых «не должно было быть»: облачный тир выключен по
  умолчанию и требует отдельного согласия, серверная сессия звонка удалена, хранение
  считается по часам сервера.
- **D50–D51**: найден и исправлен реальный дефект — словарь подсказок писал «SMS» латиницей,
  а живое распознавание выдаёт «СМС» кириллицей, из-за чего **53 строки корпуса теряли
  жёсткий сигнал**. После исправления recall на `test` 0.984 → 1.000.
- **D53**: три ошибки надёжности клиента LLM (зависание на 83 минуты при 0 % CPU,
  необработанный обрыв соединения, потеря целой партии при записи в конце).
- **D54**: английский стал **третьим языком звонка** (ru, kk, en; `mixed` — это
  казахско-русское переключение кодов, а не отдельный язык). 170 диалогов с казахстанскими
  реалиями на английском: язык нужен экспатам и иностранным резидентам, которым звонят те же
  схемы, а партнёр программы inDrive работает на десятках рынков. Вместе с этим изменением
  `shift` сдвинулся до **FPR 0.030 / recall 0.455**. Осторожно с интерпретацией: было
  0.364 [0.204, 0.549], стало 0.455 [0.281, 0.636] — **интервалы сильно пересекаются, это
  один прогон, и одновременно менялись три вещи** (английские данные, подсказка `TeamViewer`,
  переобучение). Эффект не доказан; мы фиксируем совпадение, а не причину.

### Неделя 5 (29 сентября – 1 октября) — развёртывание, браузеры, английский голос

- **D55**: сервер считает ровно теми весами, что получает браузер; деплой больше не может
  подменить модель.
- **D56**: измерено, но не исправлено — собственное предупреждение банка «никому не
  сообщайте код из СМС» может поднять шкалу; это пробел в данных, план исправления записан.
- **D57, D60**: образ собирается и стартует на Railway (без кэш-монтирований BuildKit,
  права тома); сервер начинает отвечать сразу, а демо-данные аналитика готовятся в фоне.
- **D58–D59**: кабинет аналитика сам обновляется после деплоя; для демо доступ открыт без
  ключа (переключатель, по умолчанию выключен, всё по-прежнему пишется в аудит).
- **D61**: микрофон не включался в Safari и на iPhone (браузер блокировал загрузку
  библиотеки с CDN) и в Firefox (обрыв скачивания модели через 30 с). Среда исполнения
  модели теперь раздаётся с нашего сервера; проверено в Chromium, Firefox и WebKit.
- **D62**: английский в голосовом режиме — третья модель распознавания в тестовом режиме.
  Сначала измерен риск: без «форы» английская модель забирала 4,8 % казахских и русских
  фраз; с форой 0,15 — 1,0 %, и все они в двух легитимных звонках (одна ложная тревога
  ушла, новых нет). Английские мошеннические звонки через голос: 15 из 15 (без английской
  модели — 6 из 15).

---

## 5. Что известно о слабых местах

Мы держим это в документации, а не в уме:

1. **Настоящих записей звонков нет.** Все наборы — синтетика или ручные сценарии.
   Протокол приёма реальных данных готов (`docs/DATA_INTAKE.md`), правовой разбор сделан,
   но сами данные — за стейкхолдерами.
2. **Английский не на уровне казахского и русского.** На `test` — 1.000, но это тот же
   генератор, что и в обучении. На независимом генераторе: recall 0.739 при **FPR 0.227**.
   Все 145 ложных тревог — легитимные «продающие» звонки (страхование, телемаркетинг),
   которых нет в наших английских негативах. Задача понятна и оценена.
3. **Перенос между генераторами** (`shift` 0.455) — главный открытый вопрос качества.
4. **Правовые разрывы до пилота** перечислены в `docs/LEGAL_ASSESSMENT.md` §6 (M1–M11):
   основание для данных *звонящего*, хостинг в РК, уведомления, DPA с партнёром.

---

## 6. Что дальше

| Приоритет | Работа |
|---|---|
| 1 | **Реальные звонки** (A3): приём от банка/Антифрод-центра по готовому протоколу; заблокированный held-out набор. Без этого все цифры остаются синтетическими. |
| 2 | Английские негативы в «продающем» регистре — закрыть FPR 0.227. |
| 3 | Перенос между генераторами: третий генератор для обучающих данных. |
| 4 | Правовой блок M1–M11 до любого пилота (мнение адвоката по данным звонящего — на критическом пути). |
| 5 | Мобильные устройства: бенчмарк Vosklet на Android. |

---

## 7. Где что лежит

| Документ | О чём |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Текущее состояние и передача дел — читать первым |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 59 решений, номера D1–D62 (D11, D15, D16 пропущены), с датами и измерениями |
| [`docs/eval_report.md`](docs/eval_report.md) | Все числа с интервалами, источник истины |
| [`docs/LEGAL_ASSESSMENT.md`](docs/LEGAL_ASSESSMENT.md) | Правовая оценка по законодательству РК |
| [`docs/PLAN_2026-09.md`](docs/PLAN_2026-09.md) | План после вердикта совета |
| [`qorgan-council-verdict.md`](qorgan-council-verdict.md) | Вердикт независимого совета |
| [`data/README.md`](data/README.md) | Происхождение данных (ТЗ §9) |
| [`docs/DATA_INTAKE.md`](docs/DATA_INTAKE.md) | Протокол приёма реальных звонков |

---

# Technical reference (English)

## What it does — three pages, one pipeline

The product is the static PWA under `site/`. (The Streamlit app in `app/` is a local dev
harness only — ADR D4 — it scores in the server process and its mic mode uploads audio. Do
not read it as the product.)

| Page | Persona | What happens |
|---|---|---|
| **Landing** (`index.html`) | citizen | What the tool does and does not do, in KK / RU / EN, with the AI disclosure and the download-size notice before the first model fetch. |
| **Live call** (`live.html`) | citizen | The call is analysed **turn by turn, in the browser**: utterances (replayed script, or the microphone via dual Vosk KK+RU with per-utterance voting) → rolling window → int8 `multilingual-e5-base` in a web worker → **0–100 suspicion meter** (hysteresis, hard-signal floors) → verbatim trigger phrases, tactic tags, a templated reason and tactic-specific advice in the chosen language → post-call summary → **reviewed, editable, consent-gated report** with a redaction preview. Nothing leaves the device until the citizen sends it. |
| **Analyst console** (`admin.html`) | gov analyst | Fed **only** by consented reports. KPI row, priority queue of scam organizations named by dominant tactics, novel-scheme flags, drill-down. Per-person keys, analyst vs investigator roles, a stated purpose code and an hourly budget to open a full transcript, and an HMAC-chained audit log (ADR D46). |

## Quick start

```bash
# 1. Environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"         # runtime + tests; ".[all]" adds the cloud tier, Streamlit harness, research paths
cp .env.example .env             # set the number-HMAC, analyst and audit-chain secrets (see the file); GEMINI_API_KEY only for the cloud tier / data-gen

# 2. Models + demo data in one go (idempotent; ~300 MB download on first run):
#    corpus splits + trained head weights from Hugging Face, the int8 ONNX embedder the
#    browser ships (self-hosted under site/models/), and the Level-2 demo seeds.
python scripts/deploy_bootstrap.py
#    ...or retrain the heads from the committed corpus (seconds, CPU):
python -m qorgan.data.build_corpus && python -m qorgan.classifier.linear_train

# 3. Run the site (landing + live call + analyst dashboard) -- analysis runs ON THE DEVICE
python -m qorgan.api             # http://localhost:8000
#    or the Streamlit dev harness (needs `pip install -e ".[harness]"`; not the product):
streamlit run app/streamlit_app.py
```

**On-device by design.** The pages under `site/` load the same `multilingual-e5-base`
int8 ONNX graph and the exported head weights (`site/models/`) and score transcripts in
the browser (`site/core/`, a 1:1 port of the Python classifier — `npm test` proves parity
on golden fixtures). The server embeds with the *same* int8 graph
(`QORGAN_EMBED_BACKEND=onnx`), so a verdict is identical wherever it is computed. No
route accepts audio; call content leaves the device only on an explicit report.

**No model, no key?** The app still runs — it degrades to a deterministic `mock` backend
so the demo scripts work out of the box.

**Harness microphone (optional):** `pip install -e ".[harness,live]"` (vosk, streamlit-webrtc,
sounddevice). First use downloads two small Vosk models (~100 MB) to `~/.cache/vosk`.
Put the call on speakerphone near the device. Without the extra, the Live tab's replay
mode still works and the mic modes show an install hint.

## The demo storyline (3 replay scenes + the analyst flow)
All three ship in `site/core/scenarios.json` and were verified in real Chrome.
1. **Russian scam call** (`live_scam_bank_ru`) — the meter climbs to Critical, evidence and
   advice appear mid-call, the post-call summary offers a report.
2. **Kazakh scam call** (`live_scam_bank_kk`) — the same, in Kazakh, with `secrecy`,
   `otp_request` and `safe_account` firing on verbatim cue hits.
3. **Hard negative** (`live_hard_negative_bank_ru`) — a *real* bank call does **not**
   trigger. False-positive discipline is the product's core metric.

Then the **analyst flow**: submit the report from scene 1 (use a number from a seeded org,
e.g. `+7 700 101 20 30`), open `admin.html` with an analyst key and click **Ingest into
analysis** — it lands inside that organization.

## Evaluate

```bash
# FPR-first tables, per language. `shift` is the cross-generator split -- the number to
# lead with, and the reason the others read high (they share a generator with training).
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run \
    --split test --split authored_heldout --split ood --split shift --by-language

# Streaming eval: false-latch rate (live FPR analog), time-to-alert
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.stream \
    --split test --split authored_heldout --backend linear

# Live-meter parameter sweep on cached per-turn traces (seconds, after a ~2 min trace build)
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.meter_sweep --min-turns 1,2,3 --damping 1,2,3

# Level-2 cluster quality under number-rotation stress, with stability intervals
python -m qorgan.eval.cluster --resamples 50

# Adversarial paraphrases: cue-free (A9) and legit-sounding (A9b), paired recall vs the sources
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.adversarial --split adversarial_legit

# The recogniser's register: every eval dialogue clean vs ASR-styled (lowercase, no
# punctuation, numerals as words), paired; --drop-latin is the worst case for «SMS»/«CVV»
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.asr_realism [--drop-latin]

pytest -q        # 1,329 tests, all offline
npm test         # JS core parity with Python + the DEVICE gate (browser embeddings, seconds)
npm run gate:browser   # re-capture the browser's embeddings of the gate set (headless Chromium;
                       # `npx playwright install chromium` once) after a model/runtime change

# Retrain / evaluate on the DEVICE's embeddings (ADR D33): start the bridge, then any command
# above with QORGAN_EMBED_BACKEND=device (vectors are cached; the first pass takes minutes)
npm run device:serve -- --pages 4 &
QORGAN_EMBED_BACKEND=device python -m qorgan.classifier.linear_train
```

**Numbers live in one place.** The current, measured state — FPR-first, with Clopper–Pearson
intervals, per split and per language — is the table in **§2 above**; the full methodology and
every ablation is [`docs/eval_report.md`](docs/eval_report.md). They are deliberately not
repeated here: this section used to carry its own copy, it drifted out of date, and a reviewer
caught the contradiction.

Three points that belong with the numbers rather than in the table:

- **They describe the device.** Heads are trained and evaluated on the **browser's own
  embeddings** (ADR D33), so the figures describe what the citizen's device decides, not a
  server approximation. The server's native runtime is a cosine-0.98 proxy and disagrees on
  6/200 borderline calls — reported, not hidden.
- **`shift` is the number to lead with.** 66 calls written by a *second generator* that never
  saw the corpus, the prompts or the lexicons. Every other split shares its generator with
  training data, so their recall is largely that generator's register. Until real calls exist
  (A3), the recall claim is "one generator's scams".
- **The cloud tier is the accuracy tier, and it is off by default.** On those same 66 calls the
  `llm` backend (Gemini 2.5 Pro) scores 33/33 recall at 0/33 FPR (ADR D38). It sends text
  abroad, so it requires `QORGAN_CLOUD_TIER=on` plus the citizen's per-request consent, is
  never cached and is refused outright on analyst routes (ADR D49). The gap between the device
  model and that tier is the honest cost of running on-device.

## Microphone mode (on-device speech recognition)

On desktop browsers the live page can listen to a call directly: Kazakh and Russian Vosk
models — plus English in test mode, ranked with a 0.15 handicap so it cannot take over kk/ru
speech (ADR D62) — run in the browser (Vosklet/WASM, one instance each), the best hypothesis
wins per utterance, and the transcript feeds the same on-device classifier and meter as replay —
**no audio or text leaves the device** (ADR D26). Requirements the server already meets:
`/live.html`, `/core/*` and `/vendor/*` are served cross-origin isolated (COOP/COEP) because the
recogniser needs SharedArrayBuffer; `python scripts/deploy_bootstrap.py` installs the
hash-pinned Vosklet and transformers.js/onnxruntime-web runtimes under `site/vendor/` (ADR D61)
and packages the three model tarballs (~147 MB, downloaded once by the browser) — nothing is
loaded live from a CDN, which Safari would block under COEP. Phones are disabled until the
Android bench passes (`scripts/spikes/vosklet_bench/`).

## Real calls (when they arrive)

`docs/DATA_INTAKE.md` is the policy and protocol: encrypted/on-prem delivery, a
`batch.yaml` + `calls.csv` batch format, `python scripts/ingest_partner_calls.py <batch>`
(scrub → hashed number linkage → first-come allocation into a hash-locked
`real_heldout_v2` of 60 legit / 40 scam, the rest to `real_train`). The locked set is
scored, never read, and only at release points. `data/real/` never enters git.

## Reports API (consented ingress) & privacy

`POST /api/reports` is the only way call content enters the analyst layer, and only on an
explicit user action. The server stores the transcript PII-scrubbed, the caller number as
a salted HMAC digest + prefix (`+7 700 ***`), returns exactly what it kept plus a receipt,
and `DELETE /api/reports/{receipt}` forgets it everywhere (`python -m qorgan.reports.purge`
applies the retention window). Set `QORGAN_NUMBER_HMAC_KEY` (see `.env.example`); without
it the server refuses reports that carry a number. No route accepts audio; these
invariants are enforced by `tests/test_architecture.py` (ADRs D12–D14).

## Analyst dashboard exposure

`admin.html` asks for an **analyst key** (`QORGAN_ANALYST_KEYS="id:secret:role,..."`; unset =
the console is closed) and then shows organization aggregates and excerpts; the identity in
every log line comes from the key. Reading a whole call needs the **investigator** role and a
stated purpose (`pattern_review` / `citizen_request` / `partner_request`); the server writes
`analyst · case.open · incident:<id> · purpose` to `data/processed/audit_log.jsonl` before it
answers (ADR D20). That log is a keyed hash chain (`QORGAN_AUDIT_CHAIN_KEY`):
`python -m qorgan.audit verify` names the first edited, deleted or reordered entry. Analysts
can **confirm / dismiss / merge** an organization; the verdict is stored as an append-only
event keyed by the operation's numbers, so it survives re-clustering (a dismissed operation
drops to 20 % priority; ADR D24). Per-person keys are a stand-in for SSO in a deployment.

**Open demo access (on in the hosted demo).** `QORGAN_ADMIN_OPEN_ACCESS=analyst|investigator`
(default `off`) admits a visitor *without* a key as the reserved identity `public-demo` in that
role, so a jury can open the console without credentials. Only who may enter changes: every
action is still audited, under `public-demo`, and the audit-chain key is still required. Roles
and the stated purpose still apply. A presented key keeps its own identity, and a wrong key is
still refused. Use it only on a server holding fabricated data: anyone with the URL can read
what the role allows (ADR D59).

## Partner API (`/api/v1`) — consented reports in, aggregates out

A bank fraud desk, telecom or hotline can feed confirmed cases into the analyst layer and
read back the organization-level picture — without ever sending call content it does not
have to. It is a second *consented* ingress, not a bulk feed (ADR D19):

- **Auth**: `X-API-Key` per partner; keys live in the server env
  `QORGAN_PARTNER_API_KEYS="bank_a:<secret ≥16 chars>[:daily_quota],telecom_b:<secret>"`
  (unset = the API is closed; `QORGAN_PARTNER_DAILY_QUOTA` is the default budget).
- **One report per request**, preferably **structured tactic hits**; a transcript is accepted
  only if the partner already PII-scrubbed it (the server checks and refuses otherwise,
  without echoing it). `consent_basis` (a code from the data-sharing agreement) is required.
  The caller number is hashed on receipt like a citizen report. `partner_reference` makes
  retries idempotent.
- **Limits**: 60 requests/min and a rolling 24 h quota per partner (`X-Quota-Limit` /
  `X-Quota-Remaining` on every response); a content-free audit line for every action
  (`data/processed/audit_log.jsonl`).
- **Export**: `GET /api/v1/organizations` returns aggregates only — no numbers, no digests,
  no transcripts. Partners can `DELETE` only their own receipts.

```bash
export QORGAN_PARTNER_API_KEYS="bank_a:replace-with-a-32-char-secret-00000000"
python -m qorgan.api    # OpenAPI at http://localhost:8000/docs (scheme: PartnerApiKey)

# 1. a confirmed case as structured signals (preferred shape)
curl -s -X POST http://localhost:8000/api/v1/reports \
  -H "X-API-Key: replace-with-a-32-char-secret-00000000" -H "Content-Type: application/json" \
  -d '{"consent_basis":"customer_consent","tactic_ids":["otp_request","safe_account"],
       "phone_number":"+7 700 555 66 77","partner_reference":"CASE-2026-0912"}'
# -> 201 {"receipt_id": "...", "number_prefix": "+7 700 ***", "quota": {"limit":200,"used":1,...}}
# 2. the same case again -> 200 "duplicate", nothing stored, quota untouched
# 3. the organization-level picture (aggregates only)
curl -s http://localhost:8000/api/v1/organizations?locale=ru -H "X-API-Key: replace-with-a-32-char-secret-00000000"
# 4. withdraw a report
curl -s -X DELETE http://localhost:8000/api/v1/reports/<receipt_id> -H "X-API-Key: replace-with-a-32-char-secret-00000000"
```

Signals-only reports (no transcript) are stored, receipted, deletable and counted; placing
them into organizations through the number graph alone is the next Level-2 item (PLAN C9).

## Publish model/data updates (maintainers)

After a retrain: `hf auth login` (write token) then `python scripts/hf_upload.py` — it
uploads `models/linear` **together with** `data/lexicon` (the bundle hash-validates the
lexicons) and the scrubbed dataset splits. Never widen the dataset patterns: the other
`data/processed/` files (incidents, citizen reports) and raw `data/synthetic/` must stay
off the Hub.

## Layout
`src/qorgan/` (config · taxonomy · data · classifier · explain · **live** · analytics ·
asr · eval) · `app/` (Streamlit: `streamlit_app.py`, `live_view.py`, `mic_live.py`,
`analyst_view.py`) · `data/` (taxonomy · lexicons · corpus + provenance) · `tests/` ·
`docs/` (scope, architecture, decisions, status, eval report).

## Status & limits
**Level 1 analysis already runs on the device** — `site/` is a static PWA that loads the int8
ONNX embedder and the exported heads and scores in the browser; no route accepts audio. What
remains roadmap is **mobile** (phones are gated on the Android Vosklet benchmark, ADR D26) and
**carrier integration** ([`DOCUMENTATION.md`](DOCUMENTATION.md)).

The live-mic path is real (dual Vosk KK+RU streaming, per-utterance voting) but
speakerphone-quality ASR — especially Kazakh — is the accuracy bottleneck; the meter's
confidence weighting absorbs some of it. Transient mid-call latching on legitimate calls is
measured by `eval.stream` rather than estimated: currently **3/60 on `test`** and **1/19 on the
clean authored negatives**, with the inspected anchors reported separately. English is the
weakest language and is **not at Russian/Kazakh quality** — see §5 above. Real call recordings
do not exist yet (A3); every number is synthetic or author-written.
