# Архитектура Qorğan — для пилота в экосистеме eGov

> **English summary.** This file has Mermaid diagrams of Qorğan at three levels, plus the
> scoring pipeline: (1) the **context** (citizen, device, Qorğan Level 2, analysts, partners,
> and the planned eGov / Smart Bridge / operator / Antifraud Center links); (2) the
> **containers** (the PWA's on-device modules and the FastAPI server with its file stores, purge
> thread and audit chain); (3) **deployment**, as it is today (a Railway demo outside Kazakhstan)
> and as we propose it for the pilot (a Kazakhstan data centre with TLS/mTLS, a database,
> secret store, WORM audit anchors and a gateway to ШЭП); and (4) the **on-device pipeline**:
> microphone → Vosk live pass, plus a GigaAM-Multilingual pass at each utterance endpoint
> (being built, not shipped) → e5-base int8 embeddings plus cue flags → calibrated logistic
> regression → meter, explanations and advice. Each element is marked as fact (in code today),
> plan (in progress) or proposal.

_Сверено с кодом ветки `sanzh-ts` @ `54a6f0b`. Подробнее о системе — `docs/ARCHITECTURE.md`; часть
этого файла относится к июльской версии, поэтому где файлы расходятся, верен код._

**Обозначения на диаграммах.**
- Сплошная линия — [ФАКТ], существует сейчас.
- Пунктир — [ПРЕДЛОЖЕНИЕ] или [ПЛАН].
- Пометка `(план)` или `(предл.)` стоит в подписи.

## 1. Контекст

```mermaid
flowchart TB
  CIT(["Гражданин"])
  ANA(["Аналитик / следователь<br/>(оператор системы или госорган)"])
  PRT(["Партнёр: банк, оператор связи,<br/>горячая линия"])

  subgraph DEV["Устройство гражданина"]
    PWA["Qorğan PWA (site/)<br/>анализ на устройстве"]
    EGM["eGov Mobile<br/>(предл.: WebView-хост, QR-подпись)"]
  end

  subgraph QOR["Qorğan, Уровень 2 (сервер)"]
    API["FastAPI: /api/reports, /api/v1, /api/admin"]
  end

  subgraph GOV["Экосистема eGov / антифрод"]
    SB["Smart Bridge / ВШЭП<br/>(предл.)"]
    QRS["QR-подписание NITEC-S-5096<br/>(предл.)"]
    OPS["Операторы связи: жалоба,<br/>проверка номера за 2 р.д. (предл.)"]
    AFC["Участник Антифрод-центра НБ РК<br/>(предл.)"]
    NG["Nomad Guard (предл.)"]
    MVD["МВД: 102 / «Закон и порядок»<br/>(предл.)"]
  end

  CIT -->|"громкая связь / вставка текста"| PWA
  PWA ==>|"отчёт с согласием (TLS)"| API
  CIT -.-> EGM
  EGM -.->|"WebView"| PWA
  API -.->|"запрос подписи"| SB -.-> QRS
  QRS -.->|"подпись в приложении"| EGM
  PWA -.->|"жалоба с открытым номером<br/>прямо с устройства"| OPS
  API -.->|"CaseSignal: только дайджест,<br/>после проверки аналитиком"| AFC
  API -.->|"сигнал по организациям"| NG
  PWA -.->|"обращение с фрагментами"| MVD
  ANA -->|"X-Analyst-Key"| API
  PRT -->|"X-API-Key: один отчёт /<br/>только агрегаты"| API
```

Источники для связей с экосистемой: `docs/research/2026-10/egov_integration.md` §1, §2, §5
(NITEC-S-5096 [A]: https://sb.egov.kz/services/passport/NITEC-S-5096 ; проверка номера за 2 р.д. [B]:
https://informburo.kz/novosti/nomera-s-kotoryx-zvoniat-mosenniki-budut-blokirovat-v-kazaxstane-prikaz-vstupil-v-silu ;
у Антифрод-центра нет внешнего входа [A/B]:
https://orda.kz/operatory-svjazi-i-antifrod-centr-budut-formirovat-bazu-potencialnyh-zhertv-moshennikov-411834/ ).

## 2. Контейнеры

```mermaid
flowchart LR
  subgraph BR["Браузер / WebView (Уровень 1)"]
    direction TB
    LIVE["live.html + live.js<br/>звонок, отчёт, квитанция"]
    TRY["index.html + try.js<br/>вставка текста"]
    ASRM["core/asr.js<br/>Vosklet WASM, KK+RU+EN"]
    WRK["core/embed-worker.js<br/>transformers.js + ORT WASM<br/>(self-hosted, хеши проверяются)"]
    CORE["core/session, score, head, lexicon,<br/>cue-match, meter, explain,<br/>recommend, summary, report"]
    SW["sw.js<br/>офлайн-оболочка, /api/ не кэшируется"]
    ADM["admin.html + admin.js<br/>консоль, ключ в sessionStorage"]
    LIVE --> ASRM --> CORE
    TRY --> CORE
    CORE --> WRK
  end

  subgraph SV["Сервер (один процесс uvicorn)"]
    direction TB
    MW["Middleware: лимит тела 256 KB, заголовки<br/>безопасности, COOP/COEP, 422 без эха"]
    R1["api.py: /api/health, /api/analyze"]
    R2["api_reports.py: /api/reports"]
    R3["api_admin*.py: /api/admin/*<br/>require_analyst / require_investigator"]
    R4["api_partner*.py: /api/v1/*"]
    R5["api_live.py: только чтение"]
    PREP["reports/store.prepare_report<br/>scrub_text + privacy/numbers HMAC"]
    AUD["audit.py<br/>HMAC-цепочка"]
    PUR["reports/purge.PurgeSchedule<br/>старт + каждые 24 ч"]
    AN["analytics/: intake, cluster,<br/>numbers, feedback"]
    FS[("том data/processed:<br/>citizen_reports, incidents,<br/>organizations, embeddings,<br/>org_feedback, audit_log")]
    MW --> R1 & R2 & R3 & R4 & R5
    R2 --> PREP --> FS
    R4 --> PREP
    R3 --> AN --> FS
    R3 --> AUD
    R4 --> AUD --> FS
    PUR --> FS
  end

  LIVE ==>|"POST/DELETE /api/reports"| R2
  TRY -.->|"по выбору: POST /api/analyze"| R1
  ADM -->|"/api/admin/*"| R3
  BR -->|"GET статики: модель 278 MB,<br/>Vosk ~106 MB, weights.json"| SV
```

[ФАКТ] Модули на пути гражданина (`api.py`, `api_live*.py`, `api_reports*.py`, `api_partner.py`,
`live/*`) **не могут** импортировать запись в аналитику. Это проверяет AST-тест
`test_citizen_path_modules_cannot_reach_analytics_write_paths`. Данные в анализ попадают только
через явный `POST /api/admin/ingest`.

## 3. Развёртывание

### 3.1 Сейчас — демо на Railway ([ФАКТ], `docs/DEPLOY.md` §8)

```mermaid
flowchart LR
  U["Браузер посетителя"] -->|"HTTPS (TLS терминирует Railway)"| RW
  subgraph RAIL["Railway (вне РК)"]
    RW["Контейнер qorgan<br/>uid 10001, 1 процесс, 1 реплика<br/>QORGAN_ADMIN_OPEN_ACCESS=investigator (D59)"]
    VOL[("Том /app/data/processed<br/>(необязателен: без него<br/>каждый деплой начинается с нуля)")]
    RW --- VOL
  end
  GH["GitHub (код)"] -->|"сборка Dockerfile"| RAIL
  HF["Hugging Face Hub<br/>корпус, embedder"] -->|"только при сборке"| RAIL
  U -.->|"шрифты: IP уходит в Google (S1)"| GF["fonts.googleapis.com"]
```

**Ограничения демо.**
- Данные только вымышленные.
- Хостинг вне РК ([ПРОБЕЛ] M5).
- Открытый вход в консоль (ADR D59).
- Секреты хранятся в переменных платформы.
- Для реальных отчётов это **недопустимо** (`docs/DEPLOY.md` §7–§8).

### 3.2 Предлагаемый контур пилота в РК ([ПРЕДЛОЖЕНИЕ])

```mermaid
flowchart LR
  subgraph INET["Интернет"]
    U["PWA / WebView eGov Mobile"]
  end
  subgraph DC["ЦОД на территории РК (ЦОД НИТ / QazTech или коммерческий ЦОД РК — вопрос 7)"]
    subgraph DMZ["DMZ"]
      RP["Обратный прокси<br/>TLS 1.2+, HSTS, WAF,<br/>FORWARDED_ALLOW_IPS"]
      STAT["Статика / кэш моделей"]
    end
    subgraph APP["Сегмент приложения"]
      Q["Контейнер qorgan<br/>read-only FS, без root,<br/>OPEN_ACCESS=off"]
      GW["Клиент ШЭП<br/>mTLS, сертификаты"]
    end
    subgraph DATA["Сегмент данных"]
      PG[("Postgres в РК<br/>PITR, шифрование")]
      KMS["Хранилище секретов<br/>(ключи под контролем РК)"]
      WORM[("WORM: якоря head<br/>журнала аудита")]
      LOG[("Журналы без содержимого<br/>хранение 3 года")]
    end
    BAST["Бастион / доступ администраторов"]
  end
  SOC["SOC / ОЦИБ (договор)"]
  SHEP["ВШЭП / Smart Bridge"]

  U -->|"HTTPS"| RP --> Q
  U --> STAT
  Q --> PG
  Q --> KMS
  Q --> WORM
  Q --> LOG
  Q --> GW -->|"mTLS"| SHEP
  LOG --> SOC
  BAST --> Q
```

**Что меняется по сравнению с текущим кодом** (`docs/DEPLOY.md` §7):
- JSONL заменяется на Postgres. Это снимает ограничение «один процесс», если лимиты и бюджет
  открытий переедут в общее хранилище.
- Секреты переезжают из `.env` в KMS.
- Якоря `head` журнала аудита фиксируются ежедневно в WORM.
- Аналитики входят через SSO с MFA ([ПРОБЕЛ] M9).
- Демо-наполнение (ADR D60) и `QORGAN_ADMIN_OPEN_ACCESS` выключаются.
- Облачный уровень остаётся `off`.
- Шрифты размещаются у нас (S1).

mTLS в промышленной среде Smart Bridge обязателен с 27.02.2026 [A] (https://t.me/s/sb_shep ).
Размещение ПДн в РК обязательно по п. 2 ст. 12 Закона о ПДн [S] (`docs/LEGAL_ASSESSMENT.md` §2.4).

## 4. Конвейер на устройстве

```mermaid
flowchart TB
  MIC["Микрофон (громкая связь, передний план)<br/>getUserMedia, AudioWorklet"]
  subgraph P1["Проход 1 — живой (ФАКТ)"]
    VK["Vosk small KK"]
    VR["Vosk small RU"]
    VE["Vosk small EN<br/>штраф 0.15 (D62)"]
    VOTE["Голосование по реплике:<br/>средняя уверенность слов,<br/>доля казахских букв, окно 1.2 с"]
  end
  subgraph P2["Проход 2 — в конце реплики (ПЛАН, не выпущено)"]
    GA["GigaAM-Multilingual 220M int8<br/>sherpa-onnx WASM, символьный CTC<br/>язык слова по алфавиту"]
  end
  UTT["Принятая реплика"]
  WIN["Скользящее окно: голова + хвост<br/>(core/session.js), префикс 'query: '"]
  EMB["multilingual-e5-base int8 ONNX<br/>768-d, web worker, только WASM (D32)"]
  CUE["5 флагов жёстких сигналов<br/>secrecy, otp_request, credentials_request,<br/>safe_account, remote_access<br/>(cue-match с ограниченной правкой, D39)"]
  REA["1 флаг успокоения<br/>«мы никогда не спросим код…»"]
  LR["Калиброванная ЛР: риск<br/>+ 15 ЛР по тактикам (свои пороги, D30)"]
  MET["Шкала 0–100: асимметричная EMA,<br/>порог 0.59 / 0.49,<br/>включается с 3-й реплики (D29)"]
  EXP["Объяснение: дословные фрагменты<br/>+ шаблоны RU/KK/EN"]
  ADV["Советы по тактикам<br/>advice_ru/kk/en.yaml"]
  SUM["Итог после звонка"]
  REP["Черновик отчёта → проверка →<br/>согласие → POST /api/reports"]

  MIC --> VK & VR & VE --> VOTE --> UTT
  VOTE -.->|"аудио реплики"| GA -.->|"заменяет текст реплики"| UTT
  UTT --> WIN --> EMB --> LR
  WIN --> CUE --> LR
  WIN --> REA --> LR
  LR --> MET --> EXP --> ADV --> SUM --> REP
```

**Пояснения к конвейеру.**

[ФАКТ] Проход 1 реализован в `site/core/asr.js`:
- распознаватели KK и RU работают параллельно, у каждого свой экземпляр Vosklet;
- побеждает гипотеза с большей средней уверенностью слов;
- английский распознаватель участвует со штрафом 0.15 (ADR D62).

На телефонах режим микрофона выключен, пока не пройдена проверка производительности на Android
(`docs/STATUS.md`, B9).

[ПЛАН] Проход 2 описан в `docs/research/2026-10/stt_wordlevel.md`:
- в конце реплики распознаёт её заново моделью GigaAM-Multilingual 220M int8 (MIT, 224.8 MB);
- метрики модели из её карточки: FLEURS WER kk 5.2 / ru 4.4 / en 12.2 [V,
  https://huggingface.co/ai-sage/GigaAM-Multilingual];
- язык каждого слова определяется по алфавиту;
- шкала и сопоставление сигналов работают только по тексту второго прохода.

Бюджет — около 282–365 MB, в зависимости от того, какие модели Vosk останутся. Код пока только в
`scripts/spikes/asr_wordlevel/`; в `site/` его нет.

[ФАКТ] Признаки и классификатор:
- признаки: 768 + 5 + 1;
- риск — калиброванная логистическая регрессия; 15 тактик — отдельные логистические регрессии;
- веса в `site/models/weights.json` — это и есть поставляемая модель;
- сервер считает теми же весами (ADR D55);
- Python и JS дают одинаковый результат на эталонных примерах (`tests_js/`, `scripts/export_parity_fixtures.py`).

[ФАКТ] Объяснения — только дословные фрагменты и шаблоны. **Текст LLM пользователю не
показывается** (`CLAUDE.md`).

## 5. Краткая модель угроз ([ПРЕДЛОЖЕНИЕ] — основа для ТД ИБ)

| Угроза | Где | Мера сейчас ([ФАКТ]) | Что добавить |
|---|---|---|---|
| Перехват или накопление разговоров на сервере | Уровень 1 → сервер | Аудио не принимается; `/api/analyze` ничего не хранит; серверный live-session API выведен (D49) | Подписанный публичный пакет модели и кода для устройства (`LEGAL_ASSESSMENT` §5.3) |
| Удалённое расширение сбора данных («переключатель слежки») | конфигурация | Нет порога автоотправки, управляемого сервером; отчёт — только по действию гражданина | Тест-инвариант на отсутствие удалённых флагов сбора |
| Восстановление номеров по дайджестам | Уровень 2 | HMAC с секретным ключом; ключ ≠ ключ цепочки; поиска по номеру нет | KMS; ключ недоступен аналитикам (M5, `LEGAL_ASSESSMENT` §2.3) |
| Массовое чтение транскриптов инсайдером | консоль | Роль investigator, цель, 30 открытий в час, журнал до выдачи | SSO/MFA, пересмотр прав, внешняя проверка журнала (M9) |
| Подделка журнала аудита | хранилище | HMAC-цепочка, `verify`, якорь `head` | WORM вне домена доверия сервера |
| Утечка ПДн в тексте отчёта | отчёт | `scrub_text`, предпросмотр, правка гражданином | Маскировка чисел словами, имён, IBAN (M7) |
| Подмена кода на устройстве (цепочка поставок) | PWA | Своя размещённая и проверенная по хешу среда выполнения (D61); service worker | SRI / подпись релиза, SBOM |
| Отказ в обслуживании | `/api/reports`, `/api/analyze` | Лимиты частоты, тело ≤ 256 KB | WAF, нагрузочные испытания |
