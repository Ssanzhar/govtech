# Спецификация API — Qorğan (текущие маршруты и предлагаемые сервисы передачи)

> **English summary.** Sections 1–3 describe the 17 HTTP routes **as they exist** in the FastAPI
> app (`src/qorgan/api*.py`). The route table was dumped from `app.openapi()` on 2026-10-08. For
> each route the file gives authentication, role, request and response fields, error codes,
> limits and what is written to the audit log. Section 4 is a **PROPOSAL** (nothing is built): a
> hand-off schema for (a) an operator complaint, sent from the citizen's device so the raw number
> never reaches the Qorğan server, and (b) a digest-only case signal for an Antifraud Center
> participant, with an optional, consented potential-victim flag. Section 5 is a **PROPOSED**
> eGov QR identity-verification flow. To regenerate the full OpenAPI document locally, run the
> server and fetch `/openapi.json`, or open `/docs`.

_Версия сервера: `qorgan api` 0.1.0 (`src/qorgan/api.py`). Сверено 2026-10-08._

## 0. Общие правила (все маршруты) — [ФАКТ]

- **Формат.** Запросы и ответы в JSON. Ни один маршрут не принимает `audio/*` или `multipart`, и
  WebSocket-маршрутов нет (`tests/test_architecture.py`).
- **Размер тела** ≤ 256 KB (`src/qorgan/api_limits.py::MAX_BODY_BYTES`).
- **Ошибки 422** называют проблему, но **никогда не повторяют введённое значение**
  (`api_limits.py::validation_error_handler`). Причина: введённое может быть номером карты или
  именем.
- **Заголовки на каждом ответе:** `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'`, `Permissions-Policy`.
  Ответы консоли дополнительно получают `Cache-Control: no-store`.
- **Время** — ISO 8601. Хранилище считает время без часового пояса временем UTC.
- **Тактики** — идентификаторы из `data/taxonomy/tactics.yaml`:
  - `impersonation_bank`, `impersonation_gov_police`, `impersonation_telecom_delivery`;
  - `urgency`, `fear_threat`, `secrecy`;
  - `otp_request`, `credentials_request`, `safe_account`, `payment_redirect`, `remote_access`;
  - `prize_lottery`, `investment_scam`, `mule_recruitment`, `verification_ploy`.

  Неизвестный идентификатор даёт 422 с номерами позиций, а не значениями.

## 1. Маршруты для гражданина (без аутентификации)

### 1.1 `GET /api/health`

Ответ `200`:
- `status`: `"ok"`;
- `configured_backend`: `linear` | `mock` | `llm`;
- `threshold`: 0.59;
- `linear_model_source`: `web` | `bundle` | `unavailable`.

В развёртывании ожидается `web`: сервер считает с теми же весами, что и браузер (ADR D55).

### 1.2 `POST /api/analyze` — оценка на сервере, **без сохранения**

Код: `src/qorgan/api.py::analyze`.

| Поле запроса | Тип | Ограничения |
|---|---|---|
| `transcript` | string | 1–20 000 символов, не пустая строка |
| `locale` | `ru` \| `kk` \| `en` | по умолчанию `ru` |
| `backend` | string \| null | `linear`, `mock` или `llm`; по умолчанию — из конфигурации |
| `cloud_consent` | bool | Учитывается только для `llm`: явное согласие на передачу текста в Google за пределы РК |

Ответ `200` (`AnalyzeResponse`):
- `risk` (0–1), `flagged`, `threshold`;
- `backend`, `fallback` (true, если настроенная модель недоступна и ответ посчитан `mock`);
- `tags[] {id, weight}`;
- `spans[] {text, start, end}` — дословные фрагменты;
- `explanation {reason, confidence_label, caveat, human_note}` — шаблонный текст.

Ошибки:
- `403` — облачный уровень выключен (`QORGAN_CLOUD_TIER=off`), ничего не отправлено;
- `422` — пустой текст, неизвестный бэкенд или `llm` без `cloud_consent`;
- `500` — сбой генератора объяснений.

**Ничего не записывается** (`use_cache=False`; тест `test_analyze_endpoint_writes_nothing_under_data`).

### 1.3 `POST /api/reports` — единственный вход для граждан

Код: `src/qorgan/api_reports.py::submit`.

| Поле запроса (`ReportSubmission`) | Тип | Ограничения |
|---|---|---|
| `transcript` | string | 1–20 000 символов; проверен и отредактирован гражданином. Сервер всё равно повторно очищает текст через `scrub_text` |
| `phone_number` | string \| null | ≤ 32 символов. Передаётся **только в транзите по TLS**: сохраняется HMAC-дайджест и префикс |
| `flagged_phrases` | string[] | ≤ 50 фраз по ≤ 300 символов. Сохраняются только фразы, которые дословно остались в очищенном тексте |
| `tactic_ids` | string[] | ≤ 32; только известные тактики; дубликаты убираются |
| `risk_score` | number | 0–100 |
| `consent` | `true` | Обязательно, буквально `true` |
| `consent_version` | string | Только зарегистрированная версия, сейчас `report-v1` (`reports/model.py::CITIZEN_CONSENT_VERSIONS`) |
| `timestamp` | datetime \| null | Время звонка по часам клиента. Только в окне [сейчас − 180 дн, сейчас + 5 мин] |

Ответ `201` (`ReportReceipt`) — **ровно то, что сохранено**:
- `receipt_id` (24 hex), `report_id`, `status: "stored"`;
- `number_prefix` (`+7 700 ***` или null), `stored_transcript`, `flagged_phrases`, `tactic_ids`;
- `timestamp`, `consent_version`, `expires_at` (= `received_at` + 180 дней).

Ошибки:
- `422` — валидация, окно времени или номер не распознан (значение не повторяется);
- `429` — больше 20 запросов в минуту с одного адреса;
- `503` — у сервера нет ключа хеширования номеров; отчёт с номером **не сохраняется**.

Журнал аудита: не пишется (гражданин анонимен). [ПРЕДЛОЖЕНИЕ] Добавить запись без содержимого
`report.submit receipt:…`.

### 1.4 `DELETE /api/reports/{receipt_id}`

- `receipt_id` должен соответствовать `^[0-9a-f]{24}$`, иначе `422`.
- `204` — отчёт гражданина удалён везде, куда попал: отчёты, инциденты, организации,
  эмбеддинги, следы в отзывах аналитиков (`analytics/intake.py::forget_report`).
- `404` — квитанция неизвестна **или принадлежит партнёру**: партнёрский отчёт удаляет только
  сам партнёр.

### 1.5 `GET /api/live/capabilities`, `GET /api/live/scenarios`

- `capabilities` → `{microphone: false, reason}`. Сервер прямо заявляет, что не принимает аудио.
- `scenarios` → `{scenarios[] {id, label, lines[]}}` — три демонстрационных сценария.

### 1.6 Статика

Сайт `site/` отдаётся из корня `/`:
- `index.html`, `live.html`, `admin.html`, `core/*`;
- `models/weights.json`, модель e5-base int8, Vosk-модели, `/vendor/`.

Для `live.html`, `/core/*` и `/vendor/` выставляются COOP/COEP.

## 2. Консоль аналитика `/api/admin/*` — [ФАКТ]

**Аутентификация.**
- Заголовок `X-Analyst-Key` содержит персональный секрет из `QORGAN_ANALYST_KEYS`.
- Личность аналитика берётся **только** из ключа (`src/qorgan/api_admin_auth.py::require_analyst`).
- Консоль закрыта (`503`), если не настроены ключи аналитиков или ключ цепочки аудита.
- Неверный ключ даёт `401` и запись `auth.denied` (не больше 10 в минуту с одного адреса).
- Лимит — 120 запросов в минуту на аналитика.
- Все маршруты возвращают `Cache-Control: no-store`.

| Маршрут | Роль | Запрос | Ответ (главное) | Аудит |
|---|---|---|---|---|
| `GET /session` | analyst | — | `{id, role, can_open_cases, open_purposes[], open_access}` | `session.start` |
| `GET /overview?locale=` | analyst | — | `{available, kpis{incidents, organizations, novel_schemes, pending_reports}, organizations[]{id, name, priority, incidents, numbers[] (дайджесты), last_activity, is_novel, feedback}}` | — |
| `GET /stats?locale=` | analyst | — | `{available, activity[30 дней], trend{this_week, last_week, delta_pct}, top_organizations[≤6], reports{submitted, ingested, pending, signals_only}, novel_schemes}` | — |
| `GET /organizations/{org_id}?locale=` | analyst | — | `{id, name, priority, is_novel, feedback, numbers[], tactics[]{id, name, count}, representative_script (≤200 симв.), sample_incidents[≤60]{id, date, number (префикс), risk, excerpt (≤200 симв.), has_transcript}}` | — |
| `POST /organizations/{org_id}/feedback` | analyst | `{action: confirm\|dismiss\|merge, target_org_id? (только для merge), note? ≤160, без содержимого}` | `OrgSummaryOut` | `org.<action>` **до** записи; заметка с номером или текстом → `422` |
| `GET /incidents/{id}/analysis?locale=&backend=` | analyst | — | `{incident_id, excerpt, risk, threshold, flagged, backend, fallback, tags[], spans[] (только внутри фрагмента), withheld_spans, reason, caveat}` | — ; `backend=llm` → `422` |
| `POST /incidents/{id}/open?locale=&backend=` | **investigator** | `{purpose: pattern_review\|citizen_request\|partner_request, note? ≤160}` | то же плюс `transcript` (полный, очищенный) | `case.open` с `purpose` **до** выдачи; не больше 30 в час (`429` и запись в журнал); журнал недоступен → `503`, ничего не выдаётся; роль analyst → `403` и `access.denied` |
| `POST /ingest?locale=` | analyst | — | `{ingested, organizations_total, placements[]{incident_id, org_id, org_name, org_is_novel}}` | `reports.ingest` |

Для отчёта партнёра без текста `open` и `analysis` отвечают `409`.

## 3. Партнёрский API `/api/v1/*` — [ФАКТ]

**Аутентификация.**
- Заголовок `X-API-Key` с ключом из `QORGAN_PARTNER_API_KEYS` (`partner_id:secret[:daily_quota]`).
- Без ключа или с неверным ключом — `401`. Без ключа цепочки аудита — `503`.
- Лимит — 60 запросов в минуту; суточная квота — скользящие 24 ч, по умолчанию 200.
- Заголовки ответа: `X-Quota-Limit`, `X-Quota-Remaining`.

### 3.1 `POST /api/v1/reports` (`PartnerReportIn`, `extra="forbid"`, один отчёт за запрос)

| Поле | Тип | Правило |
|---|---|---|
| `consent_basis` | string, `^[a-z][a-z0-9_]{2,63}$` | обязательно; код из договора (например, `customer_consent`). [ПРОБЕЛ] M10: с договором не сверяется |
| `tactic_ids` | string[] ≤ 32 | Предпочтительная форма. Нужны тактики и/или текст |
| `transcript` | string ≤ 20 000 \| null | **Должен быть очищен заранее**. Если `scrub_text` его меняет — `422` и запись `rejected:unscrubbed_transcript` |
| `flagged_phrases` | string[] ≤ 50 × 300 | — |
| `phone_number` | string ≤ 32 \| null | хешируется при получении |
| `risk_score` | 0–100 | по умолчанию 100 (подтверждённый случай) |
| `occurred_at` | datetime \| null | в окне хранения, не в будущем (допуск 5 мин) |
| `partner_reference` | `^[A-Za-z0-9._:-]{1,64}$` | Повтор с тем же значением → `200 duplicate`, ничего не сохраняется |

Ответы:
- `201` — `{receipt_id, report_id, status, source, partner_id, partner_reference, consent_basis, number_prefix, stored_transcript, flagged_phrases, tactic_ids, timestamp, quota{limit, used, remaining, window_hours}}`;
- `200` — дубликат;
- `422`;
- `429` — квота исчерпана;
- `503` — нет ключа хеширования или журнал недоступен.

Каждый исход записывается в журнал **до** сохранения.

### 3.2 `DELETE /api/v1/reports/{receipt_id}`

- `204` — удалён свой отчёт, каскадом.
- `404` — чужая квитанция или неизвестная (ответы неотличимы).
- Обе ситуации записываются в журнал.

### 3.3 `GET /api/v1/organizations?locale=`

Ответ: `{available, generated_at, organizations[]{id, name, incidents, tactics[]{id, count}, last_activity, is_novel, priority}}`.
- **Без номеров, без дайджестов, без текстов** (`src/qorgan/api_partner_export.py`).
- Запись в журнал: `organizations.export`.

---

## 4. [ПРЕДЛОЖЕНИЕ] Сервис передачи отчёта (hand-off) — схема v0

> **Ничего из этого раздела не реализовано.** Это проект для обсуждения с НИТ, операторами и
> участником Антифрод-центра. Для публикации через Smart Bridge понадобятся XSD/XML-примеры,
> описание аутентификации и SSL и протоколы испытаний ИБ. Синхронный тайм-аут — 30 с (Приказ
> № 123) [A] (https://zakon.uchet.kz/rus/docs/V1800016777 ). Поэтому JSON ниже будет переведён в
> XSD.

**Почему два сообщения, а не одно.**
- По приказу от 18.11.2025 оператор проверяет **номер** в течение 2 рабочих дней [B]
  (https://informburo.kz/novosti/nomera-s-kotoryx-zvoniat-mosenniki-budut-blokirovat-v-kazaxstane-prikaz-vstupil-v-silu).
  Для этого ему нужен номер в открытом виде.
- Сервер Qorğan открытый номер **никогда не хранит** (инвариант 4, `src/qorgan/privacy/numbers.py`).
- Поэтому жалоба оператору (4.1) уходит **с устройства гражданина**, а сервер Qorğan
  отправляет участникам только сигнал с дайджестом (4.2).

### 4.1 `OperatorComplaint` — с устройства в канал оператора (через eGov Mobile или канал оператора)

```json
{
  "schema": "qorgan.operator_complaint/v0",
  "complaint_type": "suspected_fraud_call",
  "submitted_at": "2026-10-08T10:15:00+05:00",
  "call": {
    "occurred_at": "2026-10-08T09:58:00+05:00",
    "caller_number": "+77000000000",
    "callee_is_complainant": true
  },
  "evidence": {
    "risk_score": 92,
    "tactic_ids": ["impersonation_bank", "otp_request", "secrecy"],
    "verbatim_spans": [
      {"text": "продиктуйте код из СМС", "tactic_id": "otp_request"},
      {"text": "никому не говорите", "tactic_id": "secrecy"}
    ],
    "model": {"name": "qorgan-linear", "weights_sha256": "<sha256 site/models/weights.json>", "threshold": 0.59},
    "ai_notice": "Оценку сделала система ИИ, она может ошибаться; решение принял гражданин."
  },
  "complainant": {
    "verified_by": "egov_qr",
    "signature_cms": "<CMS, если канал оператора требует подтверждения личности>"
  },
  "potential_victim": {
    "flag": false,
    "self_declared_actions": []
  },
  "consent": {"version": "handoff-operator-v1", "given_at": "2026-10-08T10:14:40+05:00"},
  "qorgan_receipt_id": null
}
```

Правила для 4.1:
- **Номер** (`caller_number`) — пример вымышлен. Номер существует только на устройстве и у
  получателя.
- **`qorgan_receipt_id`** заполняется, только если гражданин отдельно отправил отчёт в Qorğan.
- **`potential_victim.self_declared_actions`** — закрытый список того, что гражданин отметил сам:
  `disclosed_otp`, `disclosed_card_data`, `transferred_money`, `installed_remote_access`,
  `took_loan`. Признак ставится **только** по отметке гражданина, никогда моделью. Это
  уровень низкой автономности, пп. 1 п. 2 ст. 17 Закона об ИИ [S].
- **Согласие.** Нужна **новая** версия согласия: передача оператору — это передача третьему лицу
  (п. 6 ст. 7 Закона о ПДн) [S] (`docs/LEGAL_ASSESSMENT.md` §2.1). `report-v1` её не покрывает.
- **Транспорт** — открытый вопрос: deeplink в eGov Mobile, форма оператора или сервис НИТ.
  Это вопрос 9.

### 4.2 `CaseSignal` — с сервера Qorğan участнику Антифрод-центра / МВД (только после проверки аналитиком)

```json
{
  "schema": "qorgan.case_signal/v0",
  "signal_id": "sig_3f9c…",
  "issued_at": "2026-10-09T08:00:00Z",
  "expires_at": "2027-04-07T08:00:00Z",
  "number": {
    "digest": "<HMAC-SHA256(K_recipient, нормализованный номер)[:24]>",
    "digest_scheme": "hmac-sha256-96/recipient-key-v1",
    "prefix": "+7 700 ***"
  },
  "pattern": {
    "tactic_ids": ["impersonation_bank", "otp_request", "safe_account"],
    "verbatim_spans": ["продиктуйте код из СМС", "переведите на безопасный счёт"],
    "risk_score": 92,
    "linked_reports": 4,
    "organization_is_novel": false
  },
  "review": {"status": "confirmed", "reviewed_at": "2026-10-09T07:41:00Z", "reviewer_role": "investigator"},
  "potential_victim": {"flag": false},
  "provenance": {"source": "citizen", "consent_version": "report-v2", "third_party_transfer_consented": true}
}
```

Правила для 4.2:
- **Проверка человеком.** Сигнал уходит только после `confirmed` аналитика — правило «человек
  решает» и п. 19-1 Закона о ПДн [S] (`LEGAL_ASSESSMENT` §5.1).
- **Дайджест для каждого получателя.** Если отдать получателю общий ключ номеров, он сможет
  восстановить любой номер (`LEGAL_ASSESSMENT` §2.3). Поэтому у каждого получателя свой ключ
  `K_recipient`, и дайджест для него вычисляется **в момент получения отчёта**, пока номер ещё
  есть в транзите. Получатель сравнивает дайджест со своими номерами тем же ключом. Это схема
  N1, сопоставление с сохранением приватности по договору с общим ключом. **Следствие:**
  получателей нужно знать заранее, а смена ключа не применяется к старым отчётам.
- **Только отчёты с отдельным согласием на передачу третьим лицам.** Текущий `report-v1` этого
  не покрывает. Партнёрские отчёты (`source=partner`) обратно партнёрам не передаются.
- **`verbatim_spans`** — только фразы, которые уже прошли `scrub_text`. Полный текст не
  передаётся никогда.
- **Канал.** В Антифрод-центре нет входа для внешних систем [A/B]. Получатель — участник (банк
  или оператор) по DPA (`LEGAL_ASSESSMENT` §5.3, M10, N1). Признак «потенциальная жертва»
  ставится только по 4.1.

### 4.3 Предлагаемые маршруты публикации (через ВШЭП)

| Маршрут | Направление | Аутентификация | Аудит |
|---|---|---|---|
| `GET /api/v1/signals?since=` | участник забирает сигналы (pull) | mTLS через ШЭП + `X-API-Key` | `signals.export` |
| `POST /api/v1/signals/{signal_id}/ack` | участник сообщает исход: `number_checked`, `suspended`, `no_action` | то же | `signals.ack` |
| `POST /api/admin/organizations/{org_id}/signal` | следователь выпускает сигнал | `X-Analyst-Key`, роль investigator | `signal.issue` с целью `partner_request` |

Новые записывающие маршруты нужно добавить в `_WRITE_ROUTES` в `tests/test_architecture.py` —
это правило проекта (`CLAUDE.md`, инвариант 6).

## 5. [ПРЕДЛОЖЕНИЕ] Подтверждение отчёта через eGov QR

Последовательность — в [`DATA_FLOW_PII.md`](DATA_FLOW_PII.md) §4. Сервис-источник:
NITEC-S-5096 [A] (https://sb.egov.kz/services/passport/NITEC-S-5096 ) или SIGEX [B]
(https://sigex.kz/blog/2022-12-14-signing-via-qr/ ). Формат вызовов NITEC-S-5096 не публичен,
поэтому поля ниже — наш проект.

| Маршрут | Запрос | Ответ | Правила |
|---|---|---|---|
| `POST /api/reports/{receipt_id}/verify/start` | `{consent_version: "verify-v1"}` | `{request_id, qr_payload, deeplink, expires_at}` | Документ для подписи — SHA-256 от (receipt_id ‖ отпечаток сохранённого отчёта ‖ nonce ‖ срок). Не больше 3 попыток на квитанцию |
| `POST /api/reports/{receipt_id}/verify/complete` | `{request_id}` | `{verified: true\|false, verified_at}` | Сервер проверяет цепочку НУЦ, OCSP / CRL и совпадение документа. ИИН → HMAC отдельным ключом (`QORGAN_IIN_HMAC_KEY`, новый). **ИИН и CMS не хранятся**. Запись в журнал `report.verify` без содержимого |

`DELETE /api/reports/{receipt_id}` удаляет и отметку о проверке.
