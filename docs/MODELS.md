# Модели и ресурсы: где что лежит

Скачивать вручную ничего не нужно: `python scripts/deploy_bootstrap.py` скачивает всё сам
(~690 МБ). Ссылки на Hugging Face и на источники моделей открываются без авторизации.

## Обученные нами веса

| Что | Размер | Где |
|---|---|---|
| **Поставляемая модель** (ADR D54): калиброванная логистическая регрессия риска и 15 классификаторов тактик поверх эмбеддингов e5-base и сигнальных признаков. Её используют и браузер, и сервер | 276 КБ | репозиторий: [`site/models/weights.json`](../site/models/weights.json) |
| Предыдущая версия классификаторов в формате Python (`risk_clf.joblib`, `tactic_clf.joblib`, лексиконы), выложена 2026-09-24 | ~70 КБ | https://huggingface.co/sanzh-ts/govtech |
| Корпус (обезличен), выложен 2026-09-24: train / val / test / authored_heldout / ood / shift | — | https://huggingface.co/datasets/sanzh-ts/govtech_ds |

Bootstrap скачивает бандл с Hugging Face, видит, что лексикон в репозитории с тех пор изменился,
и сервер работает на `site/models/weights.json`. Так и задумано (ADR D55), переобучать ничего
не нужно. Поэтому при оценке выводится строка `models/linear is not usable (... hash mismatch) ...
serving the committed browser weights`, а `/api/health` показывает `"linear_model_source":"web"`.

## Предобученные модели сторонних авторов (без изменений, скачиваются скриптом)

| Что | Размер | Источник | Куда |
|---|---|---|---|
| Эмбеддер `multilingual-e5-base`, int8 ONNX | 278 МБ | https://huggingface.co/Xenova/multilingual-e5-base | `site/models/Xenova/` |
| Распознавание речи Vosk: `vosk-model-small-kz-0.42`, `vosk-model-small-ru-0.22` | 106 МБ | https://alphacephei.com/vosk/models | `site/models/vosk/` |
| Распознавание речи GigaAM multilingual 220M CTC, int8 (необязательно) | 225 МБ | https://huggingface.co/fussraider/GigaAM-Multilingual-sherpa-onnx-ctc (ревизия `9f5a77e`) | `site/models/gigaam/` |

## Запуск и проверка результатов

Проверено на чистом клоне (Python 3.11, macOS, CPU).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && cp .env.example .env
python scripts/deploy_bootstrap.py            # модели + корпус

# 1. Метрики на тестовой выборке (FPR / recall с 95% интервалами, по языкам; ~2 мин на CPU)
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --by-language
# ожидаемо: test, N=115: FPR 0.020 [0.000, 0.104], recall 0.984 [0.916, 1.000]

# 2. Один звонок через API
python -m qorgan.api                          # http://localhost:8000
curl -s localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"transcript":"Это служба безопасности банка. Назовите код из СМС. Никому не говорите.","locale":"ru"}'
# ожидаемо: "risk": 0.99…, "flagged": true, тактики otp_request, impersonation_bank, secrecy, …
```

Интерфейс: http://localhost:8000/live.html (проверка звонка), `/docs` (OpenAPI). Консоль
аналитика `/admin.html` открывается только с ключами в `.env` (см. README, «Быстрый старт»).

Тестовая выборка создана тем же генератором, что и обучающая, поэтому её числа завышены.
Главная оценка — на шести независимых наборах (`shift_local*`, 828 звонков kk/ru): FPR 0.053,
recall 0.455 (ADR D76). Эти наборы не публикуются. Все числа с интервалами — в
[`eval_report.md`](eval_report.md).
