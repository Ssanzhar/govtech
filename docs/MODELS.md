# Модели и ресурсы: где что лежит

Все ссылки публичные, без авторизации. Скачивать вручную ничего не нужно: команда
`python scripts/deploy_bootstrap.py` скачивает всё сама (~690 МБ) и проверяет файлы.

## Обученные нами веса

| Что | Размер | Где |
|---|---|---|
| **Поставляемая модель**: калиброванная логистическая регрессия риска и 15 классификаторов тактик поверх эмбеддингов e5-base и сигнальных признаков; используется браузером и сервером | 276 КБ | GitHub: [`site/models/weights.json`](../site/models/weights.json) |
| Те же классификаторы для Python (`risk_clf.joblib`, `tactic_clf.joblib`, `metadata.json`, лексиконы) | ~70 КБ | Hugging Face: https://huggingface.co/sanzh-ts/govtech |
| Корпус: train / val / test / authored_heldout / ood / shift (обезличен) | — | Hugging Face: https://huggingface.co/datasets/sanzh-ts/govtech_ds |

## Предобученные модели сторонних авторов (без изменений, скачиваются скриптом)

| Что | Размер | Источник |
|---|---|---|
| Эмбеддер `multilingual-e5-base`, int8 ONNX | 278 МБ | https://huggingface.co/Xenova/multilingual-e5-base |
| Распознавание речи Vosk: `vosk-model-small-kz-0.42`, `vosk-model-small-ru-0.22` | ~106 МБ | https://alphacephei.com/vosk/models |
| Распознавание речи GigaAM multilingual 220M CTC, int8 (необязательно) | 225 МБ | https://huggingface.co/fussraider/GigaAM-Multilingual-sherpa-onnx-ctc (ревизия `9f5a77e`) |

После bootstrap: эмбеддер — в `site/models/Xenova/`, Vosk — в `site/models/vosk/`,
GigaAM — в `site/models/gigaam/`, Python-бандл — в `models/linear/`.

## Запуск и проверка результатов

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && cp .env.example .env
python scripts/deploy_bootstrap.py            # модели + корпус

# 1. Метрики модели на тестовой выборке (FPR / recall с интервалами, по языкам; ~2 мин на CPU)
QORGAN_CLASSIFIER_BACKEND=linear python -m qorgan.eval.run --split test --by-language

# 2. Один звонок через API
python -m qorgan.api                          # http://localhost:8000
curl -s localhost:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"transcript":"Это служба безопасности банка. Назовите код из СМС. Никому не говорите.","locale":"ru"}'
# → "risk": 0.99…, "flagged": true, тактики otp_request, impersonation_bank, secrecy, …
```

Интерфейс: http://localhost:8000/live.html (проверка звонка), `/admin.html` (консоль
аналитика), `/docs` (OpenAPI). Все метрики с интервалами — в [`eval_report.md`](eval_report.md).
