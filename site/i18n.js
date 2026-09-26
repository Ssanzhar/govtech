/* Page strings for the citizen live page (site/live.html) in Kazakh, Russian and English.

   Pure module: no DOM, no storage (live.js owns both). What the MODEL produces -- tactic
   names, advice, explanation templates -- is deliberately NOT here: it comes from the
   reviewed YAML sources (taxonomy, advice_{ru,kk}.yaml, templates_{ru,kk}.yaml) through
   core/qorgan-config.json, and exists in ru/kk only; `contentLocale` maps `en` onto `ru`.

   Conventions (enforced by tests_js/i18n.test.mjs):
   - every locale has exactly the keys of `ru`, none empty, with the same `{placeholders}`;
   - keys ending in `_html` may carry <em>, <b> and <a href="admin.html"> only; every other
     value is plain text;
   - a param may itself be a key, with its own params (`{ $: "lang_name.kk" }`,
     `{ $: "report.err_status", params: { status: 500 } }`), or a list of them (joined "; "),
     so a stored message re-renders in whatever language is chosen later. */

export const LOCALES = Object.freeze(["kk", "ru", "en"]); // display order (state language first)
export const DEFAULT_LOCALE = "ru";
export const STORAGE_KEY = "qorgan.locale";
export const CONTENT_LOCALES = Object.freeze(["ru", "kk"]); // what the YAML sources exist in
// Version of the consent wording (`report.consent*` keys, all locales), sent with every report so
// the server stores what the citizen agreed to. New wording needs a new version, registered with
// its digest in src/qorgan/reports/model.py (tests/reports/test_consent_versions.py checks both).
export const CONSENT_VERSION = "report-v1";

const ru = {
  "meta.title": "Qorğan — проверка звонка",
  "meta.description": "Проверьте, не звонят ли вам мошенники. Анализ идёт на вашем устройстве, решение всегда за вами.",

  "nav.home_aria": "Qorğan — на главную",
  "nav.primary_aria": "Основное меню",
  "nav.how": "Как это работает",
  "nav.analyst": "Кабинет аналитика",
  "nav.live": "Проверка звонка",

  "hero.eyebrow": "Проверка звонка · на вашем устройстве",
  "hero.title_html": "Не спешите. <em>Проверьте звонок.</em>",
  "hero.lede": "Qorğan слушает разговор по громкой связи (или разбирает пример звонка) и по ходу подсказывает, похоже ли это на мошенничество и что делать.",

  "promise.private_title": "Разговор остаётся у вас",
  "promise.private_body": "Речь и текст обрабатываются прямо на этом устройстве. Сервер не принимает звук. Текст уходит, только если вы сами отправите сообщение о звонке.",
  "promise.human_title": "Решаете вы",
  "promise.human_body": "Qorğan только подсказывает. Он сам не кладёт трубку, ничего не блокирует и никому не сообщает. Оценку даёт искусственный интеллект — он может ошибаться.",
  "promise.why_title": "Объясняет почему",
  "promise.why_body": "Показывает слова звонящего, которые насторожили, и что делать дальше.",

  "how.summary": "Как это работает — подробнее",
  "how.pipeline_aria": "Схема: речь, окно разговора, оценка риска, шкала от 0 до 100, совет",
  "how.stage_speech": "речь",
  "how.stage_window": "окно разговора",
  "how.stage_score": "оценка риска",
  "how.stage_meter": "шкала 0–100",
  "how.stage_advice": "совет",
  "how.step1": "Речь распознаётся в браузере: две небольшие модели Vosk (казахская и русская) работают параллельно, для каждой фразы берётся более уверенный вариант.",
  "how.step2": "Каждая новая фраза добавляется к «окну» разговора (его начало и последние фразы); нейросеть e5 прямо на устройстве превращает его в набор чисел.",
  "how.step3": "Классификатор оценивает риск и находит признаки — 15 приёмов мошенников, например просьбу назвать код из SMS.",
  "how.step4": "Шкала 0–100 меняется плавно, чтобы одна фраза не вызывала ложную тревогу; сильные признаки (код из SMS, «безопасный счёт») поднимают её сразу.",
  "how.step5": "Советы и объяснения не пишет генеративная модель: они берутся из проверенных шаблонов на казахском и русском языках.",

  "stage.aria": "Проверка звонка",
  "stage.eyebrow": "Проверка",
  "stage.title_html": "Пример или <em>живой звонок</em>",
  "mode.aria": "Способ проверки",
  "mode.replay": "Пример звонка",
  "mode.mic": "Микрофон",

  "replay.scenario_label": "Пример",
  "replay.custom": "Свой текст — вставьте ниже",
  "scenario.live_scam_bank_ru": "Мошенник «из банка» (на русском)",
  "scenario.live_scam_bank_kk": "Мошенник «из банка» (на казахском)",
  "scenario.live_hard_negative_bank_ru": "Настоящий звонок из банка (на русском)",
  "replay.script_label": "Слова звонящего — каждая фраза с новой строки",
  "replay.script_placeholder": "Например: Это служба безопасности вашего банка…",
  "replay.start": "Начать проверку",
  "replay.running": "Идёт проверка…",
  "replay.empty": "Выберите пример или напишите хотя бы одну фразу.",
  "replay.failed": "Проверка не удалась ({error}). Возможно, модель ещё загружается — попробуйте ещё раз.",
  "replay.scenarios_failed": "Не удалось загрузить примеры ({error}). Вставьте свой текст.",

  "download.model_first": "При первом запуске загрузится модель анализа — около {mb} МБ. Это один раз: потом она хранится на этом устройстве. Лучше через Wi-Fi.",
  "download.model_cached": "Модель анализа уже на этом устройстве — загружать ничего не нужно.",
  "download.speech_first": "При первом включении микрофона загрузятся модели распознавания речи — около {mb} МБ, один раз; потом они хранятся на этом устройстве.",
  "download.progress": "Загружаем модель анализа: {pct}% (около {mb} МБ, только в первый раз)…",
  "model.preparing": "Готовим модель анализа…",
  "model.ready": "Модель готова. Разговор анализируется только на этом устройстве.",

  "mic.copy": "Включите на телефоне громкую связь и положите его рядом с компьютером. Qorğan слушает через микрофон компьютера и распознаёт казахскую и русскую речь прямо здесь — звук никуда не отправляется и не записывается. Пока это работает только в браузере на компьютере.",
  "mic.start": "Включить микрофон",
  "mic.stop": "Завершить звонок",
  "mic.ready": "Распознавание речи работает на этом устройстве — звук не покидает браузер.",
  "mic.unavailable": "Микрофон здесь недоступен: {reasons}.",
  "mic.reason_browser": "этот браузер не умеет распознавать речь на устройстве — откройте страницу в Chrome на компьютере",
  "mic.reason_capture": "у браузера нет доступа к микрофону",
  "mic.reason_phone": "на телефонах этот режим пока выключен: мы ещё не проверили, как он там работает",
  "mic.loading_runtime": "Загружаем распознавание речи…",
  "mic.loading_model": "Загружаем модель речи ({language}) — только в первый раз…",
  "mic.listening": "Слушаем. Звук остаётся на этом устройстве.",
  "mic.stopped": "Микрофон выключен.",
  "mic.permission": "Разрешите доступ к микрофону в окне браузера…",
  "mic.start_failed": "Не удалось включить микрофон ({error}).",
  "mic.no_speech": "Речь не распознана. Проверьте громкость и попробуйте ещё раз.",
  "mic.analysis_failed": "Не удалось проанализировать фразу ({error}).",
  "mic.recognition_error": "Ошибка распознавания речи ({error}).",
  "mic.stop_failed": "Звонок завершился с ошибкой ({error}).",

  "lang_name.kk": "казахский",
  "lang_name.ru": "русский",
  "lang_short.kk": "каз",
  "lang_short.ru": "рус",

  "call.head": "Звонок · фраз: {n}",
  "call.meter_label": "Уровень подозрения",
  "call.meter_valuetext": "{score} из 100 — {band}",
  "call.signs_label": "Замеченные признаки",
  "call.advice_title": "Что делать",
  "call.transcript_label": "Текст разговора",
  "call.line_meta": "{language} · уверенность {pct}%",
  "call.content_fallback": "Советы и объяснения показаны на русском языке.",

  "band.low": "Низкий",
  "band.medium": "Средний",
  "band.high": "Высокий",
  "band.critical": "Очень высокий",
  "band.low_desc": "явных признаков мошенничества нет",
  "band.medium_desc": "будьте внимательны — есть тревожные признаки",
  "band.high_desc": "похоже на мошенничество",
  "band.critical_desc": "сильные признаки мошенничества — не выполняйте просьбы звонящего",
  "band.pill": "{band}: {desc}",

  "summary.eyebrow": "Итоги звонка",
  "summary.signs": "Замеченные признаки",
  "summary.none": "не замечены",
  "summary.why": "Почему такая оценка",
  "summary.actions": "Что делать",
  "summary.safe_tip": "Если сомневаетесь, положите трубку и сами перезвоните в организацию по официальному номеру.",

  "report.title": "Сообщить об этом звонке?",
  "report.intro": "Сообщения помогают аналитикам находить группы мошенников. Мы ничего не отправим, пока вы не проверите сообщение и сами не нажмёте «Отправить». Любую часть можно удалить, а отправленное сообщение — удалить позже по номеру квитанции.",
  "report.open": "Посмотреть сообщение",
  "report.prepare_failed": "Не удалось подготовить сообщение ({error}).",
  "report.text_label": "Текст разговора — исправьте или удалите то, что не хотите отправлять",
  "report.preview_label": "Так текст будет сохранён: номера телефонов и карт, ИИН и e-mail скрыты",
  "report.tactics_label": "Замеченные признаки — снимите галочку, если не согласны",
  "report.phone_label": "Номер звонившего (необязательно)",
  "report.phone_help": "Номер не хранится целиком: только в защищённом виде, из которого его нельзя восстановить, и первые цифры, например +7 700 ***.",
  "report.consent": "Я проверил(а) сообщение и согласен(на) его отправить.",
  "report.send": "Отправить",
  "report.sent_html": "Отправлено. Номер квитанции: <b>{receipt}</b> — сохраните его, чтобы удалить сообщение позже. Номер звонившего сохранён как <b>{prefix}</b>. Сохранённый текст:",
  "report.queued_html": "Сообщение ждёт проверки в <a href=\"admin.html\">очереди аналитика</a>.",
  "report.delete": "Удалить моё сообщение",
  "report.deleted": "Сообщение удалено: его больше нет ни в очереди, ни в анализе.",
  "report.send_failed": "Не удалось отправить ({error}).",
  "report.delete_failed": "Не удалось удалить ({error}).",
  "report.err_rejected": "сервер отклонил сообщение: {detail}",
  "report.err_numbers": "сервер сейчас не принимает номера телефонов — уберите номер и отправьте снова",
  "report.err_rate": "слишком много сообщений с этого устройства — попробуйте через минуту",
  "report.err_status": "сервер ответил ошибкой {status}",
  "report.err_gone": "сообщение уже удалено",

  "footer.promise": "Анализ на вашем устройстве · без вашего согласия ничего не отправляется · решение всегда за вами",
  "footer.tagline": "Qorğan — распознавание приёмов телефонных мошенников · қаз · рус",
  "footer.aria": "Нижнее меню",
  "footer.home": "Главная",
  "footer.eval": "Оценка качества",
};

const kk = {
  "meta.title": "Qorğan — қоңырауды тексеру",
  "meta.description": "Сізге алаяқтар қоңырау шалып тұрған жоқ па — тексеріңіз. Талдау құрылғыңызда жүреді, шешімді әрқашан өзіңіз қабылдайсыз.",

  "nav.home_aria": "Qorğan — басты бет",
  "nav.primary_aria": "Негізгі мәзір",
  "nav.how": "Бұл қалай жұмыс істейді",
  "nav.analyst": "Талдаушы кабинеті",
  "nav.live": "Қоңырауды тексеру",

  "hero.eyebrow": "Қоңырауды тексеру · өз құрылғыңызда",
  "hero.title_html": "Асықпаңыз. <em>Қоңырауды тексеріңіз.</em>",
  "hero.lede": "Qorğan әңгімені динамик арқылы тыңдайды (немесе қоңырау үлгісін талдайды) және әңгіме барысында оның алаяқтыққа ұқсайтынын-ұқсамайтынын және не істеу керегін айтып отырады.",

  "promise.private_title": "Әңгіме өзіңізде қалады",
  "promise.private_body": "Сөз бен мәтін осы құрылғының өзінде өңделеді. Сервер дыбыс қабылдамайды. Мәтін қоңырау туралы хабарламаны өзіңіз жіберген жағдайда ғана жіберіледі.",
  "promise.human_title": "Шешімді сіз қабылдайсыз",
  "promise.human_body": "Qorğan тек кеңес береді. Ол трубканы өзі қоймайды, ештеңені бұғаттамайды және ешкімге өзі хабарламайды. Бағаны жасанды интеллект береді — ол қателесуі мүмкін.",
  "promise.why_title": "Себебін түсіндіреді",
  "promise.why_body": "Күдік тудырған сөздерді көрсетеді және әрі қарай не істеу керегін айтады.",

  "how.summary": "Бұл қалай жұмыс істейді — толығырақ",
  "how.pipeline_aria": "Сызба: сөз, әңгіме терезесі, тәуекел бағасы, 0-ден 100-ге дейінгі шкала, кеңес",
  "how.stage_speech": "сөз",
  "how.stage_window": "әңгіме терезесі",
  "how.stage_score": "тәуекел бағасы",
  "how.stage_meter": "0–100 шкаласы",
  "how.stage_advice": "кеңес",
  "how.step1": "Сөз браузерде танылады: екі шағын Vosk моделі (қазақ және орыс тілдері) қатар жұмыс істейді, әр фраза үшін сенімдірек нұсқа алынады.",
  "how.step2": "Әр жаңа фраза әңгіменің «терезесіне» (басы мен соңғы фразалары) қосылады; e5 нейрожелісі оны осы құрылғының өзінде сандар жиынына айналдырады.",
  "how.step3": "Классификатор тәуекелді бағалап, белгілерді табады — алаяқтардың 15 тәсілі, мысалы, SMS-тегі кодты сұрау.",
  "how.step4": "0–100 шкаласы бірқалыпты өзгереді, сондықтан жалғыз фраза жалған дабыл тудырмайды; күшті белгілер (SMS-тегі код, «қауіпсіз шот») оны бірден көтереді.",
  "how.step5": "Кеңестер мен түсіндірмелерді генеративтік модель жазбайды: олар қазақ және орыс тілдеріндегі тексерілген үлгілерден алынады.",

  "stage.aria": "Қоңырауды тексеру",
  "stage.eyebrow": "Тексеру",
  "stage.title_html": "Үлгі немесе <em>нақты қоңырау</em>",
  "mode.aria": "Тексеру тәсілі",
  "mode.replay": "Қоңырау үлгісі",
  "mode.mic": "Микрофон",

  "replay.scenario_label": "Үлгі",
  "replay.custom": "Өз мәтініңіз — төменге қойыңыз",
  "scenario.live_scam_bank_ru": "«Банктен» хабарласқан алаяқ (орыс тілінде)",
  "scenario.live_scam_bank_kk": "«Банктен» хабарласқан алаяқ (қазақ тілінде)",
  "scenario.live_hard_negative_bank_ru": "Банктен келген шынайы қоңырау (орыс тілінде)",
  "replay.script_label": "Қоңырау шалушының сөздері — әр фраза жаңа жолдан",
  "replay.script_placeholder": "Мысалы: Бұл банктің қауіпсіздік қызметі…",
  "replay.start": "Тексеруді бастау",
  "replay.running": "Тексеріліп жатыр…",
  "replay.empty": "Үлгіні таңдаңыз немесе кемінде бір фраза жазыңыз.",
  "replay.failed": "Тексеру сәтсіз аяқталды ({error}). Модель әлі жүктеліп жатқан болуы мүмкін — қайталап көріңіз.",
  "replay.scenarios_failed": "Үлгілер жүктелмеді ({error}). Өз мәтініңізді қойыңыз.",

  "download.model_first": "Алғаш іске қосқанда талдау моделі жүктеледі — шамамен {mb} МБ. Бұл бір рет қана болады: кейін модель осы құрылғыда сақталады. Wi-Fi арқылы жүктеген дұрыс.",
  "download.model_cached": "Талдау моделі осы құрылғыда бар — ештеңе жүктеудің қажеті жоқ.",
  "download.speech_first": "Микрофонды алғаш қосқанда сөзді тану модельдері жүктеледі — шамамен {mb} МБ, бір рет қана; кейін олар осы құрылғыда сақталады.",
  "download.progress": "Талдау моделі жүктелуде: {pct}% (шамамен {mb} МБ, тек алғаш рет)…",
  "model.preparing": "Талдау моделі дайындалуда…",
  "model.ready": "Модель дайын. Әңгіме тек осы құрылғыда талданады.",

  "mic.copy": "Телефоныңызда динамикті қосып, оны компьютердің жанына қойыңыз. Qorğan компьютердің микрофоны арқылы тыңдап, қазақ және орыс тіліндегі сөзді осы жерде таниды — дыбыс ешқайда жіберілмейді және жазылмайды. Әзірге бұл тек компьютердегі браузерде жұмыс істейді.",
  "mic.start": "Микрофонды қосу",
  "mic.stop": "Қоңырауды аяқтау",
  "mic.ready": "Сөзді тану осы құрылғыда жұмыс істейді — дыбыс браузерден шықпайды.",
  "mic.unavailable": "Микрофон мұнда қолжетімсіз: {reasons}.",
  "mic.reason_browser": "бұл браузер сөзді құрылғының өзінде тани алмайды — бетті компьютердегі Chrome браузерінде ашыңыз",
  "mic.reason_capture": "браузердің микрофонға қолжетімі жоқ",
  "mic.reason_phone": "телефондарда бұл режим әзірге өшірулі: оның телефонда қалай жұмыс істейтінін әлі тексерген жоқпыз",
  "mic.loading_runtime": "Сөзді тану жүктелуде…",
  "mic.loading_model": "Сөз моделі жүктелуде ({language}) — тек алғаш рет…",
  "mic.listening": "Тыңдап тұрмыз. Дыбыс осы құрылғыда қалады.",
  "mic.stopped": "Микрофон өшірілді.",
  "mic.permission": "Браузер терезесінде микрофонға рұқсат беріңіз…",
  "mic.start_failed": "Микрофон қосылмады ({error}).",
  "mic.no_speech": "Сөз танылмады. Дыбыс деңгейін тексеріп, қайталап көріңіз.",
  "mic.analysis_failed": "Фразаны талдау сәтсіз аяқталды ({error}).",
  "mic.recognition_error": "Сөзді тану қатесі ({error}).",
  "mic.stop_failed": "Қоңырау қатемен аяқталды ({error}).",

  "lang_name.kk": "қазақ тілі",
  "lang_name.ru": "орыс тілі",
  "lang_short.kk": "қаз",
  "lang_short.ru": "орыс",

  "call.head": "Қоңырау · фразалар саны: {n}",
  "call.meter_label": "Күдік деңгейі",
  "call.meter_valuetext": "100-ден {score} — {band}",
  "call.signs_label": "Байқалған белгілер",
  "call.advice_title": "Не істеу керек",
  "call.transcript_label": "Әңгіме мәтіні",
  "call.line_meta": "{language} · сенімділік {pct}%",
  "call.content_fallback": "Кеңестер мен түсіндірмелер орыс тілінде көрсетілген.",

  "band.low": "Төмен",
  "band.medium": "Орташа",
  "band.high": "Жоғары",
  "band.critical": "Өте жоғары",
  "band.low_desc": "алаяқтықтың айқын белгілері жоқ",
  "band.medium_desc": "абай болыңыз — күдікті белгілер бар",
  "band.high_desc": "алаяқтыққа ұқсайды",
  "band.critical_desc": "алаяқтықтың күшті белгілері — қоңырау шалушының өтініштерін орындамаңыз",
  "band.pill": "{band}: {desc}",

  "summary.eyebrow": "Қоңырау қорытындысы",
  "summary.signs": "Байқалған белгілер",
  "summary.none": "байқалмады",
  "summary.why": "Неліктен бұл баға",
  "summary.actions": "Не істеу керек",
  "summary.safe_tip": "Күмәндансаңыз, трубканы қойып, ұйымға ресми нөмір арқылы өзіңіз қоңырау шалыңыз.",

  "report.title": "Бұл қоңырау туралы хабарлайсыз ба?",
  "report.intro": "Хабарламалар талдаушыларға алаяқтар топтарын табуға көмектеседі. Хабарламаны тексеріп, «Жіберу» түймесін өзіңіз баспайынша, ештеңе жіберілмейді. Кез келген бөлігін өшіре аласыз, ал жіберілген хабарламаны кейін түбіртек нөмірі арқылы жоя аласыз.",
  "report.open": "Хабарламаны қарау",
  "report.prepare_failed": "Хабарлама дайындалмады ({error}).",
  "report.text_label": "Әңгіме мәтіні — жібергіңіз келмейтін жерін түзетіңіз немесе өшіріңіз",
  "report.preview_label": "Мәтін осылай сақталады: телефон және карта нөмірлері, ЖСН мен e-mail жасырылған",
  "report.tactics_label": "Байқалған белгілер — келіспесеңіз, құсбелгіні алып тастаңыз",
  "report.phone_label": "Қоңырау шалған нөмір (міндетті емес)",
  "report.phone_help": "Нөмір толық сақталмайды: тек қалпына келтіруге болмайтын қорғалған түрі және алғашқы цифрлары ғана сақталады, мысалы +7 700 ***.",
  "report.consent": "Хабарламаны тексердім және оны жіберуге келісемін.",
  "report.send": "Жіберу",
  "report.sent_html": "Жіберілді. Түбіртек нөмірі: <b>{receipt}</b> — хабарламаны кейін жою үшін оны сақтап қойыңыз. Қоңырау шалған нөмір <b>{prefix}</b> түрінде сақталды. Сақталған мәтін:",
  "report.queued_html": "Хабарлама <a href=\"admin.html\">талдаушы кезегінде</a> тексеруді күтуде.",
  "report.delete": "Хабарламамды жою",
  "report.deleted": "Хабарлама жойылды: ол енді кезекте де, талдауда да жоқ.",
  "report.send_failed": "Жіберілмеді ({error}).",
  "report.delete_failed": "Жойылмады ({error}).",
  "report.err_rejected": "сервер хабарламаны қабылдамады: {detail}",
  "report.err_numbers": "сервер қазір телефон нөмірлерін қабылдамайды — нөмірді алып тастап, қайта жіберіңіз",
  "report.err_rate": "бұл құрылғыдан хабарлама тым көп — бір минуттан кейін қайталаңыз",
  "report.err_status": "сервер {status} қатесін қайтарды",
  "report.err_gone": "хабарлама бұрын жойылған",

  "footer.promise": "Талдау құрылғыңызда жүреді · келісіміңізсіз ештеңе жіберілмейді · шешімді әрқашан өзіңіз қабылдайсыз",
  "footer.tagline": "Qorğan — телефон алаяқтарының тәсілдерін анықтау · қаз · орыс",
  "footer.aria": "Төменгі мәзір",
  "footer.home": "Басты бет",
  "footer.eval": "Сапаны бағалау",
};

const en = {
  "meta.title": "Qorğan — check a call",
  "meta.description": "Check whether a call is a scam. The analysis runs on your device, and you always decide.",

  "nav.home_aria": "Qorğan — home",
  "nav.primary_aria": "Primary",
  "nav.how": "How it works",
  "nav.analyst": "Analyst console",
  "nav.live": "Check a call",

  "hero.eyebrow": "Call check · on your device",
  "hero.title_html": "Don’t rush. <em>Check the call.</em>",
  "hero.lede": "Qorğan listens to the call on speakerphone (or walks through a sample call) and tells you, as it goes, whether it looks like a scam and what to do.",

  "promise.private_title": "The call stays with you",
  "promise.private_body": "Speech and text are processed right on this device. The server never accepts audio. Text leaves only if you choose to send a report about the call.",
  "promise.human_title": "You decide",
  "promise.human_body": "Qorğan only advises. It never hangs up, blocks or reports anything by itself. The assessment comes from artificial intelligence and can be wrong.",
  "promise.why_title": "It shows why",
  "promise.why_body": "It highlights the caller’s words that raised concern and tells you what to do next.",

  "how.summary": "How it works — details",
  "how.pipeline_aria": "Diagram: speech, rolling window, risk score, 0 to 100 meter, advice",
  "how.stage_speech": "speech",
  "how.stage_window": "rolling window",
  "how.stage_score": "risk score",
  "how.stage_meter": "meter 0–100",
  "how.stage_advice": "advice",
  "how.step1": "Speech is recognised in the browser: two small Vosk models (Kazakh and Russian) run side by side and the more confident one wins for each phrase.",
  "how.step2": "Each new phrase joins a rolling window of the call (its opening and latest phrases); the e5 network turns it into numbers right on this device.",
  "how.step3": "A classifier scores the risk and finds the signs — 15 scam tactics, such as asking for the code from an SMS.",
  "how.step4": "The 0–100 meter moves smoothly so a single phrase cannot raise a false alarm; strong signs (an SMS code, a “safe account”) lift it at once.",
  "how.step5": "Advice and explanations are not written by a generative model: they come from reviewed templates in Kazakh and Russian.",

  "stage.aria": "Call check",
  "stage.eyebrow": "Check",
  "stage.title_html": "A sample or <em>a live call</em>",
  "mode.aria": "How to check",
  "mode.replay": "Sample call",
  "mode.mic": "Microphone",

  "replay.scenario_label": "Sample",
  "replay.custom": "Your own text — paste below",
  "scenario.live_scam_bank_ru": "Fake “bank security” call (Russian)",
  "scenario.live_scam_bank_kk": "Fake “bank security” call (Kazakh)",
  "scenario.live_hard_negative_bank_ru": "Genuine call from a bank (Russian)",
  "replay.script_label": "The caller’s words — one phrase per line",
  "replay.script_placeholder": "For example: This is your bank’s security service…",
  "replay.start": "Start the check",
  "replay.running": "Checking…",
  "replay.empty": "Pick a sample or write at least one phrase.",
  "replay.failed": "The check failed ({error}). The model may still be downloading — please try again.",
  "replay.scenarios_failed": "Could not load the samples ({error}). Paste your own text.",

  "download.model_first": "The first check downloads the analysis model — about {mb} MB. This happens once; afterwards it stays on this device. Wi-Fi is best.",
  "download.model_cached": "The analysis model is already on this device — nothing to download.",
  "download.speech_first": "The first time you turn on the microphone, the speech-recognition models download — about {mb} MB, once; afterwards they stay on this device.",
  "download.progress": "Downloading the analysis model: {pct}% (about {mb} MB, first time only)…",
  "model.preparing": "Preparing the analysis model…",
  "model.ready": "Model ready. The call is analysed only on this device.",

  "mic.copy": "Put your phone on speaker next to the computer. Qorğan listens through the computer’s microphone and recognises Kazakh and Russian speech right here — audio is never sent anywhere or recorded. For now this works only in a desktop browser.",
  "mic.start": "Turn on the microphone",
  "mic.stop": "End the call",
  "mic.ready": "Speech recognition runs on this device — audio never leaves the browser.",
  "mic.unavailable": "The microphone is not available here: {reasons}.",
  "mic.reason_browser": "this browser cannot recognise speech on the device — open the page in Chrome on a computer",
  "mic.reason_capture": "the browser has no access to a microphone",
  "mic.reason_phone": "on phones this mode is off for now: we have not yet tested how it performs there",
  "mic.loading_runtime": "Loading speech recognition…",
  "mic.loading_model": "Loading the speech model ({language}) — first time only…",
  "mic.listening": "Listening. Audio stays on this device.",
  "mic.stopped": "Microphone off.",
  "mic.permission": "Allow microphone access in the browser prompt…",
  "mic.start_failed": "Could not turn on the microphone ({error}).",
  "mic.no_speech": "No speech was recognised. Check the volume and try again.",
  "mic.analysis_failed": "Could not analyse a phrase ({error}).",
  "mic.recognition_error": "Speech recognition error ({error}).",
  "mic.stop_failed": "The call ended with an error ({error}).",

  "lang_name.kk": "Kazakh",
  "lang_name.ru": "Russian",
  "lang_short.kk": "KK",
  "lang_short.ru": "RU",

  "call.head": "Call · phrases: {n}",
  "call.meter_label": "Suspicion level",
  "call.meter_valuetext": "{score} out of 100 — {band}",
  "call.signs_label": "Signs noticed",
  "call.advice_title": "What to do",
  "call.transcript_label": "Call transcript",
  "call.line_meta": "{language} · {pct}% confident",
  "call.content_fallback": "Advice and explanations are shown in Russian: they are written and reviewed in Kazakh and Russian only.",

  "band.low": "Low",
  "band.medium": "Medium",
  "band.high": "High",
  "band.critical": "Very high",
  "band.low_desc": "no clear signs of a scam",
  "band.medium_desc": "be careful — there are warning signs",
  "band.high_desc": "this looks like a scam",
  "band.critical_desc": "strong signs of a scam — do not do what the caller asks",
  "band.pill": "{band}: {desc}",

  "summary.eyebrow": "After the call",
  "summary.signs": "Signs noticed",
  "summary.none": "none",
  "summary.why": "Why this score",
  "summary.actions": "What to do",
  "summary.safe_tip": "If in doubt, hang up and call the organisation back yourself on its official number.",

  "report.title": "Report this call?",
  "report.intro": "Reports help analysts find scam groups. Nothing is sent until you have checked the report and pressed “Send” yourself. You can remove any part of it, and delete a sent report later with its receipt number.",
  "report.open": "Review the report",
  "report.prepare_failed": "Could not prepare the report ({error}).",
  "report.text_label": "Call transcript — edit or delete anything you do not want to send",
  "report.preview_label": "This is what will be stored: phone and card numbers, IINs and e-mails are hidden",
  "report.tactics_label": "Signs noticed — untick any you disagree with",
  "report.phone_label": "Caller’s number (optional)",
  "report.phone_help": "The number is never stored in full — only in a protected form it cannot be recovered from, plus its first digits, e.g. +7 700 ***.",
  "report.consent": "I have checked this report and agree to send it.",
  "report.send": "Send",
  "report.sent_html": "Sent. Receipt number: <b>{receipt}</b> — keep it to delete the report later. The caller’s number is stored as <b>{prefix}</b>. Stored text:",
  "report.queued_html": "It is now waiting in the <a href=\"admin.html\">analyst queue</a>.",
  "report.delete": "Delete my report",
  "report.deleted": "Report deleted — it is gone from the queue and from the analysis.",
  "report.send_failed": "Could not send ({error}).",
  "report.delete_failed": "Could not delete ({error}).",
  "report.err_rejected": "the server rejected the report: {detail}",
  "report.err_numbers": "the server does not accept caller numbers right now — remove the number and send again",
  "report.err_rate": "too many reports from this device — try again in a minute",
  "report.err_status": "the server answered with error {status}",
  "report.err_gone": "the report was already deleted",

  "footer.promise": "Analysis on your device · nothing is sent without your consent · you always decide",
  "footer.tagline": "Qorğan — phone-scam pattern detection · kk · ru",
  "footer.aria": "Footer",
  "footer.home": "Home",
  "footer.eval": "Evaluation",
};

export const STRINGS = Object.freeze({ ru: Object.freeze(ru), kk: Object.freeze(kk), en: Object.freeze(en) });

/** Stored choice if valid, else the first kk/ru browser language, else the default (ru). */
export function pickLocale({ stored = null, languages = [] } = {}) {
  if (LOCALES.includes(stored)) return stored;
  for (const tag of languages || []) {
    const primary = String(tag || "").toLowerCase().split(/[-_]/)[0];
    if (primary === "kk" || primary === "ru") return primary;
  }
  return DEFAULT_LOCALE;
}

/** The locale the model-produced content (advice, tactic names, templates) is shown in. */
export function contentLocale(locale, available = CONTENT_LOCALES) {
  return available.includes(locale) ? locale : DEFAULT_LOCALE;
}

/** `{name}` placeholders of a template, sorted and de-duplicated. */
export function placeholdersOf(template) {
  return [...new Set([...String(template).matchAll(/\{([a-z_]+)\}/g)].map((m) => m[1]))].sort();
}

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) => ESCAPES[ch]);

function lookup(locale, key) {
  return STRINGS[locale]?.[key] ?? STRINGS[DEFAULT_LOCALE][key] ?? key;
}

function resolveParam(locale, value) {
  if (Array.isArray(value)) return value.map((v) => resolveParam(locale, v)).join("; ");
  if (value && typeof value === "object" && typeof value.$ === "string") return t(locale, value.$, value.params ?? null);
  return String(value ?? "");
}

/** Does the page have a string for `key`? */
export const hasKey = (key) => Object.hasOwn(STRINGS[DEFAULT_LOCALE], key);

function fill(locale, template, params, encode) {
  return template.replace(/\{([a-z_]+)\}/g, (whole, name) =>
    params && Object.hasOwn(params, name) ? encode(resolveParam(locale, params[name])) : whole);
}

/** Plain-text string for `key` (falls back to ru, then to the key itself). */
export function t(locale, key, params = null) {
  return fill(locale, lookup(locale, key), params, (s) => s);
}

/** Markup string for an `_html` key; params are escaped. */
export function tHtml(locale, key, params = null) {
  return fill(locale, lookup(locale, key), params, escapeHtml);
}
