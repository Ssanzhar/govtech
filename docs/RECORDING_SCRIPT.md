# Recording script — team voice test set (2026-10)

**Why:** every speech number so far comes from read Wikipedia (FLEURS) or synthetic voices. Two
things only your own voices can answer:
1. **Kazakh-accented English** — does GigaAM write it in Latin (fine) or Cyrillic (then we keep a
   separate English recogniser)?
2. **Do the scam warning cues survive real speech through a phone?** (code from SMS, safe account,
   "don't tell anyone", remote-access app …).

**Time:** ~15 minutes per person. **People:** at least 3, ideally 5, mixing men and women and
different accents. Everyone reads Part A; Part B in the languages you speak comfortably.

## How to record
- **Take 1 — "speakerphone":** call a teammate. The teammate puts the call on speakerphone and
  holds their phone ~40 cm from a laptop running a voice recorder (QuickTime → New Audio Recording,
  or Voice Memos). You read the lines into your phone. This is exactly the channel the app hears.
- **Take 2 — "direct" (optional):** read the same lines straight into a laptop or phone recorder.
- Quiet room is fine; don't hunt for perfect silence. Speak naturally, at normal speed — **don't
  read word by word**. A small slip is fine; re-read only if you garbled the whole line.
- Leave ~2 s of silence between lines. One file per part is fine; I'll cut it.
- **File names:** `<initials>_<part>_<take>.<ext>` e.g. `st_A_speaker.m4a`, `ak_B_direct.wav`.
  Any format (m4a, wav, mp3).
- **Where:** a shared folder or send to the person running the experiments. **Not** into git,
  not to Hugging Face, not to any cloud AI service — voices are personal data; they stay on our
  laptops (`data/real_team/`, gitignored) and are deleted after the evaluation if you ask.
- Reading these lines is your consent to use the recordings for **testing only** (never training).

## Part A — English (read with your natural accent; 40 lines)
1. Hello, this is the security department of your bank.
2. We noticed a suspicious payment on your card a few minutes ago.
3. Please tell me the code we just sent to your phone.
4. Do not tell anyone about this call, it is a confidential investigation.
5. You need to move your money to a safe account right now.
6. Please install AnyDesk so our specialist can protect your phone.
7. Can you read me your full card number and the three digits on the back?
8. Your parcel is waiting at the post office, please confirm your address.
9. Congratulations, you have won a prize in our lottery.
10. Your SIM card contract expires today, we need to extend it.
11. Hi mom, I lost my phone, can you send me some money on this number?
12. I'm calling from the police, your son was in an accident.
13. This is your manager, I need you to make an urgent transfer today.
14. Your account will be blocked in ten minutes if you don't act.
15. We can offer you a guaranteed income from crypto trading.
16. Hi, are you coming to the meeting at three?
17. I'm stuck in traffic, I'll be about twenty minutes late.
18. Your order has been delivered, thank you for shopping with us.
19. Your appointment with the doctor is confirmed for Monday morning.
20. Did you send me the report yesterday?
21. Good afternoon, I'm calling about the apartment you posted online.
22. Can you pick up some bread and milk on the way home?
23. The bank will never ask you for your SMS code.
24. Thank you for calling, have a nice day.
25. Let's talk online tomorrow evening.
26. My flight lands at seven, can you meet me at the airport?
27. The payment for the internet is due at the end of the month.
28. Please check your email, I sent you the documents.
29. Your loan has been approved, now transfer the insurance fee.
30. Press one to speak to an operator.
31. Somebody is trying to take a loan in your name.
32. I made a mistake and sent you money, please send it back.
33. We are from the National Bank, your data was stolen.
34. Read me the six digit code, it is only to cancel the operation.
35. Okay, see you later, bye.
36. The courier will come to pick up the cash at your door.
37. Your electronic signature is expiring, we need to renew it.
38. Please keep the line open and don't hang up.
39. I'm calling from the tax office about your business.
40. Sorry, wrong number.

## Part B — Kazakh / Russian / mixed (read the ones you're comfortable with; 40 lines)
Mixed lines switch language mid-sentence on purpose — say them the way people actually talk.

**Kazakh**
1. Сәлеметсіз бе, банктің қауіпсіздік қызметінен хабарласып тұрмын.
2. Телефоныңызға келген SMS-тегі кодты айтыңыз.
3. Бұл туралы ешкімге айтпаңыз, бұл құпия операция.
4. Ақшаңызды қауіпсіз шотқа аударыңыз.
5. Мына бағдарламасын жүктеңіз, біз телефоныңызды қорғаймыз.
6. ЖСН-іңізді айтыңыз, тексеріп жатырмыз.
7. Сізге сәлемдеме келді, курьерге кодты айтыңыз.
8. Апа, ертең дәрігерге барамыз ба?
9. Мен кешігемін, жолда кептеліс бар.
10. Тапсырысыңыз дайын, ертең алып кетіңіз.

**Russian**
11. Здравствуйте, вас беспокоит служба безопасности банка.
12. Продиктуйте код из SMS, это для отмены операции.
13. Никому не говорите, это секретная операция.
14. Переведите деньги на безопасный счёт, так будет надёжнее.
15. Установите приложение AnyDesk, наш специалист поможет.
16. Назовите номер карты и три цифры на обороте.
17. На ваше имя пытаются оформить кредит.
18. Мам, я задержусь, буду через час.
19. Ваш заказ доставлен, спасибо за покупку.
20. Напоминаем о записи к врачу на понедельник.

**Mixed (kk ↔ ru, some English words)**
21. Сәлеметсіз бе, мен банктің службы безопасности қызметкерімін.
22. Телефоныңызға келген код из СМС айтыңызшы.
23. Ақшаңызды безопасный счёт деген шотқа аударыңыз.
24. Бұл туралы ешкімге айтпаңыз, это тайна следствия.
25. Здравствуйте, это Қазпошта, сізге сәлемдеме келді.
26. Апай, не волнуйтесь, бәрі жақсы болады, просто скажите код.
27. Ваш сын в полиции, тез арада переведите деньги.
28. Мына приложение орнатыңыз, это для защиты.
29. Сіздің атыңызға кредит оформили бүгін таңертең.
30. Ұлттық банктен звоним, ваши данные украли.
31. Сіздің online banking есептік жазбаңыз бұғатталды.
32. Маған verification code керек, тез айтыңыз.
33. Мен в магазине тұрмын, не алу керек?
34. Ертең meeting болады, ұмытпа.
35. Жиналыс перенесли на пятницу екен.
36. Скинь мне screenshot, пожалуйста.
37. Әжем передаёт привет барлығыңызға.
38. Отчёт отправил, тексеріп көріңізші.
39. Банктен смс пришло, жалақы түсті.
40. Ну всё, көріскенше, сау бол.

_These lines are test material only. They are never added to training data and never used to
extend the cue lexicon (ADRs D35/D43)._
