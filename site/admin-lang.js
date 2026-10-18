/* Analyst console language control -- the same ҚАЗ / РУС / ENG switch as the landing and the
   citizen page, and the same stored choice (i18n-dom.js). It localizes the shared chrome (the
   header and the footer's title block, which also take the language as their `lang`) and drives
   the content language of the console (tactic names, reasons: the `adm-loc` radios that
   admin.js reads), so one control changes the page. The console's own labels stay English
   (a small trained audience; the audit and purpose wording awaits legal review). */

import { DEFAULT_LOCALE, LOCALES, pickLocale } from "./i18n.js";
import { browserLanguages, localize, readStoredLocale, storeLocale } from "./i18n-dom.js";

const SWITCH = 'input[name="adm-lang"]';
const CONTENT = 'input[name="adm-loc"]';
const CHROME = "header.nav, footer";

function localizeChrome(locale) {
  document.querySelectorAll(CHROME).forEach((el) => {
    el.lang = locale;
    localize(el, locale);
  });
}

function apply(next, { persist = true } = {}) {
  const locale = LOCALES.includes(next) ? next : DEFAULT_LOCALE;
  if (persist) storeLocale(locale);
  document.querySelectorAll(SWITCH).forEach((radio) => { radio.checked = radio.value === locale; });
  localizeChrome(locale);
  const content = [...document.querySelectorAll(CONTENT)].find((radio) => radio.value === locale);
  if (content && !content.checked) {
    content.checked = true;
    content.dispatchEvent(new Event("change", { bubbles: true })); // admin.js reloads in it
  }
}

document.querySelectorAll(SWITCH).forEach((radio) =>
  radio.addEventListener("change", () => { if (radio.checked) apply(radio.value); })
);
// The content chips and the header switch are one choice.
document.querySelectorAll(CONTENT).forEach((radio) =>
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    storeLocale(radio.value);
    document.querySelectorAll(SWITCH).forEach((s) => { s.checked = s.value === radio.value; });
    localizeChrome(radio.value);
  })
);
apply(pickLocale({ stored: readStoredLocale(), languages: browserLanguages() }), { persist: false });
