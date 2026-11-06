/* The DOM side of site/i18n.js, shared by the landing (landing.js) and the citizen page
   (live.js). i18n.js stays pure; this is the one place that applies the markup contract:
     data-i18n="key"             -> textContent
     data-i18n-html="key"        -> innerHTML (`_html` keys only; params are escaped)
     data-i18n-attr="a:key;b:k"  -> attributes
     data-i18n-params='{"n":1}'  -> params for all of the above
   The chosen language is one stored value (`STORAGE_KEY`) for every page, so a visitor who
   picks Kazakh on the landing opens the citizen page in Kazakh. */

import { STORAGE_KEY, t as pageT, tHtml as pageTHtml } from "./i18n.js";

const PAGE_TEXT = { t: pageT, tHtml: pageTHtml };

export const I18N_SELECTOR = "[data-i18n],[data-i18n-html],[data-i18n-attr]";

/** Render one element from its data-i18n / data-i18n-html / data-i18n-attr keys. `text` is the
    dictionary's `{t, tHtml}` (the citizen pages' by default; the console has its own). */
export function renderEl(el, locale, { t, tHtml } = PAGE_TEXT) {
  const params = el.dataset.i18nParams ? JSON.parse(el.dataset.i18nParams) : null;
  // A key the loaded strings lack (t() echoes it back) keeps the markup's own text, so a page
  // newer than its i18n.js degrades to Russian instead of showing "landing.title_html".
  const known = (key, value) => value !== key;
  if (el.dataset.i18n) {
    const value = t(locale, el.dataset.i18n, params);
    if (known(el.dataset.i18n, value)) el.textContent = value;
  } else if (el.dataset.i18nHtml) {
    const value = tHtml(locale, el.dataset.i18nHtml, params);
    if (known(el.dataset.i18nHtml, value)) el.innerHTML = value;
  }
  if (el.dataset.i18nAttr) {
    for (const pair of el.dataset.i18nAttr.split(";")) {
      const [attr, key] = pair.split(":");
      const value = t(locale, key, params);
      if (known(key, value)) el.setAttribute(attr, value);
    }
  }
}

/** Render `root` (the document, or one element and its subtree). */
export function localize(root, locale, text = PAGE_TEXT) {
  if (root !== document && root.matches?.(I18N_SELECTOR)) renderEl(root, locale, text);
  root.querySelectorAll(I18N_SELECTOR).forEach((el) => renderEl(el, locale, text));
}

export function readStoredLocale() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

export function storeLocale(value) {
  try { localStorage.setItem(STORAGE_KEY, value); } catch { /* private mode: the choice lasts this visit */ }
}

export const browserLanguages = () => (navigator.languages?.length ? navigator.languages : [navigator.language]);
