# Android build path (ADR D67)

Two containers for the citizen page, for two different questions.

| | `twa/` — Trusted Web Activity | `webview-shell/` — plain WebView |
|---|---|---|
| What it is | the Android demo app: Chrome renders the page | a test harness shaped like eGov Mobile's mini-service |
| Cross-origin isolation | yes (Chrome honours COOP/COEP) | never (Android WebView does not support it) |
| Voice mode the page picks | full on desktop-class devices; on a phone, **light** until the B9 phone bench passes (D25), `?voice=full` to force | **light** (GigaAM-only + energy VAD) |
| Signed with | your upload key (never in this repo) | the SDK's local debug key (debug builds only) |

**Status (2026-10-21):** JDK 17 (`brew install openjdk@17`), the Android SDK (platform-tools,
`platforms;android-35`, `build-tools;35.0.0`, no system images) and a Gradle 8.11.1 wrapper are set
up on the dev Mac. `webview-shell/` **builds** (`./gradlew assembleDebug`, 15 KB debug APK). It has
not been installed on a phone yet. The TWA is **not built**: see "Placeholders" below. The native
text core spike is in `qorgan-core/`. For what it does and its parity numbers, see
[CORE_SPIKE.md](CORE_SPIKE.md).

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=$HOME/Library/Android/sdk
cd android/webview-shell && echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew assembleDebug                                              # start URL http://localhost:8000/live.html
./gradlew assembleDebug -PqorganUrl=https://<host>/live.html         # any https:// URL (or http://localhost)
```

**On a USB-connected phone** (USB debugging on):
```bash
python -m qorgan.api &                       # the Mac serves site/ on :8000
adb reverse tcp:8000 tcp:8000                # the phone's localhost:8000 -> the Mac (a secure context: mic allowed)
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n kz.qorgan.webviewshell/.MainActivity   # debug builds: add --es url <url> to override
```
The shell allows cleartext only to `localhost` / `127.0.0.1`. It stays on the configured origin
and grants the page the microphone only: audio capture, that origin, and only after the
`RECORD_AUDIO` runtime permission. Debug builds enable WebView remote debugging (see below).

## Placeholders to fill in

- `twa/twa-manifest.json`: `host`, `iconUrl`, `webManifestUrl` and `fullScopeUrl` use
  `qorgan.example.kz`. Put the real HTTPS host there (the one serving `site/`).
- `site/.well-known/assetlinks.json`: `sha256_cert_fingerprints` must be the SHA-256 of the
  **app signing key** (Play Console → App integrity, or `keytool -list -v -keystore <file>`).
  While it holds the placeholder, Chrome cannot verify the app and opens the page as a Custom Tab
  with a URL bar. It still runs, and it is still cross-origin isolated.
- The signing key: create it yourself (`bubblewrap init` offers to, or `keytool -genkeypair`),
  store it outside git (`*.keystore` and `*.jks` are gitignored), and keep its passwords out of
  the repo.

## TWA (Bubblewrap)

Not built. What is missing: (1) a public **HTTPS host** serving `site/` with the production
headers, to replace `qorgan.example.kz` (Bubblewrap fetches the web manifest and icons from it);
(2) an **upload keystore** created and kept outside git; (3) its SHA-256 in
`site/.well-known/assetlinks.json`, deployed. Without (3) the app still runs, as a Custom Tab with
a URL bar. `@bubblewrap/cli` itself is not installed (it is a global npm package). The JDK and SDK it
needs are now present: when `bubblewrap` asks, give it the JDK and SDK paths above.

```bash
npm i -g @bubblewrap/cli@1.25.0            # needs JDK 17 + Android SDK; bubblewrap offers to install both
cd android/twa
bubblewrap update --manifest=twa-manifest.json   # generates the Gradle project here from the manifest
bubblewrap build                            # asks for the keystore passwords; writes app-release-signed.apk
bubblewrap fingerprint add <sha256>         # or edit assetlinks.json directly, then deploy site/
adb install app-release-signed.apk
```

The generated Gradle project (`app/`, `gradle/`, `build.gradle`, ...) can be committed once it
exists; the keystore never is.

## Emulator run (functional, not performance)

```bash
brew install openjdk@17 && brew install --cask android-commandlinetools
sdkmanager "platform-tools" "emulator" "platforms;android-35" "build-tools;35.0.0" \
           "system-images;android-35;google_apis_playstore;arm64-v8a"
avdmanager create avd -n qorgan35 -k "system-images;android-35;google_apis_playstore;arm64-v8a" -d pixel_7
emulator -avd qorgan35 -memory 4096 &
node tests_js/tools/static_site.mjs --isolation on  --port 8411 &   # the production headers
node tests_js/tools/static_site.mjs --isolation off --port 8410 &   # WebView never isolates anyway
adb reverse tcp:8411 tcp:8411 && adb reverse tcp:8410 tcp:8410      # emulator localhost = a secure context
```

- **Chrome / TWA:** open `http://localhost:8411/live.html?debug=asr`. Expect
  `crossOriginIsolated === true` and the light mode (phone gate), or the full mode with
  `&voice=full`. A TWA built for `localhost` is never verified, so it shows as a Custom Tab.
- **WebView:** build `webview-shell/` (`./gradlew assembleDebug`), then
  `adb shell am start -n kz.qorgan.webviewshell/.MainActivity --es url http://localhost:8410/live.html?debug=asr`.
  Expect `crossOriginIsolated === false`, the light-mode note, and GigaAM loading.
- **Driving it:** debug builds enable WebView debugging, so
  `adb forward tcp:9222 localabstract:webview_devtools_remote_<pid>` (Chrome:
  `localabstract:chrome_devtools_remote`) lets Playwright `chromium.connectOverCDP` read
  `window.__qorganVoice` and take screenshots. The emulator's microphone is the host's, and only
  if `hw.audioInput=yes`, so speech has to be played into the host mic or replaced in the page
  through CDP.
- Emulator numbers are **functional only**. Memory and latency need a real phone (B9; one
  Firebase Test Lab run is the cheapest real device).
