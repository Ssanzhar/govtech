# Android build path (ADR D67)

Two containers for the citizen page, for two different questions.

| | `twa/` — Trusted Web Activity | `webview-shell/` — plain WebView |
|---|---|---|
| What it is | the Android demo app: Chrome renders the page | a test harness shaped like eGov Mobile's mini-service |
| Cross-origin isolation | yes (Chrome honours COOP/COEP) | never (Android WebView does not support it) |
| Voice mode the page picks | full on desktop-class devices; on a phone, **light** until the B9 phone bench passes (D25), `?voice=full` to force | **light** (GigaAM-only + energy VAD) |
| Signed with | your upload key (never in this repo) | the SDK's local debug key (debug builds only) |

Nothing here was built or run on this machine (2026-10-09): the download link was ~40 KB/s, so
a JDK, the Android SDK and a system image (several GB) would have taken many hours. The
browser evidence for the light mode is from desktop Chromium with isolation off
(`docs/research/2026-10/voice_d67/`). The steps below are what a functional emulator run needs.

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
- **WebView:** build `webview-shell/` in Android Studio (debug), then
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
