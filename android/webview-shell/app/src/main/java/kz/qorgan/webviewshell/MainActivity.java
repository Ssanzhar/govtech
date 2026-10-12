package kz.qorgan.webviewshell;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;

/**
 * A plain Android WebView around the citizen page (ADR D67): the closest stand-in for a
 * mini-service inside eGov Mobile, which embeds pages the same way. WebView never grants
 * cross-origin isolation, so the page must choose its light (GigaAM-only) voice mode here.
 * Test harness only: no signing config, no network code of its own; the page decides
 * everything. The URL comes from the launch intent extra "url" (default: the emulator's view of
 * the host's static server, made reachable with `adb reverse tcp:8410 tcp:8410`).
 */
public class MainActivity extends Activity {
    private static final String DEFAULT_URL = "http://localhost:8410/live.html";
    private static final int AUDIO_REQUEST = 1;
    private WebView webView;
    private PermissionRequest pending;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Lets Chrome DevTools / Playwright (connectOverCDP) inspect the page during test runs.
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        webView = new WebView(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                // The page asks for the microphone; grant it only with the OS permission.
                pending = request;
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    grantAudio();
                } else {
                    requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, AUDIO_REQUEST);
                }
            }
        });
        setContentView(webView);
        String url = getIntent().getStringExtra("url");
        webView.loadUrl(url != null ? url : DEFAULT_URL);
    }

    private void grantAudio() {
        if (pending == null) return;
        pending.grant(new String[] {PermissionRequest.RESOURCE_AUDIO_CAPTURE});
        pending = null;
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        if (code != AUDIO_REQUEST || pending == null) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) grantAudio();
        else { pending.deny(); pending = null; }
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }
}
