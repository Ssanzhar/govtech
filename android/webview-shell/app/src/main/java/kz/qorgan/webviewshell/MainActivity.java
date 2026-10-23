package kz.qorgan.webviewshell;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/**
 * A plain Android WebView around the citizen page (ADR D67): the closest stand-in for a
 * mini-service inside eGov Mobile, which embeds pages the same way. WebView never grants
 * cross-origin isolation, so the page must choose its light (GigaAM-only) voice mode here.
 * Test harness only: no signing config, no network code of its own; the page decides
 * everything and audio never leaves the page (it is transcribed in the WebView).
 *
 * The start URL is fixed at build time (-PqorganUrl, see app/build.gradle). Debug builds also
 * accept an override from the launch intent extra "url" (adb shell am start ... --es url ...).
 */
public class MainActivity extends Activity {
    private static final String TAG = "QorganShell";
    private static final int AUDIO_REQUEST = 1;
    private WebView webView;
    private PermissionRequest pending;
    private String allowedOrigin;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Lets Chrome DevTools / Playwright (connectOverCDP) inspect the page during test runs.
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        String override = BuildConfig.DEBUG ? getIntent().getStringExtra("url") : null;
        String url = override != null ? override : BuildConfig.QORGAN_URL;
        allowedOrigin = originOf(Uri.parse(url));

        webView = new WebView(this);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);               // the whole page is JS
        settings.setDomStorageEnabled(true);               // localStorage: UI prefs only
        settings.setMediaPlaybackRequiresUserGesture(false); // AudioContext / mic stream start
        settings.setAllowFileAccess(false);                // no file:// reach into the app
        settings.setAllowContentAccess(false);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        settings.setSupportMultipleWindows(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                // Stay on the configured origin; anything else is not loaded in the shell.
                boolean same = allowedOrigin.equals(originOf(request.getUrl()));
                if (!same) Log.w(TAG, "blocked navigation off-origin: " + request.getUrl());
                return !same;
            }
        });
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                // Only the microphone, only for the configured origin, only with the OS permission.
                if (!allowedOrigin.equals(originOf(request.getOrigin())) || !audioOnly(request.getResources())) {
                    Log.w(TAG, "denied web permission request from " + request.getOrigin());
                    request.deny();
                    return;
                }
                pending = request;
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    grantAudio();
                } else {
                    requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, AUDIO_REQUEST);
                }
            }

            @Override
            public void onPermissionRequestCanceled(PermissionRequest request) {
                if (pending == request) pending = null;
            }
        });
        setContentView(webView);
        Log.i(TAG, "loading " + url);
        webView.loadUrl(url);
    }

    private static boolean audioOnly(String[] resources) {
        if (resources == null || resources.length == 0) return false;
        for (String r : resources) {
            if (!PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(r)) return false;
        }
        return true;
    }

    private static String originOf(Uri uri) {
        if (uri == null || uri.getScheme() == null || uri.getHost() == null) return "";
        int port = uri.getPort();
        return uri.getScheme() + "://" + uri.getHost() + (port == -1 ? "" : ":" + port);
    }

    private void grantAudio() {
        if (pending == null) return;
        pending.grant(new String[] {PermissionRequest.RESOURCE_AUDIO_CAPTURE});
        pending = null;
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        if (code != AUDIO_REQUEST || pending == null) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            grantAudio();
        } else {
            pending.deny();
            pending = null;
        }
    }

    @Override
    protected void onDestroy() {
        if (webView != null) webView.destroy();
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }
}
