// Android library (AAR) built from the same Kotlin sources as :core, plus the Android builds of
// ONNX Runtime and the DJL tokenizer. Model content is never committed here: a Gradle task copies
// the config + head weights from site/ into generated assets at build time, and the 278 MB int8
// embedder + its 17 MB tokenizer are bundled only on request (-PqorganBundleModel=true); a host
// app such as eGov Mobile would deliver them on demand instead (see ../CORE_SPIKE.md).
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val repoRoot = rootProject.projectDir.parentFile.parentFile
val siteDir = File(repoRoot, "site")
val bundleModel = (project.findProperty("qorganBundleModel") ?: "false").toString().toBoolean()
val generatedAssets = layout.buildDirectory.dir("generated/qorganAssets")
val gateAssets = layout.buildDirectory.dir("generated/qorganGateAssets")

val copyQorganAssets by tasks.registering(Copy::class) {
    description = "Copies the shipped config + weights (and optionally the embedder) from site/."
    into(generatedAssets.map { it.dir("qorgan") })
    from(File(siteDir, "core/qorgan-config.json"))
    from(File(siteDir, "models/weights.json"))
    if (bundleModel) {
        from(File(siteDir, "models/Xenova/multilingual-e5-base")) {
            include("tokenizer.json", "tokenizer_config.json", "config.json", "onnx/model_quantized.onnx")
            into("multilingual-e5-base")
        }
    }
    doFirst {
        listOf("core/qorgan-config.json", "models/weights.json").forEach {
            require(File(siteDir, it).isFile) { "missing site/$it (run python -m qorgan.web.client_config / deploy_bootstrap)" }
        }
    }
}

android {
    namespace = "kz.qorgan.core"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    sourceSets["main"].apply {
        java.srcDirs("../core/src/main/kotlin")
        assets.srcDir(generatedAssets)
    }
    sourceSets["androidTest"].assets.srcDir(gateAssets)
    packaging {
        // DJL's JVM jar carries desktop tokenizer binaries (~40 MB); Android uses tokenizer-native.
        resources.excludes += "native/lib/**"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        // ONNX / tokenizer files must stay uncompressed so they can be memory-mapped or read fast.
        noCompress += listOf("onnx", "json")
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

// The 200-case runtime gate fixtures for the on-device instrumented test (read from tests_js/).
val copyGateFixtures by tasks.registering(Copy::class) {
    into(gateAssets.map { it.dir("gate") })
    from(File(repoRoot, "tests_js/fixtures")) { include("runtime_gate.json", "runtime_gate_browser.f32") }
}

tasks.named("preBuild") { dependsOn(copyQorganAssets, copyGateFixtures) }

dependencies {
    implementation(libs.serialization.json)
    api(libs.onnxruntime.android)
    api(libs.djl.tokenizers)
    implementation(libs.djl.tokenizer.native.android)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.junit)
}
