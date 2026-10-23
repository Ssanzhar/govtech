// Pure Kotlin/JVM: everything here runs (and is tested) on the Mac without a device.
// ONNX Runtime and the DJL tokenizer are compileOnly: the AAR supplies the Android builds,
// the tests supply the desktop builds of the same Java APIs.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.jvm")
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    implementation(libs.serialization.json)
    compileOnly(libs.onnxruntime.jvm)
    compileOnly(libs.djl.tokenizers)
    testImplementation(libs.junit)
    testImplementation(libs.onnxruntime.jvm)
    testImplementation(libs.djl.tokenizers)
}

val repoRoot = rootProject.projectDir.parentFile.parentFile

tasks.test {
    // Fixtures, config and weights are read in place from the repository (never duplicated).
    systemProperty("qorgan.repo", repoRoot.absolutePath)
    // The 200-transcript runtime gate embeds ~2,000 texts with the 278 MB int8 model; opt in.
    systemProperty("qorgan.runtimeGate", (project.findProperty("runtimeGate") ?: "false").toString())
    maxHeapSize = "4g"
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
