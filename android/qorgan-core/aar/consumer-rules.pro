# ONNX Runtime and the DJL tokenizer are called through JNI.
-keep class ai.onnxruntime.** { *; }
-keep class ai.djl.huggingface.tokenizers.jni.** { *; }
