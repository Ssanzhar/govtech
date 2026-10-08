# Qorğan's browser build of sherpa-onnx: offline (non-streaming) ASR only, single-threaded,
# no preloaded .data bundle. Copied over wasm/web/CMakeLists.txt of the pinned sherpa-onnx tag by
# build.sh (ADR D63). Differences from upstream's web target:
#   - exports only the offline-recognizer C API (no TTS / KWS / diarization), so the wasm is small;
#   - ENVIRONMENT=worker: it runs only inside the dedicated GigaAM worker (site/core/gigaam-worker.js);
#   - WORKERFS is linked in, so the 225 MB model is mounted from a Blob instead of being copied
#     into the in-heap MEMFS (one copy fewer at load time);
#   - no -pthread: works where the page is not cross-origin isolated (e.g. Android WebView).
if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use scripts/sherpa_wasm/build.sh")
endif()

set(exported_functions
  SherpaOnnxAcceptWaveformOffline
  SherpaOnnxCreateOfflineRecognizer
  SherpaOnnxCreateOfflineStream
  SherpaOnnxDecodeOfflineStream
  SherpaOnnxDestroyOfflineRecognizer
  SherpaOnnxDestroyOfflineRecognizerResult
  SherpaOnnxDestroyOfflineStream
  SherpaOnnxDestroyOfflineStreamResultJson
  SherpaOnnxGetOfflineStreamResult
  SherpaOnnxGetOfflineStreamResultAsJson
  SherpaOnnxOfflineStreamSetOption
  SherpaOnnxOfflineStreamGetOption
  SherpaOnnxOfflineRecognizerSetConfig
  SherpaOnnxFileExists
  SherpaOnnxGetVersionStr
  SherpaOnnxGetGitSha1
)
set(mangled_exported_functions)
foreach(x IN LISTS exported_functions)
  list(APPEND mangled_exported_functions "_${x}")
endforeach()
list(JOIN mangled_exported_functions "," all_exported_functions)

include_directories(${CMAKE_SOURCE_DIR})
set(MY_FLAGS " -sFORCE_FILESYSTEM=1 -sINITIAL_MEMORY=64MB -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB ")
string(APPEND MY_FLAGS " -sSTACK_SIZE=10485760 ")
string(APPEND MY_FLAGS " -sENVIRONMENT=worker -lworkerfs.js ")
string(APPEND MY_FLAGS " -sEXPORTED_FUNCTIONS=[_CopyHeap,_malloc,_free,${all_exported_functions}] ")
string(APPEND MY_FLAGS " -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString','HEAPU8','HEAP16','HEAP32','HEAPU32','HEAPF32','HEAPF64','FS'] ")
string(APPEND MY_FLAGS " -sMODULARIZE=1 -s 'EXPORT_NAME=\"SherpaOnnx\"' ")

set(CMAKE_EXECUTABLE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-web ${CMAKE_CURRENT_SOURCE_DIR}/qorgan-sherpa.cc)
target_link_libraries(sherpa-onnx-wasm-web sherpa-onnx-core sherpa-onnx-c-api)
set_target_properties(sherpa-onnx-wasm-web PROPERTIES LINK_FLAGS "${MY_FLAGS}")
install(TARGETS sherpa-onnx-wasm-web DESTINATION bin/wasm/web)
install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-web>/sherpa-onnx-wasm-web.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-web>/sherpa-onnx-wasm-web.wasm"
  DESTINATION
    bin/wasm/web
)
