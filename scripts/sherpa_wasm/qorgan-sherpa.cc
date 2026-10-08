// The one helper sherpa-onnx-asr.js needs beyond the C API (same as upstream's
// wasm/nodejs/sherpa-onnx-wasm-nodejs.cc): copy a config struct into the heap.
#include <algorithm>
#include <cstdint>

#include "sherpa-onnx/c-api/c-api.h"

extern "C" {

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
