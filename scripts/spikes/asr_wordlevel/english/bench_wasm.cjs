// Single-thread WASM speed of an English candidate (npm sherpa-onnx = the WASM build, Node).
//   node bench_wasm.cjs <path/to/node_modules/sherpa-onnx> <engine> <repo> [limit]
// Decodes the first <limit> `authored` phone-channel rows of data/asr_capture/english_set.jsonl
// (2-8 s utterances: the sherpa WASM Moonshine build throws on any input longer than ~9 s, so
// FLEURS' ~11 s rows cannot be used for it) and prints one JSON line (RTF = decode s / audio s, RSS).
const [pkg, engine, repo, limitArg] = process.argv.slice(2);
const sherpa = require(pkg);
const fs = require('fs');
const M = `${repo}/models/asr`;
const limit = Number(limitArg || 60);
const LEAD_PAD = 0.3, TAIL = 0.8, CHUNK = 1600;

function build() {
  if (engine === 'fc_rnnt480' || engine === 'fc_ctc480') {
    const d = engine === 'fc_rnnt480'
      ? `${M}/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8`
      : `${M}/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8`;
    const modelConfig = {
      transducer: engine === 'fc_rnnt480' ? {encoder: `${d}/encoder.int8.onnx`, decoder: `${d}/decoder.int8.onnx`, joiner: `${d}/joiner.int8.onnx`} : {encoder: '', decoder: '', joiner: ''},
      paraformer: {encoder: '', decoder: ''}, zipformer2Ctc: {model: ''},
      nemoCtc: {model: engine === 'fc_ctc480' ? `${d}/model.int8.onnx` : ''}, toneCtc: {model: ''},
      tokens: `${d}/tokens.txt`, numThreads: 1, provider: 'cpu', debug: 0, modelType: '', modelingUnit: 'cjkchar', bpeVocab: '',
    };
    return {online: true, rec: sherpa.createOnlineRecognizer({
      featConfig: {sampleRate: 16000, featureDim: 80}, modelConfig, decodingMethod: 'greedy_search', maxActivePaths: 4,
      enableEndpoint: 0, rule1MinTrailingSilence: 2.4, rule2MinTrailingSilence: 1.2, rule3MinUtteranceLength: 20,
      hotwordsFile: '', hotwordsScore: 1.5, ctcFstDecoderConfig: {graph: '', maxActive: 3000}, ruleFsts: '', ruleFars: ''})};
  }
  if (engine === 'moon_tiny' || engine === 'moon_base') {
    const d = `${M}/sherpa-onnx-moonshine-${engine === 'moon_tiny' ? 'tiny' : 'base'}-en-quantized-2026-02-27`;
    return {online: false, rec: sherpa.createOfflineRecognizer({
      featConfig: {sampleRate: 16000, featureDim: 80},
      modelConfig: {moonshine: {encoder: `${d}/encoder_model.ort`, mergedDecoder: `${d}/decoder_model_merged.ort`},
                    tokens: `${d}/tokens.txt`, numThreads: 1, provider: 'cpu', debug: 0},
      decodingMethod: 'greedy_search'})};
  }
  if (engine === 'parakeet110') {
    const d = `${M}/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8`;
    return {online: false, rec: sherpa.createOfflineRecognizer({
      featConfig: {sampleRate: 16000, featureDim: 80},
      modelConfig: {nemoCtc: {model: `${d}/model.int8.onnx`}, tokens: `${d}/tokens.txt`, numThreads: 1, provider: 'cpu', debug: 0},
      decodingMethod: 'greedy_search'})};
  }
  throw new Error(`unknown engine ${engine}`);
}

const {online, rec} = build();
const rows = fs.readFileSync(`${repo}/data/asr_capture/english_set.jsonl`, 'utf8').trim().split('\n')
  .map(JSON.parse).filter(r => r.set === 'authored' && r.channel === 'phone').slice(0, limit);
let audio = 0, secs = 0;
rows.forEach((r, i) => {
  const wave = sherpa.readWave(`${repo}/${r.wav}`);
  const samples = new Float32Array(wave.samples.length + LEAD_PAD * 16000);
  samples.set(wave.samples, LEAD_PAD * 16000);
  const t0 = process.hrtime.bigint();
  let text;
  if (online) {
    const s = rec.createStream();
    for (let o = 0; o < samples.length; o += CHUNK) {
      s.acceptWaveform(16000, samples.subarray(o, o + CHUNK));
      while (rec.isReady(s)) rec.decode(s);
    }
    s.acceptWaveform(16000, new Float32Array(TAIL * 16000));
    s.inputFinished();
    while (rec.isReady(s)) rec.decode(s);
    text = rec.getResult(s).text;
    s.free();
  } else {
    const s = rec.createStream();
    s.acceptWaveform(16000, samples);
    rec.decode(s);
    text = rec.getResult(s).text;
    s.free();
  }
  secs += Number(process.hrtime.bigint() - t0) / 1e9;
  audio += wave.samples.length / wave.sampleRate;
  if (i < 2) console.error(r.id, '|', text);
});
console.log(JSON.stringify({engine, n: rows.length, audio_s: +audio.toFixed(1), decode_s: +secs.toFixed(1),
  rtf_wasm_1t: +(secs / audio).toFixed(4), rss_mb: Math.round(process.memoryUsage().rss / 1e6)}));
