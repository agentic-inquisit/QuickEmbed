# Browser (WebGPU / WASM)

Use the ONNX export so it loads via `onnxruntime-web`:

```bash
# model: onnx-community/embeddinggemma-2-ONNX
```

```js
import { AutoTokenizer, AutoModel } from '@huggingface/transformers';

const tokenizer = await AutoTokenizer.from_pretrained('onnx-community/embeddinggemma-2-ONNX');
const model = await AutoModel.from_pretrained('onnx-community/embeddinggemma-2-ONNX', {
  device: 'webgpu',   // falls back to 'wasm' automatically if WebGPU is unsupported
  dtype: 'q4',        // INT4 quantized — keeps the download small for a first page load
});

const inputs = await tokenizer(['task: search result | query: your text here']);
const { sentence_embedding } = await model(inputs);
```

## Constraints
- WebGPU support is still inconsistent across browsers/devices (Safari and
  older Android WebViews are the main gaps) — always ship the WASM fallback
  path, don't assume WebGPU.
- First load downloads the quantized ONNX weights (tens of MB) client-side;
  cache via the Cache API / IndexedDB so repeat visits don't re-fetch.
- Text-only is the realistic browser target. Running the vision/audio
  encoders in-browser works but adds real weight and is not yet
  battle-tested per the model's own release notes — prefer routing image/audio
  requests to the cloud backend from web clients.
