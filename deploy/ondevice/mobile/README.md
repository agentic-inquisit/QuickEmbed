# Mobile (Android / iOS) — LiteRT

True edge deployment uses Google's pre-converted `.task` bundles, not the
raw safetensors checkpoint.

Pick the bundle matching the modalities you need (smaller = faster to load,
less RAM):

| Bundle | HF repo | Params |
|---|---|---|
| Text only | `litert-community/embeddinggemma-2-text-270m-litert-lm` | 270M |
| Text + vision | `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` | 440M |
| Full multimodal | `litert-community/embeddinggemma-2-740m-litert-lm` | 740M |

## Android (MediaPipe Tasks)

```kotlin
val options = TextEmbedderOptions.builder()
    .setBaseOptions(BaseOptions.builder().setModelAssetPath("embeddinggemma2_740m.task").build())
    .setQuantize(true)
    .build()
val embedder = TextEmbedder.createFromOptions(context, options)
val result = embedder.embed(inputText)
```

Place the `.task` file in `app/src/main/assets/`. Image/audio inputs use the
same `ImageEmbedder` / `AudioEmbedder` MediaPipe classes with the multimodal
bundle.

## iOS

Use the MediaPipe iOS Tasks framework (`MPPTextEmbedder`); API shape mirrors
the Android one above.

## Constraints measured by Google (Pixel 11 Pro, INT4/INT8 QAT)
- Text-only active RAM: ~191 MB
- Full multimodal (text+vision+audio) active RAM: ~567 MB
- No official hosted fallback — if the device can't fit the full bundle,
  route to the cloud backend (`deploy/cloud/`) instead of trying to force a
  bigger bundle onto constrained hardware.
