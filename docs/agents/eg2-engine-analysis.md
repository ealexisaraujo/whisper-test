# EG2 Transcribe — how the "EmbeddingGemma 2" transcriber works

Static reverse-engineering of **Embedding Gemma Transcribe 1.6** (`ai.haku.eg2demo`, The Smart Language),
done with REA (Ghidra 12.1.4 provider) plus the public model manifests. Nothing from the app was executed.

## Target identity

| Artifact | SHA-256 |
|---|---|
| `EG2-Transcribe-1.6.dmg` (from `t.co/JIyWeGTdqT` → `haku-models.tslfiles.org/eg2demo/EG2-Transcribe-1.6.dmg`) | `4ee020455bf3cd6d3184b6f8271af060cd37be0bacc3669cca9739d2a7805214` |
| `Contents/MacOS/EG2Transcribe` (arm64 Swift, links CoreML, Accelerate, AVFoundation, Sparkle) | `7bcdc0085df93f5e5538e54e943d82c9701172361c56ec6a4241e325fa182348` |

The app is a 2.3 MB shell; the models (715 MB, manifest `v5`) are downloaded on first run from
`https://haku-models.tslfiles.org/eg2demo/v5/manifest.json` and checked file-by-file with SHA-256.
App 1.6 only knows manifests `v1`–`v5`; a `v6` manifest (2026-10-20) already adds a `tech/` vocabulary pack
that this binary does not reference.

## Pipeline (observed)

```
16 kHz mono PCM
  └─ LogMel (Swift/Accelerate)          160-sample left pad, 320-sample periodic Hann, hop 160 (10 ms),
                                         512-pt real DFT, |X| (magnitude), 257×128 mel matrix, ln(x + 0.001)
       └─ [1, 2000, 128] per 20 s window + frame mask
  └─ eg2_audio_l11.mlmodelc (Core ML)   EmbeddingGemma 2 audio encoder, cut at layer 11, palettized 6–14 bit
       └─ [1, 500, 1024]  (4× subsampling → 40 ms per frame)
  └─ reader_P6.mlmodelc (Core ML)       new 12-layer/12-head d=768 "arm" (86 M params)
       └─ vec [1, 500, 768]  — one point in EG2's *text-embedding space* per frame
       └─ blank [1, 500, 1]  — CTC blank logit
  └─ scorer_blk_d128_k16_b4_t500        vec·W(768→128) · Table(128×229 460) → max over blocks of 4
       └─ top-16 blocks per frame = 64 candidate words
  └─ Rerank (Swift, concurrentPerform)  exact dot product vec · word_unit_f16[cand] (229 457 × 768 fp16)
  └─ back()                             word = best cand if  rBias + rScale·cos  >  blank[t]  else blank
                                         (rScale 39.857, rBias −0.4626 from reader_P6.json)
  └─ Stream.decode()                    greedy word-level CTC collapse; "<rep>" re-emits previous word;
                                         word start time = window_t0 + frame × 0.04 s
  └─ HeadOnlyPunct (punct_headonly)     word_unit vectors of the decoded words, 96-word blocks →
                                         punct[4] + case[3] logits, punct_case.json overrides casing
```

So it is **not** a token-by-token decoder like Whisper. Each 40 ms frame is mapped *directly* to a point
in the embedding space, and the word is found by nearest-neighbour search over a closed 229k-word vocabulary.
That is why it is fast and streamable (no autoregressive loop), and also why it can never produce a word
outside `reader_words.json`.

## Evidence

| Claim | Evidence |
|---|---|
| Models are EG2 converted to Core ML, audio encoder fine-tuned, new reading + punctuation layers | `Resources/legal/NOTICE.txt`; app strings "Compiling the audio encoder", "Loading the vocabulary (229,457 words)" |
| Tensor shapes (2000×128 mel → 500×1024 → 500×768 + blank) | `*.mlmodelc/metadata.json` inputSchema/outputSchema (manifest v5, SHA-256 verified) |
| Encoder = layer-11 output, mixed palettization | `eg2_audio_l11.mlmodelc/metadata.json` storagePrecision |
| Reader arm config, scale/bias | `reader_P6.json` (`scale 39.857`, `bias −0.4626`, `layers L11`, `d 768`, `layers_n 12`) |
| 229 457 words × 768 fp16 | `word_unit_f16.bin` = 352 445 952 B = 229 457 × 768 × 2 |
| Block top-k scorer | `scorer_blk_d128_k16_b4_t500.mlmodelc/model.mil`: matmul 768→128, matmul 128→229 460, reshape `[1,500,57365,4]`, `reduce_max`, `topk` 16 |
| Mel recipe | REA `procedure_pseudo_code` `LogMel.callAsFunction` @ `0x1000805d0` (ev_2190228f…): pad `0xa0`, `vDSP_vmul` 0x140, `vDSP_DFT_Execute` 512, `vDSP_zvabs`×0.5, `cblas_sgemm` 257→128, `vDSP_vsadd 0.001`, `vvlogf`; `hann_320.f32` equals periodic Hann (max err 2.9e-8) |
| Blank-vs-word rule | `EG2Transcriber.back` @ `0x1000754a0` (ev_72f93510…): `if (rBias + rScale * cos <= blank[t]) id = -1` |
| Rerank over scorer candidates | `Rerank.run` @ `0x100096f5c` (ev_ef8102be…), init score −2.0, `concurrentPerform` over frames |
| Greedy word CTC + `<rep>` + 40 ms timing | `Stream.decode` @ `0x100077d6c` (ev_f446f8d2…): `t = t0 + i*0.04`, compares word to `"<rep>"` |
| Windowing for long audio | `transcribeShared` @ `0x100075ae8` (ev_dad2fd81…): window 320 000 samples (20 s), step `0x46500` = 288 000 (18 s); keep frames `[25, 475)` except first (from 0) and last (to end) |
| ANE + GPU in parallel | `transcribeShared` builds two workers (`ane`, `gpu: Front`) behind semaphores |
| Punctuation in independent 96-word blocks | `HeadOnlyPunct.decide` @ `0x10007dfa0` (ev_d1ec538e…): slices of `0x60` words, no overlap |
| v6 tech pack (not used by 1.6) | `v6/manifest.json`: `tech/coreml/letters_LS.mlmodelc` (char CTC, 29 classes on the same 1024-d l11 features), `fixer.json` (16-feature MLP, τ 0.6), `terms.json`, `new_words.json` (77 words) + `new_unit_f16.bin` (77 × 768) |

## Weaknesses we can improve on

1. **Closed, English-only vocabulary.** All 229 457 entries are lowercase ASCII (`también`, `está`, `canción` are absent).
   Names, jargon, and every non-English language fall through. The vendor's own v6 pack (letter CTC + term fixer) confirms they know.
2. **Pure greedy decoding.** Per-frame argmax with a hard blank threshold; no beam search, no language model, no confidence output.
3. **Punctuation without context at block edges.** 96-word blocks with no overlap; the head sees only text embeddings, not audio.
4. **Dictation-shaped output.** Words carry only a start time; no segments, SRT, or end times.

## Unknowns

- Exact training data and real-world WER of the EG2 reader. Not measured here.
- Internals of the encoder graph (`model.mil`, 549 KB). Only its interface was used.
- Licence of the converted weights beyond the NOTICE (Apache-2.0 derivative + Gemma Prohibited Use Policy).
  Download them for local use only; do not redistribute them in this repo.
