# `fast_mode` shortcuts

`EncoderParams::fast_mode` trades byte parity with libwebp for speed. Each shortcut is a
`lossless::encoder::FastMode` field of its own so that any can be offered by itself; the
`fast_mode` flag turns on the set that measured best per method, `FastMode::for_method`.
`IMAGE_WEBP_FAST_SET=run,chain` overrides that set for measurement.

Measured on the x86 worker (i9-13900, pinned, min of 3 interleaved runs), 6 MP inputs:
`tests/reference/gallery1/3.png` resized to 3000x2000 (RGB photo), the same with the alpha
of `gallery2/3_webp_a.png` over it (RGBA photo), and that logo tiled (RGBA graphic).
Every variant decodes back to the input exactly.

| shortcut | method | RGB photo | RGBA photo | RGBA graphic |
|---|---|---|---|---|
| none (libwebp parity) | 0 | 175 ms, 5,544,036 B | 94 ms, 2,757,390 B | 58 ms, 1,265,434 B |
| `run_refs` | 0 | 45.5 ms, 5,542,768 B | 23 ms, 2,277,156 B | 24 ms, 3,044,826 B |
| none (libwebp parity) | 4 | 1737 ms, 4,816,002 B | 921 ms, 1,779,556 B | 726 ms, 665,972 B |
| `short_chain` | 4 | 811 ms, 4,816,002 B | 703 ms, 1,779,556 B | 657 ms, 666,256 B |
| `run_refs` | 4 | 486 ms, 4,816,002 B | 552 ms, 1,868,928 B | 640 ms, 2,508,124 B |
| none (libwebp parity) | 6 | 1821 ms, 4,811,944 B | 977 ms, 1,781,194 B | 778 ms, 662,184 B |
| `short_chain` | 6 | 885 ms, 4,811,944 B | 770 ms, 1,781,194 B | 741 ms, 662,792 B |
| `run_refs` | 6 | 559 ms, 4,811,944 B | 634 ms, 1,867,748 B | 711 ms, 2,494,032 B |

`run_refs`: backward references by run length only (left or upper neighbour), no hash chain;
`src/lossless/encoder/fast.rs` then counts and writes the image from the pixels without a
reference per pixel. Photos lose nothing; a graphic with far-back repeats grows (the tiled
logo 2.4x), since only LZ77 sees them.

`short_chain`: the hash chain filled with quality 0's effort (8 chain steps, a window of 16
rows) whatever the quality, the rest of the method unchanged. On photos the full search finds
nothing the short one does not, so the bytes are identical and methods 4 to 6 take half the
time; the graphic grows by under 0.1%. This is what `fast_mode` turns on at methods 1 to 6.

On top of `short_chain` at method 4 (second run, so `short_chain` alone read 855 / 704 / 661 ms there):

| knob | RGB photo | RGBA photo | RGBA graphic |
|---|---|---|---|
| `no_trace` | 807 ms, same bytes | 704 ms, same bytes | 637 ms, 671,924 B (+0.85%) |
| `no_cache` | 652 ms, same bytes | 621 ms, 1,868,928 B (+5.0%) | 616 ms, same bytes |
| both | 653 ms, same bytes | 626 ms, +5.0% | 591 ms, +0.85% |

`no_trace`: no cost-model pass over the references (libwebp runs it from quality 25).
`no_cache`: no colour cache, so none of the search over its sizes; the +5% on the RGBA photo is
what the cache was buying there, and it is also where `run_refs` lost its 5%.
Neither is in the `fast_mode` set: each costs bytes on one of the three inputs.

Predictor search knobs, on top of `short_chain` at method 4 (`short_chain` alone read
806 / 706 / 660 ms in that run):

| knob | RGB photo | RGBA photo | RGBA graphic |
|---|---|---|---|
| `pred_subset` | 687 ms, 4,813,998 B (-0.04%) | 471 ms, 1,779,604 B (+0.003%) | 422 ms, 664,810 B (-0.2%) |
| `pred_single` | 670 ms, 5,415,528 B (+12.4%) | 389 ms, 2,124,622 B (+19.4%) | 332 ms, 722,172 B (+8.4%) |

`pred_subset`: the predictor search tries four modes per tile (left, top, select, gradient)
instead of libwebp's 14. Bytes within 0.2% either way, 15-36% faster: in the `fast_mode` set
at methods 1 to 6. `pred_single`: libwebp's method-0 predictor (select) for every tile with
no search; a knob only.

`run_refs` above method 0 is not in the `fast_mode` set: 3.6x on photos at identical bytes,
but +5% on the RGBA photo and 3.8x on the graphic.
