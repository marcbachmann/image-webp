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
| none (libwebp parity) | 4 | 1732 ms, 4,816,002 B | | |

`run_refs`: backward references by run length only (left or upper neighbour), no hash chain;
`src/lossless/encoder/fast.rs` then counts and writes the image from the pixels without a
reference per pixel. Photos lose nothing; a graphic with far-back repeats grows (the tiled
logo 2.4x), since only LZ77 sees them.

`short_chain`: the hash chain filled with quality 0's effort (8 chain steps, a window of 16
rows) whatever the quality, the rest of the method unchanged. Not measured yet.
