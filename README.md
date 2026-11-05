# image-webp, libwebp port

WebP in pure Rust that produces **the same bytes as libwebp 1.6**, and is **faster than libwebp** on most of
what we measured, single-threaded, on x86-64 and on Apple silicon.

This is a fork of [image-rs/image-webp](https://github.com/image-rs/image-webp) (see
[image-rs/image-webp#195](https://github.com/image-rs/image-webp/issues/195)). The lossy and lossless encoders,
the alpha encoder and the decoder's hot paths are ported from libwebp 1.6.0, with libwebp's SSE2, AVX2 and NEON
kernels rewritten as safe Rust SIMD.

- **Byte-identical to libwebp.** A lossy encode is the bitstream `WebPEncode` writes with the default preset, at
  every quality and methods 0-6. Lossless and alpha encodes match libwebp at the same quality and method.
  Decoded pixels match libwebp's default (`dwebp`) output. Switching from libwebp changes no output and no file size.
- **Faster.** Ahead of libwebp in every case in [Performance](#performance) except three, where it is level:
  two of the fastest lossless encodes on x86-64, and lossy RGB decoding on the M3.
- **Little memory.** Encoding reads rows from a callback, and decoding hands out a band of rows at a time. A
  24 MP lossy encode peaks at 22 MB, against libwebp's 232 MB.
- **No C.** No C compiler, no FFI and no system library. The only dependencies are `byteorder-lite` and
  `quick-error`.
- **No threads of its own.** The library never spawns a thread. A caller that has threads to spare can lend them to
  a decode through `Runner`.

## Features

| | |
|---|---|
| Lossy encoding (VP8) | libwebp's encoder: quality 0-100, methods 0-6, segments, trellis, the default preset |
| Lossless encoding (VP8L) | libwebp's encoder: quality, method, `exact`, near-lossless |
| Alpha (ALPH) | compressed or raw, libwebp's filters, or encoded separately with `encode_alpha_chunk` |
| Animation encoding | `AnimationEncoder`: changed rectangles, blending, key frames, frames added whole or a band at a time |
| Decoding | every WebP feature: lossy, lossless, alpha, animation, ICC/EXIF/XMP, libwebp's demuxer rules |
| Streaming | `encode_rows`, `band_reader`, `read_image_rows`, and `start` to decode a frame before the file has arrived |
| Scaled decoding | `frame(n, Some((w, h)), alpha)` resizes with libwebp's rescaler while decoding |
| SIMD | SSE2 on every x86-64, AVX2 when the CPU has it (`IMAGE_WEBP_NO_AVX2` turns it off), NEON on aarch64; portable Rust elsewhere |

`AnimationEncoder` is this crate's own design and is measured against libwebp's `WebPAnimEncoder`; its output is not
byte-identical to it. Everything else above follows libwebp.

## Using it

```toml
[dependencies]
image-webp = { git = "https://github.com/marcbachmann/image-webp", branch = "libwebp-port" }
```

The API is upstream's with more added and one break: decoding needs a reader that is `Send`. That is why the
[`image`](https://crates.io/crates/image) crate, whose WebP decoder takes any reader, does not build against this
fork yet.

Lossless encoding defaults to libwebp's quality 75, method 4. On the inputs below that gives files 20-80% smaller
than upstream's encoder, but it takes 24-50 times as long. Set `lossless_quality` and `lossless_method`
to 0 for libwebp's fastest setting.

Lossy encode, as `cwebp -q 80 -m 4`:

```rust
use image_webp::{ColorType, EncoderParams, WebPEncoder};

let (width, height) = (640, 480);
let rgb = vec![128u8; (width * height * 3) as usize];

let mut params = EncoderParams::default();
params.use_lossy = true;
params.lossy_quality = 80;
params.lossy_method = 4;

let mut webp = Vec::new();
let mut encoder = WebPEncoder::new(&mut webp);
encoder.set_params(params);
encoder.encode(&rgb, width, height, ColorType::Rgb8)?;
```

Encode without holding the image. The callback fills row `y`, for example from a decoder or a resizer:

```rust
use image_webp::{ColorType, EncoderParams, WebPEncoder};

let (width, height) = (6000, 4000);
let mut params = EncoderParams::default();
params.use_lossy = true;
params.lossy_quality = 80;

let mut webp = Vec::new();
let mut encoder = WebPEncoder::new(&mut webp);
encoder.set_params(params);
encoder.encode_rows(width, height, ColorType::Rgb8, |y, row| row.fill((y % 256) as u8))?;
```

Decode the whole image, a band of rows at a time, or scaled:

```rust
use image_webp::WebPDecoder;
use std::io::Cursor;

let data = std::fs::read("photo.webp")?;

let mut decoder = WebPDecoder::new(Cursor::new(&data))?;
let mut pixels = vec![0; decoder.output_buffer_size().unwrap()];
decoder.read_image(&mut pixels)?;

let mut decoder = WebPDecoder::new(Cursor::new(&data))?;
let mut bands = decoder.band_reader()?;
while let Some((first_row, rows)) = bands.next_band()? {
    // `rows` holds whole rows from `first_row` on, RGB or RGBA as `read_image` writes them.
}

let mut decoder = WebPDecoder::new(Cursor::new(&data))?;
let alpha = decoder.has_alpha();
let mut frame = decoder.frame(0, Some((320, 180)), alpha)?;
let (w, h) = frame.dimensions();
let mut thumbnail = vec![0; (w * h) as usize * frame.bytes_per_pixel()];
frame.read_into(&mut thumbnail)?;
```

Encode an animation:

```rust
use image_webp::{AnimationEncoder, ColorType};

let (width, height) = (320, 240);
let mut file = std::io::Cursor::new(Vec::new());
let mut animation = AnimationEncoder::new(&mut file, width, height)?;
for i in 0..10u8 {
    let frame = vec![i * 25; (width * height * 4) as usize];
    animation.add_frame(&frame, ColorType::Rgba8, 100)?;
}
animation.finish()?;
```

## Performance

Measured 2026-11-05 at `c15a9af`. Single thread, best of several interleaved runs. x86 is an Intel i9-13900
(best of 5 decodes and 3 encodes) and ARM is an Apple M3 (best of 2-3). "upstream" is
image-rs `main` at `e43b2f9`. libwebp is `libwebp-sys` 0.14, built with SSE4.1 and AVX2 on x86 and with NEON on
ARM. Every output of this crate below is byte-identical to libwebp's: the encoded files and the decoded pixels.

Inputs: `tests/reference/gallery1/3.png` resized to 3000x2000 (6 MP) or 6000x4000 (24 MP). The alpha cases put
the alpha of `gallery2/3_webp_a.png`, resized, over it, and the graphic is `gallery2/3_webp_a.png` tiled to
3000x2000. The `.webp` files were written by `cwebp -q 80 -m 4` and `cwebp -lossless -q 75 -m 4`.

Decode (ms, x86 / M3):

| file | upstream | this, band reader | this, `read_image` | libwebp |
|---|---|---|---|---|
| lossy RGB, 24 MP | 299 / 260 | 91 / 96 | 107 / 96 | 136 / 99 |
| lossy + alpha, 24 MP | 305 / 256 | 79 / 72 | 98 / 74 | 189 / 114 |
| lossless RGB photo, 6 MP | 71 / 97 | 57 / 79 | 58 / 80 | 74 / 86 |
| lossless RGBA photo, 6 MP | 29 / 34 | 19 / 26 | 20 / 25 | 26 / 34 |
| lossless RGBA graphic, 6 MP | 22 / 26 | 11 / 19 | 12 / 18 | 14 / 22 |

Lossy encode, 24 MP, quality 80, method 4 (ms, x86 / M3):

| input | this | libwebp | size, both |
|---|---|---|---|
| RGB | 768 / 1012 | 1126 / 1124 | 1.29 MB |
| RGBA | 1563 / 1551 | 2307 / 2570 | 0.90 MB |

Lossless encode, 6 MP (ms, x86 / M3; at the same settings the sizes of this crate and libwebp are equal):

| input | upstream (default) | q0 m0: this / libwebp | q75 m4: this / libwebp |
|---|---|---|---|
| RGB photo | 38 / 37, 6.05 MB | 175 / 155 vs 192 / 198, 5.54 MB | 1724 / 1850 vs 2036 / 2284, 4.82 MB |
| RGBA photo | 37 / 38, 6.92 MB | 79 / 71 vs 79 / 95, 2.76 MB | 882 / 900 vs 1200 / 1457, 1.78 MB |
| RGBA graphic | 20 / 18, 3.27 MB | 44 / 39 vs 46 / 57, 1.27 MB | 670 / 644 vs 1045 / 1340, 0.67 MB |

Peak memory of the 24 MP lossy encode on x86: `encode_rows` 22 MB, `encode` 127 MB, and libwebp, with the 72 MB
image in memory, 232 MB.

## How the output is checked

The examples run libwebp (through `libwebp-sys`, a dev-dependency only) side by side with this crate:
`lossy_vs_libwebp`, `lossless_vs_libwebp` and `anim_vs_libwebp` for the encoders, and `decode_bench --check` and
`scaled_bench` for the decoder. The test suite compares against libwebp too, for example `lossy_matches_libwebp`
and the frame decode sweep in `tests/frame.rs`. On aarch64 the comparison is with libwebp's NEON build, whose
transforms round differently from its SSE2 ones; this crate matches libwebp on each architecture. Malformed files
can still be accepted or rejected differently from libwebp: byte parity is the target for valid files.

## Unsafe code

The crate is `#![deny(unsafe_code)]`, and every exception is an explicit `#[allow]`. The SIMD kernels are safe
Rust: intrinsics are safe inside `#[target_feature]` functions, and loads and stores go through `simd_mem`, which
checks bounds. The `unsafe` that remains is in `simd_mem.rs` (the slice-to-register loads and stores),
`wide.rs` (AVX2 detection and dispatch), and the calls into `#[target_feature]` functions after the CPU check,
commented with `// SAFETY:`. Upstream is `#![forbid(unsafe_code)]`; this fork is not.

## Minimum Rust version

1.88.

## License

The ported code is derived from libwebp, so those parts are under libwebp's BSD 3-Clause license and patent grant,
in [LICENSE-LIBWEBP](LICENSE-LIBWEBP). The crate is `(MIT OR Apache-2.0) AND BSD-3-Clause`.
