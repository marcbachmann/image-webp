//! SSE2 kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_sse2.c`, bit-exact
//! with the portable ones in `reverse_transform`. SSE2 is part of x86_64, so no runtime detection
//! is needed. Pixels are RGBA bytes; every kernel works per channel, so the order is libwebp's.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::x86_64::*;
use std::ops::Range;

use super::reverse_transform::with_top;
use crate::simd_mem::{load, store};

#[target_feature(enable = "sse2")]
#[inline]
fn load1(p: &[u8; 4]) -> __m128i {
    _mm_cvtsi32_si128(load(p))
}

#[target_feature(enable = "sse2")]
#[inline]
fn store1(p: &mut [u8; 4], v: __m128i) {
    store(p, _mm_cvtsi128_si32(v))
}

/// `(a + b) >> 1` per byte.
#[target_feature(enable = "sse2")]
#[inline]
fn average2(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1)))
}

/// Adds a prediction from the row above only, four pixels at a time: `pred(top, top_left, top_right)`.
#[target_feature(enable = "sse2")]
#[inline]
fn add_upper(out: &mut [u8], top: &[u8], pred: impl Fn(__m128i, __m128i, __m128i) -> __m128i) {
    let m = out.len() & !15;
    let (four, rest) = out.split_at_mut(m);
    for (o, [tl, t, tr]) in with_top::<16>(four, top) {
        let p = pred(load(t), load(tl), load(tr));
        let v = _mm_add_epi8(load(o), p);
        store(o, v);
    }
    for (o, [tl, t, tr]) in with_top::<4>(rest, &top[m..]) {
        let p = pred(load1(t), load1(tl), load1(tr));
        store1(o, _mm_add_epi8(load1(o), p));
    }
}

/// Adds a prediction that takes the pixel decoded just before, a pixel at a time:
/// `pred(left, top, top_left, top_right)`.
#[target_feature(enable = "sse2")]
#[inline]
fn add_serial(
    out: &mut [u8],
    top: &[u8],
    left: &[u8; 4],
    pred: impl Fn(__m128i, __m128i, __m128i, __m128i) -> __m128i,
) {
    let mut left = load1(left);
    for (o, [tl, t, tr]) in with_top::<4>(out, top) {
        let p = pred(left, load1(t), load1(tl), load1(tr));
        left = _mm_add_epi8(load1(o), p);
        store1(o, left);
    }
}

/// Predictor 11, select: the left pixel where it is further from the top-left than the top
/// pixel is. Both sums with the residual are taken before the select, so the next left pixel
/// waits only on the left one's distance.
#[target_feature(enable = "sse2")]
#[inline]
fn add_select(out: &mut [u8], top: &[u8], left: &[u8; 4]) {
    let mut left = load1(left);
    for (o, [tl, t, _]) in with_top::<4>(out, top) {
        let (up, up_left, src) = (load1(t), load1(tl), load1(o));
        let mask = _mm_cmpgt_epi32(_mm_sad_epu8(left, up_left), _mm_sad_epu8(up, up_left));
        let (from_left, from_up) = (_mm_add_epi8(src, left), _mm_add_epi8(src, up));
        left = _mm_or_si128(_mm_and_si128(mask, from_left), _mm_andnot_si128(mask, from_up));
        store1(o, left);
    }
}

/// Predictor 1, the left pixel: a prefix sum over each four pixels.
#[target_feature(enable = "sse2")]
#[inline]
fn add_left(out: &mut [u8], left: &[u8; 4]) {
    let mut prev = _mm_set1_epi32(load(left));
    let (four, rest) = out.as_chunks_mut::<16>();
    for o in four {
        let src = load(o);
        let sum0 = _mm_add_epi8(src, _mm_slli_si128::<4>(src));
        let sum1 = _mm_add_epi8(sum0, _mm_slli_si128::<8>(sum0));
        store(o, _mm_add_epi8(sum1, prev));
        // The next left pixel waits on one add, the four pixels' sum taken beside it.
        prev = _mm_add_epi8(prev, _mm_shuffle_epi32::<0xff>(sum1));
    }
    for o in rest.as_chunks_mut::<4>().0 {
        prev = _mm_add_epi8(load1(o), prev);
        store1(o, prev);
    }
}

/// Predictor 1 on `image_data[range]`, which has a pixel before it.
#[allow(unsafe_code)]
pub(crate) fn predict_left(image_data: &mut [u8], range: Range<usize>) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { predict_left_sse2(image_data, range) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn predict_left_sse2(image_data: &mut [u8], range: Range<usize>) {
    let (head, out) = image_data[..range.end].split_at_mut(range.start);
    add_left(out, head.last_chunk().unwrap())
}

/// Undoes predictor `mode` (0 to 15) over `image_data[range]`, the pixels of one row from its
/// second on, below the row above at `width` pixels before them.
#[allow(unsafe_code)]
pub(crate) fn predict(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { predict_sse2(mode, image_data, range, width) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn predict_sse2(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    // The row above from the top-left pixel of the first on, which ends before the row's pixels:
    // the top-right of the last is at most at `range.end - stride`.
    let (head, out) = image_data[..range.end].split_at_mut(range.start);
    let (top, left) = (&head[range.start - width * 4 - 4..], head.last_chunk().unwrap());
    match mode {
        1 => add_left(out, left),
        2 => add_upper(out, top, |t, _, _| t),
        3 => add_upper(out, top, |_, _, tr| tr),
        4 => add_upper(out, top, |_, tl, _| tl),
        5 => add_serial(out, top, left, |l, t, _, tr| average2(average2(l, tr), t)),
        6 => add_serial(out, top, left, |l, _, tl, _| average2(l, tl)),
        7 => add_serial(out, top, left, |l, t, _, _| average2(l, t)),
        8 => add_upper(out, top, |t, tl, _| average2(tl, t)),
        9 => add_upper(out, top, |t, _, tr| average2(t, tr)),
        10 => add_serial(out, top, left, |l, t, tl, tr| average2(average2(l, tl), average2(t, tr))),
        11 => add_select(out, top, left),
        // `clamp(l + t - tl)` as `l` saturated up by `t - tl` and down by `tl - t`, one of
        // them zero: the left pixel waits on two byte ops, not an unpack and a pack.
        12 => add_serial(out, top, left, |l, t, tl, _| {
            _mm_subs_epu8(_mm_adds_epu8(l, _mm_subs_epu8(t, tl)), _mm_subs_epu8(tl, t))
        }),
        13 => add_serial(out, top, left, |l, t, tl, _| {
            let zero = _mm_setzero_si128();
            let tl = _mm_unpacklo_epi8(tl, zero);
            let avg = _mm_srli_epi16::<1>(_mm_add_epi16(_mm_unpacklo_epi8(t, zero), _mm_unpacklo_epi8(l, zero)));
            // (avg - tl) / 2, rounded toward zero as C divides.
            let d = _mm_sub_epi16(_mm_sub_epi16(avg, tl), _mm_cmpgt_epi16(tl, avg));
            let s = _mm_add_epi16(avg, _mm_srai_epi16::<1>(d));
            _mm_packus_epi16(s, s)
        }),
        _ => add_upper(out, top, |_, _, _| _mm_set1_epi32(0xff00_0000u32 as i32)),
    }
}

/// The colour transform over whole pixels of `row`, with one multiplier triple (red to blue,
/// green to blue, green to red) per block of `block` pixels from `mults`.
#[allow(unsafe_code)]
pub(crate) fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    let n = row.len() / 4;
    assert!(mults.len() >= n.div_ceil(block) * 4);
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        return unsafe { super::avx2::color_transform_row(row, block, mults) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { color_transform_row_sse2(row, block, mults) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn color_transform_row_sse2(row: &mut [u8], block: usize, mults: &[u8]) {
    for (b, m) in row.chunks_mut(block * 4).zip(mults.chunks_exact(4)) {
        color_transform(b, m[0], m[1], m[2])
    }
}

/// libwebp's `TransformColorInverse_SSE2` over the whole pixels of `p`, with red and blue where
/// RGBA keeps them: the deltas `(t * c) >> 5` of signed bytes are the high halves of the 16-bit
/// products of `c << 8` and `t << 3`.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn color_transform(p: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
    let cst = |m: u8| i32::from(((m as i8 as i16) << 3) as u16);
    let mults_rb = _mm_set1_epi32(cst(green_to_blue) << 16 | cst(green_to_red));
    let mults_b2 = _mm_set1_epi32(cst(red_to_blue));
    let mask_ag = _mm_set1_epi32(0xff00ff00u32 as i32);
    let step = |input: __m128i| {
        let ag = _mm_and_si128(input, mask_ag);
        let g = _mm_shufflehi_epi16::<0xa0>(_mm_shufflelo_epi16::<0xa0>(ag));
        let rb = _mm_add_epi8(input, _mm_mulhi_epi16(g, mults_rb));
        // Red and blue in the high bytes of their halves; the red delta of blue goes to byte 3.
        let rb = _mm_slli_epi16::<8>(rb);
        let delta = _mm_slli_epi32::<24>(_mm_mulhi_epi16(rb, mults_b2));
        _mm_or_si128(_mm_srli_epi16::<8>(_mm_add_epi8(rb, delta)), ag)
    };
    let (four, rest) = p.as_chunks_mut::<16>();
    for c in four {
        let v = step(load(c));
        store(c, v);
    }
    for c in rest.as_chunks_mut::<4>().0 {
        store1(c, step(load1(c)));
    }
}

/// Adds green to red and blue.
#[allow(unsafe_code)]
pub(crate) fn add_green(image_data: &mut [u8]) {
    let mut i = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        i = unsafe { super::avx2::add_green(image_data) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { add_green_sse2(&mut image_data[4 * i..]) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn add_green_sse2(image_data: &mut [u8]) {
    let step = |v: __m128i| {
        let g = _mm_and_si128(_mm_srli_epi32::<8>(v), _mm_set1_epi32(0xff));
        _mm_add_epi8(v, _mm_or_si128(g, _mm_slli_epi32::<16>(g)))
    };
    let (four, rest) = image_data.as_chunks_mut::<16>();
    for c in four {
        let v = step(load(c));
        store(c, v);
    }
    for c in rest.as_chunks_mut::<4>().0 {
        store1(c, step(load1(c)));
    }
}
