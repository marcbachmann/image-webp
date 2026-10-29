//! NEON kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_neon.c`, bit-exact
//! with the portable ones in `reverse_transform`. NEON is part of aarch64, so no runtime detection
//! is needed. Pixels are RGBA bytes; every predictor works per channel, so the order is libwebp's.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::aarch64::*;
use std::ops::Range;

use super::reverse_transform::with_top;
use crate::simd_mem::{load, store};

/// One pixel, in every lane.
#[target_feature(enable = "neon")]
#[inline]
fn load1(p: &[u8; 4]) -> uint8x16_t {
    vreinterpretq_u8_u32(vdupq_n_u32(load(p)))
}

/// Lane `J` of `v` to the pixel `J` of `p`.
#[target_feature(enable = "neon")]
#[inline]
fn store_lane<const J: i32>(p: &mut [u8], v: uint8x16_t) {
    store(&mut p[4 * J as usize..], vgetq_lane_u32::<J>(vreinterpretq_u32_u8(v)))
}

/// Adds a prediction from the row above only, four pixels at a time: `pred(top, top_left, top_right)`.
#[target_feature(enable = "neon")]
#[inline]
fn add_upper(
    out: &mut [u8],
    top: &[u8],
    pred: impl Fn(uint8x16_t, uint8x16_t, uint8x16_t) -> uint8x16_t,
) {
    let m = out.len() & !15;
    let (four, rest) = out.split_at_mut(m);
    for (o, [tl, t, tr]) in with_top::<16>(four, top) {
        let p = pred(load(t), load(tl), load(tr));
        let v = vaddq_u8(load(o), p);
        store(o, v);
    }
    for (o, [tl, t, tr]) in with_top::<4>(rest, &top[m..]) {
        let p = pred(load1(t), load1(tl), load1(tr));
        let v = vaddq_u8(load1(o), p);
        store_lane::<0>(o, v);
    }
}

/// Adds a prediction that takes the pixel decoded just before, `pred(left, top, top_left,
/// top_right)`. As libwebp's NEON, four pixels share each vector: every lane is predicted and the
/// result's lane `j` kept, then rotated into lane `j + 1` as the next pixel's left.
#[target_feature(enable = "neon")]
#[inline]
fn add_serial(
    out: &mut [u8],
    top: &[u8],
    left: &[u8; 4],
    pred: impl Fn(uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) -> uint8x16_t,
) {
    let mut left = load1(left);
    let m = out.len() & !15;
    let (four, rest) = out.split_at_mut(m);
    for (o, [tl, t, tr]) in with_top::<16>(four, top) {
        let (src, tt, tl, tr) = (load(o), load(t), load(tl), load(tr));
        macro_rules! lane {
            ($j:literal) => {
                let res = vaddq_u8(src, pred(left, tt, tl, tr));
                store_lane::<$j>(o, res);
                left = vextq_u8::<12>(res, res);
            };
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
    }
    for (o, [tl, t, tr]) in with_top::<4>(rest, &top[m..]) {
        left = vaddq_u8(load1(o), pred(left, load1(t), load1(tl), load1(tr)));
        store_lane::<0>(o, left);
    }
}

/// Predictor 1, the left pixel: a prefix sum over each four pixels.
#[target_feature(enable = "neon")]
#[inline]
fn add_left(out: &mut [u8], left: &[u8; 4]) {
    let zero = vdupq_n_u8(0);
    let mut prev = load1(left);
    let (four, rest) = out.as_chunks_mut::<16>();
    for o in four {
        let src = load(o);
        let sum0 = vaddq_u8(src, vextq_u8::<12>(zero, src));
        let sum1 = vaddq_u8(sum0, vextq_u8::<8>(zero, sum0));
        let res = vaddq_u8(sum1, prev);
        store(o, res);
        prev = vreinterpretq_u8_u32(vdupq_laneq_u32::<3>(vreinterpretq_u32_u8(res)));
    }
    for o in rest.as_chunks_mut::<4>().0 {
        prev = vaddq_u8(load1(o), prev);
        store_lane::<0>(o, prev);
    }
}

/// `clamp(a + b - c)` per channel.
#[target_feature(enable = "neon")]
#[inline]
fn clamped_add_subtract_full(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    let half = |a: uint8x8_t, b: uint8x8_t, c: uint8x8_t| {
        let d = vreinterpretq_s16_u16(vsubl_u8(b, c));
        vqmovun_s16(vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(a)), d))
    };
    let lo = half(vget_low_u8(a), vget_low_u8(b), vget_low_u8(c));
    vcombine_u8(lo, half(vget_high_u8(a), vget_high_u8(b), vget_high_u8(c)))
}

/// `clamp(avg + (avg - c) / 2)` per channel, of `avg` the average of `a` and `b`, the division
/// rounding toward zero as C's: one off `c` where it is above the average.
#[target_feature(enable = "neon")]
#[inline]
fn clamped_add_subtract_half(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    let avg = vhaddq_u8(a, b);
    let c = vaddq_u8(c, vcgtq_u8(c, avg));
    let diff = vreinterpretq_s8_u8(vhsubq_u8(avg, c));
    let half = |avg: uint8x8_t, diff: int8x8_t| {
        vqmovun_s16(vaddw_s8(vreinterpretq_s16_u16(vmovl_u8(avg)), diff))
    };
    let lo = half(vget_low_u8(avg), vget_low_s8(diff));
    vcombine_u8(lo, half(vget_high_u8(avg), vget_high_s8(diff)))
}

/// Predictor 1 on `image_data[range]`, which has a pixel before it.
#[allow(unsafe_code)]
pub(crate) fn predict_left(image_data: &mut [u8], range: Range<usize>) {
    // SAFETY: NEON is part of aarch64.
    unsafe { predict_left_neon(image_data, range) }
}

#[target_feature(enable = "neon")]
#[inline]
fn predict_left_neon(image_data: &mut [u8], range: Range<usize>) {
    let (head, out) = image_data[..range.end].split_at_mut(range.start);
    add_left(out, head.last_chunk().unwrap())
}

/// Undoes predictor `mode` (0 to 15) over `image_data[range]`, the pixels of one row from its
/// second on, below the row above at `width` pixels before them.
#[allow(unsafe_code)]
pub(crate) fn predict(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { predict_neon(mode, image_data, range, width) }
}

#[target_feature(enable = "neon")]
#[inline]
fn predict_neon(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    // The row above from the top-left pixel of the first on, which ends before the row's pixels:
    // the top-right of the last is at most at `range.end - stride`.
    let (head, out) = image_data[..range.end].split_at_mut(range.start);
    let (top, left) = (&head[range.start - width * 4 - 4..], head.last_chunk().unwrap());
    match mode {
        1 => add_left(out, left),
        2 => add_upper(out, top, |t, _, _| t),
        3 => add_upper(out, top, |_, _, tr| tr),
        4 => add_upper(out, top, |_, tl, _| tl),
        5 => add_serial(out, top, left, |l, t, _, tr| vhaddq_u8(vhaddq_u8(l, tr), t)),
        6 => add_serial(out, top, left, |l, _, tl, _| vhaddq_u8(l, tl)),
        7 => add_serial(out, top, left, |l, t, _, _| vhaddq_u8(l, t)),
        8 => add_upper(out, top, |t, tl, _| vhaddq_u8(tl, t)),
        9 => add_upper(out, top, |t, _, tr| vhaddq_u8(t, tr)),
        10 => add_serial(out, top, left, |l, t, tl, tr| {
            vhaddq_u8(vhaddq_u8(l, tl), vhaddq_u8(t, tr))
        }),
        11 => add_serial(out, top, left, |l, t, tl, _| {
            // The left pixel where it is further from the top-left than the top pixel is.
            let sad = |a, b| vpaddlq_u16(vpaddlq_u8(vabdq_u8(a, b)));
            let take_top = vcleq_u32(sad(l, tl), sad(t, tl));
            vbslq_u8(vreinterpretq_u8_u32(take_top), t, l)
        }),
        12 => add_serial(out, top, left, |l, t, tl, _| {
            clamped_add_subtract_full(l, t, tl)
        }),
        13 => add_serial(out, top, left, |l, t, tl, _| {
            clamped_add_subtract_half(l, t, tl)
        }),
        _ => add_upper(out, top, |_, _, _| {
            vreinterpretq_u8_u32(vdupq_n_u32(0xff00_0000))
        }),
    }
}

/// The colour transform over whole pixels of `row`, with one multiplier triple (red to blue,
/// green to blue, green to red) per block of `block` pixels from `mults`.
#[allow(unsafe_code)]
pub(crate) fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    let n = row.len() / 4;
    assert!(mults.len() >= n.div_ceil(block) * 4);
    // SAFETY: NEON is part of aarch64.
    unsafe { color_transform_row_neon(row, block, mults) }
}

#[target_feature(enable = "neon")]
#[inline]
fn color_transform_row_neon(row: &mut [u8], block: usize, mults: &[u8]) {
    for (b, m) in row.chunks_mut(block * 4).zip(mults.chunks_exact(4)) {
        color_transform(b, m[0], m[1], m[2])
    }
}

/// libwebp's `TransformColorInverse_NEON` over the whole pixels of `p`, with red and blue where
/// RGBA keeps them: the deltas `(t * c) >> 5` of signed bytes are `vqdmulh` of `t << 8` and `c << 2`.
#[target_feature(enable = "neon")]
#[inline]
fn color_transform(p: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
    let cst = |m: u8| i32::from(i16::from(m as i8) << 2) as u32 & 0xffff;
    let mults_rb = vreinterpretq_s16_u32(vdupq_n_u32(cst(green_to_blue) << 16 | cst(green_to_red)));
    let mults_b2 = vreinterpretq_s16_u32(vdupq_n_u32(cst(red_to_blue)));
    // Green in the high byte of both 16-bit halves of its pixel.
    const G0G: [u8; 16] = [
        255, 1, 255, 1, 255, 5, 255, 5, 255, 9, 255, 9, 255, 13, 255, 13,
    ];
    let g0g = load(&G0G);
    let keep_rb = vreinterpretq_u8_u32(vdupq_n_u32(0x00ff_00ff));
    let step = |input: uint8x16_t| {
        let g = vreinterpretq_s16_u8(vqtbl1q_u8(input, g0g));
        let rb = vaddq_u8(input, vreinterpretq_u8_s16(vqdmulhq_s16(g, mults_rb)));
        // The red delta of blue, from red in the high byte of the first half, to byte 2.
        let r = vshlq_n_s16::<8>(vreinterpretq_s16_u8(rb));
        let delta = vshlq_n_u32::<16>(vreinterpretq_u32_s16(vqdmulhq_s16(r, mults_b2)));
        vbslq_u8(keep_rb, vaddq_u8(rb, vreinterpretq_u8_u32(delta)), input)
    };
    let (four, rest) = p.as_chunks_mut::<16>();
    for c in four {
        let v = step(load(c));
        store(c, v);
    }
    for c in rest.as_chunks_mut::<4>().0 {
        let v = step(load1(c));
        store_lane::<0>(c, v);
    }
}

/// Adds green to red and blue.
#[allow(unsafe_code)]
pub(crate) fn add_green(image_data: &mut [u8]) {
    // SAFETY: NEON is part of aarch64.
    unsafe { add_green_neon(image_data) }
}

#[target_feature(enable = "neon")]
#[inline]
fn add_green_neon(image_data: &mut [u8]) {
    // Green to bytes 0 and 2 of its pixel, zero to 1 and 3.
    const G_G_: [u8; 16] = [
        1, 255, 1, 255, 5, 255, 5, 255, 9, 255, 9, 255, 13, 255, 13, 255,
    ];
    let shuffle = load(&G_G_);
    let step = |v: uint8x16_t| vaddq_u8(v, vqtbl1q_u8(v, shuffle));
    let (four, rest) = image_data.as_chunks_mut::<16>();
    for c in four {
        let v = step(load(c));
        store(c, v);
    }
    for c in rest.as_chunks_mut::<4>().0 {
        let v = step(load1(c));
        store_lane::<0>(c, v);
    }
}
