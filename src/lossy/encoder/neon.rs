//! NEON kernels ported from libwebp's `dsp/enc_neon.c`, bit-exact with the scalar ones.
//! NEON is part of aarch64, so no runtime detection is needed. Each wrapper checks the
//! lengths its loads and stores rely on.

use core::arch::aarch64::*;

use super::dsp::{Matrix, BPS, I4_LOOKUP1, I4_LOOKUP2, I4_LOOKUP3, MAX_LEVEL};
use crate::simd_mem::neon::{vld3q_u8, vld4q_u8};
use crate::simd_mem::{load, load_at, store, store_at};

const KC1: i16 = 20091;
const KC2: i16 = (35468 / 2) as i16;

/// All 16 pixels of a 4x4 block, row after row. Reads 8 bytes of each row, with vector
/// loads rather than lane inserts from general registers.
#[target_feature(enable = "neon")]
#[inline]
fn load4x4(p: &[u8]) -> uint8x16_t {
    let p = &p[..3 * BPS + 8];
    let row = |y: usize| vreinterpret_u32_u8(load_at(p, y * BPS));
    let r01 = vzip1_u32(row(0), row(1));
    let r23 = vzip1_u32(row(2), row(3));
    vreinterpretq_u8_u32(vcombine_u32(r01, r23))
}

#[target_feature(enable = "neon")]
#[inline]
fn transform_pass(rows: int16x8x2_t) -> int16x8x2_t {
    let b1 = vcombine_s16(vget_high_s16(rows.0), vget_high_s16(rows.1));
    let c0 = vsraq_n_s16::<1>(b1, vqdmulhq_n_s16(b1, KC1));
    let c1 = vqdmulhq_n_s16(b1, KC2);
    let a = vqadd_s16(vget_low_s16(rows.0), vget_low_s16(rows.1));
    let b = vqsub_s16(vget_low_s16(rows.0), vget_low_s16(rows.1));
    let c = vqsub_s16(vget_low_s16(c1), vget_high_s16(c0));
    let d = vqadd_s16(vget_low_s16(c0), vget_high_s16(c1));
    let d0 = vcombine_s16(a, b);
    let d1 = vcombine_s16(d, c);
    let e0 = vqaddq_s16(d0, d1);
    let e_tmp = vqsubq_s16(d0, d1);
    let e1 = vcombine_s16(vget_high_s16(e_tmp), vget_low_s16(e_tmp));
    let tmp0 = vzipq_s16(e0, e1);
    let out = vzipq_s16(tmp0.0, tmp0.1);
    int16x8x2_t(out.0, out.1)
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn itransform_one(r: &[u8], input: &[i16], dst: &mut [u8]) {
    // SAFETY: NEON is part of aarch64.
    unsafe { itransform_one_neon(r, input, dst) }
}

#[target_feature(enable = "neon")]
#[inline]
fn itransform_one_neon(r: &[u8], input: &[i16], dst: &mut [u8]) {
    let input = &input[..16];
    let r = &r[..3 * BPS + 8];
    let dst = &mut dst[..3 * BPS + 4];
    let rows = int16x8x2_t(load(input), load_at(input, 8));
    let rows = transform_pass(transform_pass(rows));
    let px = vreinterpretq_u32_u8(load4x4(r));
    let dst01 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vget_low_u32(px))));
    let dst23 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vget_high_u32(px))));
    let out01 = vqmovun_s16(vrsraq_n_s16::<3>(dst01, rows.0));
    let out23 = vqmovun_s16(vrsraq_n_s16::<3>(dst23, rows.1));
    let (o01, o23) = (vreinterpret_u32_u8(out01), vreinterpret_u32_u8(out23));
    let rows = [
        vget_lane_u32::<0>(o01),
        vget_lane_u32::<1>(o01),
        vget_lane_u32::<0>(o23),
        vget_lane_u32::<1>(o23),
    ];
    for (y, row) in rows.into_iter().enumerate() {
        store_at(dst, y * BPS, row);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn transpose4x4_s16(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
    d: int16x4_t,
) -> (int16x8_t, int16x8_t) {
    let ab = vtrn_s16(a, b);
    let cd = vtrn_s16(c, d);
    let tmp02 = vtrn_s32(vreinterpret_s32_s16(ab.0), vreinterpret_s32_s16(cd.0));
    let tmp13 = vtrn_s32(vreinterpret_s32_s16(ab.1), vreinterpret_s32_s16(cd.1));
    (
        vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp02.0),
            vreinterpret_s64_s32(tmp13.0),
        )),
        vreinterpretq_s16_s64(vcombine_s64(
            vreinterpret_s64_s32(tmp13.1),
            vreinterpret_s64_s32(tmp02.1),
        )),
    )
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn ftransform(src: &[u8], r: &[u8], out: &mut [i16]) {
    // SAFETY: NEON is part of aarch64.
    unsafe { ftransform_neon(src, r, out) }
}

#[target_feature(enable = "neon")]
#[inline]
fn ftransform_neon(src: &[u8], r: &[u8], out: &mut [i16]) {
    let out = &mut out[..16];
    let s0 = load4x4(src);
    let r0 = load4x4(r);
    let d0d1 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s0), vget_low_u8(r0)));
    let d2d3 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s0), vget_high_u8(r0)));
    let (d0d1, d3d2) = transpose4x4_s16(
        vget_low_s16(d0d1),
        vget_high_s16(d0d1),
        vget_low_s16(d2d3),
        vget_high_s16(d2d3),
    );

    // First pass.
    let a0a1 = vaddq_s16(d0d1, d3d2);
    let a3a2 = vsubq_s16(d0d1, d3d2);
    let a0a1_2 = vshlq_n_s16::<3>(a0a1);
    let tmp0 = vadd_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
    let tmp2 = vsub_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
    let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
    let tmp1 = vshrn_n_s32::<9>(vaddq_s32(a2_p_a3, vdupq_n_s32(1812)));
    let tmp3 = vshrn_n_s32::<9>(vaddq_s32(a3_m_a2, vdupq_n_s32(937)));
    let (d0d1, d3d2) = transpose4x4_s16(tmp0, tmp1, tmp2, tmp3);

    // Second pass.
    let a0a1 = vaddq_s16(d0d1, d3d2);
    let a3a2 = vsubq_s16(d0d1, d3d2);
    let a0_k7 = vadd_s16(vget_low_s16(a0a1), vdup_n_s16(7));
    let out0 = vshr_n_s16::<4>(vadd_s16(a0_k7, vget_high_s16(a0a1)));
    let out2 = vshr_n_s16::<4>(vsub_s16(a0_k7, vget_high_s16(a0a1)));
    let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);
    let tmp1 = vaddhn_s32(a2_p_a3, vdupq_n_s32(12000 + (1 << 16)));
    let out3 = vaddhn_s32(a3_m_a2, vdupq_n_s32(51000));
    let a3_eq_0 = vreinterpret_s16_u16(vceq_s16(vget_low_s16(a3a2), vdup_n_s16(0)));
    let out1 = vadd_s16(tmp1, a3_eq_0);
    store(out, out0);
    store_at(out, 4, out1);
    store_at(out, 8, out2);
    store_at(out, 12, out3);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn sse16xn(a: &[u8], b: &[u8], rows: usize) -> i32 {
    // SAFETY: NEON is part of aarch64.
    unsafe { sse16xn_neon(a, b, rows) }
}

#[target_feature(enable = "neon")]
#[inline]
fn sse16xn_neon(a: &[u8], b: &[u8], rows: usize) -> i32 {
    let a = &a[..(rows - 1) * BPS + 16];
    let b = &b[..(rows - 1) * BPS + 16];
    let mut sum = vdupq_n_u32(0);
    for y in 0..rows {
        let abs_diff = vabdq_u8(load_at(a, y * BPS), load_at(b, y * BPS));
        let prod1 = vmull_u8(vget_low_u8(abs_diff), vget_low_u8(abs_diff));
        let prod2 = vmull_u8(vget_high_u8(abs_diff), vget_high_u8(abs_diff));
        sum = vaddq_u32(sum, vaddq_u32(vpaddlq_u16(prod1), vpaddlq_u16(prod2)));
    }
    vaddvq_u32(sum) as i32
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn sse4x4(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: NEON is part of aarch64.
    unsafe { sse4x4_neon(a, b) }
}

#[target_feature(enable = "neon")]
#[inline]
fn sse4x4_neon(a: &[u8], b: &[u8]) -> i32 {
    let abs_diff = vabdq_u8(load4x4(a), load4x4(b));
    let prod1 = vmull_u8(vget_low_u8(abs_diff), vget_low_u8(abs_diff));
    let prod2 = vmull_u8(vget_high_u8(abs_diff), vget_high_u8(abs_diff));
    vaddvq_u32(vaddq_u32(vpaddlq_u16(prod1), vpaddlq_u16(prod2))) as i32
}

/// `|weighted Hadamard sum of b - that of a| >> 5`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn tdisto4x4(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    // SAFETY: NEON is part of aarch64.
    unsafe { tdisto4x4_neon(a, b, w) }
}

#[target_feature(enable = "neon")]
#[inline]
fn tdisto4x4_neon(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let a = &a[..3 * BPS + 8];
    let b = &b[..3 * BPS + 8];
    let ld = |p: &[u8], y: usize| vreinterpret_u32_u8(load_at(p, y * BPS));
    let row = |y: usize| vreinterpret_u8_u32(vzip1_u32(ld(a, y), ld(b, y)));
    let (i0, i1, i2, i3) = (row(0), row(1), row(2), row(3));

    // Vertical pass.
    let q_a0 = vreinterpretq_s16_u16(vaddl_u8(i0, i2));
    let q_a1 = vreinterpretq_s16_u16(vaddl_u8(i1, i3));
    let q_a2 = vreinterpretq_s16_u16(vsubl_u8(i1, i3));
    let q_a3 = vreinterpretq_s16_u16(vsubl_u8(i0, i2));
    let v = [
        vaddq_s16(q_a0, q_a1),
        vaddq_s16(q_a3, q_a2),
        vsubq_s16(q_a3, q_a2),
        vsubq_s16(q_a0, q_a1),
    ];

    // Transpose.
    let t0 = vtrnq_s16(v[0], v[1]);
    let t1 = vtrnq_s16(v[2], v[3]);
    let t2 = vtrnq_s32(vreinterpretq_s32_s16(t0.0), vreinterpretq_s32_s16(t1.0));
    let t3 = vtrnq_s32(vreinterpretq_s32_s16(t0.1), vreinterpretq_s32_s16(t1.1));
    let t = [
        vreinterpretq_s16_s32(t2.0),
        vreinterpretq_s16_s32(t3.0),
        vreinterpretq_s16_s32(t2.1),
        vreinterpretq_s16_s32(t3.1),
    ];

    // Horizontal pass.
    let q_a0 = vaddq_s16(t[0], t[2]);
    let q_a1 = vaddq_s16(t[1], t[3]);
    let q_a3 = vsubq_s16(t[0], t[2]);
    let q_a2 = vsubq_s16(t[1], t[3]);
    let h = [
        vabsq_s16(vaddq_s16(q_a0, q_a1)),
        vabsq_s16(vaddq_s16(q_a3, q_a2)),
        vabdq_s16(q_a3, q_a2),
        vabdq_s16(q_a0, q_a1),
    ];

    let q_w07 = load::<int16x8_t, _>(w);
    let q_w8f = load_at::<int16x8_t, _>(w, 8);
    let wv = [
        vget_low_s16(q_w07),
        vget_high_s16(q_w07),
        vget_low_s16(q_w8f),
        vget_high_s16(q_w8f),
    ];
    let mut sums = [vdupq_n_s32(0); 4];
    for k in 0..4 {
        sums[k] = vmlsl_s16(
            vmull_s16(wv[k], vget_low_s16(h[k])),
            wv[k],
            vget_high_s16(h[k]),
        );
    }
    let sum = vaddq_s32(vaddq_s32(sums[0], sums[1]), vaddq_s32(sums[2], sums[3]));
    vaddvq_s32(sum).abs() >> 5
}

#[target_feature(enable = "neon")]
#[inline]
fn quantize8(input: &mut [i16], mtx: &Matrix, offset: usize) -> int16x8_t {
    let sharp: uint16x8_t = load_at(&mtx.sharpen, offset);
    let q: uint16x8_t = load_at(&mtx.q, offset);
    let iq: uint16x8_t = load_at(&mtx.iq, offset);
    let bias0: uint32x4_t = load_at(&mtx.bias, offset);
    let bias1: uint32x4_t = load_at(&mtx.bias, offset + 4);
    let a: int16x8_t = load_at(input, offset);
    let b = vreinterpretq_u16_s16(vabsq_s16(a));
    let sign = vshrq_n_s16::<15>(a);
    let c = vaddq_u16(b, sharp);
    let m0 = vmull_u16(vget_low_u16(c), vget_low_u16(iq));
    let m1 = vmull_u16(vget_high_u16(c), vget_high_u16(iq));
    let m2 = vhaddq_u32(m0, bias0);
    let m3 = vhaddq_u32(m1, bias1);
    let c0 = vcombine_u16(vshrn_n_u32::<16>(m2), vshrn_n_u32::<16>(m3));
    let c1 = vminq_u16(c0, vdupq_n_u16(MAX_LEVEL as u16));
    let c2 = veorq_s16(vreinterpretq_s16_u16(c1), sign);
    let c3 = vsubq_s16(c2, sign);
    store_at(input, offset, vmulq_s16(c3, vreinterpretq_s16_u16(q)));
    c3
}

const SHUFFLES: [[u8; 8]; 4] = [
    [0, 1, 2, 3, 8, 9, 16, 17],
    [10, 11, 4, 5, 6, 7, 12, 13],
    [18, 19, 24, 25, 26, 27, 20, 21],
    [14, 15, 22, 23, 28, 29, 30, 31],
];

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn quantize_block(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    // SAFETY: NEON is part of aarch64.
    unsafe { quantize_block_neon(input, out, mtx) }
}

#[target_feature(enable = "neon")]
#[inline]
fn quantize_block_neon(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    let input = &mut input[..16];
    let out = &mut out[..16];
    let out0 = quantize8(input, mtx, 0);
    let out1 = quantize8(input, mtx, 8);
    let all = uint8x8x4_t(
        vreinterpret_u8_s16(vget_low_s16(out0)),
        vreinterpret_u8_s16(vget_high_s16(out0)),
        vreinterpret_u8_s16(vget_low_s16(out1)),
        vreinterpret_u8_s16(vget_high_s16(out1)),
    );
    let mut any = vdup_n_u8(0);
    for (k, shuffle) in SHUFFLES.iter().enumerate() {
        let v = vtbl4_u8(all, load(shuffle));
        store_at(out, 4 * k, v);
        any = vorr_u8(any, v);
    }
    vget_lane_u64::<0>(vreinterpret_u64_u8(any)) != 0
}

/// The position of the last non-zero coefficient of `coeffs`, or -1: four mask bits per
/// coefficient narrowed into one word.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn last_non_zero(coeffs: &[i16]) -> i32 {
    // SAFETY: NEON is part of aarch64.
    unsafe { last_non_zero_neon(coeffs) }
}

#[target_feature(enable = "neon")]
#[inline]
fn last_non_zero_neon(coeffs: &[i16]) -> i32 {
    let coeffs = &coeffs[..16];
    let lo: int16x8_t = load(coeffs);
    let hi: int16x8_t = load_at(coeffs, 8);
    let nz = vcombine_u8(vmovn_u16(vtstq_s16(lo, lo)), vmovn_u16(vtstq_s16(hi, hi)));
    let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(nz));
    let bits = vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed));
    (63 - bits.leading_zeros() as i32) >> 2
}

/// libwebp's `Intra4Preds_NEON`: the ten 4x4 predictions picked from the averages of the
/// edge with table lookups, a whole row of four modes per store. `b[t]` is the first top
/// sample, as for `dsp::pred_luma4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn pred_luma4(dst: &mut [u8], b: &[u8], t: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { pred_luma4_neon(dst, b, t) }
}

#[target_feature(enable = "neon")]
#[inline]
fn pred_luma4_neon(dst: &mut [u8], b: &[u8], t: usize) {
    use super::dsp::{I4DC4, I4HD4, I4RD4};
    let edge = &b[t - 5..t + 11];
    let dst = &mut dst[I4DC4..I4DC4 + 7 * BPS + 8];
    let l1: uint8x16x4_t = load(&I4_LOOKUP1);
    let l2: uint8x16x4_t = load(&I4_LOOKUP2);
    let l3: uint8x16x4_t = load(&I4_LOOKUP3);
    // L K J I X A B C D E F G H
    let preload = load(edge);
    let a = vqtbl1q_u8(preload, l1.0);
    let c = vextq_u8::<2>(a, a);
    let mut avg3_all = vrhaddq_u8(vhaddq_u8(a, c), preload);
    let avg2_all = vrhaddq_u8(a, preload);
    let dc_in = vset_lane_u8::<3>(vgetq_lane_u8::<0>(preload), vget_low_u8(c));
    let dc = ((vaddlv_u8(dc_in) + 4) >> 3) as u8;
    avg3_all = vsetq_lane_u8::<15>(vgetq_lane_u8::<0>(preload), avg3_all);
    avg3_all = vsetq_lane_u8::<14>(dc, avg3_all);
    let combined = uint8x16x2_t(avg2_all, avg3_all);

    // TrueMotion: left + top - top-left, saturated.
    let sub_a = vdup_laneq_u8::<4>(preload);
    let full_b = vqtbl1q_u8(preload, l1.1);
    let abcd = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(vextq_u8::<5>(preload, preload)));
    let sub = vsubl_u8(vreinterpret_u8_u32(vdup_n_u32(abcd)), sub_a);
    let tm_lo = vqmovun_s16(vreinterpretq_s16_u16(vaddw_u8(sub, vget_low_u8(full_b))));
    let tm_hi = vqmovun_s16(vreinterpretq_s16_u16(vaddw_u8(sub, vget_high_u8(full_b))));
    let tm = [
        vget_lane_u32::<0>(vreinterpret_u32_u8(tm_lo)),
        vget_lane_u32::<1>(vreinterpret_u32_u8(tm_lo)),
        vget_lane_u32::<0>(vreinterpret_u32_u8(tm_hi)),
        vget_lane_u32::<1>(vreinterpret_u32_u8(tm_hi)),
    ];

    for (y, (tbl3, tbl2)) in [(l3.0, l2.0), (l3.1, l2.1), (l3.2, l2.2), (l3.3, l2.3)]
        .into_iter()
        .enumerate()
    {
        // DC4, TM4, VE4, HE4, then RD4, VR4, LD4, VL4.
        let r = vreinterpretq_u32_u8(vqtbl2q_u8(combined, tbl3));
        let r = vsetq_lane_u32::<1>(tm[y], r);
        store_at(dst, y * BPS, r);
        store_at(dst, I4RD4 - I4DC4 + y * BPS, vqtbl2q_u8(combined, tbl2));
    }
    // HD4, HU4
    let r0 = vqtbl2q_u8(combined, l1.2);
    let r1 = vqtbl2q_u8(combined, l1.3);
    let hd = I4HD4 - I4DC4;
    store_at(dst, hd, vget_low_u8(r0));
    store_at(dst, hd + BPS, vget_high_u8(r0));
    store_at(dst, hd + 2 * BPS, vget_low_u8(r1));
    store_at(dst, hd + 3 * BPS, vget_high_u8(r1));
}

/// libwebp's `ConvertRGBToY_NEON`: `(16839 r + 33059 g + 6420 b + 2^15) >> 16` plus 16.
#[target_feature(enable = "neon")]
#[inline]
fn rgb_to_y(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let (r, g, b) = (vmovl_u8(r), vmovl_u8(g), vmovl_u8(b));
    let half = |r: uint16x4_t, g: uint16x4_t, b: uint16x4_t| {
        let y = vmlal_n_u16(vmlal_n_u16(vmull_n_u16(r, 16839), g, 33059), b, 6420);
        vrshrn_n_u32::<16>(y)
    };
    let lo = half(vget_low_u16(r), vget_low_u16(g), vget_low_u16(b));
    let hi = half(vget_high_u16(r), vget_high_u16(g), vget_high_u16(b));
    vqmovn_u16(vaddq_u16(vcombine_u16(lo, hi), vdupq_n_u16(16)))
}

/// The luma of the leading pixels of an RGB (`STEP` 3) or RGBA (`STEP` 4) row, 16 at a time.
/// Returns how many it did.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn rgb_to_y_row<const STEP: usize>(line: &[u8], out: &mut [u8]) -> usize {
    // SAFETY: NEON is part of aarch64.
    unsafe { rgb_to_y_row_neon::<STEP>(line, out) }
}

#[target_feature(enable = "neon")]
#[inline]
fn rgb_to_y_row_neon<const STEP: usize>(line: &[u8], out: &mut [u8]) -> usize {
    let n = out.len().min(line.len() / STEP) & !15;
    let (line, out) = (&line[..n * STEP], out[..n].as_chunks_mut::<16>().0);
    let luma = |r: uint8x16_t, g: uint8x16_t, b: uint8x16_t| {
        let lo = rgb_to_y(vget_low_u8(r), vget_low_u8(g), vget_low_u8(b));
        let hi = rgb_to_y(vget_high_u8(r), vget_high_u8(g), vget_high_u8(b));
        vcombine_u8(lo, hi)
    };
    if STEP == 3 {
        for (p, out) in line.as_chunks::<48>().0.iter().zip(out) {
            let c = vld3q_u8(p);
            store(out, luma(c.0, c.1, c.2));
        }
    } else {
        for (p, out) in line.as_chunks::<64>().0.iter().zip(out) {
            let c = vld4q_u8(p);
            store(out, luma(c.0, c.1, c.2));
        }
    }
    n
}
