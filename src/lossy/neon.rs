//! NEON decoder kernels with the API of `sse2.rs`, ported from libwebp's `dsp/dec_neon.c`,
//! `dsp/upsampling_neon.c` and the SSE2 kernels, bit-exact with the portable ones. NEON is
//! part of aarch64, so no runtime detection is needed. Their loads and stores go through
//! `simd_mem`, which checks their bounds; `p` is the offset of the first pixel past the edge.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::aarch64::*;

use crate::simd_mem::{load, load_at, load_rows, neon, store, store_rows, Rows};

#[target_feature(enable = "neon")]
#[inline]
fn flip(x: uint8x16_t) -> int8x16_t {
    vreinterpretq_s8_u8(veorq_u8(x, vdupq_n_u8(0x80)))
}

#[target_feature(enable = "neon")]
#[inline]
fn flip_back(x: int8x16_t) -> uint8x16_t {
    veorq_u8(vreinterpretq_u8_s8(x), vdupq_n_u8(0x80))
}

#[target_feature(enable = "neon")]
#[inline]
fn needs_filter(
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    thresh: u8,
) -> uint8x16_t {
    let a_p0_q0 = vabdq_u8(p0, q0);
    let sum = vqaddq_u8(
        vqaddq_u8(a_p0_q0, a_p0_q0),
        vshrq_n_u8::<1>(vabdq_u8(p1, q1)),
    );
    vcgeq_u8(vdupq_n_u8(thresh), sum)
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn needs_filter2(
    p3: uint8x16_t,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    q3: uint8x16_t,
    thresh: u8,
    ithresh: u8,
) -> uint8x16_t {
    let m1 = vmaxq_u8(vabdq_u8(p3, p2), vabdq_u8(p2, p1));
    let m2 = vmaxq_u8(vabdq_u8(p1, p0), vabdq_u8(q3, q2));
    let m3 = vmaxq_u8(vabdq_u8(q2, q1), vabdq_u8(q1, q0));
    let interior = vcgeq_u8(vdupq_n_u8(ithresh), vmaxq_u8(vmaxq_u8(m1, m2), m3));
    vandq_u8(interior, needs_filter(p1, p0, q0, q1, thresh))
}

#[target_feature(enable = "neon")]
#[inline]
fn needs_hev(
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    hev: u8,
) -> uint8x16_t {
    let m = vmaxq_u8(vabdq_u8(p1, p0), vabdq_u8(q1, q0));
    vcgtq_u8(m, vdupq_n_u8(hev))
}

#[target_feature(enable = "neon")]
#[inline]
fn base_delta(p1: int8x16_t, p0: int8x16_t, q0: int8x16_t, q1: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0, p0);
    let s1 = vqaddq_s8(vqsubq_s8(p1, q1), q0_p0);
    let s2 = vqaddq_s8(q0_p0, s1);
    vqaddq_s8(q0_p0, s2)
}

#[target_feature(enable = "neon")]
#[inline]
fn base_delta0(p0: int8x16_t, q0: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0, p0);
    vqaddq_s8(q0_p0, vqaddq_s8(q0_p0, q0_p0))
}

/// libwebp's `ApplyFilter2NoFlip`: `p0 + (delta + 3) >> 3` and `q0 - (delta + 4) >> 3`.
#[target_feature(enable = "neon")]
#[inline]
fn apply_filter2(p0: int8x16_t, q0: int8x16_t, delta: int8x16_t) -> (int8x16_t, int8x16_t) {
    let d3 = vshrq_n_s8::<3>(vqaddq_s8(delta, vdupq_n_s8(3)));
    let d4 = vshrq_n_s8::<3>(vqaddq_s8(delta, vdupq_n_s8(4)));
    (vqaddq_s8(p0, d3), vqsubq_s8(q0, d4))
}

#[target_feature(enable = "neon")]
#[inline]
fn do_filter2(
    p1: uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: uint8x16_t,
    thresh: u8,
) {
    let mask = needs_filter(p1, *p0, *q0, q1, thresh);
    let (p0s, q0s) = (flip(*p0), flip(*q0));
    let delta = vandq_s8(
        base_delta(flip(p1), p0s, q0s, flip(q1)),
        vreinterpretq_s8_u8(mask),
    );
    let (p0s, q0s) = apply_filter2(p0s, q0s, delta);
    *p0 = flip_back(p0s);
    *q0 = flip_back(q0s);
}

#[target_feature(enable = "neon")]
#[inline]
fn do_filter4(
    p1: &mut uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: &mut uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) {
    let (p1s, q1s) = (flip(*p1), flip(*q1));
    let simple = vandq_u8(mask, hev_mask);
    let delta = base_delta(p1s, flip(*p0), flip(*q0), q1s);
    let (p0s, q0s) = apply_filter2(
        flip(*p0),
        flip(*q0),
        vandq_s8(delta, vreinterpretq_s8_u8(simple)),
    );
    let complex = veorq_u8(simple, mask);
    let delta0 = vandq_s8(base_delta0(p0s, q0s), vreinterpretq_s8_u8(complex));
    let a1 = vshrq_n_s8::<3>(vqaddq_s8(delta0, vdupq_n_s8(4)));
    let a2 = vshrq_n_s8::<3>(vqaddq_s8(delta0, vdupq_n_s8(3)));
    let a3 = vrshrq_n_s8::<1>(a1);
    *p0 = flip_back(vqaddq_s8(p0s, a2));
    *q0 = flip_back(vqsubq_s8(q0s, a1));
    *p1 = flip_back(vqaddq_s8(p1s, a3));
    *q1 = flip_back(vqsubq_s8(q1s, a3));
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn do_filter6(
    p2: &mut uint8x16_t,
    p1: &mut uint8x16_t,
    p0: &mut uint8x16_t,
    q0: &mut uint8x16_t,
    q1: &mut uint8x16_t,
    q2: &mut uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) {
    let (p2s, p1s, q1s, q2s) = (flip(*p2), flip(*p1), flip(*q1), flip(*q2));
    let simple = vandq_u8(mask, hev_mask);
    let delta0 = base_delta(p1s, flip(*p0), flip(*q0), q1s);
    let (p0s, q0s) = apply_filter2(
        flip(*p0),
        flip(*q0),
        vandq_s8(delta0, vreinterpretq_s8_u8(simple)),
    );
    let delta = vandq_s8(delta0, vreinterpretq_s8_u8(veorq_u8(simple, mask)));
    // With S = 9 * a - 1: (9 * a + 63) >> 7 = (S + 64) >> 7, (18 * a + 63) >> 7 =
    // (S + 32) >> 6 and (27 * a + 63) >> 7 = (S + 18 * a + 64) >> 7.
    let (lo, hi) = (vget_low_s8(delta), vget_high_s8(delta));
    let m1 = vdupq_n_s16(-1);
    let s_lo = vmlal_s8(m1, vdup_n_s8(9), lo);
    let s_hi = vmlal_s8(m1, vdup_n_s8(9), hi);
    let z_lo = vmlal_s8(s_lo, vdup_n_s8(18), lo);
    let z_hi = vmlal_s8(s_hi, vdup_n_s8(18), hi);
    let a3 = vcombine_s8(vqrshrn_n_s16::<7>(s_lo), vqrshrn_n_s16::<7>(s_hi));
    let a2 = vcombine_s8(vqrshrn_n_s16::<6>(s_lo), vqrshrn_n_s16::<6>(s_hi));
    let a1 = vcombine_s8(vqrshrn_n_s16::<7>(z_lo), vqrshrn_n_s16::<7>(z_hi));
    *p0 = flip_back(vqaddq_s8(p0s, a1));
    *q0 = flip_back(vqsubq_s8(q0s, a1));
    *p1 = flip_back(vqaddq_s8(p1s, a2));
    *q1 = flip_back(vqsubq_s8(q1s, a2));
    *p2 = flip_back(vqaddq_s8(p2s, a3));
    *q2 = flip_back(vqsubq_s8(q2s, a3));
}

/// Two side-by-side 8x8 byte transposes, one in each half of the registers.
#[target_feature(enable = "neon")]
#[inline]
fn transpose8x8x2(r: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
    let r01 = vtrnq_u8(r[0], r[1]);
    let r23 = vtrnq_u8(r[2], r[3]);
    let r45 = vtrnq_u8(r[4], r[5]);
    let r67 = vtrnq_u8(r[6], r[7]);
    let u16 = vreinterpretq_u16_u8;
    let r02 = vtrnq_u16(u16(r01.0), u16(r23.0));
    let r13 = vtrnq_u16(u16(r01.1), u16(r23.1));
    let r46 = vtrnq_u16(u16(r45.0), u16(r67.0));
    let r57 = vtrnq_u16(u16(r45.1), u16(r67.1));
    let u32 = vreinterpretq_u32_u16;
    let r04 = vtrnq_u32(u32(r02.0), u32(r46.0));
    let r26 = vtrnq_u32(u32(r02.1), u32(r46.1));
    let r15 = vtrnq_u32(u32(r13.0), u32(r57.0));
    let r37 = vtrnq_u32(u32(r13.1), u32(r57.1));
    let u8 = vreinterpretq_u8_u32;
    [
        u8(r04.0),
        u8(r15.0),
        u8(r26.0),
        u8(r37.0),
        u8(r04.1),
        u8(r15.1),
        u8(r26.1),
        u8(r37.1),
    ]
}

/// The 8 columns of 8 rows `a` (low halves) and 8 rows `b` (high halves), one register per
/// column.
#[target_feature(enable = "neon")]
#[inline]
fn columns(a: [uint8x8_t; 8], b: [uint8x8_t; 8]) -> [uint8x16_t; 8] {
    transpose8x8x2(core::array::from_fn(|k| vcombine_u8(a[k], b[k])))
}

/// The inverse of `columns`.
#[target_feature(enable = "neon")]
#[inline]
fn rows(c: [uint8x16_t; 8]) -> ([uint8x8_t; 8], [uint8x8_t; 8]) {
    let t = transpose8x8x2(c);
    (
        core::array::from_fn(|k| vget_low_u8(t[k])),
        core::array::from_fn(|k| vget_high_u8(t[k])),
    )
}

/// The 8 columns of the 16 rows of 8 bytes from `at`, the first 8 rows in the low halves.
#[target_feature(enable = "neon")]
#[inline]
fn load8x16t(buf: &[u8], at: usize, stride: usize) -> [uint8x16_t; 8] {
    columns(load_rows(buf, at, stride), load_rows(buf, at + 8 * stride, stride))
}

#[target_feature(enable = "neon")]
#[inline]
fn store8x16t(buf: &mut [u8], at: usize, stride: usize, c: [uint8x16_t; 8]) {
    let (a, b) = rows(c);
    store_rows(buf, at, stride, a);
    store_rows(buf, at + 8 * stride, stride, b);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn simple_v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { simple_v16_neon(buf, p, stride, thresh) }
}

#[target_feature(enable = "neon")]
#[inline]
fn simple_v16_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    let mut b = Rows::<4, 16>::new(buf, p - 2 * stride, stride);
    let [p1, mut p0, mut q0, q1] = core::array::from_fn(|k| b.load(k, 0));
    do_filter2(p1, &mut p0, &mut q0, q1, thresh);
    b.store(1, 0, p0);
    b.store(2, 0, q0);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn simple_h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { simple_h16_neon(buf, p, stride, thresh) }
}

#[target_feature(enable = "neon")]
#[inline]
fn simple_h16_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    let at = p - 4;
    let mut c = load8x16t(buf, at, stride);
    let [_, _, p1, mut p0, mut q0, q1, _, _] = c;
    do_filter2(p1, &mut p0, &mut q0, q1, thresh);
    (c[3], c[4]) = (p0, q0);
    store8x16t(buf, at, stride, c);
}

pub(crate) fn simple_v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    for k in 1..4 {
        simple_v16(buf, p + 4 * k * stride, stride, thresh);
    }
}

pub(crate) fn simple_h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    for k in 1..4 {
        simple_h16(buf, p + 4 * k, stride, thresh);
    }
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { v16_neon(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
fn v16_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut b = Rows::<8, 16>::new(buf, p - 4 * stride, stride);
    let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] =
        core::array::from_fn(|k| b.load(k, 0));
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );
    for (k, x) in [p2, p1, p0, q0, q1, q2].into_iter().enumerate() {
        b.store(1 + k, 0, x);
    }
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { h16_neon(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
fn h16_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let at = p - 4;
    let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] =
        load8x16t(buf, at, stride);
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );
    store8x16t(buf, at, stride, [p3, p2, p1, p0, q0, q1, q2, q3]);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { v16i_neon(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
fn v16i_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut b = Rows::<16, 16>::new(buf, p, stride);
    let row = |b: &Rows<16, 16>, k: usize| b.load::<uint8x16_t>(k, 0);
    let (mut p3, mut p2, mut p1, mut p0) = (row(&b, 0), row(&b, 1), row(&b, 2), row(&b, 3));
    for k in 0..3 {
        let e = 4 * k + 4;
        let (mut q0, mut q1, q2, q3) = (row(&b, e), row(&b, e + 1), row(&b, e + 2), row(&b, e + 3));
        let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
        let hev_mask = needs_hev(p1, p0, q0, q1, hev);
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
        for (k, x) in [p1, p0, q0, q1].into_iter().enumerate() {
            b.store(e - 2 + k, 0, x);
        }
        (p3, p2, p1, p0) = (q0, q1, q2, q3);
    }
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: NEON is part of aarch64.
    unsafe { h16i_neon(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
fn h16i_neon(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // All 16 columns: the left 8 and the right 8, each filtered edge in registers.
    let (top, bottom) = (p, p + 8 * stride);
    let t = load_rows::<uint8x16_t, 8>(buf, top, stride);
    let b = load_rows::<uint8x16_t, 8>(buf, bottom, stride);
    let [l0, l1, l2, l3, l4, l5, l6, l7] = columns(t.map(|x| vget_low_u8(x)), b.map(|x| vget_low_u8(x)));
    let [r0, r1, r2, r3, r4, r5, r6, r7] = columns(t.map(|x| vget_high_u8(x)), b.map(|x| vget_high_u8(x)));
    let mut c = [
        l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7,
    ];
    for e in [4, 8, 12] {
        let mask = needs_filter2(
            c[e - 4],
            c[e - 3],
            c[e - 2],
            c[e - 1],
            c[e],
            c[e + 1],
            c[e + 2],
            c[e + 3],
            thresh,
            ithresh,
        );
        let hev_mask = needs_hev(c[e - 2], c[e - 1], c[e], c[e + 1], hev);
        let [mut p1, mut p0, mut q0, mut q1] = [c[e - 2], c[e - 1], c[e], c[e + 1]];
        do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
        (c[e - 2], c[e - 1], c[e], c[e + 1]) = (p1, p0, q0, q1);
    }
    let (lt, lb) = rows(core::array::from_fn(|k| c[k]));
    let (rt, rb) = rows(core::array::from_fn(|k| c[k + 8]));
    store_rows::<_, 8>(buf, top, stride, core::array::from_fn(|k| vcombine_u8(lt[k], rt[k])));
    store_rows::<_, 8>(buf, bottom, stride, core::array::from_fn(|k| vcombine_u8(lb[k], rb[k])));
}

/// The 8 rows of `u` (low halves) and `v` (high halves).
#[target_feature(enable = "neon")]
#[inline]
fn load_uv(u: &Rows<8, 8>, v: &Rows<8, 8>) -> [uint8x16_t; 8] {
    core::array::from_fn(|k| vcombine_u8(u.load(k, 0), v.load(k, 0)))
}

/// `r` to the rows of `u` (low halves) and `v` (high halves) from `from` on.
#[target_feature(enable = "neon")]
#[inline]
fn store_uv<const N: usize>(u: &mut Rows<8, 8>, v: &mut Rows<8, 8>, from: usize, r: [uint8x16_t; N]) {
    for (k, x) in r.into_iter().enumerate() {
        u.store(from + k, 0, vget_low_u8(x));
        v.store(from + k, 0, vget_high_u8(x));
    }
}

#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn v8(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    // SAFETY: NEON is part of aarch64.
    unsafe { v8_neon(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn v8_neon(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    let at = p - 4 * stride;
    let (mut u, mut v) = (Rows::new(u, at, stride), Rows::new(v, at, stride));
    let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] = load_uv(&u, &v);
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );
    store_uv(&mut u, &mut v, 1, [p2, p1, p0, q0, q1, q2]);
}

#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn h8(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    // SAFETY: NEON is part of aarch64.
    unsafe { h8_neon(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn h8_neon(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    let at = p - 4;
    let [p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3] =
        columns(load_rows(u, at, stride), load_rows(v, at, stride));
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );
    let (ru, rv) = rows([p3, p2, p1, p0, q0, q1, q2, q3]);
    store_rows(u, at, stride, ru);
    store_rows(v, at, stride, rv);
}

#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn v8i(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    // SAFETY: NEON is part of aarch64.
    unsafe { v8i_neon(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn v8i_neon(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    let (mut u, mut v) = (Rows::new(u, p, stride), Rows::new(v, p, stride));
    let [p3, p2, mut p1, mut p0, mut q0, mut q1, q2, q3] = load_uv(&u, &v);
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
    store_uv(&mut u, &mut v, 2, [p1, p0, q0, q1]);
}

#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn h8i(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    // SAFETY: NEON is part of aarch64.
    unsafe { h8i_neon(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn h8i_neon(
    u: &mut [u8],
    v: &mut [u8],
    p: usize,
    stride: usize,
    thresh: u8,
    ithresh: u8,
    hev: u8,
) {
    let [p3, p2, mut p1, mut p0, mut q0, mut q1, q2, q3] =
        columns(load_rows(u, p, stride), load_rows(v, p, stride));
    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, thresh, ithresh);
    let hev_mask = needs_hev(p1, p0, q0, q1, hev);
    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);
    let (ru, rv) = rows([p3, p2, p1, p0, q0, q1, q2, q3]);
    store_rows(u, p, stride, ru);
    store_rows(v, p, stride, rv);
}

/// libwebp's `CONVERT8` of `upsampling_neon.c`: 8 pixels to R, G, B. The samples are
/// shifted up by 7 so that `vqdmulh` (`2 * a * b >> 16`) gives libwebp's `MultHi`, and
/// `u * 33050 >> 8` is `u << 7` plus `u * 282 >> 8`.
#[target_feature(enable = "neon")]
#[inline]
fn yuv444_to_rgb(y: uint8x8_t, u: uint8x8_t, v: uint8x8_t) -> [uint8x8_t; 3] {
    let y0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(y));
    let u0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(u));
    let v0 = vreinterpretq_s16_u16(vshll_n_u8::<7>(v));
    let y1 = vqdmulhq_n_s16(y0, 19077);
    let r0 = vqdmulhq_n_s16(v0, 26149);
    let g0 = vqdmulhq_n_s16(u0, 6419);
    let g1 = vqdmulhq_n_s16(v0, 13320);
    let b0 = vqdmulhq_n_s16(u0, 282);
    let r = vqaddq_s16(r0, vqaddq_s16(y1, vdupq_n_s16(-14234)));
    let g = vqsubq_s16(vqaddq_s16(y1, vdupq_n_s16(8708)), vqaddq_s16(g0, g1));
    let b = vqaddq_s16(vqaddq_s16(b0, vqaddq_s16(y1, vdupq_n_s16(-17685))), u0);
    [
        vqshrun_n_s16::<6>(r),
        vqshrun_n_s16::<6>(g),
        vqshrun_n_s16::<6>(b),
    ]
}

/// `yuv444_to_rgb` of 16 pixels.
#[target_feature(enable = "neon")]
#[inline]
fn yuv444_to_rgb16(y: &[u8; 16], u: &[u8; 16], v: &[u8; 16]) -> [uint8x16_t; 3] {
    let lo = |s: &[u8; 16]| load::<uint8x8_t, _>(&s[..8]);
    let hi = |s: &[u8; 16]| load::<uint8x8_t, _>(&s[8..]);
    let [r0, g0, b0] = yuv444_to_rgb(lo(y), lo(u), lo(v));
    let [r1, g1, b1] = yuv444_to_rgb(hi(y), hi(u), hi(v));
    [
        vcombine_u8(r0, r1),
        vcombine_u8(g0, g1),
        vcombine_u8(b0, b1),
    ]
}

/// RGB (`BPP` 3) or RGBA with alpha 255 (`BPP` 4) of the leading pixels of a row with
/// full-width chroma, 16 or 8 pixels at a time. Returns how many pixels it converted.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    // SAFETY: NEON is part of aarch64.
    unsafe { yuv_to_rgb_row_neon::<BPP>(y, u, v, out) }
}

#[target_feature(enable = "neon")]
#[inline]
fn yuv_to_rgb_row_neon<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let m = n & !15;
    let (y16, u16, v16) = (
        y.as_chunks::<16>().0,
        u.as_chunks::<16>().0,
        v.as_chunks::<16>().0,
    );
    let blocks = y16.iter().zip(u16.iter().zip(v16));
    if BPP == 3 {
        for (o, (y, (u, v))) in out.as_chunks_mut::<48>().0.iter_mut().zip(blocks) {
            let [r, g, b] = yuv444_to_rgb16(y, u, v);
            neon::vst3q_u8(o, uint8x16x3_t(r, g, b));
        }
    } else {
        for (o, (y, (u, v))) in out.as_chunks_mut::<64>().0.iter_mut().zip(blocks) {
            let [r, g, b] = yuv444_to_rgb16(y, u, v);
            neon::vst4q_u8(o, uint8x16x4_t(r, g, b, vdupq_n_u8(255)));
        }
    }
    if m + 8 > n {
        return m;
    }
    let [r, g, b] = yuv444_to_rgb(load_at(y, m), load_at(u, m), load_at(v, m));
    let o = &mut out[BPP * m..];
    if BPP == 3 {
        neon::vst3_u8(o, uint8x8x3_t(r, g, b));
    } else {
        neon::vst4_u8(o, uint8x8x4_t(r, g, b, vdup_n_u8(255)));
    }
    m + 8
}

/// A 4x4 transpose of 16-bit lanes.
#[target_feature(enable = "neon")]
#[inline]
fn transpose4x4(r: [int16x4_t; 4]) -> [int16x4_t; 4] {
    let a = vtrn_s16(r[0], r[1]);
    let b = vtrn_s16(r[2], r[3]);
    let c = vtrn_s32(vreinterpret_s32_s16(a.0), vreinterpret_s32_s16(b.0));
    let d = vtrn_s32(vreinterpret_s32_s16(a.1), vreinterpret_s32_s16(b.1));
    [
        vreinterpret_s16_s32(c.0),
        vreinterpret_s16_s32(d.0),
        vreinterpret_s16_s32(c.1),
        vreinterpret_s16_s32(d.1),
    ]
}

/// One pass of libwebp's `TransformPass_NEON`, saturating: `x * 35468 >> 16` is `vqdmulh` by
/// 17734, and `x + (x * 20091 >> 16)` adds half of `vqdmulh` by 20091.
#[target_feature(enable = "neon")]
#[inline]
fn transform_pass(i: [int16x4_t; 4]) -> [int16x4_t; 4] {
    let mul1 = |x: int16x4_t| vsra_n_s16::<1>(x, vqdmulh_n_s16(x, 20091));
    let mul2 = |x: int16x4_t| vqdmulh_n_s16(x, 17734);
    let a = vqadd_s16(i[0], i[2]);
    let b = vqsub_s16(i[0], i[2]);
    let c = vqsub_s16(mul2(i[1]), mul1(i[3]));
    let d = vqadd_s16(mul1(i[1]), mul2(i[3]));
    [
        vqadd_s16(a, d),
        vqadd_s16(b, c),
        vqsub_s16(b, c),
        vqsub_s16(a, d),
    ]
}

#[target_feature(enable = "neon")]
#[inline]
fn widen4(x: u32) -> uint8x8_t {
    vreinterpret_u8_u32(vdup_n_u32(x))
}

#[target_feature(enable = "neon")]
#[inline]
fn narrow4(x: uint8x8_t) -> u32 {
    vget_lane_u32::<0>(vreinterpret_u32_u8(x))
}

/// `t` added to the 4 pixels `x`, saturated to 8 bits.
#[target_feature(enable = "neon")]
#[inline]
fn add_row(x: u32, t: int16x4_t) -> u32 {
    let d = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(widen4(x))));
    let s = vadd_s16(d, t);
    narrow4(vqmovun_s16(vcombine_s16(s, s)))
}

/// `t` added to the 4x4 pixels at `pos` of `dst`, each row rounded and shifted by 3.
#[target_feature(enable = "neon")]
#[inline]
fn add_rows(dst: &mut [u8], pos: usize, stride: usize, t: [int16x4_t; 4]) {
    let px = load_rows::<u32, 4>(dst, pos, stride);
    let px: [u32; 4] = core::array::from_fn(|k| add_row(px[k], vrshr_n_s16::<3>(t[k])));
    store_rows(dst, pos, stride, px);
}

/// libwebp's `TransformOne_NEON`: the inverse transform of `block` (16 coefficients that fit 16
/// bits) in saturating 16 bits, added to the 4x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn transform_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { transform_add_neon(block, dst, pos, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn transform_add_neon(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let block = &block[..16];
    let t = transpose4x4(transform_pass(core::array::from_fn(|r| load_at(block, 4 * r))));
    add_rows(dst, pos, stride, transpose4x4(transform_pass(t)));
}

/// libwebp's `TransformAC3_NEON`: the inverse transform of `block`, whose only coefficients are
/// 0, 1 and 4, in saturating 16 bits, added to the 4x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn transform_ac3_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { transform_ac3_add_neon(block, dst, pos, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn transform_ac3_add_neon(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let block = &block[..16];
    let mul1 = |a: i16| (((i32::from(a) * 20091) >> 16) + i32::from(a)) as i16;
    let mul2 = |a: i16| ((i32::from(a) * 35468) >> 16) as i16;
    let (c1, d1) = (mul2(block[1]), mul1(block[1]));
    let cd = [d1, c1, c1.wrapping_neg(), d1.wrapping_neg()];
    let b = vqadd_s16(vdup_n_s16(block[0]), load(&cd));
    let (c4, d4) = (vdup_n_s16(mul2(block[4])), vdup_n_s16(mul1(block[4])));
    let rows = [
        vqadd_s16(b, d4),
        vqadd_s16(b, c4),
        vqsub_s16(b, c4),
        vqsub_s16(b, d4),
    ];
    add_rows(dst, pos, stride, rows);
}

/// The interpolated pixels `out[2 * j + 1]` and `out[2 * j + 2]` of the leading chroma pairs
/// `j` of `upsample_chroma_row`, 16 or 8 pairs at a time in 8 bits as libwebp's
/// `UPSAMPLE_16PIXELS`. Returns how many pairs it did.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    // SAFETY: NEON is part of aarch64.
    unsafe { upsample_pairs_neon(near, far, out) }
}

#[target_feature(enable = "neon")]
#[inline]
fn upsample_pairs_neon(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    if pairs < 8 {
        return 0;
    }
    let one = vdupq_n_u8(1);
    let pairs16 = |a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t| {
        let s = vrhaddq_u8(a, d);
        let t = vrhaddq_u8(b, c);
        let st = veorq_u8(s, t);
        let ad = veorq_u8(a, d);
        let bc = veorq_u8(b, c);
        let lsb = vandq_u8(vorrq_u8(vorrq_u8(ad, bc), st), one);
        let k = vsubq_u8(vrhaddq_u8(s, t), lsb);
        let get_m = |ij: uint8x16_t, x: uint8x16_t| {
            let lsb = vandq_u8(vorrq_u8(vandq_u8(ij, st), veorq_u8(k, x)), one);
            vsubq_u8(vrhaddq_u8(k, x), lsb)
        };
        uint8x16x2_t(vrhaddq_u8(a, get_m(bc, t)), vrhaddq_u8(b, get_m(ad, s)))
    };
    let m = pairs & !15;
    let (a, b) = (near.as_chunks::<16>().0, near[1..].as_chunks::<16>().0);
    let (c, d) = (far.as_chunks::<16>().0, far[1..].as_chunks::<16>().0);
    let o = out[1..].as_chunks_mut::<32>().0;
    for (o, (a, (b, (c, d)))) in o.iter_mut().zip(a.iter().zip(b.iter().zip(c.iter().zip(d)))) {
        neon::vst2q_u8(o, pairs16(load(a), load(b), load(c), load(d)));
    }
    if m + 8 > pairs {
        return m;
    }
    let half = |s: &[u8], at: usize| vcombine_u8(load_at(s, at), vdup_n_u8(0));
    let eo = pairs16(half(near, m), half(near, m + 1), half(far, m), half(far, m + 1));
    neon::vst2_u8(
        &mut out[2 * m + 1..],
        uint8x8x2_t(vget_low_u8(eo.0), vget_low_u8(eo.1)),
    );
    m + 8
}

/// libwebp's `DC16`, `DC8uv` and their `NoTop`, `NoLeft` and `NoTopLeft` variants: a
/// `SIZE`x`SIZE` block (8 or 16) at `p` set to the rounded mean of the row above and the
/// column to the left, as far as they are given.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn dc<const SIZE: usize>(
    buf: &mut [u8],
    p: usize,
    stride: usize,
    top: bool,
    left: bool,
) {
    // SAFETY: NEON is part of aarch64.
    unsafe { dc_neon::<SIZE>(buf, p, stride, top, left) }
}

#[target_feature(enable = "neon")]
#[inline]
fn dc_neon<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize, top: bool, left: bool) {
    let shift = SIZE.trailing_zeros() - 1;
    let mut sum = 0;
    if top {
        sum += if SIZE == 8 {
            u32::from(vaddlv_u8(load_at(buf, p - stride)))
        } else {
            u32::from(vaddlvq_u8(load_at(buf, p - stride)))
        };
    }
    if left {
        let col = Rows::<SIZE, 1>::new(buf, p - 1, stride);
        sum += (0..SIZE).map(|y| u32::from(col.load::<u8>(y, 0))).sum::<u32>();
    }
    let dc = match (top, left) {
        (true, true) => (sum + SIZE as u32) >> (shift + 2),
        (false, false) => 128,
        _ => (sum + SIZE as u32 / 2) >> (shift + 1),
    };
    let v = vdupq_n_u8(dc as u8);
    if SIZE == 8 {
        store_rows(buf, p, stride, [vget_low_u8(v); SIZE]);
    } else {
        store_rows(buf, p, stride, [v; SIZE]);
    }
}

/// libwebp's `TransformDC`: `dc` (the DC coefficient plus rounding, shifted) added to the
/// 4x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn add_dc(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { add_dc_neon(dc, dst, pos, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn add_dc_neon(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    let dc = vdup_n_s16(dc as i16);
    let px = load_rows::<u32, 4>(dst, pos, stride);
    store_rows(dst, pos, stride, px.map(|x| add_row(x, dc)));
}

/// `(a + 2 * b + c + 2) >> 2` per byte, as libwebp's predictors compute it.
#[target_feature(enable = "neon")]
#[inline]
fn avg3(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    vrhadd_u8(vhadd_u8(a, c), b)
}

/// The bytes of `x` from `N` on, zeros shifted in.
#[target_feature(enable = "neon")]
#[inline]
fn shr<const N: i32>(x: uint8x8_t) -> uint8x8_t {
    vext_u8::<N>(x, vdup_n_u8(0))
}

/// The low 4 bytes of each of `rows` to the 4x4 block at `p`.
#[target_feature(enable = "neon")]
#[inline]
fn store4x4(buf: &mut [u8], p: usize, stride: usize, rows: [uint8x8_t; 4]) {
    store_rows(buf, p, stride, rows.map(|x| narrow4(x)));
}

/// libwebp's `VE4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn ve4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { ve4_neon(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn ve4_neon(buf: &mut [u8], p: usize, stride: usize) {
    let t = load_at(buf, p - stride - 1);
    let avg = avg3(t, shr::<1>(t), shr::<2>(t));
    store4x4(buf, p, stride, [avg; 4]);
}

/// libwebp's `LD4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn ld4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { ld4_neon(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn ld4_neon(buf: &mut [u8], p: usize, stride: usize) {
    let t = load_at(buf, p - stride);
    let c = vext_u8::<2>(t, vdup_lane_u8::<7>(t));
    let avg = avg3(t, shr::<1>(t), c);
    store4x4(
        buf,
        p,
        stride,
        [avg, shr::<1>(avg), shr::<2>(avg), shr::<3>(avg)],
    );
}

/// libwebp's `RD4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn rd4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { rd4_neon(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn rd4_neon(buf: &mut [u8], p: usize, stride: usize) {
    let [i, j, k, l] = load_rows::<u8, 4>(buf, p - 1, stride).map(u32::from);
    let lkji = l | (k << 8) | (j << 16) | (i << 24);
    // L K J I X A B C D ...
    let xabcd = load_at(buf, p - stride - 1);
    let e = vcombine_u8(widen4(lkji), xabcd);
    let e = vextq_u8::<4>(e, e);
    let a = vrhaddq_u8(vhaddq_u8(e, vextq_u8::<2>(e, e)), vextq_u8::<1>(e, e));
    let lane = |x: uint8x16_t| vget_low_u8(x);
    store4x4(
        buf,
        p,
        stride,
        [
            lane(vextq_u8::<3>(a, a)),
            lane(vextq_u8::<2>(a, a)),
            lane(vextq_u8::<1>(a, a)),
            lane(a),
        ],
    );
}

/// libwebp's `VR4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn vr4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { vr4_neon(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn vr4_neon(buf: &mut [u8], p: usize, stride: usize) {
    let [x, i, j, k] = load_rows::<u8, 4>(buf, p - stride - 1, stride);
    let xabcd = load_at(buf, p - stride - 1);
    let abcd0 = shr::<1>(xabcd);
    let abcd = vrhadd_u8(xabcd, abcd0);
    let ixabcd = vext_u8::<7>(vdup_n_u8(i), xabcd);
    let efgh = avg3(ixabcd, xabcd, abcd0);
    let zero = vdup_n_u8(0);
    store4x4(
        buf,
        p,
        stride,
        [abcd, efgh, vext_u8::<7>(zero, abcd), vext_u8::<7>(zero, efgh)],
    );
    let avg3_1 =
        |a: u8, b: u8, c: u8| ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8;
    store_rows(buf, p + 2 * stride, stride, [avg3_1(j, i, x), avg3_1(k, j, i)]);
}

/// libwebp's `VL4`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn vl4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { vl4_neon(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn vl4_neon(buf: &mut [u8], p: usize, stride: usize) {
    let one = vdup_n_u8(1);
    let a = load_at(buf, p - stride);
    let b = shr::<1>(a);
    let c = shr::<2>(a);
    let avg1 = vrhadd_u8(a, b);
    let avg2 = vrhadd_u8(c, b);
    let avg3 = vrhadd_u8(avg1, avg2);
    let lsb1 = vand_u8(veor_u8(avg1, avg2), one);
    let abbc = vorr_u8(veor_u8(a, b), veor_u8(c, b));
    let avg4 = vqsub_u8(avg3, vand_u8(abbc, lsb1));
    store4x4(
        buf,
        p,
        stride,
        [avg1, avg4, shr::<1>(avg1), shr::<1>(avg4)],
    );
    store_rows(
        buf,
        p + 2 * stride + 3,
        stride,
        [vget_lane_u8::<4>(avg4), vget_lane_u8::<5>(avg4)],
    );
}

/// libwebp's `TrueMotion` of a `SIZE`x`SIZE` block (4, 8 or 16) at `p`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn tm<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: NEON is part of aarch64.
    unsafe { tm_neon::<SIZE>(buf, p, stride) }
}

#[target_feature(enable = "neon")]
#[inline]
fn tm_neon<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize) {
    let corner = i16::from(buf[p - stride - 1]);
    let t = match SIZE {
        4 => vcombine_u8(widen4(load_at(buf, p - stride)), vdup_n_u8(0)),
        8 => vcombine_u8(load_at(buf, p - stride), vdup_n_u8(0)),
        _ => load_at(buf, p - stride),
    };
    let t0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t)));
    let t1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t)));
    let left = load_rows::<u8, SIZE>(buf, p - 1, stride);
    let row = |y: usize| {
        let base = vdupq_n_s16(i16::from(left[y]) - corner);
        (
            vqmovun_s16(vaddq_s16(base, t0)),
            vqmovun_s16(vaddq_s16(base, t1)),
        )
    };
    match SIZE {
        4 => store_rows(buf, p, stride, core::array::from_fn::<_, SIZE, _>(|y| narrow4(row(y).0))),
        8 => store_rows(buf, p, stride, core::array::from_fn::<_, SIZE, _>(|y| row(y).0)),
        _ => store_rows(
            buf,
            p,
            stride,
            core::array::from_fn::<_, SIZE, _>(|y| {
                let (lo, hi) = row(y);
                vcombine_u8(lo, hi)
            }),
        ),
    }
}

/// The gradient alpha unfiltering of `row[x..]` (`row[x - 1]` done), 16 pixels at a time as
/// a running sum of the residuals and the top row's steps, for as long as no prediction
/// needs clamping. Returns where it stopped: at a block that clamps or with fewer than 16
/// pixels left.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn unfilter_gradient(prev: &[u8], row: &mut [u8], x: usize) -> usize {
    // SAFETY: NEON is part of aarch64.
    unsafe { unfilter_gradient_neon(prev, row, x) }
}

#[target_feature(enable = "neon")]
#[inline]
fn unfilter_gradient_neon(prev: &[u8], row: &mut [u8], mut x: usize) -> usize {
    assert!(x >= 1);
    let (done, row) = row.split_at_mut(x);
    let top_left = prev.get(x - 1..).unwrap_or_default();
    let top = top_left.get(1..).unwrap_or_default();
    let zero = vdupq_n_u8(0);
    let max = vdupq_n_u16(255);
    let mut left = vdupq_n_u8(done[x - 1]);
    let (row, top, top_left) = (
        row.as_chunks_mut::<16>().0,
        top.as_chunks::<16>().0,
        top_left.as_chunks::<16>().0,
    );
    for (r, (top, top_left)) in row.iter_mut().zip(top.iter().zip(top_left)) {
        let (top, top_left): (uint8x16_t, uint8x16_t) = (load(top), load(top_left));
        let mut s = vaddq_u8(load(r), vsubq_u8(top, top_left));
        s = vaddq_u8(s, vextq_u8::<15>(zero, s));
        s = vaddq_u8(s, vextq_u8::<14>(zero, s));
        s = vaddq_u8(s, vextq_u8::<12>(zero, s));
        s = vaddq_u8(s, vextq_u8::<8>(zero, s));
        let out = vaddq_u8(s, left);
        // The unclamped predictions, from the left pixels this block would produce.
        let before = vextq_u8::<15>(left, out);
        let pred = |b: uint8x8_t, t: uint8x8_t, tl: uint8x8_t| vsubw_u8(vaddl_u8(b, t), tl);
        let lo = pred(vget_low_u8(before), vget_low_u8(top), vget_low_u8(top_left));
        let hi = pred(
            vget_high_u8(before),
            vget_high_u8(top),
            vget_high_u8(top_left),
        );
        if vmaxvq_u16(vorrq_u16(vcgtq_u16(lo, max), vcgtq_u16(hi, max))) != 0 {
            break;
        }
        store(r, out);
        left = vdupq_laneq_u8::<15>(out);
        x += 16;
    }
    x
}

/// The horizontal alpha unfiltering of `row[x..]` (`row[x - 1]` done), a running sum 16
/// pixels at a time. Returns where it stopped, with fewer than 16 pixels left.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn unfilter_horizontal(row: &mut [u8], x: usize) -> usize {
    // SAFETY: NEON is part of aarch64.
    unsafe { unfilter_horizontal_neon(row, x) }
}

#[target_feature(enable = "neon")]
#[inline]
fn unfilter_horizontal_neon(row: &mut [u8], mut x: usize) -> usize {
    assert!(x >= 1);
    let (done, row) = row.split_at_mut(x);
    let zero = vdupq_n_u8(0);
    let mut left = vdupq_n_u8(done[x - 1]);
    for r in row.as_chunks_mut::<16>().0 {
        let mut s: uint8x16_t = load(r);
        s = vaddq_u8(s, vextq_u8::<15>(zero, s));
        s = vaddq_u8(s, vextq_u8::<14>(zero, s));
        s = vaddq_u8(s, vextq_u8::<12>(zero, s));
        s = vaddq_u8(s, vextq_u8::<8>(zero, s));
        let out = vaddq_u8(s, left);
        store(r, out);
        left = vdupq_laneq_u8::<15>(out);
        x += 16;
    }
    x
}
