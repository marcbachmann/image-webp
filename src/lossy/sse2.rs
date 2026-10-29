//! SSE2 decoder kernels ported from libwebp's `dsp/dec_sse2.c`, bit-exact with the portable
//! ones. SSE2 is part of x86_64, so no runtime detection is needed. Each kernel checks the
//! bounds its loads and stores rely on; `p` is the offset of the first pixel past the edge.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::x86_64::*;

use crate::simd_mem::{load, store, Rows};

#[target_feature(enable = "sse2")]
#[inline]
fn abs_diff(p: __m128i, q: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(q, p), _mm_subs_epu8(p, q))
}

#[target_feature(enable = "sse2")]
#[inline]
fn flip(x: __m128i) -> __m128i {
    _mm_xor_si128(x, _mm_set1_epi8(-128))
}

#[target_feature(enable = "sse2")]
#[inline]
fn signed_shift8b(x: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_srai_epi16::<11>(_mm_unpacklo_epi8(zero, x));
    let hi = _mm_srai_epi16::<11>(_mm_unpackhi_epi8(zero, x));
    _mm_packs_epi16(lo, hi)
}

#[target_feature(enable = "sse2")]
#[inline]
fn not_hev(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i, hev_thresh: u8) -> __m128i {
    let t_max = _mm_max_epu8(abs_diff(p1, p0), abs_diff(q1, q0));
    let t_max_h = _mm_subs_epu8(t_max, _mm_set1_epi8(hev_thresh as i8));
    _mm_cmpeq_epi8(t_max_h, _mm_setzero_si128())
}

#[target_feature(enable = "sse2")]
#[inline]
fn base_delta(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i) -> __m128i {
    let p1_q1 = _mm_subs_epi8(p1, q1);
    let q0_p0 = _mm_subs_epi8(q0, p0);
    let s1 = _mm_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm_adds_epi8(q0_p0, s1);
    _mm_adds_epi8(q0_p0, s2)
}

#[target_feature(enable = "sse2")]
#[inline]
fn do_simple_filter(p0: &mut __m128i, q0: &mut __m128i, fl: __m128i) {
    let v3 = signed_shift8b(_mm_adds_epi8(fl, _mm_set1_epi8(3)));
    let v4 = signed_shift8b(_mm_adds_epi8(fl, _mm_set1_epi8(4)));
    *q0 = _mm_subs_epi8(*q0, v4);
    *p0 = _mm_adds_epi8(*p0, v3);
}

#[target_feature(enable = "sse2")]
#[inline]
fn update_2_pixels(pi: &mut __m128i, qi: &mut __m128i, a0_lo: __m128i, a0_hi: __m128i) {
    let delta = _mm_packs_epi16(_mm_srai_epi16::<7>(a0_lo), _mm_srai_epi16::<7>(a0_hi));
    *pi = flip(_mm_adds_epi8(*pi, delta));
    *qi = flip(_mm_subs_epi8(*qi, delta));
}

#[target_feature(enable = "sse2")]
#[inline]
fn needs_filter(p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i, thresh: u8) -> __m128i {
    let t3 = _mm_srli_epi16::<1>(_mm_and_si128(abs_diff(p1, q1), _mm_set1_epi8(-2)));
    let t4 = abs_diff(p0, q0);
    let t6 = _mm_adds_epu8(_mm_adds_epu8(t4, t4), t3);
    let t7 = _mm_subs_epu8(t6, _mm_set1_epi8(thresh as i8));
    _mm_cmpeq_epi8(t7, _mm_setzero_si128())
}

#[target_feature(enable = "sse2")]
#[inline]
fn do_filter2(p1: __m128i, p0: &mut __m128i, q0: &mut __m128i, q1: __m128i, thresh: u8) {
    let mask = needs_filter(p1, *p0, *q0, q1, thresh);
    let (mut sp0, mut sq0) = (flip(*p0), flip(*q0));
    let a = _mm_and_si128(base_delta(flip(p1), sp0, sq0, flip(q1)), mask);
    do_simple_filter(&mut sp0, &mut sq0, a);
    *p0 = flip(sp0);
    *q0 = flip(sq0);
}

#[target_feature(enable = "sse2")]
#[inline]
fn do_filter4(
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    mask: __m128i,
    hev_thresh: u8,
) {
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp1, sp0, sq0, sq1) = (flip(*p1), flip(*p0), flip(*q0), flip(*q1));
    let mut t1 = _mm_andnot_si128(not_hev, _mm_subs_epi8(sp1, sq1));
    let t2 = _mm_subs_epi8(sq0, sp0);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_adds_epi8(t1, t2);
    t1 = _mm_and_si128(t1, mask);
    let t2 = signed_shift8b(_mm_adds_epi8(t1, _mm_set1_epi8(3)));
    let t3 = signed_shift8b(_mm_adds_epi8(t1, _mm_set1_epi8(4)));
    *p0 = flip(_mm_adds_epi8(sp0, t2));
    *q0 = flip(_mm_subs_epi8(sq0, t3));
    // signed (t3 + 1) >> 1
    let t3 = _mm_sub_epi8(
        _mm_avg_epu8(_mm_add_epi8(t3, _mm_set1_epi8(-128)), _mm_setzero_si128()),
        _mm_set1_epi8(64),
    );
    let t3 = _mm_and_si128(not_hev, t3);
    *q1 = flip(_mm_subs_epi8(sq1, t3));
    *p1 = flip(_mm_adds_epi8(sp1, t3));
}

#[target_feature(enable = "sse2")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn do_filter6(
    p2: &mut __m128i,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    q2: &mut __m128i,
    mask: __m128i,
    hev_thresh: u8,
) {
    let zero = _mm_setzero_si128();
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp2, sp1, mut sp0, mut sq0, sq1, sq2) = (
        flip(*p2),
        flip(*p1),
        flip(*p0),
        flip(*q0),
        flip(*q1),
        flip(*q2),
    );
    let a = base_delta(sp1, sp0, sq0, sq1);
    do_simple_filter(
        &mut sp0,
        &mut sq0,
        _mm_and_si128(a, _mm_andnot_si128(not_hev, mask)),
    );

    let k9 = _mm_set1_epi16(0x0900);
    let k63 = _mm_set1_epi16(63);
    let f = _mm_and_si128(a, _mm_and_si128(not_hev, mask));
    let f9_lo = _mm_mulhi_epi16(_mm_unpacklo_epi8(zero, f), k9);
    let f9_hi = _mm_mulhi_epi16(_mm_unpackhi_epi8(zero, f), k9);
    let a2_lo = _mm_add_epi16(f9_lo, k63);
    let a2_hi = _mm_add_epi16(f9_hi, k63);
    let a1_lo = _mm_add_epi16(a2_lo, f9_lo);
    let a1_hi = _mm_add_epi16(a2_hi, f9_hi);
    let a0_lo = _mm_add_epi16(a1_lo, f9_lo);
    let a0_hi = _mm_add_epi16(a1_hi, f9_hi);
    (*p2, *q2, *p1, *q1, *p0, *q0) = (sp2, sq2, sp1, sq1, sp0, sq0);
    update_2_pixels(p2, q2, a2_lo, a2_hi);
    update_2_pixels(p1, q1, a1_lo, a1_hi);
    update_2_pixels(p0, q0, a0_lo, a0_hi);
}

/// 8 rows of 4 pixels of `r` from row `k`, at byte `x`, transposed: columns 0 and 1 in `p`,
/// 2 and 3 in `q`.
#[target_feature(enable = "sse2")]
#[inline]
fn load8x4<const N: usize, const W: usize>(
    r: &Rows<N, W>,
    k: usize,
    x: usize,
) -> (__m128i, __m128i) {
    let a0 = _mm_set_epi32(
        r.load(k + 6, x),
        r.load(k + 2, x),
        r.load(k + 4, x),
        r.load(k, x),
    );
    let a1 = _mm_set_epi32(
        r.load(k + 7, x),
        r.load(k + 3, x),
        r.load(k + 5, x),
        r.load(k + 1, x),
    );
    let b0 = _mm_unpacklo_epi8(a0, a1);
    let b1 = _mm_unpackhi_epi8(a0, a1);
    let c0 = _mm_unpacklo_epi16(b0, b1);
    let c1 = _mm_unpackhi_epi16(b0, b1);
    (_mm_unpacklo_epi32(c0, c1), _mm_unpackhi_epi32(c0, c1))
}

/// Columns `x..x + 4` of 8 rows of `a` from row `ka` and 8 rows of `b` from row `kb`, one
/// register per column.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn load16x4<const N: usize, const W: usize, const M: usize, const V: usize>(
    a: &Rows<N, W>,
    ka: usize,
    b: &Rows<M, V>,
    kb: usize,
    x: usize,
) -> [__m128i; 4] {
    let (t1, t2) = load8x4(a, ka, x);
    let (p0, q1) = load8x4(b, kb, x);
    [
        _mm_unpacklo_epi64(t1, p0),
        _mm_unpackhi_epi64(t1, p0),
        _mm_unpacklo_epi64(t2, q1),
        _mm_unpackhi_epi64(t2, q1),
    ]
}

#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn store4x4<const N: usize, const W: usize>(
    mut v: __m128i,
    r: &mut Rows<N, W>,
    k: usize,
    x: usize,
) {
    for i in 0..4 {
        r.store(k + i, x, _mm_cvtsi128_si32(v));
        v = _mm_srli_si128::<4>(v);
    }
}

/// The columns of `load16x4` back in rows, four rows of four pixels to a register.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn columns_to_rows(c: [__m128i; 4]) -> [__m128i; 4] {
    let [p1, p0, q0, q1] = c;
    let p0_s = _mm_unpacklo_epi8(p1, p0);
    let p1_s = _mm_unpackhi_epi8(p1, p0);
    let q0_s = _mm_unpacklo_epi8(q0, q1);
    let q1_s = _mm_unpackhi_epi8(q0, q1);
    [
        _mm_unpacklo_epi16(p0_s, q0_s),
        _mm_unpackhi_epi16(p0_s, q0_s),
        _mm_unpacklo_epi16(p1_s, q1_s),
        _mm_unpackhi_epi16(p1_s, q1_s),
    ]
}

/// Columns `x..x + 4` of 16 rows of `r` from row `k`.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn store16x4<const N: usize, const W: usize>(
    c: [__m128i; 4],
    r: &mut Rows<N, W>,
    k: usize,
    x: usize,
) {
    for (i, v) in columns_to_rows(c).into_iter().enumerate() {
        store4x4(v, r, k + 4 * i, x);
    }
}

/// Columns `x..x + 4` of 8 rows of `u` and 8 rows of `v` from row `k`.
#[target_feature(enable = "sse2")]
#[inline]
fn store16x4_uv<const N: usize, const W: usize>(
    c: [__m128i; 4],
    u: &mut Rows<N, W>,
    v: &mut Rows<N, W>,
    k: usize,
    x: usize,
) {
    let [a, b, c, d] = columns_to_rows(c);
    store4x4(a, u, k, x);
    store4x4(b, u, k + 4, x);
    store4x4(c, v, k, x);
    store4x4(d, v, k + 4, x);
}

#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn load_uv<const N: usize, const W: usize>(
    u: &Rows<N, W>,
    v: &Rows<N, W>,
    k: usize,
    x: usize,
) -> __m128i {
    _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(u.load(k, x)),
        _mm_cvtsi64_si128(v.load(k, x)),
    )
}

#[target_feature(enable = "sse2")]
#[inline]
pub(super) fn store_uv<const N: usize, const W: usize>(
    c: __m128i,
    u: &mut Rows<N, W>,
    v: &mut Rows<N, W>,
    k: usize,
    x: usize,
) {
    u.store(k, x, _mm_cvtsi128_si64(c));
    v.store(k, x, _mm_cvtsi128_si64(_mm_srli_si128::<8>(c)));
}

#[target_feature(enable = "sse2")]
#[inline]
fn max_diff(m: __m128i, p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i) -> __m128i {
    let m = _mm_max_epu8(m, abs_diff(p1, p0));
    let m = _mm_max_epu8(m, abs_diff(p3, p2));
    _mm_max_epu8(m, abs_diff(p2, p1))
}

#[target_feature(enable = "sse2")]
#[inline]
fn complex_mask(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    thresh: u8,
    ithresh: u8,
    mask: __m128i,
) -> __m128i {
    let diff = _mm_subs_epu8(mask, _mm_set1_epi8(ithresh as i8));
    let thresh_mask = _mm_cmpeq_epi8(diff, _mm_setzero_si128());
    _mm_and_si128(thresh_mask, needs_filter(p1, p0, q0, q1, thresh))
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn simple_v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { simple_v16_sse2(buf, p, stride, thresh) }
}

#[target_feature(enable = "sse2")]
fn simple_v16_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    let mut r = Rows::<4, 16>::new(buf, p - 2 * stride, stride);
    let [p1, mut p0, mut q0, q1] = [r.load(0, 0), r.load(1, 0), r.load(2, 0), r.load(3, 0)];
    do_filter2(p1, &mut p0, &mut q0, q1, thresh);
    r.store(1, 0, p0);
    r.store(2, 0, q0);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn simple_h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { simple_h16_sse2(buf, p, stride, thresh) }
}

#[target_feature(enable = "sse2")]
fn simple_h16_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
    let mut r = Rows::<16, 4>::new(buf, p - 2, stride);
    let [p1, mut p0, mut q0, q1] = load16x4(&r, 0, &r, 8, 0);
    do_filter2(p1, &mut p0, &mut q0, q1, thresh);
    store16x4([p1, p0, q0, q1], &mut r, 0, 0);
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe { v16_sse2(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
fn v16_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut r = Rows::<8, 16>::new(buf, p - 4 * stride, stride);
    let [t1, mut p2, mut p1, mut p0] = [r.load(0, 0), r.load(1, 0), r.load(2, 0), r.load(3, 0)];
    let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
    let [mut q0, mut q1, mut q2, t1] = [r.load(4, 0), r.load(5, 0), r.load(6, 0), r.load(7, 0)];
    let mask = max_diff(mask, t1, q2, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );
    for (k, x) in [p2, p1, p0, q0, q1, q2].into_iter().enumerate() {
        r.store(k + 1, 0, x);
    }
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { h16_sse2(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
fn h16_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut r = Rows::<16, 8>::new(buf, p - 4, stride);
    let [p3, mut p2, mut p1, mut p0] = load16x4(&r, 0, &r, 8, 0);
    let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
    let [mut q0, mut q1, mut q2, q3] = load16x4(&r, 0, &r, 8, 4);
    let mask = max_diff(mask, q3, q2, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );
    store16x4([p3, p2, p1, p0], &mut r, 0, 0);
    store16x4([q0, q1, q2, q3], &mut r, 0, 4);
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { v16i_sse2(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
fn v16i_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut r = Rows::<16, 16>::new(buf, p, stride);
    let [mut p3, mut p2, mut p1, mut p0] = [r.load(0, 0), r.load(1, 0), r.load(2, 0), r.load(3, 0)];
    for b in [4, 8, 12] {
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [n3, n2, tmp1, tmp2] = [
            r.load(b, 0),
            r.load(b + 1, 0),
            r.load(b + 2, 0),
            r.load(b + 3, 0),
        ];
        (p3, p2) = (n3, n2);
        let mask = max_diff(mask, p3, p2, tmp1, tmp2);
        let mask = complex_mask(p1, p0, p3, p2, thresh, ithresh, mask);
        do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
        for (k, x) in [p1, p0, p3, p2].into_iter().enumerate() {
            r.store(b - 2 + k, 0, x);
        }
        p1 = tmp1;
        p0 = tmp2;
    }
}

#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { h16i_sse2(buf, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
fn h16i_sse2(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut r = Rows::<16, 16>::new(buf, p, stride);
    let [mut p3, mut p2, mut p1, mut p0] = load16x4(&r, 0, &r, 8, 0);
    for b in [4, 8, 12] {
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [n3, n2, tmp1, tmp2] = load16x4(&r, 0, &r, 8, b);
        (p3, p2) = (n3, n2);
        let mask = max_diff(mask, p3, p2, tmp1, tmp2);
        let mask = complex_mask(p1, p0, p3, p2, thresh, ithresh, mask);
        do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
        store16x4([p1, p0, p3, p2], &mut r, 0, b - 2);
        p1 = tmp1;
        p0 = tmp2;
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe { v8_sse2(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
fn v8_sse2(u: &mut [u8], v: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut u = Rows::<8, 8>::new(u, p - 4 * stride, stride);
    let mut v = Rows::<8, 8>::new(v, p - 4 * stride, stride);
    let [t1, mut p2, mut p1, mut p0] = [
        load_uv(&u, &v, 0, 0),
        load_uv(&u, &v, 1, 0),
        load_uv(&u, &v, 2, 0),
        load_uv(&u, &v, 3, 0),
    ];
    let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
    let [mut q0, mut q1, mut q2, t1] = [
        load_uv(&u, &v, 4, 0),
        load_uv(&u, &v, 5, 0),
        load_uv(&u, &v, 6, 0),
        load_uv(&u, &v, 7, 0),
    ];
    let mask = max_diff(mask, t1, q2, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );
    for (k, x) in [p2, p1, p0, q0, q1, q2].into_iter().enumerate() {
        store_uv(x, &mut u, &mut v, k + 1, 0);
    }
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe { h8_sse2(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
fn h8_sse2(u: &mut [u8], v: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut u = Rows::<8, 8>::new(u, p - 4, stride);
    let mut v = Rows::<8, 8>::new(v, p - 4, stride);
    let [p3, mut p2, mut p1, mut p0] = load16x4(&u, 0, &v, 0, 0);
    let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
    let [mut q0, mut q1, mut q2, q3] = load16x4(&u, 0, &v, 0, 4);
    let mask = max_diff(mask, q3, q2, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );
    store16x4_uv([p3, p2, p1, p0], &mut u, &mut v, 0, 0);
    store16x4_uv([q0, q1, q2, q3], &mut u, &mut v, 0, 4);
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe { v8i_sse2(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
fn v8i_sse2(u: &mut [u8], v: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut u = Rows::<8, 8>::new(u, p, stride);
    let mut v = Rows::<8, 8>::new(v, p, stride);
    let [t2, t1, mut p1, mut p0] = [
        load_uv(&u, &v, 0, 0),
        load_uv(&u, &v, 1, 0),
        load_uv(&u, &v, 2, 0),
        load_uv(&u, &v, 3, 0),
    ];
    let mask = max_diff(abs_diff(p1, p0), t2, t1, p1, p0);
    let [mut q0, mut q1, t1, t2] = [
        load_uv(&u, &v, 4, 0),
        load_uv(&u, &v, 5, 0),
        load_uv(&u, &v, 6, 0),
        load_uv(&u, &v, 7, 0),
    ];
    let mask = max_diff(mask, t2, t1, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev);
    for (k, x) in [p1, p0, q0, q1].into_iter().enumerate() {
        store_uv(x, &mut u, &mut v, k + 2, 0);
    }
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe { h8i_sse2(u, v, p, stride, thresh, ithresh, hev) }
}

#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
fn h8i_sse2(u: &mut [u8], v: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
    let mut u = Rows::<8, 8>::new(u, p, stride);
    let mut v = Rows::<8, 8>::new(v, p, stride);
    let [t2, t1, mut p1, mut p0] = load16x4(&u, 0, &v, 0, 0);
    let mask = max_diff(abs_diff(p1, p0), t2, t1, p1, p0);
    let [mut q0, mut q1, t1, t2] = load16x4(&u, 0, &v, 0, 4);
    let mask = max_diff(mask, t2, t1, q1, q0);
    let mask = complex_mask(p1, p0, q0, q1, thresh, ithresh, mask);
    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev);
    store16x4_uv([p1, p0, q0, q1], &mut u, &mut v, 0, 2);
}

#[target_feature(enable = "sse2")]
#[inline]
fn load_hi16(p: &[u8]) -> __m128i {
    _mm_unpacklo_epi8(_mm_setzero_si128(), _mm_cvtsi64_si128(load(p)))
}

/// libwebp's `YUV444ToRGB_SSE2`: 8 pixels to 16-bit R, G, B (to be saturated to 8 bits).
#[target_feature(enable = "sse2")]
#[inline]
fn yuv444_to_rgb(y: &[u8], u: &[u8], v: &[u8]) -> [__m128i; 3] {
    let (y0, u0, v0) = (load_hi16(y), load_hi16(u), load_hi16(v));
    let y1 = _mm_mulhi_epu16(y0, _mm_set1_epi16(19077));
    let r0 = _mm_mulhi_epu16(v0, _mm_set1_epi16(26149));
    let r2 = _mm_add_epi16(_mm_sub_epi16(y1, _mm_set1_epi16(14234)), r0);
    let g0 = _mm_mulhi_epu16(u0, _mm_set1_epi16(6419));
    let g1 = _mm_mulhi_epu16(v0, _mm_set1_epi16(13320));
    let g4 = _mm_sub_epi16(
        _mm_add_epi16(y1, _mm_set1_epi16(8708)),
        _mm_add_epi16(g0, g1),
    );
    let b0 = _mm_mulhi_epu16(u0, _mm_set1_epi16(33050u16 as i16));
    let b2 = _mm_subs_epu16(_mm_adds_epu16(b0, y1), _mm_set1_epi16(17685));
    [
        _mm_srai_epi16::<6>(r2),
        _mm_srai_epi16::<6>(g4),
        _mm_srli_epi16::<6>(b2),
    ]
}

#[target_feature(enable = "sse2")]
#[inline]
fn planar_to_24b_step(i: [__m128i; 6]) -> [__m128i; 6] {
    let m = _mm_set1_epi16(0x00ff);
    let even = |a, b| _mm_packus_epi16(_mm_and_si128(a, m), _mm_and_si128(b, m));
    let odd = |a, b| _mm_packus_epi16(_mm_srli_epi16::<8>(a), _mm_srli_epi16::<8>(b));
    [
        even(i[0], i[1]),
        even(i[2], i[3]),
        even(i[4], i[5]),
        odd(i[0], i[1]),
        odd(i[2], i[3]),
        odd(i[4], i[5]),
    ]
}

/// RGB (`BPP` 3) or RGBA with alpha 255 (`BPP` 4) of the leading pixels of a row with
/// full-width chroma, 32 or 8 pixels at a time as libwebp's `VP8YuvToRgb32_SSE2` and
/// `VP8YuvToRgba32_SSE2`. Returns how many pixels it converted.
#[allow(unsafe_code)]
pub(crate) fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let mut i = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU has AVX2.
        i = unsafe { super::avx2::yuv_to_rgb_row::<BPP>(y, u, v, out) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { yuv_to_rgb_row_from::<BPP>(y, u, v, out, i) }
}

/// `yuv_to_rgb_row` with the pixels before `i` already converted.
#[target_feature(enable = "sse2")]
pub(crate) fn yuv_to_rgb_row_from<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
    i: usize,
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let step = if BPP == 3 { 32 } else { 8 };
    let m = n.saturating_sub(i) / step * step;
    let (y, u, v) = (&y[i..i + m], &u[i..i + m], &v[i..i + m]);
    let out = &mut out[BPP * i..BPP * (i + m)];
    if BPP == 3 {
        let yuv = y
            .as_chunks::<32>()
            .0
            .iter()
            .zip(u.as_chunks::<32>().0)
            .zip(v.as_chunks::<32>().0);
        for (o, ((y, u), v)) in out.as_chunks_mut::<96>().0.iter_mut().zip(yuv) {
            let [r0, g0, b0] = yuv444_to_rgb(y, u, v);
            let [r1, g1, b1] = yuv444_to_rgb(&y[8..], &u[8..], &v[8..]);
            let [r2, g2, b2] = yuv444_to_rgb(&y[16..], &u[16..], &v[16..]);
            let [r3, g3, b3] = yuv444_to_rgb(&y[24..], &u[24..], &v[24..]);
            let mut c = [
                _mm_packus_epi16(r0, r1),
                _mm_packus_epi16(r2, r3),
                _mm_packus_epi16(g0, g1),
                _mm_packus_epi16(g2, g3),
                _mm_packus_epi16(b0, b1),
                _mm_packus_epi16(b2, b3),
            ];
            for _ in 0..5 {
                c = planar_to_24b_step(c);
            }
            for (k, x) in c.into_iter().enumerate() {
                store(&mut o[16 * k..], x);
            }
        }
    } else {
        let alpha = _mm_set1_epi16(255);
        let yuv = y
            .as_chunks::<8>()
            .0
            .iter()
            .zip(u.as_chunks::<8>().0)
            .zip(v.as_chunks::<8>().0);
        for (o, ((y, u), v)) in out.as_chunks_mut::<32>().0.iter_mut().zip(yuv) {
            let [r, g, b] = yuv444_to_rgb(y, u, v);
            let rb = _mm_packus_epi16(r, b);
            let ga = _mm_packus_epi16(g, alpha);
            let rg = _mm_unpacklo_epi8(rb, ga);
            let ba = _mm_unpackhi_epi8(rb, ga);
            store(&mut o[..16], _mm_unpacklo_epi16(rg, ba));
            store(&mut o[16..], _mm_unpackhi_epi16(rg, ba));
        }
    }
    i + m
}

#[target_feature(enable = "sse2")]
#[inline]
fn transpose_2_4x4(i0: __m128i, i1: __m128i, i2: __m128i, i3: __m128i) -> [__m128i; 4] {
    let t0_0 = _mm_unpacklo_epi16(i0, i1);
    let t0_1 = _mm_unpacklo_epi16(i2, i3);
    let t0_2 = _mm_unpackhi_epi16(i0, i1);
    let t0_3 = _mm_unpackhi_epi16(i2, i3);
    let t1_0 = _mm_unpacklo_epi32(t0_0, t0_1);
    let t1_1 = _mm_unpacklo_epi32(t0_2, t0_3);
    let t1_2 = _mm_unpackhi_epi32(t0_0, t0_1);
    let t1_3 = _mm_unpackhi_epi32(t0_2, t0_3);
    [
        _mm_unpacklo_epi64(t1_0, t1_1),
        _mm_unpackhi_epi64(t1_0, t1_1),
        _mm_unpacklo_epi64(t1_2, t1_3),
        _mm_unpackhi_epi64(t1_2, t1_3),
    ]
}

/// The two passes of libwebp's `Transform_SSE2` on the coefficient rows of one or two blocks,
/// side by side: the rows of pixel differences.
#[target_feature(enable = "sse2")]
#[inline]
fn transform_rows(rows: [__m128i; 4]) -> [__m128i; 4] {
    let k1 = _mm_set1_epi16(20091);
    let k2 = _mm_set1_epi16(-30068);
    let pass = |i0: __m128i, i1: __m128i, i2: __m128i, i3: __m128i| {
        let a = _mm_add_epi16(i0, i2);
        let b = _mm_sub_epi16(i0, i2);
        let c = _mm_add_epi16(
            _mm_sub_epi16(i1, i3),
            _mm_sub_epi16(_mm_mulhi_epi16(i1, k2), _mm_mulhi_epi16(i3, k1)),
        );
        let d = _mm_add_epi16(
            _mm_add_epi16(i1, i3),
            _mm_add_epi16(_mm_mulhi_epi16(i1, k1), _mm_mulhi_epi16(i3, k2)),
        );
        [
            _mm_add_epi16(a, d),
            _mm_add_epi16(b, c),
            _mm_sub_epi16(b, c),
            _mm_sub_epi16(a, d),
        ]
    };
    let [r0, r1, r2, r3] = rows;
    let [t0, t1, t2, t3] = pass(r0, r1, r2, r3);
    let [t0, t1, t2, t3] = transpose_2_4x4(t0, t1, t2, t3);
    let [s0, s1, s2, s3] = pass(_mm_add_epi16(t0, _mm_set1_epi16(4)), t1, t2, t3);
    transpose_2_4x4(
        _mm_srai_epi16::<3>(s0),
        _mm_srai_epi16::<3>(s1),
        _mm_srai_epi16::<3>(s2),
        _mm_srai_epi16::<3>(s3),
    )
}

/// The four rows of 4 coefficients of a block, in the low halves.
#[target_feature(enable = "sse2")]
#[inline]
fn coefficient_rows(block: &[i16]) -> [__m128i; 4] {
    let block: &[i16; 16] = block.try_into().unwrap();
    [
        _mm_cvtsi64_si128(load(&block[..4])),
        _mm_cvtsi64_si128(load(&block[4..8])),
        _mm_cvtsi64_si128(load(&block[8..12])),
        _mm_cvtsi64_si128(load(&block[12..])),
    ]
}

/// libwebp's `Transform_SSE2` for one block: the inverse transform of `block` (16
/// coefficients that fit 16 bits) added to the 4x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn transform_add(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { transform_add_sse2(block, dst, pos, stride) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn transform_add_sse2(block: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let [b0, b1, b2, b3] = coefficient_rows(&block[..16]);
    let [t0, t1, t2, t3] = transform_rows([b0, b1, b2, b3]);
    let zero = _mm_setzero_si128();
    let mut d = Rows::<4, 4>::new(dst, pos, stride);
    let d01 = _mm_unpacklo_epi32(
        _mm_cvtsi32_si128(d.load(0, 0)),
        _mm_cvtsi32_si128(d.load(1, 0)),
    );
    let d23 = _mm_unpacklo_epi32(
        _mm_cvtsi32_si128(d.load(2, 0)),
        _mm_cvtsi32_si128(d.load(3, 0)),
    );
    let d01 = _mm_unpacklo_epi8(d01, zero);
    let d23 = _mm_unpacklo_epi8(d23, zero);
    // Two rows a pack: LLVM narrows a pack of which it keeps one half with extra clamps.
    let r = _mm_packus_epi16(
        _mm_add_epi16(d01, _mm_unpacklo_epi64(t0, t1)),
        _mm_add_epi16(d23, _mm_unpacklo_epi64(t2, t3)),
    );
    store4(&mut d, 0, 0, r);
    store4(&mut d, 1, 0, _mm_srli_si128::<4>(r));
    store4(&mut d, 2, 0, _mm_srli_si128::<8>(r));
    store4(&mut d, 3, 0, _mm_srli_si128::<12>(r));
}

/// libwebp's `Transform_SSE2` for two blocks side by side (`do_two`): the inverse transforms of
/// `blocks` (32 coefficients that fit 16 bits) added to the 8x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn transform_add_two(blocks: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { transform_add_two_sse2(blocks, dst, pos, stride) }
}

#[target_feature(enable = "sse2")]
fn transform_add_two_sse2(blocks: &[i16], dst: &mut [u8], pos: usize, stride: usize) {
    let [a0, a1, a2, a3] = coefficient_rows(&blocks[..16]);
    let [b0, b1, b2, b3] = coefficient_rows(&blocks[16..32]);
    let t = transform_rows([
        _mm_unpacklo_epi64(a0, b0),
        _mm_unpacklo_epi64(a1, b1),
        _mm_unpacklo_epi64(a2, b2),
        _mm_unpacklo_epi64(a3, b3),
    ]);
    let zero = _mm_setzero_si128();
    let mut d = Rows::<4, 8>::new(dst, pos, stride);
    for (k, t) in t.into_iter().enumerate() {
        let x = _mm_unpacklo_epi8(_mm_cvtsi64_si128(d.load(k, 0)), zero);
        let x = _mm_add_epi16(x, t);
        d.store(k, 0, _mm_cvtsi128_si64(_mm_packus_epi16(x, x)));
    }
}

/// The interpolated pixels `out[2 * j + 1]` and `out[2 * j + 2]` of the leading chroma pairs
/// `j` of `upsample_chroma_row`, 8 pairs at a time. Returns how many pairs it did.
#[allow(unsafe_code)]
pub(crate) fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let mut j = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU has AVX2.
        j = unsafe { super::avx2::upsample_pairs(near, far, out) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { upsample_pairs_from(near, far, out, j) }
}

/// `upsample_pairs` with the pairs before `j` already done.
#[target_feature(enable = "sse2")]
pub(crate) fn upsample_pairs_from(near: &[u8], far: &[u8], out: &mut [u8], mut j: usize) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    // libwebp's `UPSAMPLE_32PIXELS` for the near row: 16 pairs in 8 bits.
    let m = pairs.saturating_sub(j) / 16 * 16;
    if m > 0 {
        let one = _mm_set1_epi8(1);
        let (a, b) = (&near[j..j + m], &near[j + 1..j + 1 + m]);
        let (c, d) = (&far[j..j + m], &far[j + 1..j + 1 + m]);
        let abcd = a.as_chunks::<16>().0.iter().zip(b.as_chunks::<16>().0);
        let abcd = abcd.zip(c.as_chunks::<16>().0).zip(d.as_chunks::<16>().0);
        let o = &mut out[2 * j + 1..2 * (j + m) + 1];
        for (o, (((a, b), c), d)) in o.as_chunks_mut::<32>().0.iter_mut().zip(abcd) {
            let (a, b, c, d) = (load(a), load(b), load(c), load(d));
            let s = _mm_avg_epu8(a, d);
            let t = _mm_avg_epu8(b, c);
            let st = _mm_xor_si128(s, t);
            let ad = _mm_xor_si128(a, d);
            let bc = _mm_xor_si128(b, c);
            let lsb = _mm_and_si128(_mm_or_si128(_mm_or_si128(ad, bc), st), one);
            let k = _mm_sub_epi8(_mm_avg_epu8(s, t), lsb);
            let get_m = |ij: __m128i, x: __m128i| {
                let lsb = _mm_and_si128(
                    _mm_or_si128(_mm_and_si128(ij, st), _mm_xor_si128(k, x)),
                    one,
                );
                _mm_sub_epi8(_mm_avg_epu8(k, x), lsb)
            };
            let even = _mm_avg_epu8(a, get_m(bc, t));
            let odd = _mm_avg_epu8(b, get_m(ad, s));
            store(&mut o[..16], _mm_unpacklo_epi8(even, odd));
            store(&mut o[16..], _mm_unpackhi_epi8(even, odd));
        }
        j += m;
    }
    let k8 = _mm_set1_epi16(8);
    while j + 8 <= pairs {
        let (near, far) = (&near[j..j + 9], &far[j..j + 9]);
        let a = mix(load(near), load(far));
        let b = mix(load(&near[1..]), load(&far[1..]));
        let a3 = _mm_add_epi16(a, _mm_add_epi16(a, a));
        let b3 = _mm_add_epi16(b, _mm_add_epi16(b, b));
        let e = _mm_srli_epi16::<4>(_mm_add_epi16(_mm_add_epi16(a3, b), k8));
        let o = _mm_srli_epi16::<4>(_mm_add_epi16(_mm_add_epi16(b3, a), k8));
        let eo = _mm_unpacklo_epi8(_mm_packus_epi16(e, e), _mm_packus_epi16(o, o));
        store(&mut out[2 * j + 1..2 * j + 17], eo);
        j += 8;
    }
    j
}

/// `3 * near + far` of 8 pixels in 16 bits.
#[target_feature(enable = "sse2")]
#[inline]
fn mix(near: i64, far: i64) -> __m128i {
    let zero = _mm_setzero_si128();
    let a = _mm_unpacklo_epi8(_mm_cvtsi64_si128(near), zero);
    let c = _mm_unpacklo_epi8(_mm_cvtsi64_si128(far), zero);
    _mm_add_epi16(_mm_add_epi16(a, _mm_add_epi16(a, a)), c)
}

/// libwebp's `DC16_SSE2`, `DC8uv_SSE2` and their `NoTop`, `NoLeft` and `NoTopLeft` variants:
/// a `SIZE`x`SIZE` block (8 or 16) at `p` set to the rounded mean of the row above and the
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
    // SAFETY: SSE2 is part of x86_64.
    unsafe {
        match SIZE {
            8 => dc_sse2::<8, 9>(buf, p, stride, top, left),
            _ => dc_sse2::<16, 17>(buf, p, stride, top, left),
        }
    }
}

#[target_feature(enable = "sse2")]
fn dc_sse2<const SIZE: usize, const N: usize>(
    buf: &mut [u8],
    p: usize,
    stride: usize,
    top: bool,
    left: bool,
) {
    let mut r = Rows::<N, N>::new(buf, p - stride - 1, stride);
    let shift = SIZE.trailing_zeros() - 1;
    let mut sum = 0;
    if top {
        let t = if SIZE == 8 {
            _mm_cvtsi64_si128(r.load(0, 1))
        } else {
            r.load(0, 1)
        };
        let sad = _mm_sad_epu8(t, _mm_setzero_si128());
        sum += (_mm_cvtsi128_si32(sad) + _mm_extract_epi16::<4>(sad)) as u32;
    }
    if left {
        for y in 1..=SIZE {
            sum += u32::from(r.load::<u8>(y, 0));
        }
    }
    let dc = match (top, left) {
        (true, true) => (sum + SIZE as u32) >> (shift + 2),
        (false, false) => 128,
        _ => (sum + SIZE as u32 / 2) >> (shift + 1),
    };
    let v = _mm_set1_epi8(dc as i8);
    for y in 1..=SIZE {
        if SIZE == 8 {
            r.store(y, 1, _mm_cvtsi128_si64(v));
        } else {
            r.store(y, 1, v);
        }
    }
}

/// libwebp's `TransformDC`: `dc` (the DC coefficient plus rounding, shifted) added to the
/// 4x4 pixels at `pos` of `dst`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn add_dc(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { add_dc_sse2(dc, dst, pos, stride) }
}

#[target_feature(enable = "sse2")]
fn add_dc_sse2(dc: i32, dst: &mut [u8], pos: usize, stride: usize) {
    let dc = _mm_set1_epi16(dc as i16);
    let up = _mm_packus_epi16(dc, dc);
    let down = _mm_packus_epi16(_mm_sub_epi16(_mm_setzero_si128(), dc), dc);
    let mut d = Rows::<4, 4>::new(dst, pos, stride);
    for k in 0..4 {
        let x = _mm_cvtsi32_si128(d.load(k, 0));
        store4(&mut d, k, 0, _mm_subs_epu8(_mm_adds_epu8(x, up), down));
    }
}

/// `(a + 2 * b + c + 2) >> 2` per byte, as libwebp's predictors compute it.
#[target_feature(enable = "sse2")]
#[inline]
fn avg3(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    let lsb = _mm_and_si128(_mm_xor_si128(a, c), _mm_set1_epi8(1));
    _mm_avg_epu8(_mm_subs_epu8(_mm_avg_epu8(a, c), lsb), b)
}

#[target_feature(enable = "sse2")]
#[inline]
fn store4<const N: usize, const W: usize>(r: &mut Rows<N, W>, k: usize, x: usize, v: __m128i) {
    r.store(k, x, _mm_cvtsi128_si32(v));
}

/// The rows of a 4x4 prediction at `p`: the row above from one pixel to its left on and 9
/// bytes wide, then the block's rows with the pixel to their left.
#[inline(always)]
fn rows4(buf: &mut [u8], p: usize, stride: usize) -> Rows<'_, 5, 9> {
    Rows::new(buf, p - stride - 1, stride)
}

/// libwebp's `VE4_SSE2`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn ve4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { ve4_sse2(buf, p, stride) }
}

#[target_feature(enable = "sse2")]
fn ve4_sse2(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = rows4(buf, p, stride);
    let t = _mm_cvtsi64_si128(r.load(0, 0));
    let avg = avg3(t, _mm_srli_si128::<1>(t), _mm_srli_si128::<2>(t));
    for y in 1..5 {
        store4(&mut r, y, 1, avg);
    }
}

/// libwebp's `LD4_SSE2`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn ld4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { ld4_sse2(buf, p, stride) }
}

#[target_feature(enable = "sse2")]
fn ld4_sse2(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = rows4(buf, p, stride);
    let t = _mm_cvtsi64_si128(r.load(0, 1));
    let c = _mm_insert_epi16::<3>(_mm_srli_si128::<2>(t), i32::from(r.load::<u8>(0, 8)));
    let avg = avg3(t, _mm_srli_si128::<1>(t), c);
    store4(&mut r, 1, 1, avg);
    store4(&mut r, 2, 1, _mm_srli_si128::<1>(avg));
    store4(&mut r, 3, 1, _mm_srli_si128::<2>(avg));
    store4(&mut r, 4, 1, _mm_srli_si128::<3>(avg));
}

/// libwebp's `RD4_SSE2`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn rd4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { rd4_sse2(buf, p, stride) }
}

#[target_feature(enable = "sse2")]
fn rd4_sse2(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = rows4(buf, p, stride);
    let [i, j, k, l] = [r.load::<u8>(1, 0), r.load(2, 0), r.load(3, 0), r.load(4, 0)];
    let xabcd = _mm_slli_si128::<4>(_mm_cvtsi64_si128(r.load(0, 0)));
    let lkji = _mm_cvtsi32_si128(i32::from_le_bytes([l, k, j, i]));
    let e = _mm_or_si128(lkji, xabcd);
    let avg = avg3(e, _mm_srli_si128::<1>(e), _mm_srli_si128::<2>(e));
    store4(&mut r, 4, 1, avg);
    store4(&mut r, 3, 1, _mm_srli_si128::<1>(avg));
    store4(&mut r, 2, 1, _mm_srli_si128::<2>(avg));
    store4(&mut r, 1, 1, _mm_srli_si128::<3>(avg));
}

/// libwebp's `VR4_SSE2`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn vr4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { vr4_sse2(buf, p, stride) }
}

#[target_feature(enable = "sse2")]
fn vr4_sse2(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = rows4(buf, p, stride);
    let [x, i, j, k] = [r.load::<u8>(0, 0), r.load(1, 0), r.load(2, 0), r.load(3, 0)];
    let xabcd = _mm_cvtsi64_si128(r.load(0, 0));
    let abcd0 = _mm_srli_si128::<1>(xabcd);
    let abcd = _mm_avg_epu8(xabcd, abcd0);
    let ixabcd = _mm_insert_epi16::<0>(
        _mm_slli_si128::<1>(xabcd),
        i32::from(i) | (i32::from(x) << 8),
    );
    let efgh = avg3(ixabcd, xabcd, abcd0);
    store4(&mut r, 1, 1, abcd);
    store4(&mut r, 2, 1, efgh);
    store4(&mut r, 3, 1, _mm_slli_si128::<1>(abcd));
    store4(&mut r, 4, 1, _mm_slli_si128::<1>(efgh));
    let avg3_1 =
        |a: u8, b: u8, c: u8| ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8;
    r.store(3, 1, avg3_1(j, i, x));
    r.store(4, 1, avg3_1(k, j, i));
}

/// libwebp's `VL4_SSE2`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn vl4(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { vl4_sse2(buf, p, stride) }
}

#[target_feature(enable = "sse2")]
fn vl4_sse2(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = rows4(buf, p, stride);
    let one = _mm_set1_epi8(1);
    let a = _mm_cvtsi64_si128(r.load(0, 1));
    let b = _mm_srli_si128::<1>(a);
    let c = _mm_srli_si128::<2>(a);
    let avg1 = _mm_avg_epu8(a, b);
    let avg2 = _mm_avg_epu8(c, b);
    let avg3 = _mm_avg_epu8(avg1, avg2);
    let lsb1 = _mm_and_si128(_mm_xor_si128(avg1, avg2), one);
    let abbc = _mm_or_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, b));
    let avg4 = _mm_subs_epu8(avg3, _mm_and_si128(abbc, lsb1));
    let extra = _mm_cvtsi128_si32(_mm_srli_si128::<4>(avg4)) as u32;
    store4(&mut r, 1, 1, avg1);
    store4(&mut r, 2, 1, avg4);
    store4(&mut r, 3, 1, _mm_srli_si128::<1>(avg1));
    store4(&mut r, 4, 1, _mm_srli_si128::<1>(avg4));
    r.store(3, 4, extra as u8);
    r.store(4, 4, (extra >> 8) as u8);
}

/// libwebp's `TrueMotion_SSE2` of a `SIZE`x`SIZE` block (4, 8 or 16) at `p`.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn tm<const SIZE: usize>(buf: &mut [u8], p: usize, stride: usize) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe {
        match SIZE {
            4 => tm_sse2::<4, 5>(buf, p, stride),
            8 => tm_sse2::<8, 9>(buf, p, stride),
            _ => tm_sse2::<16, 17>(buf, p, stride),
        }
    }
}

#[target_feature(enable = "sse2")]
fn tm_sse2<const SIZE: usize, const N: usize>(buf: &mut [u8], p: usize, stride: usize) {
    let mut r = Rows::<N, N>::new(buf, p - stride - 1, stride);
    let zero = _mm_setzero_si128();
    let corner = i16::from(r.load::<u8>(0, 0));
    let t = match SIZE {
        4 => _mm_cvtsi32_si128(r.load(0, 1)),
        8 => _mm_cvtsi64_si128(r.load(0, 1)),
        _ => r.load(0, 1),
    };
    let (t0, t1) = (_mm_unpacklo_epi8(t, zero), _mm_unpackhi_epi8(t, zero));
    for y in 1..=SIZE {
        let base = _mm_set1_epi16(i16::from(r.load::<u8>(y, 0)) - corner);
        let out = _mm_packus_epi16(_mm_add_epi16(base, t0), _mm_add_epi16(base, t1));
        match SIZE {
            4 => store4(&mut r, y, 1, out),
            8 => r.store(y, 1, _mm_cvtsi128_si64(out)),
            _ => r.store(y, 1, out),
        }
    }
}

/// Byte 15 of `x` in every byte.
#[target_feature(enable = "sse2")]
#[inline]
fn broadcast_last(x: __m128i) -> __m128i {
    _mm_shuffle_epi32::<0xff>(_mm_shufflehi_epi16::<0xff>(_mm_unpackhi_epi8(x, x)))
}

/// The running sum of the 16 bytes of `x`, plus `left` in every byte.
#[target_feature(enable = "sse2")]
#[inline]
fn running_sum(x: __m128i, left: __m128i) -> __m128i {
    let s = _mm_add_epi8(x, _mm_slli_si128::<1>(x));
    let s = _mm_add_epi8(s, _mm_slli_si128::<2>(s));
    let s = _mm_add_epi8(s, _mm_slli_si128::<4>(s));
    let s = _mm_add_epi8(s, _mm_slli_si128::<8>(s));
    _mm_add_epi8(s, left)
}

/// The gradient alpha unfiltering of `row[x..]` (`row[x - 1]` done), 16 pixels at a time as
/// a running sum of the residuals and the top row's steps, for as long as no prediction
/// needs clamping. Returns where it stopped: at a block that clamps or with fewer than 16
/// pixels left.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn unfilter_gradient(prev: &[u8], row: &mut [u8], x: usize) -> usize {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { unfilter_gradient_sse2(prev, row, x) }
}

#[target_feature(enable = "sse2")]
fn unfilter_gradient_sse2(prev: &[u8], row: &mut [u8], mut x: usize) -> usize {
    let n = row.len().min(prev.len());
    assert!(x >= 1);
    let m = n.saturating_sub(x) / 16 * 16;
    if m == 0 {
        return x;
    }
    let zero = _mm_setzero_si128();
    let mut left = _mm_set1_epi8(row[x - 1] as i8);
    let tops = prev[x..x + m]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(prev[x - 1..x - 1 + m].as_chunks::<16>().0);
    for (o, (t, tl)) in row[x..x + m].as_chunks_mut::<16>().0.iter_mut().zip(tops) {
        let (top, top_left) = (load(t), load(tl));
        let out = running_sum(_mm_add_epi8(load(o), _mm_sub_epi8(top, top_left)), left);
        // The unclamped predictions, from the left pixels this block would produce; one is
        // out of 0..=255 where its high byte is not zero.
        let before = _mm_or_si128(_mm_slli_si128::<1>(out), _mm_srli_si128::<15>(left));
        let pred = |b: __m128i, t: __m128i, tl: __m128i| {
            _mm_srli_epi16::<8>(_mm_sub_epi16(_mm_add_epi16(b, t), tl))
        };
        let lo = pred(
            _mm_unpacklo_epi8(before, zero),
            _mm_unpacklo_epi8(top, zero),
            _mm_unpacklo_epi8(top_left, zero),
        );
        let hi = pred(
            _mm_unpackhi_epi8(before, zero),
            _mm_unpackhi_epi8(top, zero),
            _mm_unpackhi_epi8(top_left, zero),
        );
        let in_range = _mm_cmpeq_epi8(_mm_packs_epi16(lo, hi), zero);
        if _mm_movemask_epi8(in_range) != 0xffff {
            break;
        }
        store(o, out);
        left = broadcast_last(out);
        x += 16;
    }
    x
}

/// The horizontal alpha unfiltering of `row[x..]` (`row[x - 1]` done), a running sum 16
/// pixels at a time. Returns where it stopped, with fewer than 16 pixels left.
#[allow(unsafe_code)]
#[inline(always)]
pub(crate) fn unfilter_horizontal(row: &mut [u8], x: usize) -> usize {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { unfilter_horizontal_sse2(row, x) }
}

#[target_feature(enable = "sse2")]
fn unfilter_horizontal_sse2(row: &mut [u8], x: usize) -> usize {
    assert!(x >= 1);
    let mut left = _mm_set1_epi8(row[x - 1] as i8);
    let m = row.len().saturating_sub(x) / 16 * 16;
    for o in row[x..x + m].as_chunks_mut::<16>().0 {
        let out = running_sum(load(o), left);
        store(o, out);
        left = broadcast_last(out);
    }
    x + m
}
