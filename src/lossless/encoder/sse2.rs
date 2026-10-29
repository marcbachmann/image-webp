//! SSE2 versions of the spatial predictor residuals (libwebp's `PredictorSub*_SSE2`),
//! bit-exact with the scalar ones. They handle whole groups of four pixels and return
//! how many they did; the caller finishes the rest.

use core::arch::x86_64::*;

use super::predictor::neighbour_groups;
use crate::simd_mem::{load, load_at, store, store_at};

#[target_feature(enable = "sse2")]
#[inline]
fn average2(a: __m128i, b: __m128i) -> __m128i {
    let one = _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1));
    _mm_sub_epi8(_mm_avg_epu8(a, b), one)
}

#[target_feature(enable = "sse2")]
#[inline]
fn abs_diff(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

#[target_feature(enable = "sse2")]
#[inline]
fn pixel_sums(dl: __m128i, dt: __m128i) -> __m128i {
    let s = _mm_madd_epi16(_mm_sub_epi16(dl, dt), _mm_set1_epi16(1));
    _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01))
}

/// Per pixel: `t` if sum(|l - tl|) <= sum(|t - tl|) over the four channels, else `l`.
#[target_feature(enable = "sse2")]
#[inline]
fn select(t: __m128i, l: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let dl = abs_diff(l, tl);
    let dt = abs_diff(t, tl);
    let lo = pixel_sums(_mm_unpacklo_epi8(dl, zero), _mm_unpacklo_epi8(dt, zero));
    let hi = pixel_sums(_mm_unpackhi_epi8(dl, zero), _mm_unpackhi_epi8(dt, zero));
    let s = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(lo),
        _mm_castsi128_ps(hi),
        0b10_00_10_00,
    ));
    let use_l = _mm_cmpgt_epi32(s, zero);
    _mm_or_si128(_mm_and_si128(use_l, l), _mm_andnot_si128(use_l, t))
}

#[target_feature(enable = "sse2")]
#[inline]
fn clamped_add_subtract_full(l: __m128i, t: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_sub_epi16(
        _mm_add_epi16(_mm_unpacklo_epi8(l, zero), _mm_unpacklo_epi8(t, zero)),
        _mm_unpacklo_epi8(tl, zero),
    );
    let hi = _mm_sub_epi16(
        _mm_add_epi16(_mm_unpackhi_epi8(l, zero), _mm_unpackhi_epi8(t, zero)),
        _mm_unpackhi_epi8(tl, zero),
    );
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "sse2")]
#[inline]
fn add_half_difference(a: __m128i, b: __m128i) -> __m128i {
    let d = _mm_sub_epi16(a, b);
    _mm_add_epi16(
        a,
        _mm_srai_epi16(_mm_add_epi16(d, _mm_srli_epi16(d, 15)), 1),
    )
}

#[target_feature(enable = "sse2")]
#[inline]
fn clamped_add_subtract_half(l: __m128i, t: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ave = average2(l, t);
    let lo = add_half_difference(_mm_unpacklo_epi8(ave, zero), _mm_unpacklo_epi8(tl, zero));
    let hi = add_half_difference(_mm_unpackhi_epi8(ave, zero), _mm_unpackhi_epi8(tl, zero));
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "sse2")]
#[inline]
fn predict<const M: usize>(l: __m128i, tl: __m128i, t: __m128i, tr: __m128i) -> __m128i {
    match M {
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(l, tr), t),
        6 => average2(l, tl),
        7 => average2(l, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(l, tl), average2(t, tr)),
        11 => select(t, l, tl),
        12 => clamped_add_subtract_full(l, t, tl),
        13 => clamped_add_subtract_half(l, t, tl),
        _ => _mm_set1_epi32(0xff000000u32 as i32),
    }
}

/// Residuals of `cur[i + 1]` for `i` in `0..out.len()`, with left `cur[i]` and top-left,
/// top and top-right `up[i..i + 3]`.
#[allow(unsafe_code)]
#[inline(always)]
pub(super) fn sub_batch<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        let i = unsafe { super::avx2::sub_batch::<M>(cur, up, out) };
        // SAFETY: SSE2 is part of x86_64.
        return unsafe { sub_group::<M, false>(cur, up, out, i) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { sub_batch_sse2::<M>(cur, up, out) }
}

/// The one group of four that can follow the AVX2 kernels' groups of eight: at `i` if it is
/// below `out.len() & !3` (and, for `OPAQUE`, has no fully transparent pixel). Returns how
/// many pixels are done.
#[target_feature(enable = "sse2")]
#[inline]
fn sub_group<const M: usize, const OPAQUE: bool>(
    cur: &[u32],
    up: &[u32],
    out: &mut [u32],
    i: usize,
) -> usize {
    if i >= out.len() & !3 {
        return i;
    }
    let x = load_at(cur, i + 1);
    if OPAQUE {
        let alpha_zero = _mm_cmpeq_epi32(_mm_srli_epi32(x, 24), _mm_setzero_si128());
        if _mm_movemask_epi8(alpha_zero) != 0 {
            return i;
        }
    }
    let pred = predict::<M>(load_at(cur, i), load_at(up, i), load_at(up, i + 1), load_at(up, i + 2));
    store_at(out, i, _mm_sub_epi8(x, pred));
    i + 4
}

#[target_feature(enable = "sse2")]
#[inline]
fn sub_batch_sse2<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    let groups = neighbour_groups::<4>(cur, up, n);
    for (o, [l, x, tl, t, tr]) in out[..n].as_chunks_mut::<4>().0.iter_mut().zip(groups) {
        let pred = predict::<M>(load(l), load(tl), load(t), load(tr));
        store(o, _mm_sub_epi8(load(x), pred));
    }
    n
}

/// `sub_batch` for groups of four pixels none of which is fully transparent, stopping at
/// the first group that has one. Returns how many pixels it did.
#[allow(unsafe_code)]
#[inline(always)]
pub(super) fn sub_batch_opaque<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        let i = unsafe { super::avx2::sub_batch_opaque::<M>(cur, up, out) };
        // SAFETY: SSE2 is part of x86_64.
        return unsafe { sub_group::<M, true>(cur, up, out, i) };
    }
    // SAFETY: SSE2 is part of x86_64.
    unsafe { sub_batch_opaque_sse2::<M>(cur, up, out) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn sub_batch_opaque_sse2<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    let groups = neighbour_groups::<4>(cur, up, n);
    let mut i = 0;
    for (o, [l, x, tl, t, tr]) in out[..n].as_chunks_mut::<4>().0.iter_mut().zip(groups) {
        let x = load(x);
        let alpha_zero = _mm_cmpeq_epi32(_mm_srli_epi32(x, 24), _mm_setzero_si128());
        if _mm_movemask_epi8(alpha_zero) != 0 {
            break;
        }
        let pred = predict::<M>(load(l), load(tl), load(t), load(tr));
        store(o, _mm_sub_epi8(x, pred));
        i += 4;
    }
    i
}

/// `(changed, nonzero)`: bit `j` set where `v[j + 1] != v[j]`, and where `v[j + 1] != 0`.
#[allow(unsafe_code)]
#[inline(always)]
pub(super) fn run_masks(v: &[u32; 65]) -> (u64, u64) {
    // SAFETY: SSE2 is part of x86_64.
    unsafe { run_masks_sse2(v) }
}

#[target_feature(enable = "sse2")]
#[inline]
fn run_masks_sse2(v: &[u32; 65]) -> (u64, u64) {
    let (mut changed, mut nonzero) = (0u64, 0u64);
    let zero = _mm_setzero_si128();
    for k in 0..4 {
        let at = |i: usize| load_at(v, 16 * k + i);
        let (c0, c1, c2, c3) = (at(1), at(5), at(9), at(13));
        let eq = _mm_packs_epi16(
            _mm_packs_epi32(_mm_cmpeq_epi32(c0, at(0)), _mm_cmpeq_epi32(c1, at(4))),
            _mm_packs_epi32(_mm_cmpeq_epi32(c2, at(8)), _mm_cmpeq_epi32(c3, at(12))),
        );
        let is_zero = _mm_packs_epi16(
            _mm_packs_epi32(_mm_cmpeq_epi32(c0, zero), _mm_cmpeq_epi32(c1, zero)),
            _mm_packs_epi32(_mm_cmpeq_epi32(c2, zero), _mm_cmpeq_epi32(c3, zero)),
        );
        changed |= u64::from(!_mm_movemask_epi8(eq) as u16) << (16 * k);
        nonzero |= u64::from(!_mm_movemask_epi8(is_zero) as u16) << (16 * k);
    }
    (changed, nonzero)
}
