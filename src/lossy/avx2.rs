//! AVX2 decoder kernels: libwebp has none for VP8 decoding, so these run the SSE2 kernels of
//! `sse2.rs` 256 bits wide, bit-exact with them. Callers check [`crate::wide::has_avx2`] first.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::x86_64::*;

use super::sse2::{load16x4, load_uv, store16x4, store4x4, store_uv};
use crate::simd_mem::{load, store, Rows};

/// 16 bytes from `at` in the low lane, 16 from `at + 32` in the high one.
#[target_feature(enable = "avx2")]
#[inline]
fn load_lanes(p: &[u8; 64], at: usize) -> __m256i {
    _mm256_set_m128i(load(&p[at + 32..]), load(&p[at..]))
}

/// `yuv444_to_rgb` of `sse2.rs` on 16 pixels whose luma and chroma are in the high bytes.
#[target_feature(enable = "avx2")]
#[inline]
fn yuv444_to_rgb(y0: __m256i, u0: __m256i, v0: __m256i) -> [__m256i; 3] {
    let y1 = _mm256_mulhi_epu16(y0, _mm256_set1_epi16(19077));
    let r0 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(26149));
    let r2 = _mm256_add_epi16(_mm256_sub_epi16(y1, _mm256_set1_epi16(14234)), r0);
    let g0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(6419));
    let g1 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(13320));
    let g4 = _mm256_sub_epi16(
        _mm256_add_epi16(y1, _mm256_set1_epi16(8708)),
        _mm256_add_epi16(g0, g1),
    );
    let b0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(33050u16 as i16));
    let b2 = _mm256_subs_epu16(_mm256_adds_epu16(b0, y1), _mm256_set1_epi16(17685));
    [
        _mm256_srai_epi16::<6>(r2),
        _mm256_srai_epi16::<6>(g4),
        _mm256_srli_epi16::<6>(b2),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn planar_to_24b_step(i: [__m256i; 6]) -> [__m256i; 6] {
    let m = _mm256_set1_epi16(0x00ff);
    let even = |a, b| _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
    let odd = |a, b| _mm256_packus_epi16(_mm256_srli_epi16::<8>(a), _mm256_srli_epi16::<8>(b));
    [
        even(i[0], i[1]),
        even(i[2], i[3]),
        even(i[4], i[5]),
        odd(i[0], i[1]),
        odd(i[2], i[3]),
        odd(i[4], i[5]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn store_lo(dst: &mut [u8], x: __m256i) {
    store(dst, _mm256_castsi256_si128(x));
}

#[target_feature(enable = "avx2")]
#[inline]
fn store_hi(dst: &mut [u8], x: __m256i) {
    store(dst, _mm256_extracti128_si256::<1>(x));
}

/// `sse2::yuv_to_rgb_row` 64 (RGB) or 16 (RGBA) pixels at a time: RGB runs libwebp's
/// `VP8YuvToRgb32_SSE2` on two blocks of 32 pixels, one in each lane.
#[target_feature(enable = "avx2")]
pub(crate) fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let zero = _mm256_setzero_si256();
    if BPP == 3 {
        let m = n / 64 * 64;
        let yuv = y[..m]
            .as_chunks::<64>()
            .0
            .iter()
            .zip(u[..m].as_chunks::<64>().0);
        let yuv = yuv.zip(v[..m].as_chunks::<64>().0);
        for (o, ((y, u), v)) in out[..3 * m].as_chunks_mut::<192>().0.iter_mut().zip(yuv) {
            let (ya, ua, va) = (load_lanes(y, 0), load_lanes(u, 0), load_lanes(v, 0));
            let (yb, ub, vb) = (load_lanes(y, 16), load_lanes(u, 16), load_lanes(v, 16));
            let lo = |x| _mm256_unpacklo_epi8(zero, x);
            let hi = |x| _mm256_unpackhi_epi8(zero, x);
            let [r0, g0, b0] = yuv444_to_rgb(lo(ya), lo(ua), lo(va));
            let [r1, g1, b1] = yuv444_to_rgb(hi(ya), hi(ua), hi(va));
            let [r2, g2, b2] = yuv444_to_rgb(lo(yb), lo(ub), lo(vb));
            let [r3, g3, b3] = yuv444_to_rgb(hi(yb), hi(ub), hi(vb));
            let mut c = [
                _mm256_packus_epi16(r0, r1),
                _mm256_packus_epi16(r2, r3),
                _mm256_packus_epi16(g0, g1),
                _mm256_packus_epi16(g2, g3),
                _mm256_packus_epi16(b0, b1),
                _mm256_packus_epi16(b2, b3),
            ];
            for _ in 0..5 {
                c = planar_to_24b_step(c);
            }
            for (k, x) in c.into_iter().enumerate() {
                store_lo(&mut o[16 * k..], x);
                store_hi(&mut o[96 + 16 * k..], x);
            }
        }
        m
    } else {
        let m = n / 16 * 16;
        let alpha = _mm256_set1_epi16(255);
        let widen = |p: &[u8; 16]| _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(load(p)));
        let yuv = y[..m]
            .as_chunks::<16>()
            .0
            .iter()
            .zip(u[..m].as_chunks::<16>().0);
        let yuv = yuv.zip(v[..m].as_chunks::<16>().0);
        for (o, ((y, u), v)) in out[..4 * m].as_chunks_mut::<64>().0.iter_mut().zip(yuv) {
            let [r, g, b] = yuv444_to_rgb(widen(y), widen(u), widen(v));
            let rb = _mm256_packus_epi16(r, b);
            let ga = _mm256_packus_epi16(g, alpha);
            let rg = _mm256_unpacklo_epi8(rb, ga);
            let ba = _mm256_unpackhi_epi8(rb, ga);
            let (lo, hi) = (_mm256_unpacklo_epi16(rg, ba), _mm256_unpackhi_epi16(rg, ba));
            store_lo(&mut o[..16], lo);
            store_lo(&mut o[16..], hi);
            store_hi(&mut o[32..], lo);
            store_hi(&mut o[48..], hi);
        }
        m
    }
}

/// `sse2::upsample_pairs` 32 pairs at a time, libwebp's `UPSAMPLE_32PIXELS` on 256 bits.
#[target_feature(enable = "avx2")]
pub(crate) fn upsample_pairs(near: &[u8], far: &[u8], out: &mut [u8]) -> usize {
    let n = near.len().min(far.len());
    let pairs = (out.len().saturating_sub(1) / 2).min(n.saturating_sub(1));
    let m = pairs / 32 * 32;
    if m == 0 {
        return 0;
    }
    let one = _mm256_set1_epi8(1);
    let ab = near[..m]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(near[1..m + 1].as_chunks::<32>().0);
    let cd = far[..m]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(far[1..m + 1].as_chunks::<32>().0);
    for (o, ((a, b), (c, d))) in out[1..2 * m + 1]
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .zip(ab.zip(cd))
    {
        let (a, b, c, d) = (load(a), load(b), load(c), load(d));
        let s = _mm256_avg_epu8(a, d);
        let t = _mm256_avg_epu8(b, c);
        let st = _mm256_xor_si256(s, t);
        let ad = _mm256_xor_si256(a, d);
        let bc = _mm256_xor_si256(b, c);
        let lsb = _mm256_and_si256(_mm256_or_si256(_mm256_or_si256(ad, bc), st), one);
        let k = _mm256_sub_epi8(_mm256_avg_epu8(s, t), lsb);
        let get_m = |ij: __m256i, x: __m256i| {
            let lsb = _mm256_and_si256(
                _mm256_or_si256(_mm256_and_si256(ij, st), _mm256_xor_si256(k, x)),
                one,
            );
            _mm256_sub_epi8(_mm256_avg_epu8(k, x), lsb)
        };
        let even = _mm256_avg_epu8(a, get_m(bc, t));
        let odd = _mm256_avg_epu8(b, get_m(ad, s));
        let (lo, hi) = (
            _mm256_unpacklo_epi8(even, odd),
            _mm256_unpackhi_epi8(even, odd),
        );
        store_lo(&mut o[..16], lo);
        store_lo(&mut o[16..], hi);
        store_hi(&mut o[32..], lo);
        store_hi(&mut o[48..], hi);
    }
    m
}

#[target_feature(enable = "avx2")]
#[inline]
fn abs_diff(p: __m256i, q: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_subs_epu8(q, p), _mm256_subs_epu8(p, q))
}

#[target_feature(enable = "avx2")]
#[inline]
fn flip(x: __m256i) -> __m256i {
    _mm256_xor_si256(x, _mm256_set1_epi8(-128))
}

#[target_feature(enable = "avx2")]
#[inline]
fn signed_shift8b(x: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_srai_epi16::<11>(_mm256_unpacklo_epi8(zero, x));
    let hi = _mm256_srai_epi16::<11>(_mm256_unpackhi_epi8(zero, x));
    _mm256_packs_epi16(lo, hi)
}

#[target_feature(enable = "avx2")]
#[inline]
fn not_hev(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i, hev_thresh: u8) -> __m256i {
    let t_max = _mm256_max_epu8(abs_diff(p1, p0), abs_diff(q1, q0));
    let t_max_h = _mm256_subs_epu8(t_max, _mm256_set1_epi8(hev_thresh as i8));
    _mm256_cmpeq_epi8(t_max_h, _mm256_setzero_si256())
}

#[target_feature(enable = "avx2")]
#[inline]
fn base_delta(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i) -> __m256i {
    let p1_q1 = _mm256_subs_epi8(p1, q1);
    let q0_p0 = _mm256_subs_epi8(q0, p0);
    let s1 = _mm256_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm256_adds_epi8(q0_p0, s1);
    _mm256_adds_epi8(q0_p0, s2)
}

#[target_feature(enable = "avx2")]
#[inline]
fn update_2_pixels(pi: &mut __m256i, qi: &mut __m256i, a0_lo: __m256i, a0_hi: __m256i) {
    let delta = _mm256_packs_epi16(_mm256_srai_epi16::<7>(a0_lo), _mm256_srai_epi16::<7>(a0_hi));
    *pi = flip(_mm256_adds_epi8(*pi, delta));
    *qi = flip(_mm256_subs_epi8(*qi, delta));
}

#[target_feature(enable = "avx2")]
#[inline]
fn needs_filter(p1: __m256i, p0: __m256i, q0: __m256i, q1: __m256i, thresh: u8) -> __m256i {
    let t3 = _mm256_srli_epi16::<1>(_mm256_and_si256(abs_diff(p1, q1), _mm256_set1_epi8(-2)));
    let t4 = abs_diff(p0, q0);
    let t6 = _mm256_adds_epu8(_mm256_adds_epu8(t4, t4), t3);
    let t7 = _mm256_subs_epu8(t6, _mm256_set1_epi8(thresh as i8));
    _mm256_cmpeq_epi8(t7, _mm256_setzero_si256())
}

#[target_feature(enable = "avx2")]
#[inline]
fn do_filter4(
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    mask: __m256i,
    hev_thresh: u8,
) {
    let not_hev = not_hev(*p1, *p0, *q0, *q1, hev_thresh);
    let (sp1, sp0, sq0, sq1) = (flip(*p1), flip(*p0), flip(*q0), flip(*q1));
    let mut t1 = _mm256_andnot_si256(not_hev, _mm256_subs_epi8(sp1, sq1));
    let t2 = _mm256_subs_epi8(sq0, sp0);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_adds_epi8(t1, t2);
    t1 = _mm256_and_si256(t1, mask);
    let t2 = signed_shift8b(_mm256_adds_epi8(t1, _mm256_set1_epi8(3)));
    let t3 = signed_shift8b(_mm256_adds_epi8(t1, _mm256_set1_epi8(4)));
    *p0 = flip(_mm256_adds_epi8(sp0, t2));
    *q0 = flip(_mm256_subs_epi8(sq0, t3));
    // signed (t3 + 1) >> 1
    let t3 = _mm256_sub_epi8(
        _mm256_avg_epu8(
            _mm256_add_epi8(t3, _mm256_set1_epi8(-128)),
            _mm256_setzero_si256(),
        ),
        _mm256_set1_epi8(64),
    );
    let t3 = _mm256_and_si256(not_hev, t3);
    *q1 = flip(_mm256_subs_epi8(sq1, t3));
    *p1 = flip(_mm256_adds_epi8(sp1, t3));
}

#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn do_filter6(
    p2: &mut __m256i,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    q2: &mut __m256i,
    mask: __m256i,
    hev_thresh: u8,
) {
    let zero = _mm256_setzero_si256();
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
    let fl = _mm256_and_si256(a, _mm256_andnot_si256(not_hev, mask));
    let v3 = signed_shift8b(_mm256_adds_epi8(fl, _mm256_set1_epi8(3)));
    let v4 = signed_shift8b(_mm256_adds_epi8(fl, _mm256_set1_epi8(4)));
    sq0 = _mm256_subs_epi8(sq0, v4);
    sp0 = _mm256_adds_epi8(sp0, v3);

    let k9 = _mm256_set1_epi16(0x0900);
    let k63 = _mm256_set1_epi16(63);
    let f = _mm256_and_si256(a, _mm256_and_si256(not_hev, mask));
    let f9_lo = _mm256_mulhi_epi16(_mm256_unpacklo_epi8(zero, f), k9);
    let f9_hi = _mm256_mulhi_epi16(_mm256_unpackhi_epi8(zero, f), k9);
    let a2_lo = _mm256_add_epi16(f9_lo, k63);
    let a2_hi = _mm256_add_epi16(f9_hi, k63);
    let a1_lo = _mm256_add_epi16(a2_lo, f9_lo);
    let a1_hi = _mm256_add_epi16(a2_hi, f9_hi);
    let a0_lo = _mm256_add_epi16(a1_lo, f9_lo);
    let a0_hi = _mm256_add_epi16(a1_hi, f9_hi);
    (*p2, *q2, *p1, *q1, *p0, *q0) = (sp2, sq2, sp1, sq1, sp0, sq0);
    update_2_pixels(p2, q2, a2_lo, a2_hi);
    update_2_pixels(p1, q1, a1_lo, a1_hi);
    update_2_pixels(p0, q0, a0_lo, a0_hi);
}

#[target_feature(enable = "avx2")]
#[inline]
fn max_diff(m: __m256i, p3: __m256i, p2: __m256i, p1: __m256i, p0: __m256i) -> __m256i {
    let m = _mm256_max_epu8(m, abs_diff(p1, p0));
    let m = _mm256_max_epu8(m, abs_diff(p3, p2));
    _mm256_max_epu8(m, abs_diff(p2, p1))
}

#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn complex_mask(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    thresh: u8,
    ithresh: u8,
    mask: __m256i,
) -> __m256i {
    let diff = _mm256_subs_epu8(mask, _mm256_set1_epi8(ithresh as i8));
    let thresh_mask = _mm256_cmpeq_epi8(diff, _mm256_setzero_si256());
    _mm256_and_si256(thresh_mask, needs_filter(p1, p0, q0, q1, thresh))
}

/// The rows a macroblock's normal filter reaches: a luma plane in the low lanes and the two
/// chroma planes in the high ones, the macroblock from byte `left` of their rows on.
struct Planes<'a, const YN: usize, const YW: usize, const CN: usize, const CW: usize> {
    y: Rows<'a, YN, YW>,
    u: Rows<'a, CN, CW>,
    v: Rows<'a, CN, CW>,
    left: usize,
}

impl<const YN: usize, const YW: usize, const CN: usize, const CW: usize>
    Planes<'_, YN, YW, CN, CW>
{
    /// Row `k` of the luma plane beside row `k` of the chroma planes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn row(&self, k: usize) -> __m256i {
        let x = self.left;
        _mm256_set_m128i(load_uv(&self.u, &self.v, k, x), self.y.load(k, x))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn store_row(&mut self, k: usize, c: __m256i) {
        let x = self.left;
        self.y.store(k, x, _mm256_castsi256_si128(c));
        store_uv(
            _mm256_extracti128_si256::<1>(c),
            &mut self.u,
            &mut self.v,
            k,
            x,
        );
    }
}

/// The transposition of `sse2::load8x4`, of 8 luma rows in the low lanes and 8 chroma rows in
/// the high ones as `gather!` loads them.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose8x4(a0: __m256i, a1: __m256i) -> (__m256i, __m256i) {
    let b0 = _mm256_unpacklo_epi8(a0, a1);
    let b1 = _mm256_unpackhi_epi8(a0, a1);
    let c0 = _mm256_unpacklo_epi16(b0, b1);
    let c1 = _mm256_unpackhi_epi16(b0, b1);
    (_mm256_unpacklo_epi32(c0, c1), _mm256_unpackhi_epi32(c0, c1))
}

/// Rows `k`, `k + 4`, `k + 2` and `k + 6` of luma `l` and rows `j`, ... of chroma `c`, 4
/// bytes each at byte `x`.
macro_rules! gather {
    ($l:expr, $k:expr, $c:expr, $j:expr, $x:expr) => {
        _mm256_setr_epi32(
            $l.load($k, $x),
            $l.load($k + 4, $x),
            $l.load($k + 2, $x),
            $l.load($k + 6, $x),
            $c.load($j, $x),
            $c.load($j + 4, $x),
            $c.load($j + 2, $x),
            $c.load($j + 6, $x),
        )
    };
}

/// `sse2::load16x4` of `Planes` `m`: columns `x..x + 4` of the 16 luma rows and of the 8 rows
/// of each chroma plane from row `k`. Macros, as the `#[inline(always)]` that a
/// `#[target_feature]` function cannot have: as functions LLVM keeps them out of line.
macro_rules! columns {
    ($m:expr, $k:expr, $x:expr) => {{
        let (k, x) = ($k, $x);
        let (t1, t2) = transpose8x4(
            gather!($m.y, k, $m.u, k, x),
            gather!($m.y, k + 1, $m.u, k + 1, x),
        );
        let (p0, q1) = transpose8x4(
            gather!($m.y, k + 8, $m.v, k, x),
            gather!($m.y, k + 9, $m.v, k + 1, x),
        );
        [
            _mm256_unpacklo_epi64(t1, p0),
            _mm256_unpackhi_epi64(t1, p0),
            _mm256_unpacklo_epi64(t2, q1),
            _mm256_unpackhi_epi64(t2, q1),
        ]
    }};
}

/// `columns!` stored back.
macro_rules! store_columns {
    ($m:expr, $k:expr, $x:expr, $c:expr) => {{
        let (k, x) = ($k, $x);
        let [a, b, c, d] = columns_to_rows($c);
        let lo = |v| _mm256_castsi256_si128(v);
        let hi = |v| _mm256_extracti128_si256::<1>(v);
        store4x4(lo(a), &mut $m.y, k, x);
        store4x4(hi(a), &mut $m.u, k, x);
        store4x4(lo(b), &mut $m.y, k + 4, x);
        store4x4(hi(b), &mut $m.u, k + 4, x);
        store4x4(lo(c), &mut $m.y, k + 8, x);
        store4x4(hi(c), &mut $m.v, k, x);
        store4x4(lo(d), &mut $m.y, k + 12, x);
        store4x4(hi(d), &mut $m.v, k + 4, x);
    }};
}

/// `sse2::columns_to_rows` on 256 bits.
#[target_feature(enable = "avx2")]
#[inline]
fn columns_to_rows(c: [__m256i; 4]) -> [__m256i; 4] {
    let [p1, p0, q0, q1] = c;
    let p0_s = _mm256_unpacklo_epi8(p1, p0);
    let p1_s = _mm256_unpackhi_epi8(p1, p0);
    let q0_s = _mm256_unpacklo_epi8(q0, q1);
    let q1_s = _mm256_unpackhi_epi8(q0, q1);
    [
        _mm256_unpacklo_epi16(p0_s, q0_s),
        _mm256_unpackhi_epi16(p0_s, q0_s),
        _mm256_unpacklo_epi16(p1_s, q1_s),
        _mm256_unpackhi_epi16(p1_s, q1_s),
    ]
}
/// libwebp's `DoFilter` for one macroblock with the normal filter, the luma filters (`VP8HFilter16`,
/// `VP8HFilter16i`, `VP8VFilter16`, `VP8VFilter16i`) and the chroma ones (`VP8HFilter8`, ...) on
/// the same edge in one 256-bit register: `py` and `puv` are the macroblock origins in the planes,
/// `left`, `inner` and `top` the edges to filter. False, filtering nothing, without AVX2.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_normal(
    y: &mut [u8],
    py: usize,
    ys: usize,
    u: &mut [u8],
    v: &mut [u8],
    puv: usize,
    cs: usize,
    [left, inner, top]: [bool; 3],
    limit: u8,
    ilevel: u8,
    hev: u8,
) -> bool {
    if !crate::wide::has_avx2() {
        return false;
    }
    let args = (y, py, ys, u, v, puv, cs, inner, limit, ilevel, hev);
    // SAFETY: the CPU has AVX2.
    unsafe {
        match (left, top) {
            (true, true) => filter_normal_avx2::<20, 20, 12, 12>(args),
            (true, false) => filter_normal_avx2::<16, 20, 8, 12>(args),
            (false, true) => filter_normal_avx2::<20, 16, 12, 8>(args),
            (false, false) => filter_normal_avx2::<16, 16, 8, 8>(args),
        }
    }
    true
}

type FilterArgs<'a> = (
    &'a mut [u8],
    usize,
    usize,
    &'a mut [u8],
    &'a mut [u8],
    usize,
    usize,
    bool,
    u8,
    u8,
    u8,
);

/// `filter_normal` of the `YN` luma rows of `YW` bytes and `CN` chroma rows of `CW` bytes its
/// filters reach: 4 more of each when it filters the top and the left edge. So one bounds check
/// per plane covers them all.
#[target_feature(enable = "avx2")]
fn filter_normal_avx2<const YN: usize, const YW: usize, const CN: usize, const CW: usize>(
    args: FilterArgs,
) {
    let (y, py, ys, u, v, puv, cs, inner, limit, ilevel, hev) = args;
    let (up, left) = (YN - 16, YW - 16);
    let mut m = Planes::<YN, YW, CN, CW> {
        y: Rows::new(y, py - up * ys - left, ys),
        u: Rows::new(u, puv - up * cs - left, cs),
        v: Rows::new(v, puv - up * cs - left, cs),
        left,
    };
    if left > 0 {
        let [p3, mut p2, mut p1, mut p0] = columns!(m, up, 0);
        let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
        let [mut q0, mut q1, mut q2, q3] = columns!(m, up, 4);
        let mask = max_diff(mask, q3, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, limit + 4, ilevel, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
        );
        store_columns!(m, up, 0, [p3, p2, p1, p0]);
        store_columns!(m, up, 4, [q0, q1, q2, q3]);
    }
    if inner {
        let [mut p3, mut p2, mut p1, mut p0] = columns!(m, up, left);
        for k in 0..3 {
            let b = left + 4 * k + 4;
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            let [n3, n2, tmp1, tmp2] = if k == 0 {
                columns!(m, up, b)
            } else {
                let [a, b, c, d] = load16x4(&m.y, up, &m.y, up + 8, b);
                let wide = |x| _mm256_castsi128_si256(x);
                [wide(a), wide(b), wide(c), wide(d)]
            };
            (p3, p2) = (n3, n2);
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, limit, ilevel, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            if k == 0 {
                store_columns!(m, up, b - 2, [p1, p0, p3, p2]);
            } else {
                let narrow = |x| _mm256_castsi256_si128(x);
                let c = [narrow(p1), narrow(p0), narrow(p3), narrow(p2)];
                store16x4(c, &mut m.y, up, b - 2);
            }
            (p1, p0) = (tmp1, tmp2);
        }
    }
    if up > 0 {
        let [t1, mut p2, mut p1, mut p0] = [m.row(0), m.row(1), m.row(2), m.row(3)];
        let mask = max_diff(abs_diff(p1, p0), t1, p2, p1, p0);
        let [mut q0, mut q1, mut q2, t1] = [m.row(4), m.row(5), m.row(6), m.row(7)];
        let mask = max_diff(mask, t1, q2, q1, q0);
        let mask = complex_mask(p1, p0, q0, q1, limit + 4, ilevel, mask);
        do_filter6(
            &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
        );
        for (k, x) in [p2, p1, p0, q0, q1, q2].into_iter().enumerate() {
            m.store_row(k + 1, x);
        }
    }
    if inner {
        let [mut p3, mut p2, mut p1, mut p0] =
            [m.row(up), m.row(up + 1), m.row(up + 2), m.row(up + 3)];
        for k in 0..3 {
            let b = up + 4 * k + 4;
            let mask = max_diff(abs_diff(p1, p0), p3, p2, p1, p0);
            let [n3, n2, tmp1, tmp2] = match k {
                0 => [m.row(b), m.row(b + 1), m.row(b + 2), m.row(b + 3)],
                _ => {
                    let wide = |x| _mm256_castsi128_si256(x);
                    let [a, c, d, e] = [
                        m.y.load(b, left),
                        m.y.load(b + 1, left),
                        m.y.load(b + 2, left),
                        m.y.load(b + 3, left),
                    ];
                    [wide(a), wide(c), wide(d), wide(e)]
                }
            };
            (p3, p2) = (n3, n2);
            let mask = max_diff(mask, p3, p2, tmp1, tmp2);
            let mask = complex_mask(p1, p0, p3, p2, limit, ilevel, mask);
            do_filter4(&mut p1, &mut p0, &mut p3, &mut p2, mask, hev);
            for (r, x) in [p1, p0, p3, p2].into_iter().enumerate() {
                match k {
                    0 => m.store_row(b - 2 + r, x),
                    _ => m.y.store(b - 2 + r, left, _mm256_castsi256_si128(x)),
                }
            }
            (p1, p0) = (tmp1, tmp2);
        }
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use rand::{Rng, SeedableRng};

    #[test]
    fn yuv_rows_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for len in [0usize, 15, 16, 17, 63, 64, 65, 100, 128, 191, 300] {
            let [y, u, v] = [0; 3].map(|_| (0..len).map(|_| rng.gen()).collect::<Vec<u8>>());
            for bpp in [3, 4] {
                let mut a = vec![7u8; len * bpp];
                let mut b = vec![7u8; len * bpp];
                // SAFETY: the CPU has AVX2.
                let done = unsafe {
                    match bpp {
                        3 => super::yuv_to_rgb_row::<3>(&y, &u, &v, &mut a),
                        _ => super::yuv_to_rgb_row::<4>(&y, &u, &v, &mut a),
                    }
                };
                let step = if bpp == 3 { 64 } else { 16 };
                assert_eq!(done, len / step * step);
                // SAFETY: SSE2 is part of x86_64.
                unsafe {
                    match bpp {
                        3 => super::super::sse2::yuv_to_rgb_row_from::<3>(&y, &u, &v, &mut b, 0),
                        _ => super::super::sse2::yuv_to_rgb_row_from::<4>(&y, &u, &v, &mut b, 0),
                    }
                };
                assert_eq!(a[..done * bpp], b[..done * bpp], "bpp {bpp} len {len}");
            }
        }
    }

    #[test]
    fn upsampled_pairs_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        for round in 0usize..600 {
            let width = 1 + round % 300;
            let n = width.div_ceil(2);
            let mut sample = || {
                if round % 3 == 0 {
                    rng.gen_range(0..2) * 255
                } else {
                    rng.gen()
                }
            };
            let near: Vec<u8> = (0..n).map(|_| sample()).collect();
            let far: Vec<u8> = (0..n).map(|_| sample()).collect();
            let mut a = vec![0u8; width];
            let mut b = vec![0u8; width];
            // SAFETY: the CPU has AVX2.
            let done = unsafe { super::upsample_pairs(&near, &far, &mut a) };
            // SAFETY: SSE2 is part of x86_64.
            unsafe { super::super::sse2::upsample_pairs_from(&near, &far, &mut b, 0) };
            assert_eq!(a[1..1 + 2 * done], b[1..1 + 2 * done], "{width}");
        }
    }

    #[test]
    fn macroblock_filter_matches_sse2() {
        use super::super::sse2;
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let (ys, cs) = (48, 24);
        let mut changed = 0;
        for round in 0..2000 {
            let flags = [round & 1 != 0, round & 2 != 0, round & 4 != 0];
            let (level, interior, hev) = (
                rng.gen_range(1..64u8),
                rng.gen_range(1..10u8),
                rng.gen_range(0..3),
            );
            let limit = 2 * level + interior;
            let spread = [3, 10, 30, 127][round / 8 % 4];
            let mut plane = |w: usize, h: usize| {
                let base: Vec<i32> = (0..w * h / 16)
                    .map(|_| 128 + rng.gen_range(-spread..=spread))
                    .collect();
                (0..w * h)
                    .map(|i| {
                        let b = base[(i / w / 4) * (w / 4) + i % w / 4];
                        (b + rng.gen_range(-3..=3)).clamp(0, 255) as u8
                    })
                    .collect::<Vec<u8>>()
            };
            let (y, u, v) = (plane(ys, 48), plane(cs, 24), plane(cs, 24));
            let (py, puv) = (16 * ys + 16, 8 * cs + 8);
            let (mut ya, mut ua, mut va) = (y.clone(), u.clone(), v.clone());
            assert!(super::filter_normal(
                &mut ya, py, ys, &mut ua, &mut va, puv, cs, flags, limit, interior, hev
            ));
            changed += usize::from(ya != y && ua != u);
            let (mut yb, mut ub, mut vb) = (y, u, v);
            let [left, inner, top] = flags;
            if left {
                sse2::h16(&mut yb, py, ys, limit + 4, interior, hev);
                sse2::h8(&mut ub, &mut vb, puv, cs, limit + 4, interior, hev);
            }
            if inner {
                sse2::h16i(&mut yb, py, ys, limit, interior, hev);
                sse2::h8i(&mut ub, &mut vb, puv, cs, limit, interior, hev);
            }
            if top {
                sse2::v16(&mut yb, py, ys, limit + 4, interior, hev);
                sse2::v8(&mut ub, &mut vb, puv, cs, limit + 4, interior, hev);
            }
            if inner {
                sse2::v16i(&mut yb, py, ys, limit, interior, hev);
                sse2::v8i(&mut ub, &mut vb, puv, cs, limit, interior, hev);
            }
            assert_eq!((ya, ua, va), (yb, ub, vb), "round {round}");
        }
        assert!(changed > 1000, "{changed} rounds filtered");
    }
}
