//! Resampling rows to another size with the fixed-point arithmetic of libwebp's rescaler
//! (`utils/rescaler_utils.c`, `dsp/rescaler.c`), so a scaled decode gives libwebp's pixels:
//! box averaging where a side shrinks, bilinear interpolation where it grows.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

const FIX: u32 = 32;
const ONE: u64 = 1 << FIX;
const ROUNDER: u64 = ONE >> 1;

fn frac(x: u64, y: u64) -> u32 {
    ((x << FIX) / y) as u32
}

fn mult_fix(x: u32, y: u32) -> u32 {
    ((u64::from(x) * u64::from(y) + ROUNDER) >> FIX) as u32
}

fn mult_fix_floor(x: u32, y: u32) -> u32 {
    ((u64::from(x) * u64::from(y)) >> FIX) as u32
}

fn clip(v: u32) -> u8 {
    v.min(255) as u8
}

/// Rows of `channels` interleaved 8-bit samples, `src_width`×`src_height` in,
/// `dst_width`×`dst_height` out, imported and exported a row at a time.
pub(crate) struct Rescaler {
    x_expand: bool,
    y_expand: bool,
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    channels: usize,
    /// Source pixels per output pixel where a shrinking side divides evenly.
    whole_shrink: Option<usize>,
    /// Per output pixel of any other shrink, its first source pixel (then the source width) and
    /// the weight its last one carries into the next output. Most take `run_count` source pixels.
    starts: Vec<u32>,
    weights: Vec<u32>,
    run_count: usize,
    x_add: i32,
    x_sub: i32,
    fx_scale: u32,
    y_add: i32,
    y_sub: i32,
    y_accum: i32,
    fy_scale: u32,
    fxy_scale: u32,
    src_y: usize,
    dst_y: usize,
    irow: Vec<u32>,
    frow: Vec<u32>,
}

impl Rescaler {
    pub(crate) fn new(
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
        channels: usize,
    ) -> Self {
        let x_expand = src_width < dst_width;
        let y_expand = src_height < dst_height;
        let (x_add, x_sub) = match x_expand {
            true => (dst_width as i32 - 1, src_width as i32 - 1),
            false => (src_width as i32, dst_width as i32),
        };
        let (y_add, y_sub) = match y_expand {
            true => (src_height as i32 - 1, dst_height as i32 - 1),
            false => (src_height as i32, dst_height as i32),
        };
        let fx_scale = if x_expand { 0 } else { frac(1, x_sub as u64) };
        let (fy_scale, fxy_scale) = match y_expand {
            true => (frac(1, x_add as u64), 0),
            false => {
                let ratio = dst_height as u64 * ONE / (x_add as u64 * y_add as u64);
                // A ratio of one does not fit: exported as is.
                let fxy = u32::try_from(ratio).unwrap_or(0);
                (frac(1, y_sub as u64), fxy)
            }
        };
        let whole_shrink = (!x_expand && x_add % x_sub == 0).then(|| (x_add / x_sub) as usize);
        let (mut starts, mut weights) = (Vec::new(), Vec::new());
        if !x_expand && whole_shrink.is_none() {
            let (mut accum, mut x) = (0, 0);
            for _ in 0..dst_width {
                starts.push(x);
                accum += x_add;
                while accum > 0 {
                    accum -= x_sub;
                    x += 1;
                }
                weights.push((-accum) as u32);
            }
            starts.push(x);
        }
        let count = |n: usize| {
            starts
                .windows(2)
                .filter(|s| (s[1] - s[0]) as usize == n)
                .count()
        };
        let lo = (x_add / x_sub.max(1)) as usize;
        let run_count = if count(lo + 1) > count(lo) {
            lo + 1
        } else {
            lo
        };
        Self {
            x_expand,
            y_expand,
            src_width,
            src_height,
            dst_width,
            dst_height,
            channels,
            whole_shrink,
            starts,
            weights,
            run_count,
            x_add,
            x_sub,
            fx_scale,
            y_add,
            y_sub,
            y_accum: if y_expand { y_sub } else { y_add },
            fy_scale,
            fxy_scale,
            src_y: 0,
            dst_y: 0,
            irow: vec![0; dst_width * channels],
            frow: vec![0; dst_width * channels],
        }
    }

    /// Bytes in an output row.
    pub(crate) fn dst_row_len(&self) -> usize {
        self.dst_width * self.channels
    }

    pub(crate) fn input_done(&self) -> bool {
        self.src_y >= self.src_height
    }

    /// Whether a row is ready to export; no row is imported until it is.
    pub(crate) fn has_pending_output(&self) -> bool {
        self.dst_y < self.dst_height && self.y_accum <= 0
    }

    /// Imports one row unless an output row is pending; returns whether it did.
    pub(crate) fn import(&mut self, src: &[u8]) -> bool {
        if self.has_pending_output() || self.input_done() {
            return false;
        }
        if self.y_expand {
            std::mem::swap(&mut self.irow, &mut self.frow);
        }
        if let Some(k) = self.whole_shrink {
            let sum = (!self.y_expand).then_some(&mut self.irow[..]);
            shrink_whole(
                src,
                self.channels,
                k,
                self.x_sub as u32,
                &mut self.frow,
                sum,
            );
        } else if self.x_expand {
            self.import_expand(src);
        } else {
            self.import_shrink(src);
        }
        if !self.y_expand && self.whole_shrink.is_none() {
            for (i, f) in self.irow.iter_mut().zip(&self.frow) {
                *i = i.wrapping_add(*f);
            }
        }
        self.src_y += 1;
        self.y_accum -= self.y_sub;
        true
    }

    fn import_expand(&mut self, src: &[u8]) {
        match self.channels {
            1 => self.import_expand_n::<1>(src),
            2 => self.import_expand_n::<2>(src),
            3 => self.import_expand_n::<3>(src),
            4 => self.import_expand_n::<4>(src),
            n => unreachable!("{n} channels"),
        }
    }

    fn import_shrink(&mut self, src: &[u8]) {
        match self.channels {
            1 => self.import_shrink_n::<1>(src),
            2 => self.import_shrink_n::<2>(src),
            3 => self.import_shrink_n::<3>(src),
            4 => self.import_shrink_n::<4>(src),
            n => unreachable!("{n} channels"),
        }
    }

    fn import_expand_n<const N: usize>(&mut self, src: &[u8]) {
        let px = |x: usize| -> [u32; N] { std::array::from_fn(|c| u32::from(src[x * N + c])) };
        let (x_add, x_sub) = (self.x_add, self.x_sub);
        let mut x_in = 1;
        let mut accum = x_add;
        let mut left = px(0);
        let mut right = if self.src_width > 1 { px(1) } else { left };
        for (x_out, out) in self.frow.chunks_exact_mut(N).enumerate() {
            for c in 0..N {
                out[c] = right[c]
                    .wrapping_mul(x_add as u32)
                    .wrapping_add(left[c].wrapping_sub(right[c]).wrapping_mul(accum as u32));
            }
            accum -= x_sub;
            if accum < 0 && x_out + 1 < self.dst_width {
                left = right;
                x_in += 1;
                right = px(x_in);
                accum += x_add;
            }
        }
    }

    fn import_shrink_n<const N: usize>(&mut self, src: &[u8]) {
        let src = &src[..self.src_width * N];
        let (x_sub, fx_scale) = (self.x_sub as u32, self.fx_scale);
        #[cfg(target_arch = "x86_64")]
        use sse2::shrink_spans;
        #[cfg(target_arch = "aarch64")]
        use neon::shrink_spans;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if shrink_spans(
            src,
            N,
            &self.starts,
            &self.weights,
            self.run_count,
            x_sub,
            fx_scale,
            &mut self.frow,
        ) {
            return;
        }
        for x in 0..self.dst_width {
            shrink_span::<N>(
                src,
                &self.starts,
                &self.weights,
                x_sub,
                fx_scale,
                x,
                &mut self.frow,
            );
        }
    }

    /// Exports the pending row into `dst`.
    pub(crate) fn export(&mut self, dst: &mut [u8]) {
        debug_assert!(self.has_pending_output());
        let dst = &mut dst[..self.dst_row_len()];
        if self.y_expand {
            let ab = (self.y_accum != 0).then(|| {
                let b = frac((-self.y_accum) as u64, self.y_sub as u64);
                ((ONE - u64::from(b)) as u32, b)
            });
            #[cfg(target_arch = "aarch64")]
            let done = neon::export_expand(dst, &self.frow, &self.irow, self.fy_scale, ab);
            #[cfg(not(target_arch = "aarch64"))]
            let done = 0;
            let rows = dst.iter_mut().zip(&self.frow).zip(&self.irow).skip(done);
            match ab {
                None => {
                    for ((d, &f), _) in rows {
                        *d = clip(mult_fix(f, self.fy_scale));
                    }
                }
                Some((a, b)) => {
                    for ((d, &f), &i) in rows {
                        let v = u64::from(a) * u64::from(f) + u64::from(b) * u64::from(i);
                        let j = ((v + ROUNDER) >> FIX) as u32;
                        *d = clip(mult_fix(j, self.fy_scale));
                    }
                }
            }
        } else if self.fxy_scale != 0 {
            let yscale = self.fy_scale.wrapping_mul((-self.y_accum) as u32);
            #[cfg(target_arch = "x86_64")]
            let done = sse2::export_shrink(dst, &mut self.irow, &self.frow, yscale, self.fxy_scale);
            #[cfg(target_arch = "aarch64")]
            let done = neon::export_shrink(dst, &mut self.irow, &self.frow, yscale, self.fxy_scale);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let done = 0;
            export_shrink(
                &mut dst[done..],
                &mut self.irow[done..],
                &self.frow[done..],
                yscale,
                self.fxy_scale,
            );
        } else {
            for (d, i) in dst.iter_mut().zip(self.irow.iter_mut()) {
                *d = *i as u8;
                *i = 0;
            }
        }
        self.y_accum += self.y_add;
        self.dst_y += 1;
    }
}

fn export_shrink(dst: &mut [u8], irow: &mut [u32], frow: &[u32], yscale: u32, fxy_scale: u32) {
    for ((d, i), &f) in dst.iter_mut().zip(irow.iter_mut()).zip(frow) {
        if yscale != 0 {
            let frac = mult_fix_floor(f, yscale);
            *d = clip(mult_fix(i.wrapping_sub(frac), fxy_scale));
            *i = frac;
        } else {
            *d = clip(mult_fix(*i, fxy_scale));
            *i = 0;
        }
    }
}

/// A shrink by a whole `k` source pixels per output pixel, which leaves libwebp's arithmetic no
/// fraction to carry between outputs; each output is added into `sum` as well where given.
fn shrink_whole(
    src: &[u8],
    channels: usize,
    k: usize,
    x_sub: u32,
    out: &mut [u32],
    sum: Option<&mut [u32]>,
) {
    match channels {
        1 => shrink_whole_n::<1>(src, k, x_sub, out, sum),
        2 => shrink_whole_n::<2>(src, k, x_sub, out, sum),
        3 => shrink_whole_n::<3>(src, k, x_sub, out, sum),
        4 => shrink_whole_n::<4>(src, k, x_sub, out, sum),
        n => unreachable!("{n} channels"),
    }
}

fn shrink_whole_n<const N: usize>(
    src: &[u8],
    k: usize,
    x_sub: u32,
    out: &mut [u32],
    mut sum: Option<&mut [u32]>,
) {
    #[cfg(target_arch = "x86_64")]
    let done = sse2::shrink_whole(src, N, k, x_sub, out, sum.as_deref_mut());
    #[cfg(target_arch = "aarch64")]
    let done = neon::shrink_whole(src, N, k, x_sub, out, sum.as_deref_mut());
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0;
    let pixels = src
        .chunks_exact(N * k)
        .zip(out.as_chunks_mut::<N>().0)
        .enumerate();
    for (x, (px, o)) in pixels.skip(done / N) {
        for (c, o) in o.iter_mut().enumerate() {
            *o = px
                .iter()
                .skip(c)
                .step_by(N)
                .map(|&p| u32::from(p))
                .sum::<u32>()
                * x_sub;
        }
        if let Some(sum) = sum.as_deref_mut() {
            for (s, &o) in sum[x * N..][..N].iter_mut().zip(&*o) {
                *s = s.wrapping_add(o);
            }
        }
    }
}

/// Output pixel `x` of a shrink that carries a fraction of its last source pixel into the next.
fn shrink_span<const N: usize>(
    src: &[u8],
    starts: &[u32],
    weights: &[u32],
    x_sub: u32,
    fx_scale: u32,
    x: usize,
    out: &mut [u32],
) {
    let (a, b) = (starts[x] as usize, starts[x + 1] as usize);
    for c in 0..N {
        let carry = match x {
            0 => 0,
            _ => mult_fix(
                u32::from(src[(a - 1) * N + c]).wrapping_mul(weights[x - 1]),
                fx_scale,
            ),
        };
        let pixels = src[a * N + c..b * N].iter().step_by(N);
        let sum = pixels.fold(carry, |s, &p| s.wrapping_add(u32::from(p)));
        let frac = u32::from(src[(b - 1) * N + c]).wrapping_mul(weights[x]);
        out[x * N + c] = sum.wrapping_mul(x_sub).wrapping_sub(frac);
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;

    use crate::simd_mem::{load, load_at, store};

    /// Stores the `V` vectors `f` makes of each item of `src` to the next `4 * V` outputs of
    /// `out`, adding them into `sum` as well where given; returns how many outputs it wrote.
    /// `src` and `sum` cover the whole groups of `out`.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn each_group<X, const V: usize>(
        src: impl Iterator<Item = X>,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
        f: impl Fn(X) -> [__m128i; V],
    ) -> usize {
        let out = out.as_chunks_mut::<4>().0.as_chunks_mut::<V>().0;
        match sum {
            Some(sum) => {
                let sum = sum.as_chunks_mut::<4>().0.as_chunks_mut::<V>().0;
                for ((o, s), x) in out.iter_mut().zip(sum).zip(src) {
                    for ((o, s), v) in o.iter_mut().zip(s).zip(f(x)) {
                        store(o, v);
                        let v = _mm_add_epi32(load(&*s), v);
                        store(s, v);
                    }
                }
            }
            None => {
                for (o, x) in out.iter_mut().zip(src) {
                    for (o, v) in o.iter_mut().zip(f(x)) {
                        store(o, v);
                    }
                }
            }
        }
        out.len() * 4 * V
    }

    /// [`super::shrink_whole`] for the outputs whole vectors cover; returns how many it wrote.
    /// The products go through `madd`, so `x_sub` must fit an i16.
    #[allow(unsafe_code)]
    pub(super) fn shrink_whole(
        src: &[u8],
        channels: usize,
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        mut sum: Option<&mut [u32]>,
    ) -> usize {
        let n = out.len();
        assert!(src.len() >= n * k && sum.as_ref().is_none_or(|s| s.len() >= n));
        if x_sub > i16::MAX as u32 {
            return 0;
        }
        let mut i = 0;
        if crate::wide::has_avx2() && channels == 1 {
            // SAFETY: the CPU supports AVX2.
            i = unsafe { super::avx2::shrink_whole(src, k, x_sub, out, sum.as_deref_mut()) };
        }
        let (src, out, sum) = (&src[i * k..], &mut out[i..], sum.map(|s| &mut s[i..n]));
        // SAFETY: SSE2 is part of x86_64.
        i + unsafe { shrink_whole_sse2(src, channels, k, x_sub, out, sum) }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn shrink_whole_sse2(
        src: &[u8],
        channels: usize,
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
    ) -> usize {
        let z = _mm_setzero_si128();
        let xs = _mm_set1_epi32(x_sub as i32);
        match (channels, k) {
            (1, 1) => each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                let v = load(v);
                let (lo, hi) = (_mm_unpacklo_epi8(v, z), _mm_unpackhi_epi8(v, z));
                [
                    _mm_madd_epi16(_mm_unpacklo_epi16(lo, z), xs),
                    _mm_madd_epi16(_mm_unpackhi_epi16(lo, z), xs),
                    _mm_madd_epi16(_mm_unpacklo_epi16(hi, z), xs),
                    _mm_madd_epi16(_mm_unpackhi_epi16(hi, z), xs),
                ]
            }),
            (1, 2) => {
                let xs = _mm_set1_epi16(x_sub as i16);
                each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                    let v = load(v);
                    [
                        _mm_madd_epi16(_mm_unpacklo_epi8(v, z), xs),
                        _mm_madd_epi16(_mm_unpackhi_epi8(v, z), xs),
                    ]
                })
            }
            (1, 4) => {
                let (xs, low) = (_mm_set1_epi16(x_sub as i16), _mm_set1_epi16(0xff));
                each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                    let v = load(v);
                    let pairs = _mm_add_epi16(_mm_and_si128(v, low), _mm_srli_epi16::<8>(v));
                    [_mm_madd_epi16(pairs, xs)]
                })
            }
            (1, 8) => each_group(src.as_chunks::<32>().0.iter(), out, sum, |p| {
                let a = _mm_sad_epu8(load(p), z);
                let b = _mm_sad_epu8(load(&p[16..]), z);
                let v = _mm_unpacklo_epi64(
                    _mm_shuffle_epi32::<0b1000>(a),
                    _mm_shuffle_epi32::<0b1000>(b),
                );
                [_mm_madd_epi16(v, xs)]
            }),
            (4, 2..=128) => each_group(src.chunks_exact(4 * k), out, sum, |p| {
                let (pairs, last) = p.as_chunks::<8>();
                let mut acc = z;
                for v in pairs {
                    let v = _mm_unpacklo_epi8(_mm_cvtsi64_si128(load(v)), z);
                    acc = _mm_add_epi16(acc, _mm_add_epi16(v, _mm_srli_si128::<8>(v)));
                }
                if let Some(last) = last.first_chunk::<4>() {
                    let v = _mm_cvtsi32_si128(load(last));
                    acc = _mm_add_epi16(acc, _mm_unpacklo_epi8(v, z));
                }
                [_mm_madd_epi16(_mm_unpacklo_epi16(acc, z), xs)]
            }),
            _ => 0,
        }
    }

    /// [`super::shrink_span`] for every output pixel of one channel, four outputs to a vector where
    /// four in a row take `k` source pixels each. False for more channels.
    #[allow(clippy::too_many_arguments)]
    #[allow(unsafe_code)]
    pub(super) fn shrink_spans(
        src: &[u8],
        channels: usize,
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) -> bool {
        let n = weights.len();
        assert!(starts.len() == n + 1 && out.len() >= n * channels);
        if x_sub > i16::MAX as u32 || channels != 1 {
            return false;
        }
        if crate::wide::has_avx2() {
            // SAFETY: the CPU supports AVX2.
            unsafe { super::avx2::shrink_spans(src, starts, weights, k, x_sub, fx_scale, out) };
            return true;
        }
        // SAFETY: SSE2 is part of x86_64.
        unsafe { shrink_spans_sse2(src, starts, weights, k, x_sub, fx_scale, out) };
        true
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn shrink_spans_sse2(
        src: &[u8],
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) {
        let n = weights.len();
        let mut x = 0;
        while x < n {
            if whole_run(src, starts, k, x, 4) {
                span4(src, starts, weights, k, x_sub, fx_scale, x, out);
                x += 4;
            } else {
                super::shrink_span::<1>(src, starts, weights, x_sub, fx_scale, x, out);
                x += 1;
            }
        }
    }

    /// Whether outputs `x..x + m`, not the first, each take `k` source pixels within `src`.
    #[inline(always)]
    pub(super) fn whole_run(src: &[u8], starts: &[u32], k: usize, x: usize, m: usize) -> bool {
        let n = starts.len() - 1;
        let a = starts[x] as usize;
        x >= 1
            && x + m <= n
            && a >= 1
            && starts[x + m] as usize == a + m * k
            && matches!(k, 1 | 2 | 4 | 8)
            && a + m * k <= src.len()
    }

    /// Outputs `x..x + 4` of `shrink_spans`, for which [`whole_run`] holds.
    #[target_feature(enable = "sse2")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn span4(
        src: &[u8],
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        x: usize,
        out: &mut [u32],
    ) {
        let z = _mm_setzero_si128();
        let (xs, ones) = (_mm_set1_epi32(x_sub as i32), _mm_set1_epi16(1));
        let (fx, round) = (
            _mm_set1_epi64x(i64::from(fx_scale)),
            _mm_set1_epi64x(1 << 31),
        );
        let a = starts[x] as usize;
        // The source pixel before the outputs' on, and the weight of the output before them on.
        let (p, wp) = (&src[a - 1..a + 4 * k], &weights[x - 1..x + 4]);
        let (sums, last) = match k {
            1 => {
                let v = _mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(load_at(p, 1)), z), z);
                (v, v)
            }
            2 => {
                let v = _mm_unpacklo_epi8(_mm_cvtsi64_si128(load_at(p, 1)), z);
                (_mm_madd_epi16(v, ones), _mm_srli_epi32::<16>(v))
            }
            4 => {
                let v = load_at(p, 1);
                let pairs = _mm_add_epi16(
                    _mm_and_si128(v, _mm_set1_epi16(0xff)),
                    _mm_srli_epi16::<8>(v),
                );
                (_mm_madd_epi16(pairs, ones), _mm_srli_epi32::<24>(v))
            }
            _ => {
                let (u, v) = (load_at(p, 1), load_at(p, 17));
                let (su, sv) = (_mm_sad_epu8(u, z), _mm_sad_epu8(v, z));
                let (lu, lv) = (_mm_srli_epi64::<56>(u), _mm_srli_epi64::<56>(v));
                let pick = |a, b| {
                    _mm_unpacklo_epi64(
                        _mm_shuffle_epi32::<0b1000>(a),
                        _mm_shuffle_epi32::<0b1000>(b),
                    )
                };
                (pick(su, sv), pick(lu, lv))
            }
        };
        let prev = _mm_or_si128(_mm_slli_si128::<4>(last), _mm_cvtsi32_si128(i32::from(p[0])));
        let carry = mult_fix(_mm_madd_epi16(prev, load(wp)), fx, round);
        let v = _mm_sub_epi32(
            _mm_madd_epi16(_mm_add_epi32(sums, carry), xs),
            _mm_madd_epi16(last, load(&wp[1..])),
        );
        store(&mut out[x..x + 4], v);
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn mult_fix(v: __m128i, scale: __m128i, round: __m128i) -> __m128i {
        let even = _mm_srli_epi64::<32>(_mm_add_epi64(_mm_mul_epu32(v, scale), round));
        let odd = _mm_add_epi64(_mm_mul_epu32(_mm_srli_epi64::<32>(v), scale), round);
        _mm_or_si128(even, _mm_and_si128(odd, _mm_set1_epi64x(-1 << 32)))
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn clip(v: __m128i) -> __m128i {
        let small = _mm_cmpeq_epi32(_mm_srli_epi32::<8>(v), _mm_setzero_si128());
        _mm_or_si128(
            _mm_and_si128(v, small),
            _mm_andnot_si128(small, _mm_set1_epi32(255)),
        )
    }

    /// The shrinking export for the samples whole vectors cover; returns how many it wrote.
    #[allow(unsafe_code)]
    pub(super) fn export_shrink(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
    ) -> usize {
        let mut i = 0;
        if crate::wide::has_avx2() {
            // SAFETY: the CPU supports AVX2.
            i = unsafe { super::avx2::export_shrink(dst, irow, frow, yscale, fxy_scale) };
        }
        // SAFETY: SSE2 is part of x86_64.
        unsafe { export_shrink_from(dst, irow, frow, yscale, fxy_scale, i) }
    }

    /// `export_shrink` with the samples before `i` already written.
    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) fn export_shrink_from(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
        i: usize,
    ) -> usize {
        let n = dst.len();
        let (round, floor) = (_mm_set1_epi64x(1 << 31), _mm_setzero_si128());
        let (ys, fxy) = (
            _mm_set1_epi64x(i64::from(yscale)),
            _mm_set1_epi64x(i64::from(fxy_scale)),
        );
        let dst = dst[i..].as_chunks_mut::<16>().0;
        let irow = irow[i..n].as_chunks_mut::<4>().0.as_chunks_mut::<4>().0;
        let frow = frow[i..n].as_chunks::<4>().0.as_chunks::<4>().0;
        for ((dst, irow), frow) in dst.iter_mut().zip(irow).zip(frow) {
            let mut d = [floor; 4];
            for ((d, ir), fr) in d.iter_mut().zip(irow).zip(frow) {
                let v = load(&*ir);
                let frac = match yscale {
                    0 => floor,
                    _ => mult_fix(load(fr), ys, floor),
                };
                store(ir, frac);
                *d = clip(mult_fix(_mm_sub_epi32(v, frac), fxy, round));
            }
            let bytes = _mm_packus_epi16(_mm_packs_epi32(d[0], d[1]), _mm_packs_epi32(d[2], d[3]));
            store(dst, bytes);
        }
        i + dst.len() * 16
    }
}

/// The SSE2 kernels of `mod sse2` 256 bits wide.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::sse2;
    use core::arch::x86_64::*;

    use crate::simd_mem::{load, load_at, store};

    #[target_feature(enable = "avx2")]
    #[inline]
    fn mult_fix(v: __m256i, scale: __m256i, round: __m256i) -> __m256i {
        let even = _mm256_srli_epi64::<32>(_mm256_add_epi64(_mm256_mul_epu32(v, scale), round));
        let odd = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64::<32>(v), scale), round);
        _mm256_blend_epi32::<0b1010_1010>(even, odd)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn clip(v: __m256i) -> __m256i {
        _mm256_min_epu32(v, _mm256_set1_epi32(255))
    }

    /// `sse2::each_group` with vectors of eight outputs.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn each_group<X, const V: usize>(
        src: impl Iterator<Item = X>,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
        f: impl Fn(X) -> [__m256i; V],
    ) -> usize {
        let out = out.as_chunks_mut::<8>().0.as_chunks_mut::<V>().0;
        match sum {
            Some(sum) => {
                let sum = sum.as_chunks_mut::<8>().0.as_chunks_mut::<V>().0;
                for ((o, s), x) in out.iter_mut().zip(sum).zip(src) {
                    for ((o, s), v) in o.iter_mut().zip(s).zip(f(x)) {
                        store(o, v);
                        let v = _mm256_add_epi32(load(&*s), v);
                        store(s, v);
                    }
                }
            }
            None => {
                for (o, x) in out.iter_mut().zip(src) {
                    for (o, v) in o.iter_mut().zip(f(x)) {
                        store(o, v);
                    }
                }
            }
        }
        out.len() * 8 * V
    }

    /// `sse2::shrink_whole` of one channel 8 or 32 outputs at a time; returns how many it
    /// wrote. Takes `src` of at least `k` samples an output and `x_sub` that fits an i16.
    #[target_feature(enable = "avx2")]
    pub(super) fn shrink_whole(
        src: &[u8],
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
    ) -> usize {
        let z = _mm256_setzero_si256();
        let xs = _mm256_set1_epi32(x_sub as i32);
        let xs16 = _mm256_set1_epi16(x_sub as i16);
        match k {
            1 => each_group(src.as_chunks::<32>().0.iter(), out, sum, |p| {
                let g = |at: usize| _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(load_at(p, at)));
                [g(0), g(8), g(16), g(24)].map(|v| _mm256_madd_epi16(v, xs))
            }),
            2 => each_group(src.as_chunks::<16>().0.iter(), out, sum, |p| {
                [_mm256_madd_epi16(_mm256_cvtepu8_epi16(load(p)), xs16)]
            }),
            4 => {
                let low = _mm256_set1_epi16(0xff);
                each_group(src.as_chunks::<32>().0.iter(), out, sum, |p| {
                    let v = load(p);
                    let pairs = _mm256_add_epi16(_mm256_and_si256(v, low), _mm256_srli_epi16::<8>(v));
                    [_mm256_madd_epi16(pairs, xs16)]
                })
            }
            8 => each_group(src.as_chunks::<64>().0.iter(), out, sum, |p| {
                let a = _mm256_sad_epu8(load(p), z);
                let b = _mm256_sad_epu8(load(&p[32..]), z);
                let v = _mm256_unpacklo_epi64(
                    _mm256_shuffle_epi32::<0b1000>(a),
                    _mm256_shuffle_epi32::<0b1000>(b),
                );
                [_mm256_madd_epi16(_mm256_permute4x64_epi64::<0b1101_1000>(v), xs)]
            }),
            _ => 0,
        }
    }

    /// `sse2::shrink_spans` of one channel, eight outputs to a vector where eight in a row take
    /// `k` source pixels each, else four. Takes `x_sub` that fits an i16, `starts` of one more
    /// than `weights` and `out` of as many outputs.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shrink_spans(
        src: &[u8],
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) {
        let n = weights.len();
        let z = _mm256_setzero_si256();
        let (xs, ones) = (_mm256_set1_epi32(x_sub as i32), _mm256_set1_epi16(1));
        let (fx, round) = (
            _mm256_set1_epi64x(i64::from(fx_scale)),
            _mm256_set1_epi64x(1 << 31),
        );
        let back = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
        let mut x = 0;
        while x < n {
            if !sse2::whole_run(src, starts, k, x, 8) {
                if sse2::whole_run(src, starts, k, x, 4) {
                    sse2::span4(src, starts, weights, k, x_sub, fx_scale, x, out);
                    x += 4;
                } else {
                    super::shrink_span::<1>(src, starts, weights, x_sub, fx_scale, x, out);
                    x += 1;
                }
                continue;
            }
            let a = starts[x] as usize;
            // The source pixel before the outputs' on, and the weight of the output before them on.
            let (p, wp) = (&src[a - 1..a + 8 * k], &weights[x - 1..x + 8]);
            let (sums, last) = match k {
                1 => {
                    let v = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(load_at(p, 1)));
                    (v, v)
                }
                2 => {
                    let v = _mm256_cvtepu8_epi16(load_at(p, 1));
                    (_mm256_madd_epi16(v, ones), _mm256_srli_epi32::<16>(v))
                }
                4 => {
                    let v = load_at(p, 1);
                    let pairs = _mm256_add_epi16(
                        _mm256_and_si256(v, _mm256_set1_epi16(0xff)),
                        _mm256_srli_epi16::<8>(v),
                    );
                    (_mm256_madd_epi16(pairs, ones), _mm256_srli_epi32::<24>(v))
                }
                _ => {
                    let (u, v) = (load_at(p, 1), load_at(p, 33));
                    let (su, sv) = (_mm256_sad_epu8(u, z), _mm256_sad_epu8(v, z));
                    let (lu, lv) = (_mm256_srli_epi64::<56>(u), _mm256_srli_epi64::<56>(v));
                    let pick = |a, b| {
                        _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_unpacklo_epi64(
                            _mm256_shuffle_epi32::<0b1000>(a),
                            _mm256_shuffle_epi32::<0b1000>(b),
                        ))
                    };
                    (pick(su, sv), pick(lu, lv))
                }
            };
            let prev = _mm256_blend_epi32::<1>(
                _mm256_permutevar8x32_epi32(last, back),
                _mm256_set1_epi32(i32::from(p[0])),
            );
            let carry = mult_fix(_mm256_madd_epi16(prev, load(wp)), fx, round);
            let v = _mm256_sub_epi32(
                _mm256_madd_epi16(_mm256_add_epi32(sums, carry), xs),
                _mm256_madd_epi16(last, load(&wp[1..])),
            );
            store(&mut out[x..x + 8], v);
            x += 8;
        }
    }

    /// `sse2::export_shrink` 32 samples at a time; returns how many it wrote.
    #[target_feature(enable = "avx2")]
    pub(super) fn export_shrink(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
    ) -> usize {
        let n = dst.len();
        let (round, floor) = (_mm256_set1_epi64x(1 << 31), _mm256_setzero_si256());
        let (ys, fxy) = (
            _mm256_set1_epi64x(i64::from(yscale)),
            _mm256_set1_epi64x(i64::from(fxy_scale)),
        );
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let dst = dst.as_chunks_mut::<32>().0;
        let irow = irow[..n].as_chunks_mut::<8>().0.as_chunks_mut::<4>().0;
        let frow = frow[..n].as_chunks::<8>().0.as_chunks::<4>().0;
        for ((dst, irow), frow) in dst.iter_mut().zip(irow).zip(frow) {
            let mut d = [floor; 4];
            for ((d, ir), fr) in d.iter_mut().zip(irow).zip(frow) {
                let v = load(&*ir);
                let frac = match yscale {
                    0 => floor,
                    _ => mult_fix(load(fr), ys, floor),
                };
                store(ir, frac);
                *d = clip(mult_fix(_mm256_sub_epi32(v, frac), fxy, round));
            }
            let bytes = _mm256_packus_epi16(
                _mm256_packs_epi32(d[0], d[1]),
                _mm256_packs_epi32(d[2], d[3]),
            );
            store(dst, _mm256_permutevar8x32_epi32(bytes, order));
        }
        dst.len() * 32
    }
}

/// libwebp's NEON exports (`rescaler_neon.c`), which differ from its C ones over the samples that
/// whole groups of 8 cover: they multiply by half the scale with `vqrdmulh`, on the samples as
/// signed, subtract saturating and narrow through 16 bits. The rest of a row takes the C ones.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use crate::simd_mem::{load, load_at, store};

    #[target_feature(enable = "neon")]
    #[inline]
    fn mult_fix(v: uint32x4_t, half_scale: int32x4_t) -> uint32x4_t {
        vreinterpretq_u32_s32(vqrdmulhq_s32(vreinterpretq_s32_u32(v), half_scale))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn mult_fix_floor(v: uint32x4_t, half_scale: int32x4_t) -> uint32x4_t {
        vreinterpretq_u32_s32(vqdmulhq_s32(vreinterpretq_s32_u32(v), half_scale))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn half(scale: u32) -> int32x4_t {
        vdupq_n_s32((scale >> 1) as i32)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn narrow(a: uint32x4_t, b: uint32x4_t) -> uint8x8_t {
        vqmovn_u16(vcombine_u16(vmovn_u32(a), vmovn_u32(b)))
    }

    /// [`super::mult_fix`], which libwebp's import keeps in C.
    #[target_feature(enable = "neon")]
    #[inline]
    fn mult_fix_c(v: uint32x4_t, scale: u32) -> uint32x4_t {
        let lo = vrshrn_n_u64::<32>(vmull_n_u32(vget_low_u32(v), scale));
        let hi = vrshrn_n_u64::<32>(vmull_high_n_u32(v, scale));
        vcombine_u32(lo, hi)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn load4(p: &[u8]) -> uint8x8_t {
        vreinterpret_u8_u32(vdup_n_u32(load(p)))
    }

    /// Stores the `V` vectors `f` makes of each item of `src` to the next `4 * V` outputs of
    /// `out`, adding them into `sum` as well where given; returns how many outputs it wrote.
    /// `src` and `sum` cover the whole groups of `out`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn each_group<X, const V: usize>(
        src: impl Iterator<Item = X>,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
        f: impl Fn(X) -> [uint32x4_t; V],
    ) -> usize {
        let out = out.as_chunks_mut::<4>().0.as_chunks_mut::<V>().0;
        match sum {
            Some(sum) => {
                let sum = sum.as_chunks_mut::<4>().0.as_chunks_mut::<V>().0;
                for ((o, s), x) in out.iter_mut().zip(sum).zip(src) {
                    for ((o, s), v) in o.iter_mut().zip(s).zip(f(x)) {
                        store(o, v);
                        let v = vaddq_u32(load(&*s), v);
                        store(s, v);
                    }
                }
            }
            None => {
                for (o, x) in out.iter_mut().zip(src) {
                    for (o, v) in o.iter_mut().zip(f(x)) {
                        store(o, v);
                    }
                }
            }
        }
        out.len() * 4 * V
    }

    /// [`super::shrink_whole`] for the outputs whole vectors cover; returns how many it wrote.
    #[allow(unsafe_code)]
    pub(super) fn shrink_whole(
        src: &[u8],
        channels: usize,
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
    ) -> usize {
        let n = out.len();
        assert!(src.len() >= n * k && sum.as_ref().is_none_or(|s| s.len() >= n));
        // SAFETY: NEON is part of aarch64.
        unsafe { shrink_whole_neon(src, channels, k, x_sub, out, sum) }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn shrink_whole_neon(
        src: &[u8],
        channels: usize,
        k: usize,
        x_sub: u32,
        out: &mut [u32],
        sum: Option<&mut [u32]>,
    ) -> usize {
        let xs = vdupq_n_u32(x_sub);
        let widen = |v: uint16x4_t| vmulq_u32(vmovl_u16(v), xs);
        match (channels, k) {
            (1, 1) => each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                let v = load(v);
                let (lo, hi) = (vmovl_u8(vget_low_u8(v)), vmovl_high_u8(v));
                [
                    widen(vget_low_u16(lo)),
                    widen(vget_high_u16(lo)),
                    widen(vget_low_u16(hi)),
                    widen(vget_high_u16(hi)),
                ]
            }),
            (1, 2) => each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                let v = vpaddlq_u8(load(v));
                [widen(vget_low_u16(v)), widen(vget_high_u16(v))]
            }),
            (1, 4) => each_group(src.as_chunks::<16>().0.iter(), out, sum, |v| {
                [vmulq_u32(vpaddlq_u16(vpaddlq_u8(load(v))), xs)]
            }),
            (1, 8) => each_group(src.as_chunks::<32>().0.iter(), out, sum, |p| {
                let a = vpaddlq_u16(vpaddlq_u8(load(p)));
                let b = vpaddlq_u16(vpaddlq_u8(load(&p[16..])));
                [vmulq_u32(vpaddq_u32(a, b), xs)]
            }),
            // Two pixels to a 16-bit vector: up to 256 pixels' channel sums fit.
            (4, 2..=256) => each_group(src.chunks_exact(4 * k), out, sum, |p| {
                let (pairs, last) = p.as_chunks::<8>();
                let mut acc = vdupq_n_u16(0);
                for v in pairs {
                    acc = vaddw_u8(acc, load(v));
                }
                let mut px = vadd_u16(vget_low_u16(acc), vget_high_u16(acc));
                if let Some(last) = last.first_chunk::<4>() {
                    px = vadd_u16(px, vget_low_u16(vmovl_u8(load4(last))));
                }
                [widen(px)]
            }),
            _ => 0,
        }
    }

    /// [`super::shrink_span`] for every output pixel of one channel, four outputs to a vector where
    /// four in a row take `k` source pixels each. False for more channels.
    #[allow(clippy::too_many_arguments)]
    #[allow(unsafe_code)]
    pub(super) fn shrink_spans(
        src: &[u8],
        channels: usize,
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) -> bool {
        let n = weights.len();
        assert!(starts.len() == n + 1 && out.len() >= n * channels);
        if channels != 1 {
            return false;
        }
        // SAFETY: NEON is part of aarch64.
        unsafe { shrink_spans_neon(src, starts, weights, k, x_sub, fx_scale, out) };
        true
    }

    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn shrink_spans_neon(
        src: &[u8],
        starts: &[u32],
        weights: &[u32],
        k: usize,
        x_sub: u32,
        fx_scale: u32,
        out: &mut [u32],
    ) {
        let n = weights.len();
        let xs = vdupq_n_u32(x_sub);
        let mut x = 0;
        while x < n {
            let a = starts[x] as usize;
            let whole = x >= 1 && x + 4 <= n && a >= 1 && starts[x + 4] as usize == a + 4 * k;
            if !(whole && matches!(k, 1 | 2 | 4 | 8) && a + 4 * k <= src.len()) {
                super::shrink_span::<1>(src, starts, weights, x_sub, fx_scale, x, out);
                x += 1;
                continue;
            }
            // The source pixel before the outputs' on, and the weight of the output before them on.
            let (p, wp) = (&src[a - 1..a + 4 * k], &weights[x - 1..x + 4]);
            let (sums, last) = match k {
                1 => {
                    let v = vmovl_u16(vget_low_u16(vmovl_u8(load4(&p[1..]))));
                    (v, v)
                }
                2 => {
                    let v: uint8x8_t = load_at(p, 1);
                    let last = vshr_n_u16::<8>(vreinterpret_u16_u8(v));
                    (vmovl_u16(vpaddl_u8(v)), vmovl_u16(last))
                }
                4 => {
                    let v: uint8x16_t = load_at(p, 1);
                    let last = vshrq_n_u32::<24>(vreinterpretq_u32_u8(v));
                    (vpaddlq_u16(vpaddlq_u8(v)), last)
                }
                _ => {
                    let (u, v): (uint8x16_t, uint8x16_t) = (load_at(p, 1), load_at(p, 17));
                    let sums =
                        vpaddq_u32(vpaddlq_u16(vpaddlq_u8(u)), vpaddlq_u16(vpaddlq_u8(v)));
                    let lu = vmovn_u64(vshrq_n_u64::<56>(vreinterpretq_u64_u8(u)));
                    let lv = vmovn_u64(vshrq_n_u64::<56>(vreinterpretq_u64_u8(v)));
                    (sums, vcombine_u32(lu, lv))
                }
            };
            let prev = vextq_u32::<3>(vdupq_n_u32(u32::from(p[0])), last);
            let carry = mult_fix_c(vmulq_u32(prev, load(wp)), fx_scale);
            let v = vsubq_u32(vmulq_u32(vaddq_u32(sums, carry), xs), vmulq_u32(last, load(&wp[1..])));
            store(&mut out[x..x + 4], v);
            x += 4;
        }
    }

    /// `RescalerExportRowExpand_NEON`, interpolating `frow` and `irow` by `ab` where given;
    /// returns how many samples it wrote.
    #[allow(unsafe_code)]
    pub(super) fn export_expand(
        dst: &mut [u8],
        frow: &[u32],
        irow: &[u32],
        fy_scale: u32,
        ab: Option<(u32, u32)>,
    ) -> usize {
        // SAFETY: NEON is part of aarch64.
        unsafe { export_expand_neon(dst, frow, irow, fy_scale, ab) }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn export_expand_neon(
        dst: &mut [u8],
        frow: &[u32],
        irow: &[u32],
        fy_scale: u32,
        ab: Option<(u32, u32)>,
    ) -> usize {
        let n = dst.len() & !7;
        let fy = half(fy_scale);
        let row = |f: &[u32; 4], g: &[u32; 4]| {
            let f = load(f);
            let Some((a, b)) = ab else { return f };
            let g = load(g);
            let lo = vmlal_n_u32(vmull_n_u32(vget_low_u32(f), a), vget_low_u32(g), b);
            let hi = vmlal_n_u32(vmull_n_u32(vget_high_u32(f), a), vget_high_u32(g), b);
            vcombine_u32(vrshrn_n_u64::<32>(lo), vrshrn_n_u64::<32>(hi))
        };
        let dst = dst[..n].as_chunks_mut::<8>().0;
        let frow = frow[..n].as_chunks::<4>().0.as_chunks::<2>().0;
        let irow = irow[..n].as_chunks::<4>().0.as_chunks::<2>().0;
        for ((d, f), g) in dst.iter_mut().zip(frow).zip(irow) {
            store(d, narrow(mult_fix(row(&f[0], &g[0]), fy), mult_fix(row(&f[1], &g[1]), fy)));
        }
        n
    }

    /// `RescalerExportRowShrink_NEON`; returns how many samples it wrote.
    #[allow(unsafe_code)]
    pub(super) fn export_shrink(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
    ) -> usize {
        // SAFETY: NEON is part of aarch64.
        unsafe { export_shrink_neon(dst, irow, frow, yscale, fxy_scale) }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn export_shrink_neon(
        dst: &mut [u8],
        irow: &mut [u32],
        frow: &[u32],
        yscale: u32,
        fxy_scale: u32,
    ) -> usize {
        let n = dst.len() & !7;
        let (ys, fxy) = (half(yscale), half(fxy_scale));
        let dst = dst[..n].as_chunks_mut::<8>().0;
        let irow = irow[..n].as_chunks_mut::<4>().0.as_chunks_mut::<2>().0;
        let frow = frow[..n].as_chunks::<4>().0.as_chunks::<2>().0;
        for ((d, [i0, i1]), [f0, f1]) in dst.iter_mut().zip(irow).zip(frow) {
            let (mut a0, mut a1) = (vdupq_n_u32(0), vdupq_n_u32(0));
            if yscale != 0 {
                a0 = mult_fix_floor(load(f0), ys);
                a1 = mult_fix_floor(load(f1), ys);
            }
            let b0 = vqsubq_u32(load(&*i0), a0);
            let b1 = vqsubq_u32(load(&*i1), a1);
            store(d, narrow(mult_fix(b0, fxy), mult_fix(b1, fxy)));
            store(i0, a0);
            store(i1, a1);
        }
        n
    }
}

/// libwebp's `WebPRescalerGetScaledDimensions` leaves given sizes as they are; the decoders
/// turn off the loop filter where both sides shrink below three quarters.
pub(crate) fn bypasses_filter(width: u32, height: u32, scaled: (u32, u32)) -> bool {
    scaled.0 < width * 3 / 4 && scaled.1 < height * 3 / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> Vec<u8> {
        let mut r = Rescaler::new(sw, sh, dw, dh, ch);
        let mut out = vec![0; dw * dh * ch];
        let mut y = 0;
        let mut rows = src.chunks(sw * ch);
        while y < dh {
            if r.has_pending_output() {
                r.export(&mut out[y * dw * ch..]);
                y += 1;
            } else {
                assert!(r.import(rows.next().unwrap()));
            }
        }
        out
    }

    #[test]
    fn identity_and_halving() {
        let src: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        assert_eq!(run(&src, 8, 8, 8, 8, 1), src);
        let half = run(&src, 8, 8, 4, 4, 1);
        assert_eq!(half[0], ((0 + 4 + 32 + 36) as f32 / 4.0).round() as u8);
        let up = run(&[0, 255], 2, 1, 4, 1, 1);
        assert_eq!(up, [0, 85, 170, 255]);
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// libwebp's shrinking import, a source pixel at a time.
    fn reference_shrink(
        src: &[u8],
        n: usize,
        x_add: i32,
        x_sub: i32,
        fx_scale: u32,
        out: &mut [u32],
    ) {
        let mut pixels = src.chunks_exact(n);
        let mut sum = vec![0u32; n];
        let mut accum = 0i32;
        for out in out.chunks_exact_mut(n) {
            let mut base = vec![0u32; n];
            accum += x_add;
            while accum > 0 {
                accum -= x_sub;
                let p = pixels.next().unwrap();
                for c in 0..n {
                    base[c] = u32::from(p[c]);
                    sum[c] = sum[c].wrapping_add(base[c]);
                }
            }
            for c in 0..n {
                let frac = base[c].wrapping_mul((-accum) as u32);
                out[c] = sum[c].wrapping_mul(x_sub as u32).wrapping_sub(frac);
                sum[c] = mult_fix(frac, fx_scale);
            }
        }
    }

    #[test]
    fn span_shrinks_match_the_pixel_loop() {
        let mut rng = Rng(11);
        for channels in 1..=4 {
            for (sw, dw) in [
                (3, 2),
                (5, 2),
                (7, 3),
                (101, 50),
                (6001, 3000),
                (3001, 751),
                (1001, 999),
                (17, 2),
                (9, 8),
                (3001, 3000),
                (6001, 1500),
                (6001, 750),
                (12003, 1500),
                (2000, 15),
            ] {
                let src: Vec<u8> = (0..sw * channels).map(|_| rng.next() as u8).collect();
                let mut r = Rescaler::new(sw, 4, dw, 2, channels);
                assert_eq!(r.whole_shrink, None);
                assert!(r.import(&src));
                let mut want = vec![0; dw * channels];
                reference_shrink(&src, channels, r.x_add, r.x_sub, r.fx_scale, &mut want);
                assert_eq!(r.frow, want, "{channels} channels, {sw} -> {dw}");
            }
        }
    }

    #[test]
    fn whole_shrinks_match_the_fraction_carrying_import() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for channels in 1..=4 {
            for k in [1, 2, 3, 4, 8, 9] {
                for dw in [1, 3, 4, 7, 16, 33, 100] {
                    let sw = dw * k;
                    let src: Vec<u8> = (0..sw * channels).map(|_| rng.next() as u8).collect();
                    let rows = |rng: &mut Rng, whole: bool, y_expand: bool| {
                        let mut r =
                            Rescaler::new(sw, 3, dw, if y_expand { 5 } else { 2 }, channels);
                        assert_eq!(r.whole_shrink, Some(k));
                        r.irow.iter_mut().for_each(|i| *i = rng.next() as u32);
                        if whole {
                            assert!(r.import(&src));
                        } else {
                            if y_expand {
                                std::mem::swap(&mut r.irow, &mut r.frow);
                            }
                            reference_shrink(
                                &src,
                                channels,
                                r.x_add,
                                r.x_sub,
                                r.fx_scale,
                                &mut r.frow,
                            );
                            if !y_expand {
                                for (i, f) in r.irow.iter_mut().zip(&r.frow) {
                                    *i = i.wrapping_add(*f);
                                }
                            }
                        }
                        (r.irow, r.frow)
                    };
                    let seed = rng.0;
                    let whole = (rows(&mut rng, true, false), rows(&mut rng, true, true));
                    rng.0 = seed;
                    assert_eq!(
                        whole,
                        (rows(&mut rng, false, false), rows(&mut rng, false, true)),
                        "{channels} channels, {sw} -> {dw}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    #[test]
    fn avx2_whole_shrinks_match_the_sums() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = Rng(12);
        for k in [1, 2, 4, 8] {
            for n in [0, 1, 7, 8, 9, 31, 32, 33, 70] {
                let src: Vec<u8> = (0..n * k).map(|_| rng.next() as u8).collect();
                let x_sub = (rng.next() % 32768) as u32;
                let base: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
                let (mut out, mut sum) = (vec![0; n], base.clone());
                // SAFETY: the CPU has AVX2.
                let done = unsafe { avx2::shrink_whole(&src, k, x_sub, &mut out, Some(&mut sum)) };
                assert_eq!(done, if k == 1 { n / 32 * 32 } else { n / 8 * 8 });
                for x in 0..done {
                    let v = src[x * k..][..k].iter().map(|&p| u32::from(p)).sum::<u32>() * x_sub;
                    assert_eq!((out[x], sum[x]), (v, base[x].wrapping_add(v)), "k {k} n {n}");
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    #[test]
    fn avx2_export_matches_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = Rng(11);
        for n in [0, 1, 31, 32, 33, 64, 95, 200] {
            for case in 0..64 {
                let frow: Vec<u32> = (0..n).map(|_| (rng.next() % (1 << 30)) as u32).collect();
                let irow: Vec<u32> = frow.iter().map(|&f| f + (rng.next() % 4096) as u32).collect();
                let yscale = if case % 4 == 0 { 0 } else { rng.next() as u32 };
                let fxy_scale = rng.next() as u32;
                let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
                let (mut ia, mut ib) = (irow.clone(), irow);
                // SAFETY: the CPU has AVX2.
                let done = unsafe { avx2::export_shrink(&mut a, &mut ia, &frow, yscale, fxy_scale) };
                assert_eq!(done, n / 32 * 32);
                // SAFETY: SSE2 is part of x86_64.
                unsafe { sse2::export_shrink_from(&mut b, &mut ib, &frow, yscale, fxy_scale, 0) };
                assert_eq!((&a[..done], &ia[..done]), (&b[..done], &ib[..done]), "{n}");
            }
        }
    }

    /// Exported rows against libwebp's own `WebPRescalerExportRowShrink` and
    /// `WebPRescalerExportRowExpand` on this target, at lengths around its vectors.
    #[allow(unsafe_code)]
    #[test]
    fn export_matches_libwebp() {
        #[repr(C)]
        #[derive(Default)]
        struct WebPRescaler {
            x_expand: i32,
            y_expand: i32,
            num_channels: i32,
            fx_scale: u32,
            fy_scale: u32,
            fxy_scale: u32,
            y_accum: i32,
            y_add: i32,
            y_sub: i32,
            x_add: i32,
            x_sub: i32,
            src_width: i32,
            src_height: i32,
            dst_width: i32,
            dst_height: i32,
            src_y: i32,
            dst_y: i32,
            dst: Option<core::ptr::NonNull<u8>>,
            dst_stride: i32,
            irow: Option<core::ptr::NonNull<u32>>,
            frow: Option<core::ptr::NonNull<u32>>,
        }
        type Export = unsafe extern "C" fn(*mut WebPRescaler);
        extern "C" {
            fn WebPRescalerDspInit();
            static WebPRescalerExportRowShrink: Option<Export>;
            static WebPRescalerExportRowExpand: Option<Export>;
        }
        unsafe {
            assert!(libwebp_sys::WebPGetDecoderVersion() > 0);
            WebPRescalerDspInit();
        }
        let scales = [
            1,
            0x0100_0001,
            0x4000_0000,
            0x8000_0001,
            0x1234_5679,
            u32::MAX,
        ];
        let mut rng = Rng(7);
        for n in [1, 7, 8, 15, 16, 17, 24, 48, 61] {
            for case in 0..256 {
                let expand = case % 2 == 1;
                let small = case % 4 < 2;
                // A row's sum holds its last input row, and stays below 2^31: past that, libwebp's
                // own kernels disagree with each other.
                let mut sample = || match small {
                    true => (rng.next() % 4096) as u32,
                    false => (rng.next() % (1 << 30)) as u32,
                };
                let frow: Vec<u32> = (0..n).map(|_| sample()).collect();
                let irow: Vec<u32> = frow.iter().map(|&f| f + sample()).collect();
                let pick = |rng: &mut Rng| scales[rng.next() as usize % scales.len()];
                let (fy_scale, fxy_scale) = (pick(&mut rng), pick(&mut rng));
                let y_sub = 1 + (rng.next() % 5000) as i32;
                let y_accum = match rng.next() % 3 {
                    0 => 0,
                    _ => -((rng.next() % y_sub as u64) as i32),
                };

                let mut r = match expand {
                    true => Rescaler::new(n, 2, n, 4, 1),
                    false => Rescaler::new(2 * n, 4, n, 2, 1),
                };
                (r.irow, r.frow, r.fy_scale, r.fxy_scale, r.y_accum, r.y_sub) = (
                    irow.clone(),
                    frow.clone(),
                    fy_scale,
                    fxy_scale,
                    y_accum,
                    y_sub,
                );
                let mut ours = vec![0; n];
                r.export(&mut ours);

                let (mut theirs, mut irow, mut frow) = (vec![0u8; n], irow, frow);
                let mut w = WebPRescaler {
                    y_expand: i32::from(expand),
                    num_channels: 1,
                    fy_scale,
                    fxy_scale,
                    y_accum,
                    y_sub,
                    dst_width: n as i32,
                    dst_height: 1,
                    dst: core::ptr::NonNull::new(theirs.as_mut_ptr()),
                    irow: core::ptr::NonNull::new(irow.as_mut_ptr()),
                    frow: core::ptr::NonNull::new(frow.as_mut_ptr()),
                    ..Default::default()
                };
                unsafe {
                    match expand {
                        true => WebPRescalerExportRowExpand.unwrap()(&mut w),
                        false => WebPRescalerExportRowShrink.unwrap()(&mut w),
                    }
                }
                assert_eq!(
                    (ours, r.irow),
                    (theirs, irow),
                    "{n} samples, expand {expand}, fy {fy_scale:x}, fxy {fxy_scale:x}, accum {y_accum}/{y_sub}"
                );
            }
        }
    }
}
