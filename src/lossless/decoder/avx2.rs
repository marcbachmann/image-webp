//! AVX2 kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_avx2.c`, run behind
//! [`crate::wide::has_avx2`]. Each takes whole groups of eight pixels, leaving the rest to the
//! `sse2` kernel it widens. Pixels are RGBA bytes.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.

use core::arch::x86_64::*;

use super::sse2;
use crate::simd_mem::{load, store};

/// libwebp's `TransformColorInverse_AVX2` over the whole eights of pixels of `p`, with red and
/// blue where RGBA keeps them: the number of pixels done.
#[target_feature(enable = "avx2")]
#[inline]
fn color_transform8(p: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) -> usize {
    let cst = |m: u8| i32::from(((m as i8 as i16) << 3) as u16);
    let mults_rb = _mm256_set1_epi32(cst(green_to_blue) << 16 | cst(green_to_red));
    let mults_b2 = _mm256_set1_epi32(cst(red_to_blue) << 16);
    let mask_ag = _mm256_set1_epi32(0xff00ff00u32 as i32);
    #[rustfmt::skip]
    let perm1 = _mm256_setr_epi8(
        -1, 1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13,
        -1, 1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13,
    );
    #[rustfmt::skip]
    let perm2 = _mm256_setr_epi8(
        -1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12,
        -1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12,
    );
    let eights = p.as_chunks_mut::<32>().0;
    for c in eights.iter_mut() {
        let a = load(c);
        let d = _mm256_add_epi8(a, _mm256_mulhi_epi16(_mm256_shuffle_epi8(a, perm1), mults_rb));
        let g = _mm256_add_epi8(d, _mm256_mulhi_epi16(_mm256_shuffle_epi8(d, perm2), mults_b2));
        store(c, _mm256_blendv_epi8(g, a, mask_ag));
    }
    eights.len() * 8
}

/// `sse2::color_transform_row` with the kernel above.
#[target_feature(enable = "avx2")]
pub(crate) fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    for (b, m) in row.chunks_mut(block * 4).zip(mults.as_chunks::<4>().0) {
        let done = color_transform8(b, m[0], m[1], m[2]);
        sse2::color_transform(&mut b[4 * done..], m[0], m[1], m[2]);
    }
}

/// libwebp's `AddGreenToBlueAndRed_AVX2` over the whole eights of pixels of `image_data`: the
/// number of pixels done.
#[target_feature(enable = "avx2")]
pub(crate) fn add_green(image_data: &mut [u8]) -> usize {
    #[rustfmt::skip]
    let green = _mm256_setr_epi8(
        1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1,
        1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1,
    );
    let eights = image_data.as_chunks_mut::<32>().0;
    for c in eights.iter_mut() {
        let v = load(c);
        store(c, _mm256_add_epi8(v, _mm256_shuffle_epi8(v, green)));
    }
    eights.len() * 8
}

/// Drops the alpha of the first whole eights of the pixels of `rgba` into `rgb`, where the CPU
/// has AVX2: the number of pixels done.
#[allow(unsafe_code)]
pub(crate) fn rgb_from_rgba(rgba: &[u8], rgb: &mut [u8]) -> usize {
    if !crate::wide::has_avx2() {
        return 0;
    }
    // SAFETY: the CPU supports AVX2.
    unsafe { rgb_from_rgba8(rgba, rgb) }
}

#[target_feature(enable = "avx2")]
fn rgb_from_rgba8(rgba: &[u8], rgb: &mut [u8]) -> usize {
    let n = (rgba.len() / 4).min(rgb.len() / 3);
    let mut rgb = &mut rgb[..3 * n];
    #[rustfmt::skip]
    let pack = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
    );
    let lanes = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 7);
    let eights = rgba[..4 * n].as_chunks::<32>().0;
    for px in eights {
        let v = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(load(px), pack), lanes);
        // The 24 bytes, as 32 where the next eight overwrite the last 8.
        if let Some(d) = rgb.first_chunk_mut::<32>() {
            store(d, v);
        } else {
            store(rgb, _mm256_castsi256_si128(v));
            store(&mut rgb[16..], _mm_cvtsi128_si64(_mm256_extracti128_si256::<1>(v)));
        }
        rgb = &mut core::mem::take(&mut rgb)[24..];
    }
    eights.len() * 8
}

/// Inserts the first whole eights of the RGBA pixels `pixels` into a colour cache of `bits`
/// bits, in order, where the CPU has AVX2: the number of pixels done. The eight hashes are
/// taken at once, the stores one at a time, so a later pixel still wins a shared slot.
#[allow(unsafe_code)]
pub(crate) fn color_cache_insert(cache: &mut [u32; 1 << 11], bits: u8, pixels: &[u8]) -> usize {
    if !crate::wide::has_avx2() {
        return 0;
    }
    // SAFETY: the CPU supports AVX2.
    unsafe { color_cache_insert8(cache, bits, pixels) }
}

#[target_feature(enable = "avx2")]
fn color_cache_insert8(cache: &mut [u32; 1 << 11], bits: u8, pixels: &[u8]) -> usize {
    // RGBA to libwebp's ARGB words.
    #[rustfmt::skip]
    let argb = _mm256_setr_epi8(
        2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
        2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
    );
    let shift = _mm_cvtsi32_si128(32 - i32::from(bits));
    let mut slots = [0u32; 8];
    let (quads, _) = pixels.as_chunks::<32>();
    for q in quads {
        let hash = _mm256_mullo_epi32(_mm256_shuffle_epi8(load(q), argb), _mm256_set1_epi32(0x1e35a7bd));
        store(&mut slots, _mm256_srl_epi32(hash, shift));
        for (&slot, px) in slots.iter().zip(q.as_chunks::<4>().0) {
            cache[slot as usize & ((1 << 11) - 1)] = u32::from_le_bytes(*px);
        }
    }
    quads.len() * 8
}

/// libwebp's `WebPExtractGreen` over the first whole 32s of the RGBA pixels `rgba` into
/// `green`, where the CPU has AVX2: the number of pixels done.
#[allow(unsafe_code)]
pub(crate) fn extract_green(rgba: &[u8], green: &mut [u8]) -> usize {
    if !crate::wide::has_avx2() {
        return 0;
    }
    // SAFETY: the CPU supports AVX2.
    unsafe { extract_green32(rgba, green) }
}

#[target_feature(enable = "avx2")]
fn extract_green32(rgba: &[u8], green: &mut [u8]) -> usize {
    let (src, dst) = (rgba.as_chunks::<128>().0, green.as_chunks_mut::<32>().0);
    let mut done = 0;
    for (p, d) in src.iter().zip(dst) {
        let g = |k: usize| _mm256_and_si256(_mm256_srli_epi32::<8>(load(&p[32 * k..])), _mm256_set1_epi32(0xff));
        let bytes = _mm256_packus_epi16(_mm256_packus_epi32(g(0), g(1)), _mm256_packus_epi32(g(2), g(3)));
        // The packs interleave the four eights a lane each; dword k of lane l holds eight k's
        // pixels 4l to 4l + 3.
        store(d, _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
        done += 32;
    }
    done
}

/// libwebp's `WebPDispatchAlpha` over the first whole eights of `alpha` into the alpha of the
/// RGBA pixels `rgba`, where the CPU has AVX2: the number of pixels done.
#[allow(unsafe_code)]
pub(crate) fn dispatch_alpha(alpha: &[u8], rgba: &mut [u8]) -> usize {
    if !crate::wide::has_avx2() {
        return 0;
    }
    // SAFETY: the CPU supports AVX2.
    unsafe { dispatch_alpha8(alpha, rgba) }
}

#[target_feature(enable = "avx2")]
fn dispatch_alpha8(alpha: &[u8], rgba: &mut [u8]) -> usize {
    let (src, dst) = (alpha.as_chunks::<8>().0, rgba.as_chunks_mut::<32>().0);
    let rgb = _mm256_set1_epi32(0x00ff_ffff);
    let mut done = 0;
    for (a, p) in src.iter().zip(dst) {
        let a = _mm256_slli_epi32::<24>(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(load(a))));
        let v = _mm256_or_si256(_mm256_and_si256(load(p), rgb), a);
        store(p, v);
        done += 8;
    }
    done
}

/// The eight-pixel kernel alone, for the tests.
#[cfg(test)]
#[target_feature(enable = "avx2")]
fn color_transform8_alone(p: &mut [u8], m: [u8; 4]) -> usize {
    color_transform8(p, m[0], m[1], m[2])
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::super::reverse_transform as scalar;

    fn random(rng: &mut impl Rng, len: usize) -> Vec<u8> {
        let mut data = vec![0u8; len];
        rng.fill(&mut data[..]);
        // Small values and equal neighbours bring the ties and wraps out.
        if rng.gen() {
            data.iter_mut().for_each(|b| *b = (*b % 4).wrapping_sub(2));
        }
        data
    }

    #[test]
    fn rgb_from_rgba_drops_the_alpha() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(15);
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 64, 67] {
            let rgba = random(&mut rng, n * 4);
            for slack in [0, 1, 8, 9] {
                let mut rgb = vec![0xa5; n * 3 + slack];
                let done = super::rgb_from_rgba(&rgba, &mut rgb);
                assert_eq!(done, if crate::wide::has_avx2() { n / 8 * 8 } else { 0 });
                let want: Vec<u8> = rgba.chunks_exact(4).take(done).flat_map(|p| p[..3].to_vec()).collect();
                assert_eq!(rgb[..done * 3], want[..], "n {n} slack {slack}");
                assert!(rgb[n * 3..].iter().all(|&b| b == 0xa5), "n {n} slack {slack}");
            }
        }
    }

    #[test]
    fn color_cache_insert_matches_the_scalar_one() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(16);
        for bits in 1..=11u8 {
            for n in [0, 1, 7, 8, 9, 16, 33, 200] {
                // Few colours, so slots are shared and the order of the stores shows.
                let pixels: Vec<u8> = (0..n * 4).map(|_| rng.gen_range(0..3)).collect();
                let mut want = Box::new([0u32; 1 << 11]);
                for px in pixels.chunks_exact(4) {
                    let argb = u32::from_le_bytes(px.try_into().unwrap()).swap_bytes().rotate_right(8);
                    want[(0x1e35a7bdu32.wrapping_mul(argb) >> (32 - bits)) as usize] = u32::from_le_bytes(px.try_into().unwrap());
                }
                let mut got = Box::new([0u32; 1 << 11]);
                let done = super::color_cache_insert(&mut got, bits, &pixels);
                assert_eq!(done, if crate::wide::has_avx2() { n / 8 * 8 } else { 0 });
                for px in pixels[done * 4..].chunks_exact(4) {
                    let argb = u32::from_le_bytes(px.try_into().unwrap()).swap_bytes().rotate_right(8);
                    got[(0x1e35a7bdu32.wrapping_mul(argb) >> (32 - bits)) as usize] = u32::from_le_bytes(px.try_into().unwrap());
                }
                assert!(want == got, "bits {bits} n {n}");
            }
        }
    }

    #[test]
    fn extract_green_matches_the_scalar_one() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        for n in [0, 1, 7, 8, 9, 31, 32, 33, 64, 95, 200] {
            let rgba = random(&mut rng, n * 4);
            let mut green = vec![0xa5; n + 3];
            let done = super::extract_green(&rgba, &mut green);
            assert_eq!(done, if crate::wide::has_avx2() { n / 32 * 32 } else { 0 });
            assert!(green[..done].iter().zip(rgba.chunks_exact(4)).all(|(&g, p)| g == p[1]), "n {n}");
            assert!(green[done..].iter().all(|&b| b == 0xa5));
        }
    }

    #[test]
    fn dispatch_alpha_matches_the_scalar_one() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(18);
        for n in [0, 1, 7, 8, 9, 31, 32, 33, 64, 95, 200] {
            let (rgba, alpha) = (random(&mut rng, n * 4), random(&mut rng, n + 2));
            let mut got = rgba.clone();
            let done = super::dispatch_alpha(&alpha, &mut got);
            assert_eq!(done, if crate::wide::has_avx2() { n / 8 * 8 } else { 0 });
            let mut want = rgba;
            want.chunks_exact_mut(4).zip(&alpha).take(done).for_each(|(px, &a)| px[3] = a);
            assert_eq!(got, want, "n {n}");
        }
    }

    #[test]
    fn color_transform_rows_match_the_portable_ones() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(14);
        for width in [1u16, 5, 9, 17, 64, 67, 130] {
            for size_bits in 2..=6 {
                let data = random(&mut rng, usize::from(width) * 4);
                let mut mults = vec![0u8; usize::from(super::super::subsample_size(width, size_bits)) * 4];
                rng.fill(&mut mults[..]);
                let (mut a, mut b) = (data.clone(), data);
                scalar::apply_color_transform_portable(&mut a, width, 0, size_bits, &mults);
                unsafe { super::color_transform_row(&mut b, 1 << size_bits, &mults) };
                assert_eq!(a, b, "width {width} size_bits {size_bits}");
            }
        }
    }

    #[test]
    fn color_transform_and_add_green_match_the_portable_ones() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        for width in [1u16, 7, 8, 9, 16, 23, 64, 67, 255] {
            for _ in 0..20 {
                let data = random(&mut rng, usize::from(width) * 4);
                let mults: [u8; 4] = rng.gen();
                let (mut a, mut b) = (data.clone(), data.clone());
                scalar::apply_color_transform_portable(&mut a, width, 0, 9, &mults);
                let n = usize::from(width);
                let done = unsafe { super::color_transform8_alone(&mut b, mults) };
                assert_eq!(done, n / 8 * 8);
                assert_eq!(a[..done * 4], b[..done * 4], "width {width} {mults:?}");
                assert_eq!(data[done * 4..], b[done * 4..]);
                let (mut a, mut b) = (data.clone(), data.clone());
                scalar::apply_subtract_green_portable(&mut a);
                let done = unsafe { super::add_green(&mut b) };
                assert_eq!(done, n / 8 * 8);
                assert_eq!(a[..done * 4], b[..done * 4]);
                assert_eq!(data[done * 4..], b[done * 4..]);
            }
        }
    }
}
