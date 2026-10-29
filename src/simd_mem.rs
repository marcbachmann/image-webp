//! Loads and stores between slices and SIMD registers, each checking that the bytes it
//! touches are in bounds. With these the SIMD kernels are safe code: the rest of their
//! intrinsics are safe inside `#[target_feature]` functions.
#![allow(unsafe_code)]

/// Types every bit pattern of which is a valid value, with no padding.
///
/// # Safety
/// Implementors must be such types.
pub(crate) unsafe trait Pod: Copy {}

macro_rules! pod {
    ($($t:ty),*) => { $(unsafe impl crate::simd_mem::Pod for $t {})* };
}

pod!(u8, i8, u16, i16, u32, i32, u64, i64);

#[cfg(target_arch = "x86_64")]
pod!(
    core::arch::x86_64::__m128i,
    core::arch::x86_64::__m256i
);

#[cfg(target_arch = "aarch64")]
mod aarch64_pod {
    use core::arch::aarch64::*;
    pod!(
        uint8x8_t, uint8x16_t, int8x8_t, int8x16_t, uint16x4_t, uint16x8_t, int16x4_t,
        int16x8_t, uint32x2_t, uint32x4_t, int32x2_t, int32x4_t, uint64x1_t, uint64x2_t,
        int64x1_t, int64x2_t, uint8x16x2_t, uint8x16x4_t
    );
}

/// A `V` from the start of `src`.
#[inline(always)]
pub(crate) fn load<V: Pod, T: Pod>(src: &[T]) -> V {
    assert!(size_of_val(src) >= size_of::<V>());
    // SAFETY: `src` holds at least `size_of::<V>()` bytes, any of which make a valid `V`.
    unsafe { src.as_ptr().cast::<V>().read_unaligned() }
}

/// Writes `v` to the start of `dst`.
#[inline(always)]
pub(crate) fn store<V: Pod, T: Pod>(dst: &mut [T], v: V) {
    assert!(size_of_val(dst) >= size_of::<V>());
    // SAFETY: `dst` holds at least `size_of::<V>()` bytes, and any bytes make a valid `T`.
    unsafe { dst.as_mut_ptr().cast::<V>().write_unaligned(v) }
}

/// How many `T` a `V` spans.
#[inline(always)]
const fn span<V, T>() -> usize {
    size_of::<V>().div_ceil(size_of::<T>())
}

/// A `V` from `src[at..]`, in one comparison: `at` against a bound that loops hoist.
#[inline(always)]
pub(crate) fn load_at<V: Pod, T: Pod>(src: &[T], at: usize) -> V {
    assert!(src.len() >= span::<V, T>() && at <= src.len() - span::<V, T>());
    // SAFETY: `src[at..]` holds at least `size_of::<V>()` bytes, any of which make a valid `V`.
    unsafe { src.as_ptr().add(at).cast::<V>().read_unaligned() }
}

/// Writes `v` to `dst[at..]`, in one comparison as `load_at`.
#[inline(always)]
pub(crate) fn store_at<V: Pod, T: Pod>(dst: &mut [T], at: usize, v: V) {
    assert!(dst.len() >= span::<V, T>() && at <= dst.len() - span::<V, T>());
    // SAFETY: `dst[at..]` holds at least `size_of::<V>()` bytes, and any bytes make a valid `T`.
    unsafe { dst.as_mut_ptr().add(at).cast::<V>().write_unaligned(v) }
}

/// Whether `n` rows of `size` bytes, `stride` apart from `at`, lie within `len` bytes.
#[inline(always)]
fn rows_fit(len: usize, at: usize, stride: usize, n: usize, size: usize) -> bool {
    n > 0
        && stride
            .checked_mul(n - 1)
            .and_then(|s| s.checked_add(at))
            .and_then(|s| s.checked_add(size))
            .is_some_and(|end| end <= len)
}

/// `N` values of `V` from `src`, at `at`, `at + stride`, ..., checked once.
#[inline(always)]
pub(crate) fn load_rows<V: Pod, const N: usize>(src: &[u8], at: usize, stride: usize) -> [V; N] {
    assert!(rows_fit(src.len(), at, stride, N, size_of::<V>()));
    let p = src.as_ptr();
    // SAFETY: every row is in bounds by the assert above.
    core::array::from_fn(|k| unsafe { p.add(at + k * stride).cast::<V>().read_unaligned() })
}

/// Writes `rows` to `dst` at `at`, `at + stride`, ..., checked once.
#[inline(always)]
pub(crate) fn store_rows<V: Pod, const N: usize>(dst: &mut [u8], at: usize, stride: usize, rows: [V; N]) {
    assert!(rows_fit(dst.len(), at, stride, N, size_of::<V>()));
    let p = dst.as_mut_ptr();
    for (k, v) in rows.into_iter().enumerate() {
        // SAFETY: every row is in bounds by the assert above.
        unsafe { p.add(at + k * stride).cast::<V>().write_unaligned(v) }
    }
}

/// The interleaving loads and stores of NEON, which `load` and `store` cannot express.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use core::arch::aarch64 as arch;
    use core::arch::aarch64::*;

    macro_rules! ld {
        ($($name:ident($t:ty) -> $v:ty, $n:expr;)*) => {$(
            #[inline(always)]
            pub(crate) fn $name(src: &[$t]) -> $v {
                assert!(src.len() >= $n);
                // SAFETY: `src` holds the elements read; NEON is part of aarch64.
                unsafe { arch::$name(src.as_ptr()) }
            }
        )*};
    }

    macro_rules! st {
        ($($name:ident($t:ty, $v:ty), $n:expr;)*) => {$(
            #[inline(always)]
            pub(crate) fn $name(dst: &mut [$t], v: $v) {
                assert!(dst.len() >= $n);
                // SAFETY: `dst` holds the elements written; NEON is part of aarch64.
                unsafe { arch::$name(dst.as_mut_ptr(), v) }
            }
        )*};
    }

    ld! {
        vld3q_u8(u8) -> uint8x16x3_t, 48;
        vld4q_u8(u8) -> uint8x16x4_t, 64;
        vld3_u8(u8) -> uint8x8x3_t, 24;
        vld4_u8(u8) -> uint8x8x4_t, 32;
    }

    st! {
        vst2_u8(u8, uint8x8x2_t), 16;
        vst2q_u8(u8, uint8x16x2_t), 32;
        vst3_u8(u8, uint8x8x3_t), 24;
        vst3q_u8(u8, uint8x16x3_t), 48;
        vst4_u8(u8, uint8x8x4_t), 32;
        vst4q_u8(u8, uint8x16x4_t), 64;
    }
}

/// AVX2 gathers from 256-entry tables, the indices taken modulo 256.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::*;

    /// `table[idx & 255]` for the eight 32-bit lanes of `idx`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn gather32(table: &[u32; 256], idx: __m256i) -> __m256i {
        let idx = _mm256_and_si256(idx, _mm256_set1_epi32(255));
        // SAFETY: every index is below 256.
        unsafe { _mm256_i32gather_epi32::<4>(table.as_ptr().cast(), idx) }
    }

    /// `table[idx & 255]` for the four 32-bit lanes of `idx`, as 64-bit lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn gather64(table: &[u64; 256], idx: __m128i) -> __m256i {
        let idx = _mm_and_si128(idx, _mm_set1_epi32(255));
        // SAFETY: every index is below 256.
        unsafe { _mm256_i32gather_epi64::<8>(table.as_ptr().cast(), idx) }
    }
}
