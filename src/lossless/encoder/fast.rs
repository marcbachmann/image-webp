//! The fast mode's entropy coding over runs read straight from the pixels. A pixel that
//! repeats neither its left nor its upper neighbour is a literal; a run of either is a copy,
//! with no colour cache. Both passes, counting the histograms and writing the codes, walk
//! the pixels with [`next_run`], so no reference is held per pixel and a stretch of literals
//! is one tight loop.

use super::backward_refs::{
    distance_to_plane_code, find_match_length, max_find_copy_length, PixOrCopy, MIN_LENGTH,
};
use super::bit_writer::BitWriter;
use super::entropy::{prefix_encode, sub_sample_size};
use super::histogram::Histogram;
use super::huffman::HuffmanTreeCode;

/// The plane codes of the distances 1 and `xsize`.
fn plane_codes(xsize: u32) -> (u32, u32) {
    (
        distance_to_plane_code(xsize, 1),
        distance_to_plane_code(xsize, xsize),
    )
}

/// From `i`: how many literals come first, then the copy that follows them as its plane
/// code and length, or `None` at the end of the image.
#[inline]
fn next_run(
    argb: &[u32],
    xsize: usize,
    mut i: usize,
    left: u32,
    up: u32,
) -> (usize, Option<(u32, u32)>) {
    let n = argb.len();
    let start = i;
    loop {
        while i < n {
            let p = argb[i];
            if i > 0 && (p == argb[i - 1] || (i >= xsize && p == argb[i - xsize])) {
                break;
            }
            i += 1;
        }
        if i >= n {
            return (i - start, None);
        }
        let max_len = max_find_copy_length((n - i) as u32) as usize;
        let rle_len = find_match_length(argb, i, i - 1, 0, max_len);
        let prev_row_len = if i < xsize {
            0
        } else {
            find_match_length(argb, i, i - xsize, 0, max_len)
        };
        if rle_len >= prev_row_len && rle_len >= MIN_LENGTH as usize {
            return (i - start, Some((left, rle_len as u32)));
        }
        if prev_row_len >= MIN_LENGTH as usize {
            return (i - start, Some((up, prev_row_len as u32)));
        }
        i += 1;
    }
}

/// A stretch of literals that stays in one row and one histogram tile, or a copy, at `x`, `y`.
enum Run<'a> {
    Literals(u32, u32, &'a [u32]),
    Copy(u32, u32, u32, u32),
}

/// Walks the runs of `argb` with `histo_bits` tiles, handing each to `f`.
fn walk(argb: &[u32], width: u32, histo_bits: u32, mut f: impl FnMut(Run<'_>)) {
    let (left, up) = plane_codes(width);
    let xsize = width as usize;
    let tile = 1u32 << histo_bits;
    let (mut x, mut y) = (0u32, 0u32);
    let mut i = 0;
    while i < argb.len() {
        let (n, run) = next_run(argb, xsize, i, left, up);
        let mut rest = n;
        while rest > 0 {
            let end = (x + rest as u32).min(width).min((x | (tile - 1)) + 1);
            let chunk = (end - x) as usize;
            f(Run::Literals(x, y, &argb[i..i + chunk]));
            i += chunk;
            rest -= chunk;
            x = end;
            if x == width {
                x = 0;
                y += 1;
            }
        }
        if let Some((code, len)) = run {
            f(Run::Copy(x, y, code, len));
            i += len as usize;
            x += len;
            while x >= width {
                x -= width;
                y += 1;
            }
        }
    }
}

/// Fills the `histo_bits` tile histograms of `argb`, as `get_histo_image_symbols` does
/// from references.
pub(crate) fn count(argb: &[u32], width: u32, histo_bits: u32, histos: &mut [Histogram]) {
    let histo_xsize = sub_sample_size(width, histo_bits);
    let index = |x: u32, y: u32| ((y >> histo_bits) * histo_xsize + (x >> histo_bits)) as usize;
    walk(argb, width, histo_bits, |run| match run {
        Run::Literals(x, y, pixels) => {
            let h = &mut histos[index(x, y)];
            let green = &mut h.literal[..256];
            for &p in pixels {
                h.alpha[(p >> 24) as usize] += 1;
                h.red[((p >> 16) & 0xff) as usize] += 1;
                green[((p >> 8) & 0xff) as usize] += 1;
                h.blue[(p & 0xff) as usize] += 1;
            }
        }
        Run::Copy(x, y, code, len) => {
            histos[index(x, y)].add_single(&PixOrCopy::copy(code, len), None)
        }
    });
}

/// Writes the codes of `argb`, as `store_image_to_bit_mask` does from references.
pub(crate) fn write(
    bw: &mut BitWriter,
    argb: &[u32],
    width: u32,
    histo_bits: u32,
    histogram_symbols: &[u32],
    huffman_codes: &[HuffmanTreeCode],
) {
    let histo_xsize = sub_sample_size(width, histo_bits);
    let histogram_ix = |x: u32, y: u32| {
        let tile = ((y >> histo_bits) * histo_xsize + (x >> histo_bits)) as usize;
        ((histogram_symbols[tile] >> 8) & 0xffff) as usize
    };
    // A literal's four codes and lengths as one word each, in order green, red, blue, alpha.
    let literal_tables: Vec<[[u32; 256]; 4]> = huffman_codes
        .chunks_exact(5)
        .map(|c| {
            let mut t = [[0u32; 256]; 4];
            for (k, code) in c[..4].iter().enumerate() {
                for s in 0..256 {
                    t[k][s] = u32::from(code.codes[s]) | u32::from(code.code_lengths[s]) << 16;
                }
            }
            t
        })
        .collect();
    let mut w = std::mem::take(bw);
    walk(argb, width, histo_bits, |run| match run {
        Run::Literals(x, y, pixels) => {
            let t = &literal_tables[histogram_ix(x, y)];
            for &p in pixels {
                let tg = t[0][((p >> 8) & 0xff) as usize];
                let tr = t[1][((p >> 16) & 0xff) as usize];
                let tb = t[2][(p & 0xff) as usize];
                let ta = t[3][(p >> 24) as usize];
                let (gl, rl) = (tg >> 16, tr >> 16);
                w.put_bits((tg & 0xffff) | (tr & 0xffff) << gl, gl + rl);
                let (bl, al) = (tb >> 16, ta >> 16);
                w.put_bits((tb & 0xffff) | (ta & 0xffff) << bl, bl + al);
            }
        }
        Run::Copy(x, y, dist_code, len) => {
            let ix = histogram_ix(x, y);
            let codes = &huffman_codes[5 * ix..5 * ix + 5];
            let (code, n_bits, bits) = prefix_encode(len);
            let c = &codes[0];
            let depth = u32::from(c.code_lengths[256 + code as usize]);
            let symbol = u32::from(c.codes[256 + code as usize]);
            w.put_bits((bits << depth) | symbol, depth + n_bits);
            let (code, n_bits, bits) = prefix_encode(dist_code);
            w.put_bits(
                u32::from(codes[4].codes[code as usize]),
                u32::from(codes[4].code_lengths[code as usize]),
            );
            w.put_bits(bits, n_bits);
        }
    });
    *bw = w;
}
