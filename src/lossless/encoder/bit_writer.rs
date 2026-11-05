//! libwebp's `VP8LBitWriter`: bits are packed LSB first into little-endian 32-bit words.

#[derive(Clone, Default)]
pub(crate) struct BitWriter {
    buf: Vec<u8>,
    bits: u64,
    used: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            bits: 0,
            used: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn put_bits(&mut self, bits: u32, n_bits: u32) {
        if n_bits > 0 {
            if self.used >= 32 {
                self.buf
                    .extend_from_slice(&(self.bits as u32).to_le_bytes());
                self.bits >>= 32;
                self.used -= 32;
            }
            self.bits |= u64::from(bits) << self.used;
            self.used += n_bits;
        }
    }

    /// Bytes written so far, counting a partial byte as a whole one.
    /// Puts up to 60 bits at once. Where they fit the accumulator, its whole bytes go out
    /// with one 8-byte store and no branch per bit group; otherwise two [`put_bits`](Self::put_bits).
    #[inline(always)]
    pub(crate) fn put_bits64(&mut self, bits: u64, n_bits: u32) {
        if self.used + n_bits < 64 {
            self.bits |= bits << self.used;
            self.used += n_bits;
            let len = self.buf.len();
            self.buf.extend_from_slice(&self.bits.to_le_bytes());
            let bytes = self.used >> 3;
            self.buf.truncate(len + bytes as usize);
            self.bits >>= 8 * bytes;
            self.used &= 7;
        } else {
            let lo = n_bits.min(32);
            self.put_bits((bits & ((1u64 << lo) - 1)) as u32, lo);
            self.put_bits((bits >> lo) as u32, n_bits - lo);
        }
    }

    pub(crate) fn num_bytes(&self) -> usize {
        self.buf.len() + ((self.used + 7) >> 3) as usize
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        while self.used > 0 {
            self.buf.push(self.bits as u8);
            self.bits >>= 8;
            self.used = self.used.saturating_sub(8);
        }
        self.buf
    }
}
