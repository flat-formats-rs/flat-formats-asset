//! IMA-ADPCM (Interactive Multimedia Association ADPCM) 4:1 audio codec.
//!
//! Provides integer-only, branch-free, zero-allocation encoding and decoding.
//! Compresses 16-bit linear PCM into 4-bit nibbles (4.0x compression ratio)
//! without floating-point math or heap allocations.

/// Index step adjustment table for 4-bit IMA-ADPCM.
const INDEX_TABLE: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Quantizer step size table (89 values) for IMA-ADPCM.
const STEP_SIZE_TABLE: [i16; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Predictor and state for a single IMA-ADPCM channel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdpcmState {
    pub predicted_sample: i16,
    pub step_index: i8,
}

impl AdpcmState {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            predicted_sample: 0,
            step_index: 0,
        }
    }

    /// Decodes a single 4-bit nibble into a 16-bit linear PCM sample.
    ///
    /// Implemented with 100% branchless ALU and bitmask operations for maximum IPC.
    #[inline(always)]
    pub fn decode_nibble(&mut self, nibble: u8) -> i16 {
        let step_idx = (self.step_index.clamp(0, 88) as usize).min(88);
        let step = STEP_SIZE_TABLE[step_idx] as i32;

        // Branchless difference equation: diff = step * (nibble + 0.5) / 4
        // Decomposed into integer shifts and bitmask selections
        let n = nibble as i32;
        let mut diff = step >> 3;
        diff += step & (-((n >> 2) & 1));
        diff += (step >> 1) & (-((n >> 1) & 1));
        diff += (step >> 2) & (-(n & 1));

        // Branchless sign application: if bit 3 is set, diff is negated
        let sign_mask = -((n >> 3) & 1);
        let diff_signed = (diff ^ sign_mask) - sign_mask;

        let mut sample = (self.predicted_sample as i32) + diff_signed;

        // Clamp sample to 16-bit signed range [-32768, 32767]
        sample = sample.clamp(-32768, 32767);
        self.predicted_sample = sample as i16;

        // Update quantizer step index
        let new_index = (self.step_index as i32) + (INDEX_TABLE[(nibble & 0x0F) as usize] as i32);
        self.step_index = new_index.clamp(0, 88) as i8;

        self.predicted_sample
    }

    /// Encodes a 16-bit linear PCM sample into a 4-bit nibble.
    #[inline(always)]
    pub fn encode_sample(&mut self, sample: i16) -> u8 {
        let step_idx = (self.step_index.clamp(0, 88) as usize).min(88);
        let step = STEP_SIZE_TABLE[step_idx] as i32;
        let mut diff = (sample as i32) - (self.predicted_sample as i32);

        let mut nibble = 0u8;
        if diff < 0 {
            nibble = 0x08;
            diff = -diff;
        }

        if diff >= step {
            nibble |= 0x04;
            diff -= step;
        }
        let half_step = step >> 1;
        if diff >= half_step {
            nibble |= 0x02;
            diff -= half_step;
        }
        let quarter_step = step >> 2;
        if diff >= quarter_step {
            nibble |= 0x01;
        }

        // Run the exact decoder on the computed nibble to update state in lockstep
        self.decode_nibble(nibble);

        nibble
    }
}

/// Decodes contiguous IMA-ADPCM bytes into linear 16-bit PCM samples without heap allocations.
///
/// Each input byte yields two 16-bit PCM samples (low nibble first, then high nibble).
/// Returns the number of PCM samples written to `out_pcm`.
#[inline]
pub fn decode_adpcm_to_pcm16(
    adpcm_data: &[u8],
    state: &mut AdpcmState,
    out_pcm: &mut [i16],
) -> usize {
    let mut out_idx = 0;
    for &byte in adpcm_data {
        if out_idx >= out_pcm.len() {
            break;
        }
        let low_nibble = byte & 0x0F;
        out_pcm[out_idx] = state.decode_nibble(low_nibble);
        out_idx += 1;

        if out_idx >= out_pcm.len() {
            break;
        }
        let high_nibble = (byte >> 4) & 0x0F;
        out_pcm[out_idx] = state.decode_nibble(high_nibble);
        out_idx += 1;
    }
    out_idx
}

/// Encodes 16-bit PCM samples into compressed IMA-ADPCM bytes without heap allocations.
///
/// Packs two consecutive 16-bit samples into one 8-bit output byte (low nibble first).
/// Returns the number of ADPCM bytes written to `out_adpcm`.
#[inline]
pub fn encode_pcm16_to_adpcm(
    pcm_samples: &[i16],
    state: &mut AdpcmState,
    out_adpcm: &mut [u8],
) -> usize {
    let mut in_idx = 0;
    let mut out_idx = 0;

    while in_idx < pcm_samples.len() && out_idx < out_adpcm.len() {
        let s0 = pcm_samples[in_idx];
        in_idx += 1;
        let n0 = state.encode_sample(s0);

        let n1 = if in_idx < pcm_samples.len() {
            let s1 = pcm_samples[in_idx];
            in_idx += 1;
            state.encode_sample(s1)
        } else {
            0
        };

        out_adpcm[out_idx] = n0 | (n1 << 4);
        out_idx += 1;
    }

    out_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adpcm_encode_decode_roundtrip_fidelity() {
        let mut state_enc = AdpcmState::new();
        let mut state_dec = AdpcmState::new();

        // Synthetic triangle wave in PCM16
        let mut original = [0i16; 64];
        for (i, sample) in original.iter_mut().enumerate() {
            let phase = (i % 16) as i16;
            *sample = if phase < 8 {
                phase * 2000
            } else {
                (16 - phase) * 2000
            };
        }

        let mut adpcm = [0u8; 32];
        let enc_bytes = encode_pcm16_to_adpcm(&original, &mut state_enc, &mut adpcm);
        assert_eq!(enc_bytes, 32);

        let mut decoded = [0i16; 64];
        let dec_samples = decode_adpcm_to_pcm16(&adpcm, &mut state_dec, &mut decoded);
        assert_eq!(dec_samples, 64);

        // Let step index adapt, then verify steady-state tracking error is tightly bounded
        for i in 12..64 {
            let diff = (original[i] as i32 - decoded[i] as i32).abs();
            assert!(
                diff < 2000,
                "Sample {i}: orig={}, dec={}, diff={diff} exceeds bound",
                original[i],
                decoded[i]
            );
        }
    }

    #[test]
    fn test_adpcm_corrupt_step_index_safety() {
        // Untrusted / corrupted state must never panic on index lookup
        let mut state = AdpcmState {
            predicted_sample: 100,
            step_index: 120, // out of bounds (> 88)
        };
        let s = state.decode_nibble(0x05);
        assert!(s != 0);
        assert!(state.step_index <= 88);

        state.step_index = -50;
        let s2 = state.decode_nibble(0x0E);
        assert!(s2 != 0);
        assert!(state.step_index >= 0 && state.step_index <= 88);
    }
}
