use std::mem::align_of;
use std::panic::{AssertUnwindSafe, catch_unwind};

use aec3::audio_processing::audio_buffer::AudioBuffer;

use crate::{
    META_AEC3_INVALID_ARGUMENT, META_AEC3_INVALID_CONFIG, META_AEC3_NULL_POINTER, META_AEC3_PANIC,
};

pub(crate) const FLOAT_S16_SCALE: f32 = 32_768.0;

pub(crate) fn ffi_status<F>(operation: F) -> i32
where
    F: FnOnce() -> Result<i32, i32>,
{
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => META_AEC3_PANIC,
    }
}

pub(crate) fn ffi_create<T, F>(operation: F) -> *mut T
where
    F: FnOnce() -> Result<T, i32>,
{
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(instance)) => Box::into_raw(Box::new(instance)),
        _ => std::ptr::null_mut(),
    }
}

pub(crate) fn is_aligned<T>(pointer: *const T) -> bool {
    (pointer as usize).is_multiple_of(align_of::<T>())
}

pub(crate) unsafe fn checked_ref<'a, T>(pointer: *const T) -> Result<&'a T, i32> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(META_AEC3_NULL_POINTER);
    }
    // SAFETY: The caller has supplied a non-null pointer with the required alignment.
    Ok(unsafe { &*pointer })
}

pub(crate) unsafe fn checked_mut<'a, T>(pointer: *mut T) -> Result<&'a mut T, i32> {
    if pointer.is_null() || !is_aligned(pointer) {
        return Err(META_AEC3_NULL_POINTER);
    }
    // SAFETY: The caller has supplied a non-null pointer with the required alignment.
    Ok(unsafe { &mut *pointer })
}

pub(crate) unsafe fn checked_f32_slice<'a>(
    pointer: *const f32,
    length: i32,
) -> Result<&'a [f32], i32> {
    if pointer.is_null() || length < 0 || !is_aligned(pointer) {
        return Err(META_AEC3_NULL_POINTER);
    }
    // SAFETY: The caller guarantees that `pointer` references `length` readable f32 values.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length as usize) })
}

pub(crate) unsafe fn checked_f32_slice_mut<'a>(
    pointer: *mut f32,
    length: i32,
) -> Result<&'a mut [f32], i32> {
    if pointer.is_null() || length < 0 || !is_aligned(pointer) {
        return Err(META_AEC3_NULL_POINTER);
    }
    // SAFETY: The caller guarantees that `pointer` references `length` writable f32 values.
    Ok(unsafe { std::slice::from_raw_parts_mut(pointer, length as usize) })
}

pub(crate) fn valid_sample_rate(sample_rate_hz: i32) -> Result<usize, i32> {
    match sample_rate_hz {
        16_000 | 32_000 | 48_000 => Ok(sample_rate_hz as usize),
        _ => Err(META_AEC3_INVALID_CONFIG),
    }
}

pub(crate) fn valid_channels(channels: i32, maximum: usize) -> Result<usize, i32> {
    match channels {
        1.. if (channels as usize) <= maximum => Ok(channels as usize),
        _ => Err(META_AEC3_INVALID_CONFIG),
    }
}

pub(crate) fn checked_frame_length(
    samples: &[f32],
    channels: usize,
    frames: usize,
) -> Result<(), i32> {
    if samples.len() == frames.saturating_mul(channels) {
        Ok(())
    } else {
        Err(META_AEC3_INVALID_ARGUMENT)
    }
}

pub(crate) fn checked_frame_batch_length(
    samples: &[f32],
    channels: usize,
    frames: usize,
) -> Result<usize, i32> {
    let samples_per_frame = channels.saturating_mul(frames);
    if samples_per_frame == 0
        || samples.is_empty()
        || !samples.len().is_multiple_of(samples_per_frame)
    {
        return Err(META_AEC3_INVALID_ARGUMENT);
    }
    Ok(samples.len() / samples_per_frame)
}

/// A reusable, 10-ms `AudioBuffer` bridge. Public FFI samples are normalized
/// floating point (-1.0..=1.0); WebRTC processing modules use float-s16 scale.
pub(crate) struct AudioFrameIo {
    channels: usize,
    frames: usize,
    buffer: AudioBuffer,
}

impl AudioFrameIo {
    pub(crate) fn new(sample_rate_hz: usize, channels: usize) -> Self {
        let frames = sample_rate_hz / 100;
        Self {
            channels,
            frames,
            buffer: AudioBuffer::from_sample_rates(
                sample_rate_hz,
                channels,
                sample_rate_hz,
                channels,
                sample_rate_hz,
            ),
        }
    }

    pub(crate) fn frames(&self) -> usize {
        self.frames
    }

    pub(crate) fn samples_per_frame(&self) -> usize {
        self.frames * self.channels
    }

    pub(crate) fn load(&mut self, interleaved: &[f32]) -> Result<(), i32> {
        checked_frame_length(interleaved, self.channels, self.frames)?;
        for channel in 0..self.channels {
            let output = self.buffer.channel_mut(channel);
            for (frame, sample) in output.iter_mut().enumerate() {
                *sample = interleaved[frame * self.channels + channel] * FLOAT_S16_SCALE;
            }
        }
        Ok(())
    }

    pub(crate) fn export(&self, interleaved: &mut [f32]) -> Result<(), i32> {
        checked_frame_length(interleaved, self.channels, self.frames)?;
        for channel in 0..self.channels {
            let input = self.buffer.channel(channel);
            for (frame, &sample) in input.iter().enumerate() {
                interleaved[frame * self.channels + channel] =
                    (sample / FLOAT_S16_SCALE).clamp(-1.0, 1.0);
            }
        }
        Ok(())
    }

    pub(crate) fn buffer(&self) -> &AudioBuffer {
        &self.buffer
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut AudioBuffer {
        &mut self.buffer
    }

    pub(crate) fn audio_levels(&self) -> (f32, f32) {
        let mut sum_squares = 0.0;
        let mut peak = 0.0f32;
        let mut count = 0usize;
        for channel in 0..self.channels {
            for &sample in self.buffer.channel(channel) {
                let unit = (sample / FLOAT_S16_SCALE).clamp(-1.0, 1.0);
                sum_squares += unit * unit;
                peak = peak.max(unit.abs());
                count += 1;
            }
        }
        let rms = if count != 0 {
            (sum_squares / count as f32).sqrt()
        } else {
            0.0
        };
        (rms, peak)
    }

    pub(crate) fn voice_probability(
        &self,
        vad: &mut aec3::audio_processing::agc2::vad_wrapper::VoiceActivityDetectorWrapper,
    ) -> f32 {
        let channels = (0..self.channels)
            .map(|channel| self.buffer.channel(channel))
            .collect::<Vec<_>>();
        vad.analyze(&channels)
    }
}

pub(crate) fn optional_output<'a>(
    pointer: *mut f32,
    length: i32,
    required: usize,
) -> Result<Option<&'a mut [f32]>, i32> {
    if pointer.is_null() {
        return Ok(None);
    }
    if length < required as i32 || !is_aligned(pointer) {
        return Err(META_AEC3_INVALID_ARGUMENT);
    }
    // SAFETY: The caller guarantees a writable f32 buffer of `length` elements.
    Ok(Some(unsafe {
        std::slice::from_raw_parts_mut(pointer, length as usize)
    }))
}

pub(crate) fn bool_from_ffi(value: i32) -> bool {
    value != 0
}

pub(crate) fn i32_from_usize(value: usize) -> i32 {
    value.min(i32::MAX as usize) as i32
}
