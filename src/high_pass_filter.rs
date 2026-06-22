use aec3::audio_processing::high_pass_filter::HighPassFilter;

use crate::common::{i32_from_usize, valid_channels, valid_sample_rate};
use crate::{META_AEC3_OK, MetaAec3HighPassConfig, MetaAec3HighPassStats};

pub(crate) struct HighPassHandle {
    config: MetaAec3HighPassConfig,
    filter: HighPassFilter,
    channels: usize,
    planar: Vec<Vec<f32>>,
    total_processed_samples: u64,
}

impl HighPassHandle {
    pub(crate) fn new(config: MetaAec3HighPassConfig) -> Result<Self, i32> {
        let sample_rate_hz = valid_sample_rate(config.sample_rate_hz)?;
        let channels = valid_channels(config.channels, 8)?;
        let frame_capacity = sample_rate_hz / 100;
        Ok(Self {
            config,
            filter: HighPassFilter::new(sample_rate_hz as i32, channels),
            channels,
            planar: vec![vec![0.0; frame_capacity]; channels],
            total_processed_samples: 0,
        })
    }

    pub(crate) fn config(&self) -> MetaAec3HighPassConfig {
        self.config
    }

    pub(crate) fn reset(&mut self) {
        self.filter.reset();
    }

    pub(crate) fn reconfigure(&mut self, config: MetaAec3HighPassConfig) -> Result<(), i32> {
        *self = Self::new(config)?;
        Ok(())
    }

    pub(crate) fn process(
        &mut self,
        audio: &mut [f32],
        stats: Option<&mut MetaAec3HighPassStats>,
    ) -> Result<i32, i32> {
        if audio.is_empty() || !audio.len().is_multiple_of(self.channels) {
            return Err(crate::META_AEC3_INVALID_ARGUMENT);
        }
        let frames = audio.len() / self.channels;
        if self.planar[0].len() < frames {
            for channel in &mut self.planar {
                channel.resize(frames, 0.0);
            }
        }
        for channel in 0..self.channels {
            for frame in 0..frames {
                self.planar[channel][frame] = audio[frame * self.channels + channel];
            }
        }
        let channels = &mut self.planar;
        for channel in channels.iter_mut() {
            channel.truncate(frames);
        }
        self.filter.process(channels);
        for channel in 0..self.channels {
            for frame in 0..frames {
                audio[frame * self.channels + channel] = self.planar[channel][frame];
            }
        }
        self.total_processed_samples = self
            .total_processed_samples
            .saturating_add(audio.len() as u64);

        if let Some(stats) = stats {
            let mut sum_squares = 0.0;
            let mut peak = 0.0f32;
            for &sample in audio.iter() {
                sum_squares += sample * sample;
                peak = peak.max(sample.abs());
            }
            *stats = MetaAec3HighPassStats {
                struct_size: std::mem::size_of::<MetaAec3HighPassStats>() as i32,
                sample_rate_hz: self.config.sample_rate_hz,
                channels: self.config.channels,
                processed_samples: i32_from_usize(audio.len()),
                total_processed_samples: self.total_processed_samples,
                output_rms: (sum_squares / audio.len() as f32).sqrt(),
                output_peak: peak,
            };
        }
        Ok(META_AEC3_OK)
    }

    pub(crate) fn samples_per_10ms(&self) -> i32 {
        i32_from_usize(self.config.sample_rate_hz as usize / 100 * self.channels)
    }
}
