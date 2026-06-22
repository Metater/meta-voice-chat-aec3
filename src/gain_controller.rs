use aec3::audio_processing::agc2::cpu_features::get_available_cpu_features;
use aec3::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use aec3::audio_processing::agc2::vad_wrapper::VoiceActivityDetectorWrapper;
use aec3::audio_processing::gain_controller2::{GainController2, GainController2Config};

use crate::common::{
    AudioFrameIo, bool_from_ffi, checked_frame_batch_length, i32_from_usize, valid_channels,
    valid_sample_rate,
};
use crate::{META_AEC3_INVALID_ARGUMENT, META_AEC3_OK, MetaAec3Agc2Config, MetaAec3Agc2Stats};

pub(crate) struct GainControllerHandle {
    config: MetaAec3Agc2Config,
    controller: GainController2,
    io: AudioFrameIo,
    vad: VoiceActivityDetectorWrapper,
    last_recommended_input_volume: i32,
    total_processed_samples: u64,
}

impl GainControllerHandle {
    pub(crate) fn new(config: MetaAec3Agc2Config) -> Result<Self, i32> {
        let sample_rate_hz = valid_sample_rate(config.sample_rate_hz)?;
        let channels = valid_channels(config.channels, 8)?;
        let (controller_config, input_volume_config) = Self::make_configs(config)?;
        if !GainController2::validate(&controller_config) {
            return Err(crate::META_AEC3_INVALID_CONFIG);
        }

        Ok(Self {
            config,
            controller: GainController2::new(
                controller_config,
                input_volume_config,
                sample_rate_hz,
                channels,
                bool_from_ffi(config.use_internal_vad),
            ),
            io: AudioFrameIo::new(sample_rate_hz, channels),
            vad: VoiceActivityDetectorWrapper::new(get_available_cpu_features(), sample_rate_hz),
            last_recommended_input_volume: -1,
            total_processed_samples: 0,
        })
    }

    fn make_configs(
        raw: MetaAec3Agc2Config,
    ) -> Result<(GainController2Config, InputVolumeControllerConfig), i32> {
        if !raw.fixed_gain_db.is_finite()
            || !raw.adaptive_headroom_db.is_finite()
            || !raw.adaptive_max_gain_db.is_finite()
            || !raw.adaptive_initial_gain_db.is_finite()
            || !raw.adaptive_max_gain_change_db_per_second.is_finite()
            || !raw.adaptive_max_output_noise_level_dbfs.is_finite()
            || !(0.0..=1.0).contains(&raw.ivc_clipped_ratio_threshold)
            || !(0.0..=1.0).contains(&raw.ivc_speech_probability_threshold)
            || !(0.0..=1.0).contains(&raw.ivc_speech_ratio_threshold)
            || !(0..=255).contains(&raw.ivc_min_input_volume)
            || !(0..=255).contains(&raw.ivc_clipped_level_min)
            || raw.ivc_clipped_level_step <= 0
            || raw.ivc_clipped_wait_frames < 0
            || raw.ivc_update_input_volume_wait_frames < 0
        {
            return Err(crate::META_AEC3_INVALID_CONFIG);
        }

        let mut config = GainController2Config::default();
        config.fixed_digital.gain_db = raw.fixed_gain_db;
        config.adaptive_digital.enabled = bool_from_ffi(raw.enable_adaptive_digital);
        config.adaptive_digital.headroom_db = raw.adaptive_headroom_db;
        config.adaptive_digital.max_gain_db = raw.adaptive_max_gain_db;
        config.adaptive_digital.initial_gain_db = raw.adaptive_initial_gain_db;
        config.adaptive_digital.max_gain_change_db_per_second =
            raw.adaptive_max_gain_change_db_per_second;
        config.adaptive_digital.max_output_noise_level_dbfs =
            raw.adaptive_max_output_noise_level_dbfs;
        config.input_volume_controller.enabled = bool_from_ffi(raw.enable_input_volume_controller);

        let input_volume_config = InputVolumeControllerConfig {
            min_input_volume: raw.ivc_min_input_volume,
            clipped_level_min: raw.ivc_clipped_level_min,
            clipped_level_step: raw.ivc_clipped_level_step,
            clipped_ratio_threshold: raw.ivc_clipped_ratio_threshold,
            clipped_wait_frames: raw.ivc_clipped_wait_frames,
            enable_clipping_predictor: bool_from_ffi(raw.ivc_enable_clipping_predictor),
            target_range_max_dbfs: raw.ivc_target_range_max_dbfs,
            target_range_experimental_max_dbfs: raw.ivc_target_range_experimental_max_dbfs,
            target_range_min_dbfs: raw.ivc_target_range_min_dbfs,
            update_input_volume_wait_frames: raw.ivc_update_input_volume_wait_frames,
            speech_probability_threshold: raw.ivc_speech_probability_threshold,
            speech_ratio_threshold: raw.ivc_speech_ratio_threshold,
        };
        Ok((config, input_volume_config))
    }

    pub(crate) fn config(&self) -> MetaAec3Agc2Config {
        self.config
    }

    pub(crate) fn reset(&mut self) -> Result<(), i32> {
        let config = self.config;
        *self = Self::new(config)?;
        Ok(())
    }

    pub(crate) fn reconfigure(&mut self, config: MetaAec3Agc2Config) -> Result<(), i32> {
        *self = Self::new(config)?;
        Ok(())
    }

    pub(crate) fn set_fixed_gain_db(&mut self, gain_db: f32) -> Result<(), i32> {
        if !gain_db.is_finite() || !(0.0..50.0).contains(&gain_db) {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        self.config.fixed_gain_db = gain_db;
        self.controller.set_fixed_gain_db(gain_db);
        Ok(())
    }

    pub(crate) fn set_capture_output_used(&mut self, used: bool) {
        self.config.capture_output_used = i32::from(used);
        self.controller.set_capture_output_used(used);
    }

    pub(crate) fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        applied_input_volume: i32,
        input_volume_changed: bool,
        stats: Option<&mut MetaAec3Agc2Stats>,
    ) -> Result<i32, i32> {
        if !(0..=255).contains(&applied_input_volume) {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        let samples_per_frame = self.io.samples_per_frame();
        let batches =
            checked_frame_batch_length(input, self.config.channels as usize, self.io.frames())?;
        let needed = batches * samples_per_frame;
        if output.len() < needed {
            return Err(crate::META_AEC3_BUFFER_TOO_SMALL);
        }

        let mut max_voice_probability = 0.0f32;
        let mut output_rms = 0.0f32;
        let mut output_peak = 0.0f32;
        self.last_recommended_input_volume = -1;
        self.controller
            .set_capture_output_used(bool_from_ffi(self.config.capture_output_used));
        for frame_index in 0..batches {
            let offset = frame_index * samples_per_frame;
            self.io.load(&input[offset..offset + samples_per_frame])?;
            self.controller
                .analyze(applied_input_volume, self.io.buffer());
            self.controller
                .process(input_volume_changed, self.io.buffer_mut());
            max_voice_probability =
                max_voice_probability.max(self.io.voice_probability(&mut self.vad));
            let (rms, peak) = self.io.audio_levels();
            output_rms = output_rms.max(rms);
            output_peak = output_peak.max(peak);
            self.io
                .export(&mut output[offset..offset + samples_per_frame])?;
            self.last_recommended_input_volume = self
                .controller
                .recommended_input_volume()
                .unwrap_or(self.last_recommended_input_volume);
        }
        self.total_processed_samples = self.total_processed_samples.saturating_add(needed as u64);

        if let Some(stats) = stats {
            *stats = MetaAec3Agc2Stats {
                struct_size: std::mem::size_of::<MetaAec3Agc2Stats>() as i32,
                sample_rate_hz: self.config.sample_rate_hz,
                channels: self.config.channels,
                processed_samples: i32_from_usize(needed),
                total_processed_samples: self.total_processed_samples,
                applied_input_volume,
                recommended_input_volume: self.last_recommended_input_volume,
                voice_probability: max_voice_probability,
                output_rms,
                output_peak,
            };
        }
        Ok(META_AEC3_OK)
    }

    pub(crate) fn samples_per_10ms(&self) -> i32 {
        i32_from_usize(self.io.samples_per_frame())
    }
}
