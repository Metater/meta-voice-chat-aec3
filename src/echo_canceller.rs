use aec3::api::config::EchoCanceller3Config;
use aec3::api::control::{EchoControl, Metrics};
use aec3::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use aec3::audio_processing::agc2::cpu_features::get_available_cpu_features;
use aec3::audio_processing::agc2::vad_wrapper::VoiceActivityDetectorWrapper;

use crate::common::{
    AudioFrameIo, bool_from_ffi, checked_frame_batch_length, i32_from_usize, valid_channels,
    valid_sample_rate,
};
use crate::{META_AEC3_INVALID_ARGUMENT, META_AEC3_OK, MetaAec3AecConfig, MetaAec3AecStats};

pub(crate) struct EchoCancellerHandle {
    config: MetaAec3AecConfig,
    echo: EchoCanceller3,
    render_io: AudioFrameIo,
    capture_io: AudioFrameIo,
    linear_io: Option<AudioFrameIo>,
    vad: VoiceActivityDetectorWrapper,
    last_metrics: Metrics,
    total_render_samples: u64,
    total_capture_samples: u64,
}

impl EchoCancellerHandle {
    pub(crate) fn new(config: MetaAec3AecConfig) -> Result<Self, i32> {
        let sample_rate_hz = valid_sample_rate(config.sample_rate_hz)?;
        let render_channels = valid_channels(config.render_channels, 8)?;
        let capture_channels = valid_channels(config.capture_channels, 2)?;
        if !(0.0..=1.0).contains(&config.vad_threshold) || !config.vad_threshold.is_finite() {
            return Err(crate::META_AEC3_INVALID_CONFIG);
        }
        if config.initial_delay_ms < 0 {
            return Err(crate::META_AEC3_INVALID_CONFIG);
        }

        let echo_config = Self::make_echo_config(config, render_channels, capture_channels)?;
        let mut echo = EchoCanceller3::new(
            echo_config,
            sample_rate_hz as i32,
            render_channels,
            capture_channels,
        );
        echo.set_audio_buffer_delay(config.initial_delay_ms);

        Ok(Self {
            config,
            echo,
            render_io: AudioFrameIo::new(sample_rate_hz, render_channels),
            capture_io: AudioFrameIo::new(sample_rate_hz, capture_channels),
            // AEC3 exposes its linear-filter tap as a 16-kHz low-band frame.
            linear_io: bool_from_ffi(config.export_linear_aec_output)
                .then(|| AudioFrameIo::new(16_000, capture_channels)),
            vad: VoiceActivityDetectorWrapper::new(get_available_cpu_features(), sample_rate_hz),
            last_metrics: Metrics::default(),
            total_render_samples: 0,
            total_capture_samples: 0,
        })
    }

    fn non_negative(value: i32) -> Result<usize, i32> {
        usize::try_from(value).map_err(|_| crate::META_AEC3_INVALID_CONFIG)
    }

    fn make_echo_config(
        raw: MetaAec3AecConfig,
        render_channels: usize,
        capture_channels: usize,
    ) -> Result<EchoCanceller3Config, i32> {
        let mut config = EchoCanceller3::create_default_config(render_channels, capture_channels);

        config.buffering.excess_render_detection_interval_blocks =
            Self::non_negative(raw.excess_render_detection_interval_blocks)?;
        config.buffering.max_allowed_excess_render_blocks =
            Self::non_negative(raw.max_allowed_excess_render_blocks)?;
        config.delay.default_delay = Self::non_negative(raw.delay_default_blocks)?;
        config.delay.down_sampling_factor = Self::non_negative(raw.delay_down_sampling_factor)?;
        config.delay.num_filters = Self::non_negative(raw.delay_num_filters)?;
        config.delay.delay_headroom_samples = Self::non_negative(raw.delay_headroom_samples)?;
        config.delay.hysteresis_limit_blocks =
            Self::non_negative(raw.delay_hysteresis_limit_blocks)?;
        config.delay.fixed_capture_delay_samples =
            Self::non_negative(raw.fixed_capture_delay_samples)?;
        config.delay.delay_estimate_smoothing = raw.delay_estimate_smoothing;
        config.delay.delay_candidate_detection_threshold = raw.delay_candidate_detection_threshold;
        config.delay.delay_selection_thresholds.initial = raw.delay_selection_threshold_initial;
        config.delay.delay_selection_thresholds.converged = raw.delay_selection_threshold_converged;
        config.delay.use_external_delay_estimator = bool_from_ffi(raw.delay_use_external_estimator);
        config.delay.log_warning_on_delay_changes = bool_from_ffi(raw.delay_log_warnings);
        config.delay.render_alignment_mixing.downmix = bool_from_ffi(raw.render_alignment_downmix);
        config.delay.render_alignment_mixing.adaptive_selection =
            bool_from_ffi(raw.render_alignment_adaptive_selection);
        config
            .delay
            .render_alignment_mixing
            .activity_power_threshold = raw.render_alignment_activity_power_threshold;
        config
            .delay
            .render_alignment_mixing
            .prefer_first_two_channels =
            bool_from_ffi(raw.render_alignment_prefer_first_two_channels);
        config.delay.capture_alignment_mixing.downmix =
            bool_from_ffi(raw.capture_alignment_downmix);
        config.delay.capture_alignment_mixing.adaptive_selection =
            bool_from_ffi(raw.capture_alignment_adaptive_selection);
        config
            .delay
            .capture_alignment_mixing
            .activity_power_threshold = raw.capture_alignment_activity_power_threshold;
        config
            .delay
            .capture_alignment_mixing
            .prefer_first_two_channels =
            bool_from_ffi(raw.capture_alignment_prefer_first_two_channels);

        config.filter.main.length_blocks = Self::non_negative(raw.main_filter_length_blocks)?;
        config.filter.main.leakage_converged = raw.main_filter_leakage_converged;
        config.filter.main.leakage_diverged = raw.main_filter_leakage_diverged;
        config.filter.main.error_floor = raw.main_filter_error_floor;
        config.filter.main.error_ceil = raw.main_filter_error_ceil;
        config.filter.main.noise_gate = raw.main_filter_noise_gate;
        config.filter.main_initial.length_blocks =
            Self::non_negative(raw.main_initial_filter_length_blocks)?;
        config.filter.main_initial.leakage_converged = raw.main_initial_filter_leakage_converged;
        config.filter.main_initial.leakage_diverged = raw.main_initial_filter_leakage_diverged;
        config.filter.main_initial.error_floor = raw.main_initial_filter_error_floor;
        config.filter.main_initial.error_ceil = raw.main_initial_filter_error_ceil;
        config.filter.main_initial.noise_gate = raw.main_initial_filter_noise_gate;
        config.filter.shadow.length_blocks = Self::non_negative(raw.shadow_filter_length_blocks)?;
        config.filter.shadow_initial.length_blocks =
            Self::non_negative(raw.shadow_initial_filter_length_blocks)?;
        config.filter.shadow.rate = raw.shadow_filter_rate;
        config.filter.shadow.noise_gate = raw.shadow_filter_noise_gate;
        config.filter.shadow_initial.rate = raw.shadow_initial_filter_rate;
        config.filter.shadow_initial.noise_gate = raw.shadow_initial_filter_noise_gate;
        config.filter.config_change_duration_blocks =
            Self::non_negative(raw.filter_config_change_duration_blocks)?;
        config.filter.initial_state_seconds = raw.filter_initial_state_seconds;
        config.filter.shadow_reset_hangover_blocks =
            Self::non_negative(raw.shadow_reset_hangover_blocks)?;
        config.filter.use_shadow_reset_hangover = bool_from_ffi(raw.use_shadow_reset_hangover);
        config.filter.conservative_initial_phase = bool_from_ffi(raw.conservative_initial_phase);
        config.filter.enable_shadow_filter_output_usage =
            bool_from_ffi(raw.enable_shadow_filter_output_usage);
        config.filter.use_linear_filter = bool_from_ffi(raw.use_linear_filter);
        config.filter.export_linear_aec_output = bool_from_ffi(raw.export_linear_aec_output);

        config.erle.min = raw.erle_min;
        config.erle.max_l = raw.erle_max_low;
        config.erle.max_h = raw.erle_max_high;
        config.erle.onset_detection = bool_from_ffi(raw.erle_onset_detection);
        config.erle.num_sections = Self::non_negative(raw.erle_num_sections)?;
        config.erle.clamp_quality_estimate_to_zero = bool_from_ffi(raw.erle_clamp_to_zero);
        config.erle.clamp_quality_estimate_to_one = bool_from_ffi(raw.erle_clamp_to_one);

        config.ep_strength.default_gain = raw.ep_strength_default_gain;
        config.ep_strength.default_len = raw.ep_strength_default_len;
        config.ep_strength.echo_can_saturate = bool_from_ffi(raw.ep_strength_echo_can_saturate);
        config.ep_strength.bounded_erl = bool_from_ffi(raw.ep_strength_bounded_erl);
        config.echo_audibility.low_render_limit = raw.echo_audibility_low_render_limit;
        config.echo_audibility.normal_render_limit = raw.echo_audibility_normal_render_limit;
        config.echo_audibility.floor_power = raw.echo_audibility_floor_power;
        config.echo_audibility.audibility_threshold_lf = raw.echo_audibility_threshold_lf;
        config.echo_audibility.audibility_threshold_mf = raw.echo_audibility_threshold_mf;
        config.echo_audibility.audibility_threshold_hf = raw.echo_audibility_threshold_hf;
        config.echo_audibility.use_stationarity_properties =
            bool_from_ffi(raw.echo_audibility_use_stationarity_properties);
        config.echo_audibility.use_stationarity_properties_at_init =
            bool_from_ffi(raw.echo_audibility_use_stationarity_properties_at_init);
        config.render_levels.active_render_limit = raw.render_levels_active_render_limit;
        config.render_levels.poor_excitation_render_limit =
            raw.render_levels_poor_excitation_render_limit;
        config.render_levels.poor_excitation_render_limit_ds8 =
            raw.render_levels_poor_excitation_render_limit_ds8;
        config.render_levels.render_power_gain_db = raw.render_levels_render_power_gain_db;
        config.echo_removal_control.has_clock_drift = bool_from_ffi(raw.has_clock_drift);
        config.echo_removal_control.linear_and_stable_echo_path =
            bool_from_ffi(raw.linear_and_stable_echo_path);
        config.transparent_mode.enabled = bool_from_ffi(raw.enable_transparent_mode);
        config.transparent_mode.use_hmm = bool_from_ffi(raw.transparent_mode_use_hmm);

        config.echo_model.noise_floor_hold = Self::non_negative(raw.echo_model_noise_floor_hold)?;
        config.echo_model.min_noise_floor_power = raw.echo_model_min_noise_floor_power;
        config.echo_model.stationary_gate_slope = raw.echo_model_stationary_gate_slope;
        config.echo_model.noise_gate_power = raw.echo_model_noise_gate_power;
        config.echo_model.noise_gate_slope = raw.echo_model_noise_gate_slope;
        config.echo_model.render_pre_window_size =
            Self::non_negative(raw.echo_model_render_pre_window_size)?;
        config.echo_model.render_post_window_size =
            Self::non_negative(raw.echo_model_render_post_window_size)?;

        config.suppressor.nearend_average_blocks =
            Self::non_negative(raw.suppressor_nearend_average_blocks)?;
        config.suppressor.normal_tuning.mask_lf.enr_transparent =
            raw.suppressor_normal_lf_enr_transparent;
        config.suppressor.normal_tuning.mask_lf.enr_suppress =
            raw.suppressor_normal_lf_enr_suppress;
        config.suppressor.normal_tuning.mask_lf.emr_transparent =
            raw.suppressor_normal_lf_emr_transparent;
        config.suppressor.normal_tuning.mask_hf.enr_transparent =
            raw.suppressor_normal_hf_enr_transparent;
        config.suppressor.normal_tuning.mask_hf.enr_suppress =
            raw.suppressor_normal_hf_enr_suppress;
        config.suppressor.normal_tuning.mask_hf.emr_transparent =
            raw.suppressor_normal_hf_emr_transparent;
        config.suppressor.normal_tuning.max_inc_factor = raw.suppressor_normal_max_inc_factor;
        config.suppressor.normal_tuning.max_dec_factor_lf = raw.suppressor_normal_max_dec_factor_lf;
        config.suppressor.nearend_tuning.mask_lf.enr_transparent =
            raw.suppressor_nearend_lf_enr_transparent;
        config.suppressor.nearend_tuning.mask_lf.enr_suppress =
            raw.suppressor_nearend_lf_enr_suppress;
        config.suppressor.nearend_tuning.mask_lf.emr_transparent =
            raw.suppressor_nearend_lf_emr_transparent;
        config.suppressor.nearend_tuning.mask_hf.enr_transparent =
            raw.suppressor_nearend_hf_enr_transparent;
        config.suppressor.nearend_tuning.mask_hf.enr_suppress =
            raw.suppressor_nearend_hf_enr_suppress;
        config.suppressor.nearend_tuning.mask_hf.emr_transparent =
            raw.suppressor_nearend_hf_emr_transparent;
        config.suppressor.nearend_tuning.max_inc_factor = raw.suppressor_nearend_max_inc_factor;
        config.suppressor.nearend_tuning.max_dec_factor_lf =
            raw.suppressor_nearend_max_dec_factor_lf;
        config.suppressor.dominant_nearend_detection.enr_threshold =
            raw.dominant_nearend_enr_threshold;
        config
            .suppressor
            .dominant_nearend_detection
            .enr_exit_threshold = raw.dominant_nearend_enr_exit_threshold;
        config.suppressor.dominant_nearend_detection.snr_threshold =
            raw.dominant_nearend_snr_threshold;
        config.suppressor.dominant_nearend_detection.hold_duration =
            Self::non_negative(raw.dominant_nearend_hold_duration)?;
        config
            .suppressor
            .dominant_nearend_detection
            .trigger_threshold = Self::non_negative(raw.dominant_nearend_trigger_threshold)?;
        config
            .suppressor
            .dominant_nearend_detection
            .use_during_initial_phase =
            bool_from_ffi(raw.dominant_nearend_use_during_initial_phase);
        config
            .suppressor
            .subband_nearend_detection
            .nearend_average_blocks = Self::non_negative(raw.subband_nearend_average_blocks)?;
        config.suppressor.subband_nearend_detection.subband1.low =
            Self::non_negative(raw.subband1_low)?;
        config.suppressor.subband_nearend_detection.subband1.high =
            Self::non_negative(raw.subband1_high)?;
        config.suppressor.subband_nearend_detection.subband2.low =
            Self::non_negative(raw.subband2_low)?;
        config.suppressor.subband_nearend_detection.subband2.high =
            Self::non_negative(raw.subband2_high)?;
        config
            .suppressor
            .subband_nearend_detection
            .nearend_threshold = raw.subband_nearend_threshold;
        config.suppressor.subband_nearend_detection.snr_threshold = raw.subband_snr_threshold;
        config.suppressor.use_subband_nearend_detection =
            bool_from_ffi(raw.enable_subband_nearend_detection);
        config.suppressor.high_bands_suppression.enr_threshold = raw.high_bands_enr_threshold;
        config
            .suppressor
            .high_bands_suppression
            .max_gain_during_echo = raw.high_bands_max_gain_during_echo;
        config
            .suppressor
            .high_bands_suppression
            .anti_howling_activation_threshold = raw.high_bands_anti_howling_activation_threshold;
        config.suppressor.high_bands_suppression.anti_howling_gain =
            raw.high_bands_anti_howling_gain;
        config.suppressor.floor_first_increase = raw.suppressor_floor_first_increase;

        if !config.validate() {
            return Err(crate::META_AEC3_INVALID_CONFIG);
        }
        Ok(config)
    }

    pub(crate) fn config(&self) -> MetaAec3AecConfig {
        self.config
    }

    pub(crate) fn reset(&mut self) -> Result<(), i32> {
        let config = self.config;
        *self = Self::new(config)?;
        Ok(())
    }

    pub(crate) fn reconfigure(&mut self, config: MetaAec3AecConfig) -> Result<(), i32> {
        *self = Self::new(config)?;
        Ok(())
    }

    pub(crate) fn set_stream_delay_ms(&mut self, delay_ms: i32) -> Result<(), i32> {
        if delay_ms < 0 {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        self.config.initial_delay_ms = delay_ms;
        self.echo.set_audio_buffer_delay(delay_ms);
        Ok(())
    }

    pub(crate) fn set_echo_leakage_status(&mut self, detected: bool) {
        self.echo.update_echo_leakage_status(detected);
    }

    pub(crate) fn process_render(&mut self, input: &[f32]) -> Result<i32, i32> {
        let samples_per_frame = self.render_io.samples_per_frame();
        let batches = checked_frame_batch_length(
            input,
            self.config.render_channels as usize,
            self.render_io.frames(),
        )?;
        for frame_index in 0..batches {
            let offset = frame_index * samples_per_frame;
            self.render_io
                .load(&input[offset..offset + samples_per_frame])?;
            let render = self.render_io.buffer_mut();
            render.split_into_frequency_bands();
            self.echo.analyze_render(render);
        }
        self.total_render_samples = self.total_render_samples.saturating_add(input.len() as u64);
        Ok(META_AEC3_OK)
    }

    pub(crate) fn process_capture(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        mut linear_output: Option<&mut [f32]>,
        input_volume_changed: bool,
        stats: Option<&mut MetaAec3AecStats>,
    ) -> Result<i32, i32> {
        let samples_per_frame = self.capture_io.samples_per_frame();
        let batches = checked_frame_batch_length(
            input,
            self.config.capture_channels as usize,
            self.capture_io.frames(),
        )?;
        let needed = batches * samples_per_frame;
        if output.len() < needed {
            return Err(crate::META_AEC3_BUFFER_TOO_SMALL);
        }
        if linear_output.is_some() && self.linear_io.is_none() {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        let linear_samples_per_frame = self
            .linear_io
            .as_ref()
            .map(AudioFrameIo::samples_per_frame)
            .unwrap_or(0);
        let linear_needed = batches * linear_samples_per_frame;
        if let Some(linear) = linear_output.as_ref()
            && linear.len() < linear_needed
        {
            return Err(crate::META_AEC3_BUFFER_TOO_SMALL);
        }

        let mut max_vad_probability = 0.0f32;
        let mut output_rms = 0.0f32;
        let mut output_peak = 0.0f32;
        for frame_index in 0..batches {
            let offset = frame_index * samples_per_frame;
            self.capture_io
                .load(&input[offset..offset + samples_per_frame])?;
            {
                let capture = self.capture_io.buffer_mut();
                self.echo.analyze_capture(capture);
                capture.split_into_frequency_bands();
                if let Some(linear_io) = self.linear_io.as_mut() {
                    self.echo.process_capture_with_linear_output(
                        capture,
                        linear_io.buffer_mut(),
                        input_volume_changed,
                    );
                } else {
                    self.echo.process_capture(capture, input_volume_changed);
                }
                capture.merge_frequency_bands();
            }
            self.last_metrics = self.echo.metrics();
            max_vad_probability =
                max_vad_probability.max(self.capture_io.voice_probability(&mut self.vad));
            let (rms, peak) = self.capture_io.audio_levels();
            output_rms = output_rms.max(rms);
            output_peak = output_peak.max(peak);
            self.capture_io
                .export(&mut output[offset..offset + samples_per_frame])?;
            if let (Some(linear), Some(linear_io)) =
                (linear_output.as_deref_mut(), self.linear_io.as_ref())
            {
                let linear_offset = frame_index * linear_samples_per_frame;
                linear_io
                    .export(&mut linear[linear_offset..linear_offset + linear_samples_per_frame])?;
            }
        }
        self.total_capture_samples = self.total_capture_samples.saturating_add(needed as u64);

        if let Some(stats) = stats {
            *stats = MetaAec3AecStats {
                struct_size: std::mem::size_of::<MetaAec3AecStats>() as i32,
                sample_rate_hz: self.config.sample_rate_hz,
                render_channels: self.config.render_channels,
                capture_channels: self.config.capture_channels,
                processed_samples: i32_from_usize(needed),
                output_samples: i32_from_usize(needed),
                total_render_samples: self.total_render_samples,
                total_capture_samples: self.total_capture_samples,
                voice_probability: max_vad_probability,
                voice_detected: i32::from(max_vad_probability >= self.config.vad_threshold),
                output_rms,
                output_peak,
                echo_return_loss: self.last_metrics.echo_return_loss,
                echo_return_loss_enhancement: self.last_metrics.echo_return_loss_enhancement,
                delay_ms: self.last_metrics.delay_ms,
                render_jitter_min: self.last_metrics.render_jitter_min,
                render_jitter_max: self.last_metrics.render_jitter_max,
                capture_jitter_min: self.last_metrics.capture_jitter_min,
                capture_jitter_max: self.last_metrics.capture_jitter_max,
            };
        }
        Ok(META_AEC3_OK)
    }

    pub(crate) fn metrics(&self) -> Metrics {
        self.last_metrics
    }

    pub(crate) fn render_samples_per_10ms(&self) -> i32 {
        i32_from_usize(self.render_io.samples_per_frame())
    }

    pub(crate) fn capture_samples_per_10ms(&self) -> i32 {
        i32_from_usize(self.capture_io.samples_per_frame())
    }
}
