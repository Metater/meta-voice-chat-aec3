#![allow(clippy::missing_safety_doc)] // The C ABI uses one shared pointer-safety contract.

//! C ABI wrapper around the low-level AEC3-RS processors.
//!
//! Input and output audio is interleaved, normalized `f32` PCM in the
//! `[-1.0, 1.0]` range. Every handle is intentionally single-threaded. Create
//! one handle for each independent audio path and call its functions in order:
//! high-pass filter -> AEC3 -> AGC2. WebRTC noise suppression is deliberately
//! not part of this wrapper so an external RNNoise/RNN noise processor can sit
//! between AEC3 and AGC2.

mod common;
mod echo_canceller;
mod gain_controller;
mod high_pass_filter;

use std::mem::size_of;

use common::{
    checked_f32_slice, checked_f32_slice_mut, checked_mut, checked_ref, ffi_create, ffi_status,
    optional_output,
};
use echo_canceller::EchoCancellerHandle;
use gain_controller::GainControllerHandle;
use high_pass_filter::HighPassHandle;

/// Operation completed successfully.
pub const META_AEC3_OK: i32 = 0;
/// A required pointer was null or not suitably aligned.
pub const META_AEC3_NULL_POINTER: i32 = -1;
/// A creation or reconfiguration option is unsupported or internally invalid.
pub const META_AEC3_INVALID_CONFIG: i32 = -2;
/// An operation argument is invalid for the handle's fixed audio format.
pub const META_AEC3_INVALID_ARGUMENT: i32 = -3;
/// A supplied output buffer cannot hold the requested result.
pub const META_AEC3_BUFFER_TOO_SMALL: i32 = -4;
/// Rust caught an internal panic before it could cross the C ABI boundary.
pub const META_AEC3_PANIC: i32 = -99;

/// Creation options for a high-pass filter.
///
/// The underlying WebRTC high-pass filter supports 16, 32, and 48 kHz. Audio
/// passed to `meta_aec3_high_pass_process` may contain any positive frame size.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MetaAec3HighPassConfig {
    pub sample_rate_hz: i32,
    pub channels: i32,
}

impl Default for MetaAec3HighPassConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            channels: 1,
        }
    }
}

/// Optional analysis returned by `meta_aec3_high_pass_process`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetaAec3HighPassStats {
    pub struct_size: i32,
    pub sample_rate_hz: i32,
    pub channels: i32,
    pub processed_samples: i32,
    pub total_processed_samples: u64,
    pub output_rms: f32,
    pub output_peak: f32,
}

/// Creation options for AEC3. Initialize with
/// `meta_aec3_aec_default_config` and alter only the controls you need.
///
/// Render supports 1-8 interleaved channels; capture supports mono or stereo.
/// Both sides use the same 16, 32, or 48 kHz rate. Processing accepts one or
/// more complete 10-ms frames per call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MetaAec3AecConfig {
    pub sample_rate_hz: i32,
    pub render_channels: i32,
    pub capture_channels: i32,
    pub initial_delay_ms: i32,
    pub fixed_capture_delay_samples: i32,

    pub excess_render_detection_interval_blocks: i32,
    pub max_allowed_excess_render_blocks: i32,
    pub delay_default_blocks: i32,
    pub delay_down_sampling_factor: i32,
    pub delay_num_filters: i32,
    pub delay_headroom_samples: i32,
    pub delay_hysteresis_limit_blocks: i32,
    pub delay_estimate_smoothing: f32,
    pub delay_candidate_detection_threshold: f32,
    pub delay_selection_threshold_initial: i32,
    pub delay_selection_threshold_converged: i32,
    pub delay_use_external_estimator: i32,
    pub delay_log_warnings: i32,

    pub render_alignment_downmix: i32,
    pub render_alignment_adaptive_selection: i32,
    pub render_alignment_activity_power_threshold: f32,
    pub render_alignment_prefer_first_two_channels: i32,
    pub capture_alignment_downmix: i32,
    pub capture_alignment_adaptive_selection: i32,
    pub capture_alignment_activity_power_threshold: f32,
    pub capture_alignment_prefer_first_two_channels: i32,

    pub main_filter_length_blocks: i32,
    pub main_filter_leakage_converged: f32,
    pub main_filter_leakage_diverged: f32,
    pub main_filter_error_floor: f32,
    pub main_filter_error_ceil: f32,
    pub main_filter_noise_gate: f32,
    pub main_initial_filter_length_blocks: i32,
    pub main_initial_filter_leakage_converged: f32,
    pub main_initial_filter_leakage_diverged: f32,
    pub main_initial_filter_error_floor: f32,
    pub main_initial_filter_error_ceil: f32,
    pub main_initial_filter_noise_gate: f32,
    pub shadow_filter_length_blocks: i32,
    pub shadow_initial_filter_length_blocks: i32,
    pub shadow_filter_rate: f32,
    pub shadow_filter_noise_gate: f32,
    pub shadow_initial_filter_rate: f32,
    pub shadow_initial_filter_noise_gate: f32,
    pub filter_config_change_duration_blocks: i32,
    pub filter_initial_state_seconds: f32,
    pub shadow_reset_hangover_blocks: i32,
    pub use_shadow_reset_hangover: i32,
    pub conservative_initial_phase: i32,
    pub enable_shadow_filter_output_usage: i32,
    pub use_linear_filter: i32,
    pub export_linear_aec_output: i32,

    pub erle_min: f32,
    pub erle_max_low: f32,
    pub erle_max_high: f32,
    pub erle_onset_detection: i32,
    pub erle_num_sections: i32,
    pub erle_clamp_to_zero: i32,
    pub erle_clamp_to_one: i32,
    pub ep_strength_default_gain: f32,
    pub ep_strength_default_len: f32,
    pub ep_strength_echo_can_saturate: i32,
    pub ep_strength_bounded_erl: i32,
    pub echo_audibility_low_render_limit: f32,
    pub echo_audibility_normal_render_limit: f32,
    pub echo_audibility_floor_power: f32,
    pub echo_audibility_threshold_lf: f32,
    pub echo_audibility_threshold_mf: f32,
    pub echo_audibility_threshold_hf: f32,
    pub echo_audibility_use_stationarity_properties: i32,
    pub echo_audibility_use_stationarity_properties_at_init: i32,
    pub render_levels_active_render_limit: f32,
    pub render_levels_poor_excitation_render_limit: f32,
    pub render_levels_poor_excitation_render_limit_ds8: f32,
    pub render_levels_render_power_gain_db: f32,
    pub has_clock_drift: i32,
    pub linear_and_stable_echo_path: i32,
    pub enable_transparent_mode: i32,
    pub transparent_mode_use_hmm: i32,

    pub echo_model_noise_floor_hold: i32,
    pub echo_model_min_noise_floor_power: f32,
    pub echo_model_stationary_gate_slope: f32,
    pub echo_model_noise_gate_power: f32,
    pub echo_model_noise_gate_slope: f32,
    pub echo_model_render_pre_window_size: i32,
    pub echo_model_render_post_window_size: i32,

    pub suppressor_nearend_average_blocks: i32,
    pub suppressor_normal_lf_enr_transparent: f32,
    pub suppressor_normal_lf_enr_suppress: f32,
    pub suppressor_normal_lf_emr_transparent: f32,
    pub suppressor_normal_hf_enr_transparent: f32,
    pub suppressor_normal_hf_enr_suppress: f32,
    pub suppressor_normal_hf_emr_transparent: f32,
    pub suppressor_normal_max_inc_factor: f32,
    pub suppressor_normal_max_dec_factor_lf: f32,
    pub suppressor_nearend_lf_enr_transparent: f32,
    pub suppressor_nearend_lf_enr_suppress: f32,
    pub suppressor_nearend_lf_emr_transparent: f32,
    pub suppressor_nearend_hf_enr_transparent: f32,
    pub suppressor_nearend_hf_enr_suppress: f32,
    pub suppressor_nearend_hf_emr_transparent: f32,
    pub suppressor_nearend_max_inc_factor: f32,
    pub suppressor_nearend_max_dec_factor_lf: f32,
    pub dominant_nearend_enr_threshold: f32,
    pub dominant_nearend_enr_exit_threshold: f32,
    pub dominant_nearend_snr_threshold: f32,
    pub dominant_nearend_hold_duration: i32,
    pub dominant_nearend_trigger_threshold: i32,
    pub dominant_nearend_use_during_initial_phase: i32,
    pub subband_nearend_average_blocks: i32,
    pub subband1_low: i32,
    pub subband1_high: i32,
    pub subband2_low: i32,
    pub subband2_high: i32,
    pub subband_nearend_threshold: f32,
    pub subband_snr_threshold: f32,
    pub enable_subband_nearend_detection: i32,
    pub high_bands_enr_threshold: f32,
    pub high_bands_max_gain_during_echo: f32,
    pub high_bands_anti_howling_activation_threshold: f32,
    pub high_bands_anti_howling_gain: f32,
    pub suppressor_floor_first_increase: f32,

    /// Only affects the exported VAD flag in `MetaAec3AecStats`.
    pub vad_threshold: f32,
}

impl Default for MetaAec3AecConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            render_channels: 1,
            capture_channels: 1,
            initial_delay_ms: 0,
            fixed_capture_delay_samples: 0,
            excess_render_detection_interval_blocks: 250,
            max_allowed_excess_render_blocks: 8,
            delay_default_blocks: 5,
            delay_down_sampling_factor: 4,
            delay_num_filters: 5,
            delay_headroom_samples: 32,
            delay_hysteresis_limit_blocks: 1,
            delay_estimate_smoothing: 0.7,
            delay_candidate_detection_threshold: 0.2,
            delay_selection_threshold_initial: 5,
            delay_selection_threshold_converged: 20,
            delay_use_external_estimator: 0,
            delay_log_warnings: 0,
            render_alignment_downmix: 0,
            render_alignment_adaptive_selection: 1,
            render_alignment_activity_power_threshold: 10_000.0,
            render_alignment_prefer_first_two_channels: 1,
            capture_alignment_downmix: 0,
            capture_alignment_adaptive_selection: 1,
            capture_alignment_activity_power_threshold: 10_000.0,
            capture_alignment_prefer_first_two_channels: 0,
            main_filter_length_blocks: 13,
            main_filter_leakage_converged: 0.00005,
            main_filter_leakage_diverged: 0.05,
            main_filter_error_floor: 0.001,
            main_filter_error_ceil: 2.0,
            main_filter_noise_gate: 20_075_344.0,
            main_initial_filter_length_blocks: 12,
            main_initial_filter_leakage_converged: 0.005,
            main_initial_filter_leakage_diverged: 0.5,
            main_initial_filter_error_floor: 0.001,
            main_initial_filter_error_ceil: 2.0,
            main_initial_filter_noise_gate: 20_075_344.0,
            shadow_filter_length_blocks: 13,
            shadow_initial_filter_length_blocks: 12,
            shadow_filter_rate: 0.7,
            shadow_filter_noise_gate: 20_075_344.0,
            shadow_initial_filter_rate: 0.9,
            shadow_initial_filter_noise_gate: 20_075_344.0,
            filter_config_change_duration_blocks: 250,
            filter_initial_state_seconds: 2.5,
            shadow_reset_hangover_blocks: 25,
            use_shadow_reset_hangover: 1,
            conservative_initial_phase: 0,
            enable_shadow_filter_output_usage: 1,
            use_linear_filter: 1,
            export_linear_aec_output: 0,
            erle_min: 1.0,
            erle_max_low: 4.0,
            erle_max_high: 1.5,
            erle_onset_detection: 1,
            erle_num_sections: 1,
            erle_clamp_to_zero: 1,
            erle_clamp_to_one: 1,
            ep_strength_default_gain: 1.0,
            ep_strength_default_len: 0.83,
            ep_strength_echo_can_saturate: 1,
            ep_strength_bounded_erl: 0,
            echo_audibility_low_render_limit: 256.0,
            echo_audibility_normal_render_limit: 64.0,
            echo_audibility_floor_power: 128.0,
            echo_audibility_threshold_lf: 10.0,
            echo_audibility_threshold_mf: 10.0,
            echo_audibility_threshold_hf: 10.0,
            echo_audibility_use_stationarity_properties: 0,
            echo_audibility_use_stationarity_properties_at_init: 0,
            render_levels_active_render_limit: 100.0,
            render_levels_poor_excitation_render_limit: 150.0,
            render_levels_poor_excitation_render_limit_ds8: 20.0,
            render_levels_render_power_gain_db: 0.0,
            has_clock_drift: 0,
            linear_and_stable_echo_path: 0,
            enable_transparent_mode: 1,
            transparent_mode_use_hmm: 0,
            echo_model_noise_floor_hold: 50,
            echo_model_min_noise_floor_power: 1_638_400.0,
            echo_model_stationary_gate_slope: 10.0,
            echo_model_noise_gate_power: 27_509.42,
            echo_model_noise_gate_slope: 0.3,
            echo_model_render_pre_window_size: 1,
            echo_model_render_post_window_size: 1,
            suppressor_nearend_average_blocks: 4,
            suppressor_normal_lf_enr_transparent: 0.3,
            suppressor_normal_lf_enr_suppress: 0.4,
            suppressor_normal_lf_emr_transparent: 0.3,
            suppressor_normal_hf_enr_transparent: 0.07,
            suppressor_normal_hf_enr_suppress: 0.1,
            suppressor_normal_hf_emr_transparent: 0.3,
            suppressor_normal_max_inc_factor: 2.0,
            suppressor_normal_max_dec_factor_lf: 0.25,
            suppressor_nearend_lf_enr_transparent: 1.09,
            suppressor_nearend_lf_enr_suppress: 1.1,
            suppressor_nearend_lf_emr_transparent: 0.3,
            suppressor_nearend_hf_enr_transparent: 0.1,
            suppressor_nearend_hf_enr_suppress: 0.3,
            suppressor_nearend_hf_emr_transparent: 0.3,
            suppressor_nearend_max_inc_factor: 2.0,
            suppressor_nearend_max_dec_factor_lf: 0.25,
            dominant_nearend_enr_threshold: 0.25,
            dominant_nearend_enr_exit_threshold: 10.0,
            dominant_nearend_snr_threshold: 30.0,
            dominant_nearend_hold_duration: 50,
            dominant_nearend_trigger_threshold: 12,
            dominant_nearend_use_during_initial_phase: 1,
            subband_nearend_average_blocks: 1,
            subband1_low: 1,
            subband1_high: 1,
            subband2_low: 1,
            subband2_high: 1,
            subband_nearend_threshold: 1.0,
            subband_snr_threshold: 1.0,
            enable_subband_nearend_detection: 0,
            high_bands_enr_threshold: 1.0,
            high_bands_max_gain_during_echo: 1.0,
            high_bands_anti_howling_activation_threshold: 400.0,
            high_bands_anti_howling_gain: 1.0,
            suppressor_floor_first_increase: 0.00001,
            vad_threshold: 0.5,
        }
    }
}

/// Per-capture AEC3 telemetry. Pass a null `stats` pointer to skip it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetaAec3AecStats {
    pub struct_size: i32,
    pub sample_rate_hz: i32,
    pub render_channels: i32,
    pub capture_channels: i32,
    pub processed_samples: i32,
    pub output_samples: i32,
    pub total_render_samples: u64,
    pub total_capture_samples: u64,
    pub voice_probability: f32,
    pub voice_detected: i32,
    pub output_rms: f32,
    pub output_peak: f32,
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
    pub render_jitter_min: i32,
    pub render_jitter_max: i32,
    pub capture_jitter_min: i32,
    pub capture_jitter_max: i32,
}

/// Lightweight AEC3 metrics that can be queried between capture calls.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetaAec3AecMetrics {
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
    pub render_jitter_min: i32,
    pub render_jitter_max: i32,
    pub capture_jitter_min: i32,
    pub capture_jitter_max: i32,
}

/// Creation options for WebRTC AGC2. Initialize with
/// `meta_aec3_agc2_default_config` before changing values.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MetaAec3Agc2Config {
    pub sample_rate_hz: i32,
    pub channels: i32,
    pub fixed_gain_db: f32,
    pub enable_adaptive_digital: i32,
    pub adaptive_headroom_db: f32,
    pub adaptive_max_gain_db: f32,
    pub adaptive_initial_gain_db: f32,
    pub adaptive_max_gain_change_db_per_second: f32,
    pub adaptive_max_output_noise_level_dbfs: f32,
    pub enable_input_volume_controller: i32,
    pub use_internal_vad: i32,
    pub capture_output_used: i32,

    pub ivc_min_input_volume: i32,
    pub ivc_clipped_level_min: i32,
    pub ivc_clipped_level_step: i32,
    pub ivc_clipped_ratio_threshold: f32,
    pub ivc_clipped_wait_frames: i32,
    pub ivc_enable_clipping_predictor: i32,
    pub ivc_target_range_max_dbfs: i32,
    pub ivc_target_range_experimental_max_dbfs: i32,
    pub ivc_target_range_min_dbfs: i32,
    pub ivc_update_input_volume_wait_frames: i32,
    pub ivc_speech_probability_threshold: f32,
    pub ivc_speech_ratio_threshold: f32,
}

impl Default for MetaAec3Agc2Config {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            channels: 1,
            fixed_gain_db: 0.0,
            enable_adaptive_digital: 1,
            adaptive_headroom_db: 5.0,
            adaptive_max_gain_db: 50.0,
            adaptive_initial_gain_db: 15.0,
            adaptive_max_gain_change_db_per_second: 6.0,
            adaptive_max_output_noise_level_dbfs: -50.0,
            enable_input_volume_controller: 0,
            use_internal_vad: 1,
            capture_output_used: 1,
            ivc_min_input_volume: 20,
            ivc_clipped_level_min: 70,
            ivc_clipped_level_step: 15,
            ivc_clipped_ratio_threshold: 0.1,
            ivc_clipped_wait_frames: 300,
            ivc_enable_clipping_predictor: 1,
            ivc_target_range_max_dbfs: -30,
            ivc_target_range_experimental_max_dbfs: -12,
            ivc_target_range_min_dbfs: -50,
            ivc_update_input_volume_wait_frames: 100,
            ivc_speech_probability_threshold: 0.7,
            ivc_speech_ratio_threshold: 0.6,
        }
    }
}

/// Optional analysis returned by `meta_aec3_agc2_process`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MetaAec3Agc2Stats {
    pub struct_size: i32,
    pub sample_rate_hz: i32,
    pub channels: i32,
    pub processed_samples: i32,
    pub total_processed_samples: u64,
    pub applied_input_volume: i32,
    /// -1 when the AGC2 input-volume controller is disabled or has no update.
    pub recommended_input_volume: i32,
    pub voice_probability: f32,
    pub output_rms: f32,
    pub output_peak: f32,
}

/// Opaque high-pass filter state. One handle is not safe for concurrent use.
pub struct MetaAec3HighPass(HighPassHandle);
/// Opaque AEC3 state. Feed render before its corresponding capture frame.
pub struct MetaAec3Aec(EchoCancellerHandle);
/// Opaque AGC2 state. One handle is not safe for concurrent use.
pub struct MetaAec3Agc2(GainControllerHandle);

fn optional_stats<'a, T>(pointer: *mut T) -> Result<Option<&'a mut T>, i32> {
    if pointer.is_null() {
        Ok(None)
    } else {
        // SAFETY: The FFI caller promises a valid writable stats record.
        unsafe { checked_mut(pointer).map(Some) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_default_config(
    config: *mut MetaAec3HighPassConfig,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before dereference.
        let config = unsafe { checked_mut(config)? };
        *config = MetaAec3HighPassConfig::default();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_create(
    config: *const MetaAec3HighPassConfig,
) -> *mut MetaAec3HighPass {
    ffi_create(|| {
        let config = if config.is_null() {
            MetaAec3HighPassConfig::default()
        } else {
            // SAFETY: Checked before dereference.
            unsafe { *checked_ref(config)? }
        };
        HighPassHandle::new(config).map(MetaAec3HighPass)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_free(handle: *mut MetaAec3HighPass) {
    if !handle.is_null() {
        // SAFETY: Ownership was returned by `meta_aec3_high_pass_create`.
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_get_config(
    handle: *const MetaAec3HighPass,
    config: *mut MetaAec3HighPassConfig,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_ref(handle)? };
        let config = unsafe { checked_mut(config)? };
        *config = handle.0.config();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_reconfigure(
    handle: *mut MetaAec3HighPass,
    config: *const MetaAec3HighPassConfig,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_mut(handle)? };
        let config = unsafe { *checked_ref(config)? };
        handle.0.reconfigure(config)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_reset(handle: *mut MetaAec3HighPass) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        let handle = unsafe { checked_mut(handle)? };
        handle.0.reset();
        Ok(META_AEC3_OK)
    })
}

/// Processes interleaved audio in place. `stats` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_process(
    handle: *mut MetaAec3HighPass,
    audio: *mut f32,
    audio_length: i32,
    stats: *mut MetaAec3HighPassStats,
) -> i32 {
    ffi_status(|| {
        // SAFETY: The caller supplies a valid high-pass handle and writable audio buffer.
        let handle = unsafe { checked_mut(handle)? };
        let audio = unsafe { checked_f32_slice_mut(audio, audio_length)? };
        let stats = optional_stats(stats)?;
        handle.0.process(audio, stats)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_high_pass_samples_per_10ms(
    handle: *const MetaAec3HighPass,
) -> i32 {
    match (|| {
        // SAFETY: Checked before use.
        let handle = unsafe { checked_ref(handle)? };
        Ok::<i32, i32>(handle.0.samples_per_10ms())
    })() {
        Ok(samples) => samples,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_default_config(config: *mut MetaAec3AecConfig) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before dereference.
        let config = unsafe { checked_mut(config)? };
        *config = MetaAec3AecConfig::default();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_create(
    config: *const MetaAec3AecConfig,
) -> *mut MetaAec3Aec {
    ffi_create(|| {
        let config = if config.is_null() {
            MetaAec3AecConfig::default()
        } else {
            // SAFETY: Checked before dereference.
            unsafe { *checked_ref(config)? }
        };
        EchoCancellerHandle::new(config).map(MetaAec3Aec)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_free(handle: *mut MetaAec3Aec) {
    if !handle.is_null() {
        // SAFETY: Ownership was returned by `meta_aec3_aec_create`.
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_get_config(
    handle: *const MetaAec3Aec,
    config: *mut MetaAec3AecConfig,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_ref(handle)? };
        let config = unsafe { checked_mut(config)? };
        *config = handle.0.config();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_reconfigure(
    handle: *mut MetaAec3Aec,
    config: *const MetaAec3AecConfig,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_mut(handle)? };
        let config = unsafe { *checked_ref(config)? };
        handle.0.reconfigure(config)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_reset(handle: *mut MetaAec3Aec) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }.0.reset()?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_set_stream_delay_ms(
    handle: *mut MetaAec3Aec,
    delay_ms: i32,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }
            .0
            .set_stream_delay_ms(delay_ms)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_set_echo_leakage_status(
    handle: *mut MetaAec3Aec,
    leakage_detected: i32,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }
            .0
            .set_echo_leakage_status(leakage_detected != 0);
        Ok(META_AEC3_OK)
    })
}

/// Queues one or more 10-ms render frames. Render must be called before the
/// corresponding capture frame, but render and capture do not need to arrive
/// in a strict one-for-one pattern.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_process_render(
    handle: *mut MetaAec3Aec,
    render: *const f32,
    render_length: i32,
) -> i32 {
    ffi_status(|| {
        // SAFETY: The caller supplies a valid handle and readable input buffer.
        let handle = unsafe { checked_mut(handle)? };
        let render = unsafe { checked_f32_slice(render, render_length)? };
        handle.0.process_render(render)
    })
}

/// Processes one or more 10-ms capture frames. `linear_output` is optional;
/// it is valid only if `export_linear_aec_output` was enabled at creation. It
/// receives the native 16-kHz linear-filter tap (160 samples per capture
/// channel for each 10-ms frame). `stats` may be null. Input and output
/// buffers must not overlap.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn meta_aec3_aec_process_capture(
    handle: *mut MetaAec3Aec,
    capture: *const f32,
    capture_length: i32,
    output: *mut f32,
    output_length: i32,
    linear_output: *mut f32,
    linear_output_length: i32,
    input_volume_changed: i32,
    stats: *mut MetaAec3AecStats,
) -> i32 {
    ffi_status(|| {
        // SAFETY: The caller supplies valid non-overlapping input/output buffers.
        let handle = unsafe { checked_mut(handle)? };
        let capture = unsafe { checked_f32_slice(capture, capture_length)? };
        let output = unsafe { checked_f32_slice_mut(output, output_length)? };
        let linear_output = optional_output(linear_output, linear_output_length, 0)?;
        let stats = optional_stats(stats)?;
        handle.0.process_capture(
            capture,
            output,
            linear_output,
            input_volume_changed != 0,
            stats,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_get_metrics(
    handle: *const MetaAec3Aec,
    metrics: *mut MetaAec3AecMetrics,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_ref(handle)? };
        let metrics = unsafe { checked_mut(metrics)? };
        let current = handle.0.metrics();
        *metrics = MetaAec3AecMetrics {
            echo_return_loss: current.echo_return_loss,
            echo_return_loss_enhancement: current.echo_return_loss_enhancement,
            delay_ms: current.delay_ms,
            render_jitter_min: current.render_jitter_min,
            render_jitter_max: current.render_jitter_max,
            capture_jitter_min: current.capture_jitter_min,
            capture_jitter_max: current.capture_jitter_max,
        };
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_render_samples_per_10ms(handle: *const MetaAec3Aec) -> i32 {
    match (|| {
        // SAFETY: Checked before use.
        Ok::<i32, i32>(unsafe { checked_ref(handle)? }.0.render_samples_per_10ms())
    })() {
        Ok(samples) => samples,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_aec_capture_samples_per_10ms(handle: *const MetaAec3Aec) -> i32 {
    match (|| {
        // SAFETY: Checked before use.
        Ok::<i32, i32>(unsafe { checked_ref(handle)? }.0.capture_samples_per_10ms())
    })() {
        Ok(samples) => samples,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_default_config(config: *mut MetaAec3Agc2Config) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before dereference.
        let config = unsafe { checked_mut(config)? };
        *config = MetaAec3Agc2Config::default();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_create(
    config: *const MetaAec3Agc2Config,
) -> *mut MetaAec3Agc2 {
    ffi_create(|| {
        let config = if config.is_null() {
            MetaAec3Agc2Config::default()
        } else {
            // SAFETY: Checked before dereference.
            unsafe { *checked_ref(config)? }
        };
        GainControllerHandle::new(config).map(MetaAec3Agc2)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_free(handle: *mut MetaAec3Agc2) {
    if !handle.is_null() {
        // SAFETY: Ownership was returned by `meta_aec3_agc2_create`.
        unsafe { drop(Box::from_raw(handle)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_get_config(
    handle: *const MetaAec3Agc2,
    config: *mut MetaAec3Agc2Config,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_ref(handle)? };
        let config = unsafe { checked_mut(config)? };
        *config = handle.0.config();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_reconfigure(
    handle: *mut MetaAec3Agc2,
    config: *const MetaAec3Agc2Config,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Both pointers are checked before use.
        let handle = unsafe { checked_mut(handle)? };
        let config = unsafe { *checked_ref(config)? };
        handle.0.reconfigure(config)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_reset(handle: *mut MetaAec3Agc2) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }.0.reset()?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_set_fixed_gain_db(
    handle: *mut MetaAec3Agc2,
    gain_db: f32,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }
            .0
            .set_fixed_gain_db(gain_db)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_set_capture_output_used(
    handle: *mut MetaAec3Agc2,
    used: i32,
) -> i32 {
    ffi_status(|| {
        // SAFETY: Checked before use.
        unsafe { checked_mut(handle)? }
            .0
            .set_capture_output_used(used != 0);
        Ok(META_AEC3_OK)
    })
}

/// Processes one or more 10-ms frames. Input and output buffers must not
/// overlap. `stats` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_process(
    handle: *mut MetaAec3Agc2,
    input: *const f32,
    input_length: i32,
    output: *mut f32,
    output_length: i32,
    applied_input_volume: i32,
    input_volume_changed: i32,
    stats: *mut MetaAec3Agc2Stats,
) -> i32 {
    ffi_status(|| {
        // SAFETY: The caller supplies valid non-overlapping input/output buffers.
        let handle = unsafe { checked_mut(handle)? };
        let input = unsafe { checked_f32_slice(input, input_length)? };
        let output = unsafe { checked_f32_slice_mut(output, output_length)? };
        let stats = optional_stats(stats)?;
        handle.0.process(
            input,
            output,
            applied_input_volume,
            input_volume_changed != 0,
            stats,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_agc2_samples_per_10ms(handle: *const MetaAec3Agc2) -> i32 {
    match (|| {
        // SAFETY: Checked before use.
        Ok::<i32, i32>(unsafe { checked_ref(handle)? }.0.samples_per_10ms())
    })() {
        Ok(samples) => samples,
        Err(status) => status,
    }
}

/// Lets C/C# callers verify that their `struct_size` agrees with this binary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_sizeof_high_pass_stats() -> i32 {
    size_of::<MetaAec3HighPassStats>() as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_sizeof_aec_stats() -> i32 {
    size_of::<MetaAec3AecStats>() as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn meta_aec3_sizeof_agc2_stats() -> i32 {
    size_of::<MetaAec3Agc2Stats>() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_pass_handle_processes_in_place() {
        // SAFETY: All pointers below come from live Rust values with matching lengths.
        unsafe {
            let handle = meta_aec3_high_pass_create(std::ptr::null());
            assert!(!handle.is_null());
            // 40 ms of 48-kHz mono audio.
            let mut audio = vec![0.25f32; 1_920];
            let mut stats = MetaAec3HighPassStats::default();
            assert_eq!(
                meta_aec3_high_pass_process(
                    handle,
                    audio.as_mut_ptr(),
                    audio.len() as i32,
                    &mut stats
                ),
                META_AEC3_OK
            );
            assert_eq!(stats.processed_samples, 1_920);
            assert!(stats.output_peak.is_finite());
            meta_aec3_high_pass_free(handle);
        }
    }

    #[test]
    fn aec_accepts_eight_channel_render_and_exports_metrics() {
        // SAFETY: All pointers below come from live Rust values with matching lengths.
        unsafe {
            let config = MetaAec3AecConfig {
                render_channels: 8,
                export_linear_aec_output: 1,
                ..Default::default()
            };
            let handle = meta_aec3_aec_create(&config);
            assert!(!handle.is_null());

            // Two 10-ms frames arrive together (20 ms total).
            let render = vec![0.0f32; 480 * 8 * 2];
            assert_eq!(
                meta_aec3_aec_process_render(handle, render.as_ptr(), render.len() as i32),
                META_AEC3_OK
            );
            let capture = vec![0.0f32; 960];
            let mut output = vec![0.0f32; 960];
            let mut linear = vec![0.0f32; 320];
            let mut stats = MetaAec3AecStats::default();
            assert_eq!(
                meta_aec3_aec_process_capture(
                    handle,
                    capture.as_ptr(),
                    capture.len() as i32,
                    output.as_mut_ptr(),
                    output.len() as i32,
                    linear.as_mut_ptr(),
                    linear.len() as i32,
                    0,
                    &mut stats,
                ),
                META_AEC3_OK
            );
            assert_eq!(stats.processed_samples, 960);
            assert!(stats.voice_probability.is_finite());
            meta_aec3_aec_free(handle);
        }
    }

    #[test]
    fn agc2_processes_after_aec_output() {
        // SAFETY: All pointers below come from live Rust values with matching lengths.
        unsafe {
            let handle = meta_aec3_agc2_create(std::ptr::null());
            assert!(!handle.is_null());
            let input = vec![0.01f32; 960];
            let mut output = vec![0.0f32; 960];
            let mut stats = MetaAec3Agc2Stats::default();
            assert_eq!(
                meta_aec3_agc2_process(
                    handle,
                    input.as_ptr(),
                    input.len() as i32,
                    output.as_mut_ptr(),
                    output.len() as i32,
                    255,
                    0,
                    &mut stats,
                ),
                META_AEC3_OK
            );
            assert_eq!(stats.processed_samples, 960);
            assert!(stats.output_peak.is_finite());
            meta_aec3_agc2_free(handle);
        }
    }
}
