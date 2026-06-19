use std::mem::{align_of, size_of};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use aec3::api::control::{EchoControl, Metrics};
use aec3::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use aec3::audio_processing::agc2::cpu_features::get_available_cpu_features;
use aec3::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use aec3::audio_processing::agc2::limiter::Limiter;
use aec3::audio_processing::agc2::vad_wrapper::VoiceActivityDetectorWrapper;
use aec3::audio_processing::audio_buffer::AudioBuffer;
use aec3::audio_processing::gain_controller2::{GainController2, GainController2Config};
use aec3::audio_processing::high_pass_filter::HighPassFilter;
use aec3::audio_processing::ns::{NoiseSuppressor, NsConfig, SuppressionLevel};
use aec3::audio_processing::resampler::push_sinc_resampler::PushSincResampler;
use aec3::audio_processing::stream_config::StreamConfig;
use rustfft::{FftPlanner, num_complex::Complex32};

pub const META_AEC3_OK: i32 = 0;
pub const META_AEC3_NEEDS_RNNOISE: i32 = 1;
pub const META_AEC3_NULL_POINTER: i32 = -1;
pub const META_AEC3_INVALID_CONFIG: i32 = -2;
pub const META_AEC3_INVALID_ARGUMENT: i32 = -3;
pub const META_AEC3_BUFFER_TOO_SMALL: i32 = -4;
pub const META_AEC3_NO_PENDING_RNNOISE: i32 = -5;
pub const META_AEC3_PANIC: i32 = -99;

pub const META_AEC3_NS_NONE: i32 = 0;
pub const META_AEC3_NS_WEBRTC: i32 = 1;
pub const META_AEC3_NS_RNNOISE: i32 = 2;

pub const META_AEC3_NS_LEVEL_6_DB: i32 = 0;
pub const META_AEC3_NS_LEVEL_12_DB: i32 = 1;
pub const META_AEC3_NS_LEVEL_18_DB: i32 = 2;
pub const META_AEC3_NS_LEVEL_21_DB: i32 = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetaAec3Config {
    pub sample_rate_hz: i32,
    pub render_sample_rate_hz: i32,
    pub frame_size_ms: i32,
    pub capture_channels: i32,
    pub render_channels: i32,
    pub enable_high_pass_filter: i32,
    pub enable_aec3: i32,
    pub noise_suppression_mode: i32,
    pub noise_suppression_level: i32,
    pub enable_agc2: i32,
    pub agc2_fixed_gain_db: f32,
    pub agc2_adaptive_digital: i32,
    pub agc2_input_volume_controller: i32,
    pub applied_input_volume: i32,
    pub capture_output_used: i32,
    pub user_microphone_gain: f32,
    pub enable_post_limiter: i32,
    pub initial_delay_ms: i32,
    pub vad_threshold: f32,
    pub export_linear_aec_output: i32,
}

impl Default for MetaAec3Config {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            render_sample_rate_hz: 48_000,
            frame_size_ms: 10,
            capture_channels: 1,
            render_channels: 2,
            enable_high_pass_filter: 1,
            enable_aec3: 1,
            noise_suppression_mode: META_AEC3_NS_WEBRTC,
            noise_suppression_level: META_AEC3_NS_LEVEL_12_DB,
            enable_agc2: 1,
            agc2_fixed_gain_db: 0.0,
            agc2_adaptive_digital: 1,
            agc2_input_volume_controller: 0,
            applied_input_volume: 255,
            capture_output_used: 1,
            user_microphone_gain: 1.0,
            enable_post_limiter: 1,
            initial_delay_ms: 0,
            vad_threshold: 0.5,
            export_linear_aec_output: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct MetaAec3Stats {
    pub struct_size: i32,
    pub status: i32,
    pub processed_samples: i32,
    pub output_samples: i32,
    pub aec_tap_samples: i32,
    pub rnnoise_input_samples: i32,
    pub speech_16k_samples: i32,
    pub sample_rate_hz: i32,
    pub frame_size_ms: i32,
    pub capture_channels: i32,
    pub render_channels: i32,
    pub internal_sample_rate_hz: i32,
    pub aec_enabled: i32,
    pub high_pass_enabled: i32,
    pub noise_suppression_mode: i32,
    pub vad_probability: f32,
    pub vad_is_voice: i32,
    pub rms: f32,
    pub peak: f32,
    pub recommended_input_volume: i32,
    pub post_limiter_applied: i32,
    pub echo_return_loss: f64,
    pub echo_return_loss_enhancement: f64,
    pub delay_ms: i32,
    pub render_jitter_min: i32,
    pub render_jitter_max: i32,
    pub capture_jitter_min: i32,
    pub capture_jitter_max: i32,
    pub fft_magnitudes: *mut f32,
    pub fft_capacity: i32,
    pub fft_bins_written: i32,
    pub fft_size: i32,
    pub fft_sample_rate_hz: i32,
}

impl Default for MetaAec3Stats {
    fn default() -> Self {
        Self {
            struct_size: size_of::<MetaAec3Stats>() as i32,
            status: META_AEC3_OK,
            processed_samples: 0,
            output_samples: 0,
            aec_tap_samples: 0,
            rnnoise_input_samples: 0,
            speech_16k_samples: 0,
            sample_rate_hz: 0,
            frame_size_ms: 0,
            capture_channels: 0,
            render_channels: 0,
            internal_sample_rate_hz: 0,
            aec_enabled: 0,
            high_pass_enabled: 0,
            noise_suppression_mode: META_AEC3_NS_NONE,
            vad_probability: 0.0,
            vad_is_voice: 0,
            rms: 0.0,
            peak: 0.0,
            recommended_input_volume: -1,
            post_limiter_applied: 0,
            echo_return_loss: 0.0,
            echo_return_loss_enhancement: 0.0,
            delay_ms: 0,
            render_jitter_min: 0,
            render_jitter_max: 0,
            capture_jitter_min: 0,
            capture_jitter_max: 0,
            fft_magnitudes: ptr::null_mut(),
            fft_capacity: 0,
            fft_bins_written: 0,
            fft_size: 0,
            fft_sample_rate_hz: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoiseSuppressionMode {
    None,
    WebRtc,
    Rnnoise,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeConfig {
    raw: MetaAec3Config,
    capture_rate: usize,
    render_rate: usize,
    internal_rate: usize,
    frame_ms: usize,
    chunks_per_frame: usize,
    capture_channels: usize,
    render_channels: usize,
    enable_high_pass_filter: bool,
    enable_aec3: bool,
    noise_suppression_mode: NoiseSuppressionMode,
    noise_suppression_level: SuppressionLevel,
    enable_agc2: bool,
    enable_post_limiter: bool,
    applied_input_volume: i32,
    capture_output_used: bool,
    user_microphone_gain: f32,
    vad_threshold: f32,
}

impl RuntimeConfig {
    fn from_ffi(mut raw: MetaAec3Config) -> Result<Self, i32> {
        let capture_rate = parse_rate(raw.sample_rate_hz)?;
        let render_rate = if raw.render_sample_rate_hz <= 0 {
            capture_rate
        } else {
            parse_rate(raw.render_sample_rate_hz)?
        };

        let frame_ms = match raw.frame_size_ms {
            10 | 20 | 40 => raw.frame_size_ms as usize,
            _ => return Err(META_AEC3_INVALID_CONFIG),
        };

        let capture_channels = match raw.capture_channels {
            1 | 2 => raw.capture_channels as usize,
            _ => return Err(META_AEC3_INVALID_CONFIG),
        };

        let render_channels = match raw.render_channels {
            1..=8 => raw.render_channels as usize,
            _ => return Err(META_AEC3_INVALID_CONFIG),
        };

        let noise_suppression_mode = match raw.noise_suppression_mode {
            META_AEC3_NS_NONE => NoiseSuppressionMode::None,
            META_AEC3_NS_WEBRTC => NoiseSuppressionMode::WebRtc,
            META_AEC3_NS_RNNOISE => NoiseSuppressionMode::Rnnoise,
            _ => return Err(META_AEC3_INVALID_CONFIG),
        };

        let noise_suppression_level = match raw.noise_suppression_level {
            META_AEC3_NS_LEVEL_6_DB => SuppressionLevel::K6dB,
            META_AEC3_NS_LEVEL_12_DB => SuppressionLevel::K12dB,
            META_AEC3_NS_LEVEL_18_DB => SuppressionLevel::K18dB,
            META_AEC3_NS_LEVEL_21_DB => SuppressionLevel::K21dB,
            _ => return Err(META_AEC3_INVALID_CONFIG),
        };

        if !raw.user_microphone_gain.is_finite() || raw.user_microphone_gain < 0.0 {
            return Err(META_AEC3_INVALID_CONFIG);
        }
        if !raw.vad_threshold.is_finite() {
            return Err(META_AEC3_INVALID_CONFIG);
        }
        if !raw.agc2_fixed_gain_db.is_finite()
            || raw.agc2_fixed_gain_db < 0.0
            || raw.agc2_fixed_gain_db >= 50.0
        {
            return Err(META_AEC3_INVALID_CONFIG);
        }

        raw.sample_rate_hz = capture_rate as i32;
        raw.render_sample_rate_hz = render_rate as i32;
        raw.frame_size_ms = frame_ms as i32;
        raw.capture_channels = capture_channels as i32;
        raw.render_channels = render_channels as i32;
        raw.applied_input_volume = raw.applied_input_volume.clamp(0, 255);
        raw.vad_threshold = raw.vad_threshold.clamp(0.0, 1.0);

        Ok(Self {
            raw,
            capture_rate,
            render_rate,
            internal_rate: internal_rate_for(capture_rate, render_rate),
            frame_ms,
            chunks_per_frame: frame_ms / 10,
            capture_channels,
            render_channels,
            enable_high_pass_filter: raw.enable_high_pass_filter != 0,
            enable_aec3: raw.enable_aec3 != 0,
            noise_suppression_mode,
            noise_suppression_level,
            enable_agc2: raw.enable_agc2 != 0,
            enable_post_limiter: raw.enable_post_limiter != 0,
            applied_input_volume: raw.applied_input_volume,
            capture_output_used: raw.capture_output_used != 0,
            user_microphone_gain: raw.user_microphone_gain,
            vad_threshold: raw.vad_threshold,
        })
    }

    fn capture_samples_per_frame(self) -> usize {
        frames_for_ms(self.capture_rate, self.frame_ms) * self.capture_channels
    }

    fn render_samples_per_frame_for_channels(self, channels: usize) -> usize {
        frames_for_ms(self.render_rate, self.frame_ms) * channels
    }

    fn output_samples_per_frame(self) -> usize {
        self.capture_samples_per_frame()
    }

    fn rnnoise_samples_per_frame(self) -> usize {
        frames_for_ms(48_000, self.frame_ms) * self.capture_channels
    }

    fn speech_16k_samples_per_frame(self) -> usize {
        frames_for_ms(16_000, self.frame_ms)
    }

    fn ten_ms_internal_frames(self) -> usize {
        self.internal_rate / 100
    }

    fn ten_ms_capture_frames(self) -> usize {
        self.capture_rate / 100
    }

    fn ten_ms_render_frames(self) -> usize {
        self.render_rate / 100
    }
}

struct AudioInputBuffer {
    stream_config: StreamConfig,
    audio_buffer: AudioBuffer,
    planar: Vec<Vec<f32>>,
}

impl AudioInputBuffer {
    fn new(input_rate: usize, channels: usize, internal_rate: usize) -> Self {
        let stream_config = StreamConfig::new(input_rate, channels, false);
        let audio_buffer = AudioBuffer::from_sample_rates(
            input_rate,
            channels,
            internal_rate,
            channels,
            input_rate,
        );
        Self {
            stream_config,
            audio_buffer,
            planar: vec![vec![0.0; input_rate / 100]; channels],
        }
    }

    fn load_interleaved(
        &mut self,
        interleaved: &[f32],
        input_channels: usize,
    ) -> Result<&mut AudioBuffer, i32> {
        let frames = self.stream_config.num_frames();
        if input_channels == 0 || interleaved.len() != frames * input_channels {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }

        copy_interleaved_to_planar_adjusted(
            interleaved,
            input_channels,
            self.stream_config.num_channels(),
            frames,
            &mut self.planar,
        );

        let refs: Vec<&[f32]> = self
            .planar
            .iter()
            .map(|channel| &channel[..frames])
            .collect();
        self.audio_buffer.copy_from(&refs, &self.stream_config);
        Ok(&mut self.audio_buffer)
    }
}

struct InterleavedExporter {
    source_rate: usize,
    dest_rate: usize,
    channels: usize,
    source_frames: usize,
    dest_frames: usize,
    resamplers: Vec<PushSincResampler>,
    scratch: Vec<Vec<f32>>,
}

impl InterleavedExporter {
    fn new(source_rate: usize, dest_rate: usize, channels: usize) -> Self {
        let source_frames = source_rate / 100;
        let dest_frames = dest_rate / 100;
        let resamplers = if source_frames != dest_frames {
            (0..channels)
                .map(|_| PushSincResampler::new(source_frames, dest_frames))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            source_rate,
            dest_rate,
            channels,
            source_frames,
            dest_frames,
            resamplers,
            scratch: vec![vec![0.0; dest_frames]; channels],
        }
    }

    fn samples_per_chunk(&self) -> usize {
        self.dest_frames * self.channels
    }

    fn export_audio_buffer(&mut self, audio: &AudioBuffer, out: &mut [f32]) -> usize {
        debug_assert_eq!(audio.num_channels(), self.channels);
        debug_assert_eq!(audio.num_frames(), self.source_frames);
        debug_assert!(out.len() >= self.samples_per_chunk());

        if self.source_rate == self.dest_rate {
            for frame in 0..self.dest_frames {
                for channel in 0..self.channels {
                    out[frame * self.channels + channel] =
                        float_s16_to_unit(audio.channel(channel)[frame]);
                }
            }
            return self.samples_per_chunk();
        }

        for channel in 0..self.channels {
            self.resamplers[channel].resample_f32(
                audio.channel(channel),
                &mut self.scratch[channel][..self.dest_frames],
            );
        }

        for frame in 0..self.dest_frames {
            for channel in 0..self.channels {
                out[frame * self.channels + channel] =
                    float_s16_to_unit(self.scratch[channel][frame]);
            }
        }

        self.samples_per_chunk()
    }
}

struct MonoExporter {
    source_rate: usize,
    dest_rate: usize,
    source_frames: usize,
    dest_frames: usize,
    resampler: Option<PushSincResampler>,
    mono: Vec<f32>,
    resampled: Vec<f32>,
}

impl MonoExporter {
    fn new(source_rate: usize, dest_rate: usize) -> Self {
        let source_frames = source_rate / 100;
        let dest_frames = dest_rate / 100;
        Self {
            source_rate,
            dest_rate,
            source_frames,
            dest_frames,
            resampler: (source_frames != dest_frames)
                .then(|| PushSincResampler::new(source_frames, dest_frames)),
            mono: vec![0.0; source_frames],
            resampled: vec![0.0; dest_frames],
        }
    }

    fn export_audio_buffer(&mut self, audio: &AudioBuffer, out: &mut [f32]) -> usize {
        debug_assert_eq!(audio.num_frames(), self.source_frames);
        debug_assert!(out.len() >= self.dest_frames);

        mix_audio_buffer_to_mono(audio, &mut self.mono);
        if self.source_rate == self.dest_rate {
            for (dst, &src) in out.iter_mut().take(self.dest_frames).zip(self.mono.iter()) {
                *dst = float_s16_to_unit(src);
            }
        } else {
            self.resampler
                .as_mut()
                .expect("resampler must exist when rates differ")
                .resample_f32(&self.mono, &mut self.resampled);
            for (dst, &src) in out
                .iter_mut()
                .take(self.dest_frames)
                .zip(self.resampled.iter())
            {
                *dst = float_s16_to_unit(src);
            }
        }

        self.dest_frames
    }
}

pub struct MetaAec3Processor {
    config: RuntimeConfig,
    render_io: AudioInputBuffer,
    capture_io: AudioInputBuffer,
    rnnoise_io: AudioInputBuffer,
    final_exporter: InterleavedExporter,
    aec_tap_exporter: InterleavedExporter,
    rnnoise_exporter: InterleavedExporter,
    speech_16k_exporter: MonoExporter,
    echo: Option<EchoCanceller3>,
    high_pass: Option<HighPassFilter>,
    noise_suppressor: Option<NoiseSuppressor>,
    agc2: Option<GainController2>,
    post_limiter: Limiter,
    vad: VoiceActivityDetectorWrapper,
    last_metrics: Metrics,
    last_applied_input_volume: Option<i32>,
    last_recommended_input_volume: i32,
    post_limiter_applied: bool,
    pending_rnnoise: bool,
    analysis_mono: Vec<f32>,
}

impl MetaAec3Processor {
    fn new(raw_config: MetaAec3Config) -> Result<Self, i32> {
        let config = RuntimeConfig::from_ffi(raw_config)?;
        let mut echo = None;
        if config.enable_aec3 {
            let mut echo_config = EchoCanceller3::create_default_config(
                config.render_channels,
                config.capture_channels,
            );
            echo_config.filter.export_linear_aec_output = raw_config.export_linear_aec_output != 0;
            let mut instance = EchoCanceller3::new(
                echo_config,
                config.internal_rate as i32,
                config.render_channels,
                config.capture_channels,
            );
            if raw_config.initial_delay_ms >= 0 {
                instance.set_audio_buffer_delay(raw_config.initial_delay_ms);
            }
            echo = Some(instance);
        }

        let high_pass = config
            .enable_high_pass_filter
            .then(|| HighPassFilter::new(config.internal_rate as i32, config.capture_channels));

        let ns_config = NsConfig {
            target_level: config.noise_suppression_level,
            analyze_linear_aec_output_when_available: false,
        };
        let noise_suppressor = (config.noise_suppression_mode == NoiseSuppressionMode::WebRtc)
            .then(|| {
                NoiseSuppressor::new(ns_config, config.internal_rate, config.capture_channels)
            });

        let agc2 = config.enable_agc2.then(|| {
            let mut agc_config = GainController2Config::default();
            agc_config.fixed_digital.gain_db = raw_config.agc2_fixed_gain_db;
            agc_config.adaptive_digital.enabled = raw_config.agc2_adaptive_digital != 0;
            agc_config.input_volume_controller.enabled =
                raw_config.agc2_input_volume_controller != 0;
            GainController2::new(
                agc_config,
                InputVolumeControllerConfig::default(),
                config.internal_rate,
                config.capture_channels,
                true,
            )
        });

        Ok(Self {
            config,
            render_io: AudioInputBuffer::new(
                config.render_rate,
                config.render_channels,
                config.internal_rate,
            ),
            capture_io: AudioInputBuffer::new(
                config.capture_rate,
                config.capture_channels,
                config.internal_rate,
            ),
            rnnoise_io: AudioInputBuffer::new(
                48_000,
                config.capture_channels,
                config.internal_rate,
            ),
            final_exporter: InterleavedExporter::new(
                config.internal_rate,
                config.capture_rate,
                config.capture_channels,
            ),
            aec_tap_exporter: InterleavedExporter::new(
                config.internal_rate,
                config.capture_rate,
                config.capture_channels,
            ),
            rnnoise_exporter: InterleavedExporter::new(
                config.internal_rate,
                48_000,
                config.capture_channels,
            ),
            speech_16k_exporter: MonoExporter::new(config.internal_rate, 16_000),
            echo,
            high_pass,
            noise_suppressor,
            agc2,
            post_limiter: Limiter::new(config.ten_ms_internal_frames()),
            vad: VoiceActivityDetectorWrapper::new(
                get_available_cpu_features(),
                config.internal_rate,
            ),
            last_metrics: Metrics::default(),
            last_applied_input_volume: None,
            last_recommended_input_volume: -1,
            post_limiter_applied: false,
            pending_rnnoise: false,
            analysis_mono: Vec::with_capacity(
                config.ten_ms_internal_frames() * config.chunks_per_frame,
            ),
        })
    }

    fn process_render(&mut self, input: &[f32], input_channels: usize) -> Result<i32, i32> {
        if !(1..=8).contains(&input_channels) {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        let expected = self
            .config
            .render_samples_per_frame_for_channels(input_channels);
        if input.len() != expected {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        if !self.config.enable_aec3 {
            return Ok(META_AEC3_OK);
        }

        let chunk_samples = self.config.ten_ms_render_frames() * input_channels;
        for chunk in input.chunks_exact(chunk_samples) {
            let render = self.render_io.load_interleaved(chunk, input_channels)?;
            render.split_into_frequency_bands();
            self.echo
                .as_mut()
                .expect("AEC3 is enabled")
                .analyze_render(render);
        }

        Ok(META_AEC3_OK)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_capture(
        &mut self,
        input: &[f32],
        output: Option<&mut [f32]>,
        aec_tap: Option<&mut [f32]>,
        rnnoise_input: Option<&mut [f32]>,
        speech_16k: Option<&mut [f32]>,
        stats: Option<&mut MetaAec3Stats>,
    ) -> Result<i32, i32> {
        if input.len() != self.config.capture_samples_per_frame() {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }

        let using_rnnoise = self.config.noise_suppression_mode == NoiseSuppressionMode::Rnnoise;
        if !using_rnnoise && output.is_none() {
            return Err(META_AEC3_NULL_POINTER);
        }
        if using_rnnoise && rnnoise_input.is_none() {
            return Err(META_AEC3_NULL_POINTER);
        }

        self.pending_rnnoise = false;
        self.post_limiter_applied = false;
        self.last_recommended_input_volume = -1;
        self.analysis_mono.clear();

        let input_chunk_samples =
            self.config.ten_ms_capture_frames() * self.config.capture_channels;
        let output_chunk_samples = self.final_exporter.samples_per_chunk();
        let rnnoise_chunk_samples = self.rnnoise_exporter.samples_per_chunk();
        let speech_chunk_samples = self.speech_16k_exporter.dest_frames;

        let mut output = output;
        let mut aec_tap = aec_tap;
        let mut rnnoise_input = rnnoise_input;
        let mut speech_16k = speech_16k;
        let mut max_vad = 0.0f32;

        for (chunk_index, input_chunk) in input.chunks_exact(input_chunk_samples).enumerate() {
            let capture = self
                .capture_io
                .load_interleaved(input_chunk, self.config.capture_channels)?;

            if self.config.enable_high_pass_filter {
                apply_high_pass(self.high_pass.as_mut(), capture);
            }

            if self.config.enable_aec3 {
                let echo = self.echo.as_mut().expect("AEC3 is enabled");
                echo.analyze_capture(capture);
                capture.split_into_frequency_bands();
                echo.process_capture(capture, false);
                self.last_metrics = echo.metrics();
                capture.merge_frequency_bands();
            }

            append_mono_from_audio_buffer(capture, &mut self.analysis_mono);
            max_vad = max_vad.max(analyze_vad_for_chunk(&mut self.vad, capture));

            if let Some(tap) = aec_tap.as_deref_mut() {
                let offset = chunk_index * output_chunk_samples;
                self.aec_tap_exporter
                    .export_audio_buffer(capture, &mut tap[offset..offset + output_chunk_samples]);
            }

            if let Some(speech) = speech_16k.as_deref_mut() {
                let offset = chunk_index * speech_chunk_samples;
                self.speech_16k_exporter.export_audio_buffer(
                    capture,
                    &mut speech[offset..offset + speech_chunk_samples],
                );
            }

            if let Some(rnnoise) = rnnoise_input.as_deref_mut() {
                let offset = chunk_index * rnnoise_chunk_samples;
                self.rnnoise_exporter.export_audio_buffer(
                    capture,
                    &mut rnnoise[offset..offset + rnnoise_chunk_samples],
                );
            }

            if using_rnnoise {
                continue;
            }

            apply_native_noise_suppression(
                self.config.noise_suppression_mode,
                &mut self.noise_suppressor,
                capture,
            );
            apply_agc_user_gain_and_limiter(
                self.config,
                &mut self.agc2,
                &mut self.post_limiter,
                &mut self.last_applied_input_volume,
                &mut self.last_recommended_input_volume,
                &mut self.post_limiter_applied,
                capture,
            );

            if let Some(out) = output.as_deref_mut() {
                let offset = chunk_index * output_chunk_samples;
                self.final_exporter
                    .export_audio_buffer(capture, &mut out[offset..offset + output_chunk_samples]);
            }
        }

        let status = if using_rnnoise {
            self.pending_rnnoise = true;
            META_AEC3_NEEDS_RNNOISE
        } else {
            META_AEC3_OK
        };

        self.write_stats(stats, status, max_vad, using_rnnoise);
        Ok(status)
    }

    fn finish_rnnoise(
        &mut self,
        rnnoise_output: &[f32],
        output: &mut [f32],
        stats: Option<&mut MetaAec3Stats>,
    ) -> Result<i32, i32> {
        if !self.pending_rnnoise {
            return Err(META_AEC3_NO_PENDING_RNNOISE);
        }
        if rnnoise_output.len() != self.config.rnnoise_samples_per_frame() {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        if output.len() < self.config.output_samples_per_frame() {
            return Err(META_AEC3_BUFFER_TOO_SMALL);
        }

        self.post_limiter_applied = false;
        self.last_recommended_input_volume = -1;

        let input_chunk_samples = (48_000 / 100) * self.config.capture_channels;
        let output_chunk_samples = self.final_exporter.samples_per_chunk();
        for (chunk_index, input_chunk) in
            rnnoise_output.chunks_exact(input_chunk_samples).enumerate()
        {
            let audio = self
                .rnnoise_io
                .load_interleaved(input_chunk, self.config.capture_channels)?;
            apply_agc_user_gain_and_limiter(
                self.config,
                &mut self.agc2,
                &mut self.post_limiter,
                &mut self.last_applied_input_volume,
                &mut self.last_recommended_input_volume,
                &mut self.post_limiter_applied,
                audio,
            );

            let offset = chunk_index * output_chunk_samples;
            self.final_exporter
                .export_audio_buffer(audio, &mut output[offset..offset + output_chunk_samples]);
        }

        self.pending_rnnoise = false;
        self.write_stats(stats, META_AEC3_OK, 0.0, false);
        Ok(META_AEC3_OK)
    }

    fn write_stats(
        &mut self,
        stats: Option<&mut MetaAec3Stats>,
        status: i32,
        vad_probability: f32,
        rnnoise_pending: bool,
    ) {
        let Some(stats) = stats else {
            return;
        };
        let fft_ptr = stats.fft_magnitudes;
        let fft_capacity = stats.fft_capacity;
        *stats = MetaAec3Stats::default();
        stats.fft_magnitudes = fft_ptr;
        stats.fft_capacity = fft_capacity;

        stats.status = status;
        stats.processed_samples = self.config.capture_samples_per_frame() as i32;
        stats.output_samples = if rnnoise_pending {
            0
        } else {
            self.config.output_samples_per_frame() as i32
        };
        stats.aec_tap_samples = self.config.output_samples_per_frame() as i32;
        stats.rnnoise_input_samples = self.config.rnnoise_samples_per_frame() as i32;
        stats.speech_16k_samples = self.config.speech_16k_samples_per_frame() as i32;
        stats.sample_rate_hz = self.config.capture_rate as i32;
        stats.frame_size_ms = self.config.frame_ms as i32;
        stats.capture_channels = self.config.capture_channels as i32;
        stats.render_channels = self.config.render_channels as i32;
        stats.internal_sample_rate_hz = self.config.internal_rate as i32;
        stats.aec_enabled = self.config.enable_aec3 as i32;
        stats.high_pass_enabled = self.config.enable_high_pass_filter as i32;
        stats.noise_suppression_mode = match self.config.noise_suppression_mode {
            NoiseSuppressionMode::None => META_AEC3_NS_NONE,
            NoiseSuppressionMode::WebRtc => META_AEC3_NS_WEBRTC,
            NoiseSuppressionMode::Rnnoise => META_AEC3_NS_RNNOISE,
        };
        stats.vad_probability = vad_probability;
        stats.vad_is_voice = (vad_probability >= self.config.vad_threshold) as i32;
        let (rms, peak) = rms_peak_unit_from_float_s16(&self.analysis_mono);
        stats.rms = rms;
        stats.peak = peak;
        stats.recommended_input_volume = self.last_recommended_input_volume;
        stats.post_limiter_applied = self.post_limiter_applied as i32;
        stats.echo_return_loss = self.last_metrics.echo_return_loss;
        stats.echo_return_loss_enhancement = self.last_metrics.echo_return_loss_enhancement;
        stats.delay_ms = self.last_metrics.delay_ms;
        stats.render_jitter_min = self.last_metrics.render_jitter_min;
        stats.render_jitter_max = self.last_metrics.render_jitter_max;
        stats.capture_jitter_min = self.last_metrics.capture_jitter_min;
        stats.capture_jitter_max = self.last_metrics.capture_jitter_max;
        fill_fft_stats(stats, &self.analysis_mono, self.config.internal_rate);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_default_config(config: *mut MetaAec3Config) -> i32 {
    ffi_status(|| {
        let config = checked_mut(config)?;
        *config = MetaAec3Config::default();
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_create(config: *const MetaAec3Config) -> *mut MetaAec3Processor {
    match catch_unwind(AssertUnwindSafe(|| {
        let raw = if config.is_null() {
            MetaAec3Config::default()
        } else {
            unsafe { *config }
        };
        MetaAec3Processor::new(raw)
            .map(|processor| Box::into_raw(Box::new(processor)))
            .unwrap_or(ptr::null_mut())
    })) {
        Ok(ptr) => ptr,
        Err(_) => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_free(processor: *mut MetaAec3Processor) {
    if !processor.is_null() {
        unsafe {
            let _ = Box::from_raw(processor);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_configure(
    processor: *mut MetaAec3Processor,
    config: *const MetaAec3Config,
) -> i32 {
    ffi_status(|| {
        let processor = checked_mut(processor)?;
        let config = checked_ref(config)?;
        let replacement = MetaAec3Processor::new(*config)?;
        *processor = replacement;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_get_config(
    processor: *const MetaAec3Processor,
    config: *mut MetaAec3Config,
) -> i32 {
    ffi_status(|| {
        let processor = checked_ref(processor)?;
        let config = checked_mut(config)?;
        *config = processor.config.raw;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_reset(processor: *mut MetaAec3Processor) -> i32 {
    ffi_status(|| {
        let processor = checked_mut(processor)?;
        let raw = processor.config.raw;
        *processor = MetaAec3Processor::new(raw)?;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_set_stream_delay_ms(
    processor: *mut MetaAec3Processor,
    delay_ms: i32,
) -> i32 {
    ffi_status(|| {
        let processor = checked_mut(processor)?;
        if delay_ms < 0 {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        processor.config.raw.initial_delay_ms = delay_ms;
        if let Some(echo) = processor.echo.as_mut() {
            echo.set_audio_buffer_delay(delay_ms);
        }
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_set_user_microphone_gain(
    processor: *mut MetaAec3Processor,
    gain: f32,
) -> i32 {
    ffi_status(|| {
        let processor = checked_mut(processor)?;
        if !gain.is_finite() || gain < 0.0 {
            return Err(META_AEC3_INVALID_ARGUMENT);
        }
        processor.config.raw.user_microphone_gain = gain;
        processor.config.user_microphone_gain = gain;
        Ok(META_AEC3_OK)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_capture_samples_per_frame(processor: *const MetaAec3Processor) -> i32 {
    match checked_ref(processor) {
        Ok(processor) => processor.config.capture_samples_per_frame() as i32,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_output_samples_per_frame(processor: *const MetaAec3Processor) -> i32 {
    match checked_ref(processor) {
        Ok(processor) => processor.config.output_samples_per_frame() as i32,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_rnnoise_samples_per_frame(processor: *const MetaAec3Processor) -> i32 {
    match checked_ref(processor) {
        Ok(processor) => processor.config.rnnoise_samples_per_frame() as i32,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_speech_16k_samples_per_frame(
    processor: *const MetaAec3Processor,
) -> i32 {
    match checked_ref(processor) {
        Ok(processor) => processor.config.speech_16k_samples_per_frame() as i32,
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_fft_bins_per_frame(processor: *const MetaAec3Processor) -> i32 {
    match checked_ref(processor) {
        Ok(processor) => {
            let fft_size = (processor.config.ten_ms_internal_frames()
                * processor.config.chunks_per_frame)
                .next_power_of_two();
            (fft_size / 2 + 1) as i32
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_process_render(
    processor: *mut MetaAec3Processor,
    samples: *const f32,
    samples_len: i32,
    channels: i32,
) -> i32 {
    ffi_status(|| {
        let processor = checked_mut(processor)?;
        let channels = usize_from_positive_i32(channels)?;
        let samples = checked_f32_slice(samples, samples_len)?;
        processor.process_render(samples, channels)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn meta_aec3_process_capture(
    processor: *mut MetaAec3Processor,
    capture_samples: *const f32,
    capture_samples_len: i32,
    output_samples: *mut f32,
    output_samples_len: i32,
    aec_tap_samples: *mut f32,
    aec_tap_samples_len: i32,
    rnnoise_input_samples: *mut f32,
    rnnoise_input_samples_len: i32,
    speech_16k_samples: *mut f32,
    speech_16k_samples_len: i32,
    stats: *mut MetaAec3Stats,
) -> i32 {
    let status = ffi_status(|| {
        let processor = checked_mut(processor)?;
        let capture = checked_f32_slice(capture_samples, capture_samples_len)?;
        let output = optional_f32_slice_mut(
            output_samples,
            output_samples_len,
            processor.config.output_samples_per_frame(),
        )?;
        let aec_tap = optional_f32_slice_mut(
            aec_tap_samples,
            aec_tap_samples_len,
            processor.config.output_samples_per_frame(),
        )?;
        let rnnoise = optional_f32_slice_mut(
            rnnoise_input_samples,
            rnnoise_input_samples_len,
            processor.config.rnnoise_samples_per_frame(),
        )?;
        let speech_16k = optional_f32_slice_mut(
            speech_16k_samples,
            speech_16k_samples_len,
            processor.config.speech_16k_samples_per_frame(),
        )?;
        let stats = optional_stats_mut(stats)?;
        processor.process_capture(capture, output, aec_tap, rnnoise, speech_16k, stats)
    });

    if status < META_AEC3_OK {
        write_error_status(stats, status);
    }
    status
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_finish_rnnoise_frame(
    processor: *mut MetaAec3Processor,
    rnnoise_output_samples: *const f32,
    rnnoise_output_samples_len: i32,
    output_samples: *mut f32,
    output_samples_len: i32,
    stats: *mut MetaAec3Stats,
) -> i32 {
    let status = ffi_status(|| {
        let processor = checked_mut(processor)?;
        let rnnoise_output = checked_f32_slice(rnnoise_output_samples, rnnoise_output_samples_len)?;
        let output = checked_f32_slice_mut(output_samples, output_samples_len)?;
        if output.len() < processor.config.output_samples_per_frame() {
            return Err(META_AEC3_BUFFER_TOO_SMALL);
        }
        let stats = optional_stats_mut(stats)?;
        processor.finish_rnnoise(rnnoise_output, output, stats)
    });

    if status < META_AEC3_OK {
        write_error_status(stats, status);
    }
    status
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_status_ok() -> i32 {
    META_AEC3_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn meta_aec3_status_needs_rnnoise() -> i32 {
    META_AEC3_NEEDS_RNNOISE
}

fn parse_rate(rate: i32) -> Result<usize, i32> {
    match rate {
        8_000 | 12_000 | 16_000 | 24_000 | 48_000 => Ok(rate as usize),
        _ => Err(META_AEC3_INVALID_CONFIG),
    }
}

fn internal_rate_for(capture_rate: usize, render_rate: usize) -> usize {
    let min_rate = capture_rate.min(render_rate);
    if min_rate <= 16_000 {
        16_000
    } else if min_rate <= 32_000 {
        32_000
    } else {
        48_000
    }
}

fn frames_for_ms(rate: usize, frame_ms: usize) -> usize {
    rate * frame_ms / 1000
}

fn ffi_status<F>(f: F) -> i32
where
    F: FnOnce() -> Result<i32, i32>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => META_AEC3_PANIC,
    }
}

fn checked_ref<'a, T>(ptr: *const T) -> Result<&'a T, i32> {
    if ptr.is_null() {
        return Err(META_AEC3_NULL_POINTER);
    }
    Ok(unsafe { &*ptr })
}

fn checked_mut<'a, T>(ptr: *mut T) -> Result<&'a mut T, i32> {
    if ptr.is_null() {
        return Err(META_AEC3_NULL_POINTER);
    }
    Ok(unsafe { &mut *ptr })
}

fn checked_f32_slice<'a>(ptr: *const f32, len: i32) -> Result<&'a [f32], i32> {
    if ptr.is_null() {
        return Err(META_AEC3_NULL_POINTER);
    }
    if len <= 0 || !is_aligned_for_f32(ptr) {
        return Err(META_AEC3_INVALID_ARGUMENT);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len as usize) })
}

fn checked_f32_slice_mut<'a>(ptr: *mut f32, len: i32) -> Result<&'a mut [f32], i32> {
    if ptr.is_null() {
        return Err(META_AEC3_NULL_POINTER);
    }
    if len <= 0 || !is_aligned_for_f32(ptr) {
        return Err(META_AEC3_INVALID_ARGUMENT);
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len as usize) })
}

fn optional_f32_slice_mut<'a>(
    ptr: *mut f32,
    len: i32,
    required_len: usize,
) -> Result<Option<&'a mut [f32]>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    if len < 0 || !is_aligned_for_f32(ptr) {
        return Err(META_AEC3_INVALID_ARGUMENT);
    }
    if (len as usize) < required_len {
        return Err(META_AEC3_BUFFER_TOO_SMALL);
    }
    Ok(Some(unsafe {
        std::slice::from_raw_parts_mut(ptr, len as usize)
    }))
}

fn optional_stats_mut<'a>(ptr: *mut MetaAec3Stats) -> Result<Option<&'a mut MetaAec3Stats>, i32> {
    if ptr.is_null() {
        return Ok(None);
    }
    Ok(Some(unsafe { &mut *ptr }))
}

fn write_error_status(stats: *mut MetaAec3Stats, status: i32) {
    if stats.is_null() {
        return;
    }
    unsafe {
        let fft_ptr = (*stats).fft_magnitudes;
        let fft_capacity = (*stats).fft_capacity;
        *stats = MetaAec3Stats::default();
        (*stats).fft_magnitudes = fft_ptr;
        (*stats).fft_capacity = fft_capacity;
        (*stats).status = status;
    }
}

fn usize_from_positive_i32(value: i32) -> Result<usize, i32> {
    if value <= 0 {
        Err(META_AEC3_INVALID_ARGUMENT)
    } else {
        Ok(value as usize)
    }
}

fn is_aligned_for_f32(ptr: *const f32) -> bool {
    (ptr as usize) % align_of::<f32>() == 0
}

fn copy_interleaved_to_planar_adjusted(
    interleaved: &[f32],
    input_channels: usize,
    target_channels: usize,
    frames: usize,
    planar: &mut [Vec<f32>],
) {
    debug_assert_eq!(planar.len(), target_channels);
    if target_channels == 1 && input_channels > 1 {
        let mono = &mut planar[0][..frames];
        for frame in 0..frames {
            let mut sum = 0.0f32;
            for channel in 0..input_channels {
                sum += interleaved[frame * input_channels + channel];
            }
            mono[frame] = sum / input_channels as f32;
        }
        return;
    }

    for target_channel in 0..target_channels {
        let source_channel = if target_channel < input_channels {
            target_channel
        } else {
            0
        };
        let output = &mut planar[target_channel][..frames];
        for frame in 0..frames {
            output[frame] = interleaved[frame * input_channels + source_channel];
        }
    }
}

fn apply_high_pass(filter: Option<&mut HighPassFilter>, audio: &mut AudioBuffer) {
    let Some(filter) = filter else {
        return;
    };
    let channels = audio.num_channels();
    let mut working = (0..channels)
        .map(|channel| audio.channel(channel).to_vec())
        .collect::<Vec<_>>();
    filter.process(&mut working);
    for (channel, source) in working.iter().enumerate() {
        audio.channel_mut(channel).copy_from_slice(source);
    }
}

fn apply_native_noise_suppression(
    mode: NoiseSuppressionMode,
    noise_suppressor: &mut Option<NoiseSuppressor>,
    audio: &mut AudioBuffer,
) {
    if mode != NoiseSuppressionMode::WebRtc {
        return;
    }
    if let Some(ns) = noise_suppressor.as_mut() {
        audio.split_into_frequency_bands();
        ns.analyze(audio);
        ns.process(audio);
        audio.merge_frequency_bands();
    }
}

fn apply_agc_user_gain_and_limiter(
    config: RuntimeConfig,
    agc2: &mut Option<GainController2>,
    post_limiter: &mut Limiter,
    last_applied_input_volume: &mut Option<i32>,
    last_recommended_input_volume: &mut i32,
    post_limiter_applied: &mut bool,
    audio: &mut AudioBuffer,
) {
    if let Some(agc2) = agc2.as_mut() {
        agc2.set_capture_output_used(config.capture_output_used);
        agc2.analyze(config.applied_input_volume, audio);
        let input_volume_changed = *last_applied_input_volume != Some(config.applied_input_volume);
        agc2.process(input_volume_changed, audio);
        *last_applied_input_volume = Some(config.applied_input_volume);
        *last_recommended_input_volume = agc2.recommended_input_volume().unwrap_or(-1);
    }

    let channels = audio.num_channels();
    let mut working = (0..channels)
        .map(|channel| audio.channel(channel).to_vec())
        .collect::<Vec<_>>();

    let gain = config.user_microphone_gain;
    if (gain - 1.0).abs() > f32::EPSILON {
        for channel in &mut working {
            for sample in channel {
                *sample *= gain;
                if sample.abs() > 32767.0 {
                    *post_limiter_applied = true;
                }
            }
        }
    }

    if config.enable_post_limiter {
        let mut refs = working
            .iter_mut()
            .map(|channel| channel.as_mut_slice())
            .collect::<Vec<_>>();
        post_limiter.set_samples_per_channel(audio.num_frames());
        post_limiter.process(&mut refs);
    }

    for (channel, source) in working.iter().enumerate() {
        audio.channel_mut(channel).copy_from_slice(source);
    }
}

fn analyze_vad_for_chunk(vad: &mut VoiceActivityDetectorWrapper, audio: &AudioBuffer) -> f32 {
    let frame = (0..audio.num_channels())
        .map(|channel| audio.channel(channel))
        .collect::<Vec<_>>();
    vad.analyze(&frame)
}

fn append_mono_from_audio_buffer(audio: &AudioBuffer, output: &mut Vec<f32>) {
    let frames = audio.num_frames();
    let channels = audio.num_channels();
    for frame in 0..frames {
        let mut sample = 0.0f32;
        for channel in 0..channels {
            sample += audio.channel(channel)[frame];
        }
        output.push(sample / channels as f32);
    }
}

fn mix_audio_buffer_to_mono(audio: &AudioBuffer, output: &mut [f32]) {
    let frames = audio.num_frames();
    let channels = audio.num_channels();
    debug_assert!(output.len() >= frames);
    for frame in 0..frames {
        let mut sample = 0.0f32;
        for channel in 0..channels {
            sample += audio.channel(channel)[frame];
        }
        output[frame] = sample / channels as f32;
    }
}

fn float_s16_to_unit(value: f32) -> f32 {
    value.clamp(-32768.0, 32767.0) / 32768.0
}

fn rms_peak_unit_from_float_s16(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_squares = 0.0f32;
    let mut peak = 0.0f32;
    for &sample in samples {
        let unit = float_s16_to_unit(sample);
        sum_squares += unit * unit;
        peak = peak.max(unit.abs());
    }
    ((sum_squares / samples.len() as f32).sqrt(), peak)
}

fn fill_fft_stats(stats: &mut MetaAec3Stats, mono_float_s16: &[f32], sample_rate: usize) {
    stats.fft_sample_rate_hz = sample_rate as i32;
    if mono_float_s16.is_empty() {
        return;
    }

    let fft_size = mono_float_s16.len().next_power_of_two();
    let bins = fft_size / 2 + 1;
    stats.fft_size = fft_size as i32;

    if stats.fft_magnitudes.is_null() || stats.fft_capacity <= 0 {
        return;
    }
    if !is_aligned_for_f32(stats.fft_magnitudes) {
        return;
    }

    let write_bins = bins.min(stats.fft_capacity as usize);
    let output = unsafe { std::slice::from_raw_parts_mut(stats.fft_magnitudes, write_bins) };

    let mut spectrum = vec![Complex32::new(0.0, 0.0); fft_size];
    for (dst, &src) in spectrum.iter_mut().zip(mono_float_s16.iter()) {
        dst.re = float_s16_to_unit(src);
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut spectrum);

    let scale = 1.0 / fft_size as f32;
    for index in 0..write_bins {
        output[index] = spectrum[index].norm() * scale;
    }
    stats.fft_bins_written = write_bins as i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_creates_processor() {
        let mut config = MetaAec3Config::default();
        assert_eq!(META_AEC3_OK, meta_aec3_default_config(&mut config));
        let ptr = meta_aec3_create(&config);
        assert!(!ptr.is_null());
        assert_eq!(480, meta_aec3_capture_samples_per_frame(ptr));
        meta_aec3_free(ptr);
    }

    #[test]
    fn processes_silence_with_native_path() {
        let mut config = MetaAec3Config::default();
        config.capture_channels = 1;
        config.render_channels = 2;
        config.frame_size_ms = 10;
        let ptr = meta_aec3_create(&config);
        assert!(!ptr.is_null());

        let render = vec![0.0f32; 480 * 2];
        assert_eq!(
            META_AEC3_OK,
            meta_aec3_process_render(ptr, render.as_ptr(), render.len() as i32, 2)
        );

        let capture = vec![0.0f32; 480];
        let mut output = vec![0.0f32; 480];
        let mut tap = vec![0.0f32; 480];
        let mut speech = vec![0.0f32; 160];
        let mut fft = vec![0.0f32; meta_aec3_fft_bins_per_frame(ptr) as usize];
        let mut stats = MetaAec3Stats {
            fft_magnitudes: fft.as_mut_ptr(),
            fft_capacity: fft.len() as i32,
            ..MetaAec3Stats::default()
        };

        let status = meta_aec3_process_capture(
            ptr,
            capture.as_ptr(),
            capture.len() as i32,
            output.as_mut_ptr(),
            output.len() as i32,
            tap.as_mut_ptr(),
            tap.len() as i32,
            ptr::null_mut(),
            0,
            speech.as_mut_ptr(),
            speech.len() as i32,
            &mut stats,
        );

        assert_eq!(META_AEC3_OK, status);
        assert_eq!(META_AEC3_OK, stats.status);
        assert_eq!(480, stats.output_samples);
        assert!(stats.fft_bins_written > 0);
        meta_aec3_free(ptr);
    }

    #[test]
    fn rnnoise_path_requires_finish() {
        let mut config = MetaAec3Config::default();
        config.noise_suppression_mode = META_AEC3_NS_RNNOISE;
        let ptr = meta_aec3_create(&config);
        assert!(!ptr.is_null());

        let capture = vec![0.0f32; 480];
        let mut rnnoise_input = vec![0.0f32; 480];
        let mut stats = MetaAec3Stats::default();
        let status = meta_aec3_process_capture(
            ptr,
            capture.as_ptr(),
            capture.len() as i32,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            rnnoise_input.as_mut_ptr(),
            rnnoise_input.len() as i32,
            ptr::null_mut(),
            0,
            &mut stats,
        );
        assert_eq!(META_AEC3_NEEDS_RNNOISE, status);

        let mut output = vec![0.0f32; 480];
        let finish_status = meta_aec3_finish_rnnoise_frame(
            ptr,
            rnnoise_input.as_ptr(),
            rnnoise_input.len() as i32,
            output.as_mut_ptr(),
            output.len() as i32,
            &mut stats,
        );
        assert_eq!(META_AEC3_OK, finish_status);
        meta_aec3_free(ptr);
    }

    #[test]
    fn supports_8k_capture_and_wide_render_input() {
        let mut config = MetaAec3Config::default();
        config.sample_rate_hz = 8_000;
        config.render_sample_rate_hz = 48_000;
        config.frame_size_ms = 20;
        config.capture_channels = 2;
        config.render_channels = 2;
        config.noise_suppression_mode = META_AEC3_NS_NONE;

        let ptr = meta_aec3_create(&config);
        assert!(!ptr.is_null());
        assert_eq!(320, meta_aec3_capture_samples_per_frame(ptr));
        assert_eq!(320, meta_aec3_output_samples_per_frame(ptr));
        assert_eq!(320, meta_aec3_speech_16k_samples_per_frame(ptr));

        let render = vec![0.0f32; 960 * 8];
        assert_eq!(
            META_AEC3_OK,
            meta_aec3_process_render(ptr, render.as_ptr(), render.len() as i32, 8)
        );

        let capture = vec![0.0f32; 320];
        let mut output = vec![0.0f32; 320];
        let mut stats = MetaAec3Stats::default();
        let status = meta_aec3_process_capture(
            ptr,
            capture.as_ptr(),
            capture.len() as i32,
            output.as_mut_ptr(),
            output.len() as i32,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            &mut stats,
        );
        assert_eq!(META_AEC3_OK, status);
        assert_eq!(16_000, stats.internal_sample_rate_hz);
        assert_eq!(320, stats.output_samples);
        meta_aec3_free(ptr);
    }
}
