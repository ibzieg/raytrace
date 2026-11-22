//! A render running on a background thread, which the UI polls each frame.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use glam::{DMat4, DVec3};
use raytrace::scene::{PhotonSettings, RenderSettings};
use raytrace::{Image, PhotonMaps, Renderer, Scene, World};

/// Renders smaller than this many pixels skip the low-res preview pass.
const PREVIEW_MIN_PIXELS: usize = 200_000;
const PREVIEW_DOWNSCALE: u32 = 4;

#[derive(Clone, Debug)]
pub enum Stage {
    Photons,
    Preview,
    Rendering,
    Done {
        photons: Option<Duration>,
        render: Duration,
    },
    /// A preview-only job (e.g. while dragging the camera) finished.
    Previewed,
    Cancelled,
}

pub struct Job {
    pub width: u32,
    pub height: u32,
    pub cancel: AtomicBool,
    /// Linear radiance, rows top-down. Tone mapped by the UI.
    pub pixels: Mutex<Vec<DVec3>>,
    /// Set whenever `pixels` changes; the UI clears it after uploading.
    pub updated: AtomicBool,
    pub rows_done: AtomicUsize,
    pub stage: Mutex<Stage>,
    /// Photon maps built by this job, for the UI to cache.
    pub built_maps: Mutex<Option<Arc<PhotonMaps>>>,
    /// The finished full-precision image, for saving.
    pub image: Mutex<Option<Image>>,
}

pub struct JobInput {
    pub scene: Arc<Scene>,
    pub world: Arc<World>,
    pub render: RenderSettings,
    pub photon: PhotonSettings,
    /// World-to-camera matrix.
    pub view: DMat4,
    /// Stop after the low-res preview pass.
    pub preview_only: bool,
    /// Reused when the photon settings haven't changed since they were built.
    pub cached_maps: Option<Arc<PhotonMaps>>,
}

impl Job {
    pub fn spawn(input: JobInput, ctx: egui::Context) -> Arc<Self> {
        let (width, height) = input.scene.image_size_for(&input.render);
        let (width, height) = (width.max(1), height.max(1));
        let job = Arc::new(Self {
            width,
            height,
            cancel: AtomicBool::new(false),
            pixels: Mutex::new(vec![DVec3::ZERO; (width * height) as usize]),
            updated: AtomicBool::new(true),
            rows_done: AtomicUsize::new(0),
            stage: Mutex::new(Stage::Rendering),
            built_maps: Mutex::new(None),
            image: Mutex::new(None),
        });
        let worker = job.clone();
        std::thread::spawn(move || worker.run(input, ctx));
        job
    }

    pub fn stage(&self) -> Stage {
        self.stage.lock().unwrap().clone()
    }

    pub fn is_running(&self) -> bool {
        matches!(self.stage(), Stage::Photons | Stage::Preview | Stage::Rendering)
    }

    pub fn progress(&self) -> f32 {
        self.rows_done.load(Ordering::Relaxed) as f32 / self.height as f32
    }

    fn set_stage(&self, stage: Stage, ctx: &egui::Context) {
        *self.stage.lock().unwrap() = stage;
        ctx.request_repaint();
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn run(&self, input: JobInput, ctx: egui::Context) {
        let mut photon_time = None;
        let maps = if input.photon.enabled {
            match input.cached_maps {
                Some(m) => Some(m),
                None => {
                    self.set_stage(Stage::Photons, &ctx);
                    let t = Instant::now();
                    let m = Arc::new(PhotonMaps::build(&input.world, &input.photon, 1));
                    photon_time = Some(t.elapsed());
                    *self.built_maps.lock().unwrap() = Some(m.clone());
                    Some(m)
                }
            }
        } else {
            None
        };
        if self.cancelled() {
            return self.set_stage(Stage::Cancelled, &ctx);
        }

        let cam = &input.scene.file.camera;
        let preview_settings = RenderSettings { samples: 1, ..input.render.clone() };
        let mut renderer = Renderer {
            world: &input.world,
            settings: &input.render,
            photons: maps.as_deref(),
            photon_settings: &input.photon,
            view: input.view,
            window: cam.window,
            near: cam.near,
            far: cam.far,
        };
        let t = Instant::now();

        if input.preview_only || (self.width * self.height) as usize >= PREVIEW_MIN_PIXELS {
            self.set_stage(Stage::Preview, &ctx);
            renderer.settings = &preview_settings;
            let (pw, ph) = (self.width / PREVIEW_DOWNSCALE, self.height / PREVIEW_DOWNSCALE);
            if let Some(preview) = renderer.render_with(pw.max(1), ph.max(1), Some(&self.cancel), &|_, _| {}) {
                self.blit_upscaled(&preview);
                ctx.request_repaint();
            }
            renderer.settings = &input.render;
            if input.preview_only {
                let stage = if self.cancelled() { Stage::Cancelled } else { Stage::Previewed };
                return self.set_stage(stage, &ctx);
            }
        }

        self.set_stage(Stage::Rendering, &ctx);
        let w = self.width as usize;
        let image = renderer.render_with(self.width, self.height, Some(&self.cancel), &|row, pixels| {
            let mut out = self.pixels.lock().unwrap();
            out[row * w..(row + 1) * w].copy_from_slice(pixels);
            drop(out);
            self.rows_done.fetch_add(1, Ordering::Relaxed);
            self.updated.store(true, Ordering::Relaxed);
            ctx.request_repaint();
        });

        match image {
            Some(image) => {
                *self.image.lock().unwrap() = Some(image);
                self.set_stage(Stage::Done { photons: photon_time, render: t.elapsed() }, &ctx);
            }
            None => self.set_stage(Stage::Cancelled, &ctx),
        }
    }

    /// Nearest-neighbor upscale of a preview into the display buffer.
    fn blit_upscaled(&self, preview: &Image) {
        let mut out = self.pixels.lock().unwrap();
        for y in 0..self.height {
            let sy = (y * preview.height / self.height).min(preview.height - 1);
            for x in 0..self.width {
                let sx = (x * preview.width / self.width).min(preview.width - 1);
                out[(y * self.width + x) as usize] = preview.pixels[(sy * preview.width + sx) as usize];
            }
        }
        self.updated.store(true, Ordering::Relaxed);
    }
}
