//! Interactive viewer: edit render / photon settings, orbit the camera, and
//! watch the image re-render progressively.

mod camera;
mod job;
mod overlay;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use eframe::egui::{self, Color32, ColorImage, DragValue, PointerButton, TextureHandle, TextureOptions};
use glam::DMat4;
use raytrace::scene::{DirectLighting, PhotonSettings, RenderSettings};
use raytrace::{Display, PhotonMaps, Scene, World};

use camera::OrbitCamera;
use job::{Job, JobInput, Stage};
use overlay::{OverlayOptions, OverlayPoint};

/// After the last camera movement, wait this long before the full render.
const SETTLE_TIME: Duration = Duration::from_millis(150);

const DEFAULT_SCENE: &str = "assets/scenes/box_scene.toml";

fn main() -> eframe::Result {
    let path = std::env::args_os().nth(1).map(PathBuf::from).or_else(|| {
        let default = PathBuf::from(DEFAULT_SCENE);
        default.exists().then_some(default)
    });
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("raytrace").with_inner_size([1180.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native("raytrace", options, Box::new(move |_cc| Ok(Box::new(Viewer::new(path)))))
}

struct LoadedScene {
    path: PathBuf,
    scene: Arc<Scene>,
    world: Arc<World>,
    /// The scene camera as an orbit camera, where navigation starts from.
    initial_orbit: OrbitCamera,
}

struct Viewer {
    loaded: Option<LoadedScene>,
    render: RenderSettings,
    photon: PhotonSettings,
    /// Photon maps and the settings they were built with.
    maps: Option<(PhotonSettings, Arc<PhotonMaps>)>,
    job: Option<Arc<Job>>,
    texture: Option<TextureHandle>,
    auto_render: bool,
    needs_render: bool,
    message: Option<String>,
    /// Set once the user moves the camera; None means the scene's own view.
    camera: Option<OrbitCamera>,
    /// When the camera last moved, while a navigation gesture is in progress.
    last_camera_move: Option<Instant>,
    display: Display,
    /// Re-upload the texture (display settings changed).
    display_dirty: bool,
    show_photons: bool,
    overlay_opts: OverlayOptions,
    overlay: Vec<OverlayPoint>,
    /// What `overlay` was computed for: (maps pointer, view, options).
    overlay_key: Option<(usize, DMat4, OverlayOptions)>,
}

impl Viewer {
    fn new(path: Option<PathBuf>) -> Self {
        let mut v = Self {
            loaded: None,
            render: RenderSettings::default(),
            photon: PhotonSettings::default(),
            maps: None,
            job: None,
            texture: None,
            auto_render: true,
            needs_render: false,
            message: None,
            camera: None,
            last_camera_move: None,
            display: Display::default(),
            display_dirty: false,
            show_photons: false,
            overlay_opts: OverlayOptions::default(),
            overlay: Vec::new(),
            overlay_key: None,
        };
        if let Some(p) = path {
            v.open(&p);
        }
        v
    }

    fn open(&mut self, path: &Path) {
        match load(path) {
            Ok(loaded) => {
                self.render = loaded.scene.file.render.clone();
                self.photon = loaded.scene.file.photon.clone();
                self.loaded = Some(loaded);
                self.maps = None;
                self.camera = None;
                self.overlay_key = None;
                self.needs_render = true;
                self.message = None;
            }
            Err(e) => self.message = Some(format!("{e:#}")),
        }
    }

    fn view(&self) -> Option<DMat4> {
        let loaded = self.loaded.as_ref()?;
        Some(self.camera.map_or(loaded.scene.view, |c| c.view()))
    }

    fn start_render(&mut self, ctx: &egui::Context, preview_only: bool) {
        self.cancel();
        let Some(view) = self.view() else { return };
        let Some(loaded) = &self.loaded else { return };
        let cached_maps = match &self.maps {
            Some((settings, maps)) if settings.same_emission(&self.photon) => Some(maps.clone()),
            _ => None,
        };
        self.job = Some(Job::spawn(
            JobInput {
                scene: loaded.scene.clone(),
                world: loaded.world.clone(),
                render: self.render.clone(),
                photon: self.photon.clone(),
                view,
                preview_only,
                cached_maps,
            },
            ctx.clone(),
        ));
        self.needs_render = false;
    }

    fn cancel(&mut self) {
        if let Some(job) = &self.job {
            job.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Uploads new pixels and collects photon maps from the current job.
    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        if job.updated.swap(false, Ordering::Relaxed) | std::mem::take(&mut self.display_dirty) {
            let pixels = job
                .pixels
                .lock()
                .unwrap()
                .iter()
                .map(|&c| self.display.map(c))
                .map(|[r, g, b]| Color32::from_rgb(r, g, b))
                .collect();
            let image = ColorImage::new([job.width as usize, job.height as usize], pixels);
            match &mut self.texture {
                Some(t) if t.size() == image.size => t.set(image, TextureOptions::LINEAR),
                _ => self.texture = Some(ctx.load_texture("render", image, TextureOptions::LINEAR)),
            }
        }
        if let Some(maps) = job.built_maps.lock().unwrap().take() {
            self.maps = Some((self.photon.clone(), maps));
        }
    }

    fn save_png(&mut self) {
        let Some(job) = &self.job else { return };
        let guard = job.image.lock().unwrap();
        let Some(image) = guard.as_ref() else { return };
        let name =
            self.loaded.as_ref().and_then(|l| l.path.file_stem()).map_or("render".into(), |s| s.to_string_lossy());
        let Some(path) =
            rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name(format!("{name}.png")).save_file()
        else {
            return;
        };
        self.message = Some(match image.save_png_with(&path, self.display) {
            Ok(()) => format!("Saved {}", path.display()),
            Err(e) => format!("Save failed: {e:#}"),
        });
    }

    /// Writes the current settings back into a scene file.
    fn save_scene(&mut self) {
        let Some(loaded) = &self.loaded else { return };
        let mut dialog = rfd::FileDialog::new().add_filter("Scene", &["toml"]);
        if let (Some(dir), Some(name)) = (loaded.path.parent(), loaded.path.file_name()) {
            dialog = dialog.set_directory(dir).set_file_name(name.to_string_lossy());
        }
        let Some(path) = dialog.save_file() else { return };
        let mut file = loaded.scene.file.clone();
        file.render = self.render.clone();
        file.photon = self.photon.clone();
        if let Some(camera) = &self.camera {
            camera.write_to(&mut file.camera);
        }
        // Mesh paths are relative to the scene file; only valid if it stays
        // in the same directory.
        let same_dir = path.parent() == loaded.path.parent();
        let result =
            toml::to_string(&file).map_err(anyhow::Error::from).and_then(|text| Ok(std::fs::write(&path, text)?));
        self.message = Some(match result {
            Ok(()) if same_dir => format!("Saved {}", path.display()),
            Ok(()) => format!("Saved {} (mesh paths are relative to the original folder)", path.display()),
            Err(e) => format!("Save failed: {e:#}"),
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Open…").clicked()
                && let Some(p) = rfd::FileDialog::new().add_filter("Scene", &["toml"]).pick_file()
            {
                self.open(&p);
            }
            if ui.add_enabled(self.loaded.is_some(), egui::Button::new("Reload")).clicked()
                && let Some(p) = self.loaded.as_ref().map(|l| l.path.clone())
            {
                self.open(&p);
            }
        });
        if let Some(l) = &self.loaded {
            ui.label(l.path.file_name().unwrap_or_default().to_string_lossy());
            ui.weak(format!(
                "{} triangles, {} spheres, {} lights",
                l.world.triangles.len(),
                l.world.spheres.len(),
                l.world.lights.len()
            ));
        }
        ui.separator();

        let before = (self.render.clone(), self.photon.clone());
        let r = &mut self.render;
        egui::CollapsingHeader::new("Render").default_open(true).show(ui, |ui| {
            egui::Grid::new("render").num_columns(2).show(ui, |ui| {
                ui.label("Height");
                ui.add(DragValue::new(&mut r.height).range(16..=4096).suffix(" px"));
                ui.end_row();
                ui.label("Samples");
                ui.add(DragValue::new(&mut r.samples).range(1..=256));
                ui.end_row();
                ui.label("Max depth");
                ui.add(DragValue::new(&mut r.max_depth).range(0..=16));
                ui.end_row();
                ui.label("Background");
                let mut bg = r.background.map(|c| c as f32);
                if ui.color_edit_button_rgb(&mut bg).changed() {
                    r.background = bg.map(f64::from);
                }
                ui.end_row();
            });
            ui.checkbox(&mut r.checker_background, "Checker background");
            ui.checkbox(&mut r.reflections, "Reflections");
            ui.checkbox(&mut r.refractions, "Refractions");
            ui.checkbox(&mut r.shadows, "Shadows");
            ui.add_enabled_ui(r.shadows, |ui| {
                ui.checkbox(&mut r.soft_shadows, "Soft shadows");
                ui.add_enabled_ui(r.soft_shadows, |ui| {
                    egui::Grid::new("soft").num_columns(2).show(ui, |ui| {
                        ui.label("Rays (n²)");
                        ui.add(DragValue::new(&mut r.shadow_rays).range(1..=16));
                        ui.end_row();
                        ui.label("Jitter");
                        ui.add(DragValue::new(&mut r.shadow_jitter).range(0.0..=2.0).speed(0.01));
                        ui.end_row();
                    });
                });
            });
        });

        let p = &mut self.photon;
        egui::CollapsingHeader::new("Photon mapping").default_open(true).show(ui, |ui| {
            ui.checkbox(&mut p.enabled, "Enabled");
            ui.add_enabled_ui(p.enabled, |ui| {
                egui::Grid::new("photon").num_columns(2).show(ui, |ui| {
                    ui.label("Photons / light");
                    ui.add(DragValue::new(&mut p.photons_per_light).range(1_000..=10_000_000).speed(1000));
                    ui.end_row();
                    ui.label("Bounces");
                    ui.add(DragValue::new(&mut p.max_depth).range(0..=16));
                    ui.end_row();
                    ui.label("Light power");
                    ui.add(DragValue::new(&mut p.light_power).range(0.0..=1e6).speed(10.0));
                    ui.end_row();
                    ui.label("Direct light");
                    egui::ComboBox::from_id_salt("direct_lighting")
                        .selected_text(match p.direct_lighting {
                            DirectLighting::Rays => "Shadow rays",
                            DirectLighting::Map => "Photon map",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut p.direct_lighting, DirectLighting::Rays, "Shadow rays");
                            ui.selectable_value(&mut p.direct_lighting, DirectLighting::Map, "Photon map");
                        });
                    ui.end_row();
                    let direct_map = p.direct_lighting == DirectLighting::Map;
                    for (label, uses_map, radius, gather, exposure) in [
                        ("Direct", direct_map, &mut p.radius.direct, &mut p.gather.direct, &mut p.exposure.direct),
                        ("Indirect", true, &mut p.radius.indirect, &mut p.gather.indirect, &mut p.exposure.indirect),
                        ("Caustic", true, &mut p.radius.caustic, &mut p.gather.caustic, &mut p.exposure.caustic),
                    ] {
                        ui.add_enabled(uses_map, egui::Label::new(format!("{label} radius")))
                            .on_hover_text("Maximum search radius");
                        ui.add_enabled(uses_map, DragValue::new(radius).range(0.01..=10.0).speed(0.01));
                        ui.end_row();
                        ui.add_enabled(uses_map, egui::Label::new(format!("{label} photons")))
                            .on_hover_text("Nearest photons gathered; 0 takes all within the radius");
                        ui.add_enabled(uses_map, DragValue::new(gather).range(0..=5000).speed(1.0));
                        ui.end_row();
                        ui.label(format!("{label} exposure"));
                        ui.add(DragValue::new(exposure).range(0.0..=100.0).speed(0.01));
                        ui.end_row();
                    }
                });
            });
            if let Some((_, maps)) = &self.maps {
                ui.weak(format!(
                    "Cached maps: {} direct, {} indirect, {} caustic",
                    maps.direct.len(),
                    maps.indirect.len(),
                    maps.caustic.len()
                ));
            }
            ui.add_enabled_ui(self.maps.is_some(), |ui| {
                ui.checkbox(&mut self.show_photons, "Show photons");
                ui.add_enabled_ui(self.show_photons, |ui| {
                    let o = &mut self.overlay_opts;
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut o.direct, egui::RichText::new("Direct").color(overlay::DIRECT_COLOR));
                        ui.checkbox(&mut o.indirect, egui::RichText::new("Indirect").color(overlay::INDIRECT_COLOR));
                        ui.checkbox(&mut o.caustic, egui::RichText::new("Caustic").color(overlay::CAUSTIC_COLOR));
                    });
                    ui.checkbox(&mut o.hide_occluded, "Hide occluded");
                });
            });
        });
        if (self.render.clone(), self.photon.clone()) != before {
            self.needs_render = true;
        }

        egui::CollapsingHeader::new("Display").default_open(true).show(ui, |ui| {
            let before = self.display;
            let d = &mut self.display;
            egui::Grid::new("display").num_columns(2).show(ui, |ui| {
                ui.label("Exposure");
                ui.add(egui::Slider::new(&mut d.exposure, -6.0..=6.0).suffix(" EV"));
                ui.end_row();
                ui.label("Gamma");
                ui.add(egui::Slider::new(&mut d.gamma, 1.0..=3.0));
                ui.end_row();
            });
            ui.horizontal(|ui| {
                if ui.button("Original (γ 1.0)").clicked() {
                    *d = Display::default();
                }
                if ui.button("sRGB-ish (γ 2.2)").clicked() {
                    *d = Display { gamma: 2.2, ..*d };
                }
            });
            self.display_dirty |= self.display != before;
        });

        egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| {
            ui.weak("Drag: orbit · Right-drag or Shift-drag: pan · Scroll: zoom");
            if ui.add_enabled(self.camera.is_some(), egui::Button::new("Reset to scene camera")).clicked() {
                self.camera = None;
                self.needs_render = true;
            }
        });

        ui.separator();
        ui.checkbox(&mut self.auto_render, "Re-render on change");
        let running = self.job.as_ref().is_some_and(|j| j.is_running());
        ui.horizontal(|ui| {
            if ui.add_enabled(self.loaded.is_some(), egui::Button::new("Render")).clicked() {
                self.start_render(ui.ctx(), false);
            }
            if ui.add_enabled(running, egui::Button::new("Cancel")).clicked() {
                self.cancel();
            }
        });
        ui.horizontal(|ui| {
            let done = self.job.as_ref().is_some_and(|j| j.image.lock().unwrap().is_some());
            if ui.add_enabled(done, egui::Button::new("Save PNG…")).clicked() {
                self.save_png();
            }
            if ui.add_enabled(self.loaded.is_some(), egui::Button::new("Save scene…")).clicked() {
                self.save_scene();
            }
        });

        ui.separator();
        if let Some(job) = &self.job {
            let status = match job.stage() {
                Stage::Photons => "Tracing photons…".to_string(),
                Stage::Preview => "Preview…".to_string(),
                Stage::Rendering => format!("Rendering {}×{}", job.width, job.height),
                Stage::Done { photons, render } => match photons {
                    Some(p) => format!("Done: photons {p:.2?}, render {render:.2?}"),
                    None => format!("Done in {render:.2?}"),
                },
                Stage::Previewed => "Preview".to_string(),
                Stage::Cancelled => "Cancelled".to_string(),
            };
            ui.add(egui::ProgressBar::new(job.progress()).text(status));
        }
        if let Some(m) = &self.message {
            ui.label(m);
        }
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job(&ctx);

        egui::Panel::left("controls").resizable(true).default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });

        if self.needs_render && self.auto_render && self.last_camera_move.is_none() {
            self.start_render(&ctx, false);
        }

        egui::CentralPanel::default().show(ui, |ui| {
            let Some(tex) = &self.texture else {
                ui.centered_and_justified(|ui| ui.label("Open a scene (.toml) to render"));
                return;
            };
            // Fit the image in the panel, never upscaling beyond 2×.
            let avail = ui.available_rect_before_wrap();
            let size = tex.size_vec2();
            let scale = (avail.width() / size.x).min(avail.height() / size.y).min(2.0);
            let rect = egui::Rect::from_center_size(avail.center(), size * scale);
            ui.painter().rect_filled(avail, 0.0, Color32::from_gray(24));
            ui.painter().image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );

            let response = ui.allocate_rect(avail, egui::Sense::click_and_drag());
            self.navigate(ui, &response, rect);
            if self.show_photons {
                self.draw_overlay(ui, rect);
            }
        });
    }
}

impl Viewer {
    /// Mouse camera controls. Renders previews while moving and a full image
    /// once the camera settles.
    fn navigate(&mut self, ui: &egui::Ui, response: &egui::Response, image_rect: egui::Rect) {
        let Some(loaded) = &self.loaded else { return };
        let cam_def = &loaded.scene.file.camera;
        let mut camera = self.camera.unwrap_or(loaded.initial_orbit);
        let before = camera;

        let delta = response.drag_delta();
        if response.dragged() && delta != egui::Vec2::ZERO {
            let pan = response.dragged_by(PointerButton::Secondary)
                || response.dragged_by(PointerButton::Middle)
                || ui.input(|i| i.modifiers.shift);
            if pan {
                let k = camera.world_per_point(&cam_def.window, cam_def.near, image_rect.height() as f64);
                camera.pan(delta.x as f64, delta.y as f64, k);
            } else {
                camera.orbit(delta.x as f64, delta.y as f64);
            }
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                camera.zoom(scroll as f64);
            }
        }

        // Until the user moves it, `camera` stays None so renders use the
        // scene's exact view matrix.
        let moved = camera != before;
        if moved {
            self.camera = Some(camera);
            self.last_camera_move = Some(Instant::now());
            self.start_render(ui.ctx(), true);
        } else if let Some(t) = self.last_camera_move {
            if !response.dragged() && t.elapsed() >= SETTLE_TIME {
                self.last_camera_move = None;
                self.start_render(ui.ctx(), false);
            } else {
                ui.ctx().request_repaint_after(SETTLE_TIME);
            }
        }
    }

    fn draw_overlay(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        let (Some((_, maps)), Some(view), Some(loaded)) = (&self.maps, self.view(), &self.loaded) else { return };
        let key = (Arc::as_ptr(maps) as usize, view, self.overlay_opts);
        if self.overlay_key != Some(key) {
            let cam = &loaded.scene.file.camera;
            self.overlay = overlay::project(maps, &loaded.world, view, &cam.window, cam.near, self.overlay_opts);
            self.overlay_key = Some(key);
        }
        let mut mesh = egui::Mesh::default();
        let half = egui::vec2(1.0, 1.0);
        for p in &self.overlay {
            let center = rect.min + p.uv.to_vec2() * rect.size();
            mesh.add_colored_rect(egui::Rect::from_center_size(center, half * 2.0), p.color);
        }
        ui.painter().add(egui::Shape::mesh(mesh));
    }
}

fn load(path: &Path) -> Result<LoadedScene> {
    let scene = Scene::load(path).with_context(|| format!("loading {}", path.display()))?;
    let world = World::from_scene(&scene);
    let initial_orbit = OrbitCamera::from_view(&scene.file.camera, scene.view, &world);
    Ok(LoadedScene { path: path.to_path_buf(), scene: Arc::new(scene), world: Arc::new(world), initial_orbit })
}
