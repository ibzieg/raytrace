//! Whitted-style ray tracing with optional photon-map shading.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::Result;
use glam::{DMat4, DVec3};
use rayon::prelude::*;

use crate::geometry::{Hit, PointLight, Ray, World};
use crate::optics::{reflect, refract};
use crate::photon::{ESTIMATE_SCALE, PhotonMaps};
use crate::rng::Rng;
use crate::scene::{DirectLighting, PhotonSettings, RenderSettings, Scene, Window};

/// Linear RGB, rows stored top to bottom.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<DVec3>,
}

impl Image {
    /// 8-bit output with the original's mapping: clamp to [0, 1], no gamma.
    pub fn to_rgb8(&self) -> image::RgbImage {
        self.to_rgb8_with(Display::default())
    }

    pub fn to_rgb8_with(&self, display: Display) -> image::RgbImage {
        image::RgbImage::from_fn(self.width, self.height, |x, y| {
            image::Rgb(display.map(self.pixels[(y * self.width + x) as usize]))
        })
    }

    pub fn save_png(&self, path: &Path) -> Result<()> {
        self.save_png_with(path, Display::default())
    }

    pub fn save_png_with(&self, path: &Path, display: Display) -> Result<()> {
        self.to_rgb8_with(display).save(path)?;
        Ok(())
    }
}

/// Maps linear radiance to 8-bit display values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    /// Exposure in stops: values are scaled by 2^exposure.
    pub exposure: f64,
    /// 1.0 reproduces the original renderer's look; 2.2 approximates sRGB.
    pub gamma: f64,
}

impl Default for Display {
    fn default() -> Self {
        Self { exposure: 0.0, gamma: 1.0 }
    }
}

impl Display {
    pub fn map(&self, c: DVec3) -> [u8; 3] {
        let c = (c * self.exposure.exp2()).clamp(DVec3::ZERO, DVec3::ONE);
        let c = if self.gamma == 1.0 { c } else { c.powf(1.0 / self.gamma) };
        (c * 255.0).round().to_array().map(|v| v as u8)
    }
}

pub struct Renderer<'a> {
    pub world: &'a World,
    pub settings: &'a RenderSettings,
    pub photons: Option<&'a PhotonMaps>,
    /// How the photon maps are read.
    pub photon_settings: &'a PhotonSettings,
    /// World-to-camera matrix.
    pub view: DMat4,
    pub window: Window,
    pub near: f64,
    pub far: f64,
}

impl<'a> Renderer<'a> {
    pub fn new(scene: &'a Scene, world: &'a World, photons: Option<&'a PhotonMaps>) -> Self {
        let cam = &scene.file.camera;
        Self {
            world,
            settings: &scene.file.render,
            photons,
            photon_settings: &scene.file.photon,
            view: scene.view,
            window: cam.window,
            near: cam.near,
            far: cam.far,
        }
    }

    /// Renders all rows in parallel. `progress(rows_done, rows_total)` is
    /// called from worker threads as rows finish.
    pub fn render(&self, width: u32, height: u32, progress: &(dyn Fn(usize, usize) + Sync)) -> Image {
        let done = AtomicUsize::new(0);
        self.render_with(width, height, None, &|_, _| {
            progress(done.fetch_add(1, Ordering::Relaxed) + 1, height as usize)
        })
        .expect("render without a cancel flag always completes")
    }

    /// Like [`Renderer::render`], but calls `on_row(row, pixels)` with each
    /// finished row (top-down index, any order) and stops early once `cancel`
    /// is set, returning None.
    pub fn render_with(
        &self,
        width: u32,
        height: u32,
        cancel: Option<&AtomicBool>,
        on_row: &(dyn Fn(usize, &[DVec3]) + Sync),
    ) -> Option<Image> {
        let w = width as usize;
        let mut pixels = vec![DVec3::ZERO; w * height as usize];
        let px = DVec3::new(
            (self.window.right - self.window.left) / width as f64,
            (self.window.top - self.window.bottom) / height as f64,
            0.0,
        );
        let camera_to_world = self.view.inverse();
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));

        pixels.par_chunks_mut(w).enumerate().for_each(|(row, out)| {
            if cancelled() {
                return;
            }
            // Image rows run top-down; camera y runs bottom-up.
            let y = height as usize - 1 - row;
            for (x, pixel) in out.iter_mut().enumerate() {
                let mut rng = Rng::new(((y as u64) << 32) | x as u64);
                let corner =
                    DVec3::new(self.window.left + x as f64 * px.x, self.window.bottom + y as f64 * px.y, -self.near);
                *pixel = self.pixel(corner, px, &camera_to_world, &mut rng);
            }
            on_row(row, out);
        });

        (!cancelled()).then_some(Image { width, height, pixels })
    }

    /// `corner` and `px` are in camera space.
    fn pixel(&self, corner: DVec3, px: DVec3, camera_to_world: &DMat4, rng: &mut Rng) -> DVec3 {
        let eye = camera_to_world.transform_point3(DVec3::ZERO);
        let primary = |p: DVec3| Ray::new(eye, camera_to_world.transform_vector3(p));
        let n = self.settings.samples.max(1) as usize;
        if n == 1 {
            return self.trace(&primary(corner + px * 0.5), 0, false);
        }
        // N-rooks: one sample per row and per column of an n×n sub-grid.
        let mut cols: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            cols.swap(i, rng.below(i + 1));
        }
        let mut sum = DVec3::ZERO;
        for (i, &j) in cols.iter().enumerate() {
            let offset =
                DVec3::new((i as f64 + rng.next_f64()) / n as f64, (j as f64 + rng.next_f64()) / n as f64, 0.0);
            sum += self.trace(&primary(corner + px * offset), 0, false);
        }
        sum / n as f64
    }

    /// Color seen along `ray`; the background when nothing is hit.
    pub fn trace(&self, ray: &Ray, depth: u32, internal: bool) -> DVec3 {
        let Some(hit) = self.world.intersect(ray, f64::INFINITY) else {
            return self.background(ray);
        };
        let s = self.settings;
        let m = &self.world.materials[hit.material];

        let (base, specular) = match self.photons {
            Some(maps) => {
                let estimate = maps.irradiance(&hit, m.kd, self.photon_settings);
                match self.photon_settings.direct_lighting {
                    DirectLighting::Rays => {
                        let (diffuse, specular) = self.direct_from_photon_power(ray, &hit);
                        (estimate + diffuse, specular)
                    }
                    DirectLighting::Map => (estimate, DVec3::ZERO),
                }
            }
            None => self.shade(ray, &hit),
        };

        let mut reflect_color = DVec3::ZERO;
        let mut reflect_factor = 0.0;
        // No reflections from inside an object, as in the original.
        if s.reflections && !internal && depth < s.max_depth && m.reflectivity > 0.0 {
            reflect_color = self.trace(&reflect(ray, &hit), depth + 1, false);
            reflect_factor = m.reflectivity;
        }

        let mut refract_color = DVec3::ZERO;
        let mut refract_factor = 0.0;
        if s.refractions && depth < s.max_depth && m.transmissivity > 0.0 {
            let (refracted, now_internal) = refract(ray, &hit, internal, m.effective_ior());
            refract_color = self.trace(&refracted, depth + 1, now_internal);
            refract_factor = m.transmissivity;
        }

        // Specular highlights go on top, unscaled.
        base * (1.0 - reflect_factor - refract_factor)
            + refract_color * refract_factor
            + reflect_color * reflect_factor
            + specular
    }

    /// Phong shading. Returns (ambient + diffuse, specular).
    fn shade(&self, ray: &Ray, hit: &Hit) -> (DVec3, DVec3) {
        let (diffuse, specular) = self.direct(ray, hit, |light, _| (light.diffuse, light.specular));
        // Ambient is the material's Ka alone; light ambient terms were unused.
        (self.world.materials[hit.material].ka + diffuse, specular)
    }

    /// Phong lighting from each light's share of the photon power, falling off
    /// with distance, on the same scale as the photon map estimates. No
    /// ambient term: the indirect map provides that light.
    fn direct_from_photon_power(&self, ray: &Ray, hit: &Hit) -> (DVec3, DVec3) {
        let p = self.photon_settings;
        let per_light = p.light_power / self.world.lights.len().max(1) as f64;
        let scale = per_light / (4.0 * std::f64::consts::PI) * ESTIMATE_SCALE * p.exposure.direct;
        self.direct(ray, hit, |_, dist| {
            let intensity = DVec3::splat(scale / (dist * dist));
            (intensity, intensity)
        })
    }

    /// Diffuse and specular light from each visible light, where
    /// `intensity(light, distance)` gives its diffuse and specular colors.
    fn direct(&self, ray: &Ray, hit: &Hit, intensity: impl Fn(&PointLight, f64) -> (DVec3, DVec3)) -> (DVec3, DVec3) {
        let m = &self.world.materials[hit.material];
        let n = hit.normal;
        let v = -ray.dir;
        let mut diffuse = DVec3::ZERO;
        let mut specular = DVec3::ZERO;

        for light in &self.world.lights {
            let to_light = light.position - hit.point;
            let dist = to_light.length();
            let l = to_light / dist;

            let visibility = if !self.settings.shadows || !self.occluded(hit.point, l, dist) {
                1.0
            } else if self.settings.soft_shadows {
                self.penumbra(hit, light.position)
            } else {
                continue;
            };

            let ln = l.dot(n).max(0.0);
            let r = crate::optics::reflect_dir(-l, n);
            let rv = if m.shininess > 0.0 { r.dot(v).max(0.0).powf(m.shininess) } else { 0.0 };
            let (light_diffuse, light_specular) = intensity(light, dist);
            diffuse += m.kd * light_diffuse * ln * visibility;
            specular += m.ks * light_specular * rv * visibility;
        }
        (diffuse, specular)
    }

    fn occluded(&self, from: DVec3, dir: DVec3, dist: f64) -> bool {
        self.world.intersect(&Ray { origin: from, dir }, dist).is_some()
    }

    /// Fraction of the light visible from a disc of sample points around the
    /// hit point in its tangent plane. Only called when the center is in
    /// shadow, so it softens the inside edge of shadows.
    fn penumbra(&self, hit: &Hit, light: DVec3) -> f64 {
        let dim = self.settings.shadow_rays.max(1);
        let spread = self.settings.shadow_jitter;
        let (t, b) = hit.normal.any_orthonormal_pair();
        let mut visible = 0;
        for i in 0..dim {
            for j in 0..dim {
                // The original used j/dim radians (under 60°), likely meant to
                // be a full turn.
                let angle = std::f64::consts::TAU * j as f64 / dim as f64;
                let radius = spread * i as f64;
                let p = hit.point + t * (radius * angle.sin()) + b * (radius * angle.cos());
                let to_light = light - p;
                let dist = to_light.length();
                if !self.occluded(p, to_light / dist, dist) {
                    visible += 1;
                }
            }
        }
        visible as f64 / (dim * dim) as f64
    }

    fn background(&self, ray: &Ray) -> DVec3 {
        let clear = DVec3::from(self.settings.background);
        if !self.settings.checker_background {
            return clear;
        }
        // Checkerboard on the camera's far plane.
        let ray = Ray { origin: self.view.transform_point3(ray.origin), dir: self.view.transform_vector3(ray.dir) };
        if ray.dir.z >= 0.0 {
            return clear;
        }
        const CHECKER_SIZE: f64 = 125.0;
        let t = (-self.far - ray.origin.z) / ray.dir.z;
        let p = ray.at(t);
        let cx = (p.x.abs() / CHECKER_SIZE + 0.5) as i64;
        let cy = (p.y.abs() / CHECKER_SIZE + 0.5) as i64;
        if cx % 2 == cy % 2 { clear } else { DVec3::ONE - clear }
    }
}
