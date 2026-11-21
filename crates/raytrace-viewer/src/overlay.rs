//! Projects photons onto the image so their density and spread can be seen
//! when tuning the gather radii.

use eframe::egui::{Color32, Pos2, pos2};
use glam::{DMat4, DVec3};
use rayon::prelude::*;
use raytrace::geometry::Ray;
use raytrace::scene::Window;
use raytrace::{PhotonMaps, World};

/// Photons drawn per map; larger maps are subsampled evenly.
const MAX_PER_MAP: usize = 25_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OverlayOptions {
    pub direct: bool,
    pub indirect: bool,
    pub caustic: bool,
    /// Hide photons not visible from the camera.
    pub hide_occluded: bool,
}

impl Default for OverlayOptions {
    fn default() -> Self {
        Self { direct: true, indirect: true, caustic: true, hide_occluded: true }
    }
}

pub const DIRECT_COLOR: Color32 = Color32::from_rgb(255, 210, 60);
pub const INDIRECT_COLOR: Color32 = Color32::from_rgb(80, 220, 120);
pub const CAUSTIC_COLOR: Color32 = Color32::from_rgb(240, 80, 240);

/// A photon's position in normalized image coordinates (0..1, top-left
/// origin) and its color.
pub struct OverlayPoint {
    pub uv: Pos2,
    pub color: Color32,
}

/// Normalized image coordinates (0..1, top-left origin) of a world point, or
/// None if it is behind the camera or outside the image.
pub fn project_point(view: DMat4, window: &Window, near: f64, p: DVec3) -> Option<(f64, f64)> {
    let c = view.transform_point3(p);
    if c.z >= -near * 1e-3 {
        return None;
    }
    let s = near / -c.z;
    let u = (c.x * s - window.left) / (window.right - window.left);
    let v = (window.top - c.y * s) / (window.top - window.bottom);
    ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some((u, v))
}

pub fn project(
    maps: &PhotonMaps,
    world: &World,
    view: DMat4,
    window: &Window,
    near: f64,
    opts: OverlayOptions,
) -> Vec<OverlayPoint> {
    let eye = view.inverse().transform_point3(DVec3::ZERO);
    let layers = [
        (opts.direct, &maps.direct, DIRECT_COLOR),
        (opts.indirect, &maps.indirect, INDIRECT_COLOR),
        (opts.caustic, &maps.caustic, CAUSTIC_COLOR),
    ];
    layers
        .into_iter()
        .filter(|(on, _, _)| *on)
        .flat_map(|(_, map, color)| {
            let photons = map.photons();
            let step = photons.len().div_ceil(MAX_PER_MAP).max(1);
            photons
                .par_iter()
                .step_by(step)
                .filter_map(|p| {
                    let (u, v) = project_point(view, window, near, p.pos)?;
                    if opts.hide_occluded {
                        let to_photon = p.pos - eye;
                        let dist = to_photon.length();
                        // Photons sit on surfaces, so stop just short of them.
                        let blocker = world.intersect(&Ray::new(eye, to_photon), dist * (1.0 - 1e-4));
                        if blocker.is_some() {
                            return None;
                        }
                    }
                    Some(OverlayPoint { uv: pos2(u as f32, v as f32), color })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use raytrace::scene::Window;

    use super::*;

    /// The overlay must put a world point exactly where the renderer's
    /// primary ray for that pixel lands.
    #[test]
    fn projection_matches_primary_rays() {
        let window = Window { left: -2.0, right: 3.0, bottom: -1.0, top: 1.5 };
        let near = 4.0;
        let view = raytrace::scene::look_at(DVec3::new(1.0, 2.0, 8.0), DVec3::new(0.0, 0.5, 0.0), DVec3::Y);
        let camera_to_world = view.inverse();
        let (w, h) = (40.0, 20.0);
        for (x, y) in [(0.5, 0.5), (12.5, 3.5), (39.5, 19.5)] {
            // Renderer convention: camera y runs bottom-up from window.bottom.
            let on_near = DVec3::new(
                window.left + x / w * (window.right - window.left),
                window.bottom + y / h * (window.top - window.bottom),
                -near,
            );
            let point = camera_to_world.transform_point3(on_near * 2.5);
            let (u, v) = project_point(view, &window, near, point).unwrap();
            assert!((u - x / w).abs() < 1e-9, "u {u} vs {}", x / w);
            assert!((v - (1.0 - y / h)).abs() < 1e-9, "v {v} vs {}", 1.0 - y / h);
        }
        // Behind the camera.
        assert!(project_point(view, &window, near, camera_to_world.transform_point3(DVec3::Z)).is_none());
    }
}
