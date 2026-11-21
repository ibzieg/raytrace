//! Orbit camera for mouse navigation. World +Y is up.

use std::f64::consts::FRAC_PI_2;

use glam::{DMat4, DVec3};
use raytrace::World;
use raytrace::scene::{CameraDef, Window, look_at};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    pub target: DVec3,
    pub distance: f64,
    /// Rotation about +Y; 0 looks down -Z.
    pub yaw: f64,
    pub pitch: f64,
}

const MAX_PITCH: f64 = FRAC_PI_2 - 0.01;

impl OrbitCamera {
    /// Starts from the scene camera. The orbit pivot is the camera target if
    /// the scene has one, otherwise the point in front of the eye at the depth
    /// of the scene's center.
    pub fn from_view(def: &CameraDef, view: DMat4, world: &World) -> Self {
        let camera_to_world = view.inverse();
        let eye = camera_to_world.transform_point3(DVec3::ZERO);
        let forward = camera_to_world.transform_vector3(DVec3::NEG_Z).normalize();
        let target = def.target.map(DVec3::from).unwrap_or_else(|| {
            let depth = (scene_center(world) - eye).dot(forward);
            eye + forward * if depth > 1e-3 { depth } else { 10.0 }
        });
        let offset = eye - target;
        let distance = offset.length().max(1e-3);
        Self {
            target,
            distance,
            yaw: offset.x.atan2(offset.z),
            pitch: (offset.y / distance).clamp(-1.0, 1.0).asin().clamp(-MAX_PITCH, MAX_PITCH),
        }
    }

    pub fn eye(&self) -> DVec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + DVec3::new(sy * cp, sp, cy * cp) * self.distance
    }

    pub fn view(&self) -> DMat4 {
        look_at(self.eye(), self.target, DVec3::Y)
    }

    /// Rotate by a mouse drag of (dx, dy) points.
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        const SPEED: f64 = 0.008;
        self.yaw -= dx * SPEED;
        self.pitch = (self.pitch + dy * SPEED).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Move the pivot so the scene follows the mouse. `world_per_point` is the
    /// size of one screen point at the target's depth.
    pub fn pan(&mut self, dx: f64, dy: f64, world_per_point: f64) {
        let camera_to_world = self.view().inverse();
        let right = camera_to_world.transform_vector3(DVec3::X);
        let up = camera_to_world.transform_vector3(DVec3::Y);
        self.target += (-right * dx + up * dy) * world_per_point;
    }

    /// Dolly toward the target; positive `amount` zooms in.
    pub fn zoom(&mut self, amount: f64) {
        self.distance = (self.distance * (-amount * 0.002).exp()).max(0.05);
    }

    /// World size of one screen point at the target's depth, for an image
    /// displayed `display_height` points tall.
    pub fn world_per_point(&self, window: &Window, near: f64, display_height: f64) -> f64 {
        (window.top - window.bottom) / near * self.distance / display_height
    }

    pub fn write_to(&self, def: &mut CameraDef) {
        def.eye = Some(self.eye().into());
        def.target = Some(self.target.into());
        def.up = Some([0.0, 1.0, 0.0]);
        def.view_matrix = None;
    }
}

fn scene_center(world: &World) -> DVec3 {
    let points: Vec<DVec3> =
        world.triangles.iter().map(|t| t.v0).chain(world.spheres.iter().map(|s| s.center)).collect();
    if points.is_empty() { DVec3::ZERO } else { points.iter().sum::<DVec3>() / points.len() as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_round_trips_through_orbit() {
        let def = CameraDef {
            eye: Some([3.0, 2.0, 10.0]),
            target: Some([0.0, 0.0, -1.0]),
            up: Some([0.0, 1.0, 0.0]),
            view_matrix: None,
            near: 1.0,
            far: 100.0,
            window: Window { left: -1.0, right: 1.0, bottom: -1.0, top: 1.0 },
        };
        let view = def.view_matrix().unwrap();
        let world = World::empty();
        let orbit = OrbitCamera::from_view(&def, view, &world);
        assert!(orbit.view().abs_diff_eq(view, 1e-9), "{:?} vs {view:?}", orbit.view());
    }
}
