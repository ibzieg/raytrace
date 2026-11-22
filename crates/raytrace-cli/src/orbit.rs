//! Camera paths around a scene's target for animations.

use glam::{DMat4, DQuat, DVec3};
use raytrace::scene::{CameraDef, look_at};
use raytrace::{Scene, World};

/// The scene camera, as an eye circling a pivot about the camera's up axis.
pub struct Orbit {
    eye: DVec3,
    target: DVec3,
    up: DVec3,
}

impl Orbit {
    /// The pivot is the camera's target, or if the scene gives a view matrix,
    /// the point ahead of the camera at the depth of the scene's center.
    pub fn new(scene: &Scene, world: &World) -> Self {
        let camera_to_world = scene.view.inverse();
        let eye = camera_to_world.transform_point3(DVec3::ZERO);
        let up = camera_to_world.transform_vector3(DVec3::Y).normalize();
        let target = target(&scene.file.camera, camera_to_world, eye, world);
        Self { eye, target, up }
    }

    /// The view with the eye turned `degrees` about the pivot (positive turns
    /// the camera to its right).
    pub fn view(&self, degrees: f64) -> DMat4 {
        let turn = DQuat::from_axis_angle(self.up, -degrees.to_radians());
        look_at(self.target + turn * (self.eye - self.target), self.target, self.up)
    }
}

fn target(def: &CameraDef, camera_to_world: DMat4, eye: DVec3, world: &World) -> DVec3 {
    if let Some(t) = def.target {
        return t.into();
    }
    let forward = camera_to_world.transform_vector3(DVec3::NEG_Z).normalize();
    let points: Vec<DVec3> =
        world.triangles.iter().map(|t| t.v0).chain(world.spheres.iter().map(|s| s.center)).collect();
    let center = if points.is_empty() { eye + forward } else { points.iter().sum::<DVec3>() / points.len() as f64 };
    let depth = (center - eye).dot(forward);
    eye + forward * if depth > 1e-3 { depth } else { 10.0 }
}

/// Camera angle in degrees for each frame.
pub fn angles(frames: usize, degrees: f64, turntable: bool) -> Vec<f64> {
    (0..frames)
        .map(|i| {
            if turntable {
                // Stops one step short of a full sweep so the loop is seamless.
                degrees * i as f64 / frames as f64
            } else {
                // Swing across the start view, easing in and out.
                let t = if frames > 1 { i as f64 / (frames - 1) as f64 } else { 0.5 };
                let s = 0.5 - 0.5 * (std::f64::consts::PI * t).cos();
                degrees * (s - 0.5)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swing_is_centered_and_turntable_loops() {
        let swing = angles(5, 40.0, false);
        assert_eq!(swing[0], -20.0);
        assert!((swing[2]).abs() < 1e-12);
        assert!((swing[4] - 20.0).abs() < 1e-12);
        assert_eq!(angles(4, 360.0, true), vec![0.0, 90.0, 180.0, 270.0]);
    }

    #[test]
    fn zero_angle_keeps_the_view_and_turning_keeps_the_distance() {
        let orbit = Orbit { eye: DVec3::new(3.0, 2.0, 10.0), target: DVec3::new(0.0, 1.0, 0.0), up: DVec3::Y };
        let start = look_at(orbit.eye, orbit.target, orbit.up);
        assert!(orbit.view(0.0).abs_diff_eq(start, 1e-12));
        let turned_eye = orbit.view(90.0).inverse().transform_point3(DVec3::ZERO);
        assert!((turned_eye.distance(orbit.target) - orbit.eye.distance(orbit.target)).abs() < 1e-9);
        assert!((turned_eye.y - orbit.eye.y).abs() < 1e-9);
    }
}
