//! Lights: points, glowing spheres and one-sided rectangular panels.
//!
//! Emission is described per unit of the light's power: a point sends
//! 1 / (4π) per steradian in every direction; a point on an area light's
//! surface sends cos θ / π, θ measured from the surface normal, like a matte
//! emitter.

use std::f64::consts::{PI, TAU};

use glam::DVec3;

use crate::geometry::{EPSILON, Ray};
use crate::scene::LightDef;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightShape {
    Point,
    Sphere {
        radius: f64,
    },
    /// A rectangle facing `normal`, spanning ±`half_u` and ±`half_v` around
    /// the light's position.
    Rect {
        half_u: DVec3,
        half_v: DVec3,
        normal: DVec3,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub position: DVec3,
    pub diffuse: DVec3,
    pub specular: DVec3,
    pub shape: LightShape,
}

/// A point on a light chosen for a shadow ray or photon.
pub struct LightSample {
    pub point: DVec3,
    /// Surface normal there; None for a point light.
    pub normal: Option<DVec3>,
    /// Share of the light's surface this sample stands for, relative to
    /// sampling all of it (1/2 when only the near half of a sphere is used).
    pub weight: f64,
}

impl Light {
    pub fn point(position: DVec3, diffuse: DVec3, specular: DVec3) -> Self {
        Self { position, diffuse, specular, shape: LightShape::Point }
    }

    pub fn from_def(def: &LightDef) -> Self {
        let shape = if let Some(radius) = def.radius {
            LightShape::Sphere { radius }
        } else if let Some([w, h]) = def.size {
            let normal = DVec3::from(def.normal).normalize();
            // Width runs along X where the panel allows, otherwise along Z.
            let axis = if normal.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            let u = (axis - normal * axis.dot(normal)).normalize();
            let v = normal.cross(u);
            LightShape::Rect { half_u: u * (w / 2.0), half_v: v * (h / 2.0), normal }
        } else {
            LightShape::Point
        };
        Self { position: def.position.into(), diffuse: def.diffuse.into(), specular: def.specular.into(), shape }
    }

    pub fn is_point(&self) -> bool {
        self.shape == LightShape::Point
    }

    pub fn area(&self) -> f64 {
        match self.shape {
            LightShape::Point => 0.0,
            LightShape::Sphere { radius } => 4.0 * PI * radius * radius,
            LightShape::Rect { half_u, half_v, .. } => 4.0 * half_u.length() * half_v.length(),
        }
    }

    /// The point for (u, v) in [0, 1)². For a sphere, `toward` picks the half
    /// facing that point; None samples the whole sphere.
    pub fn sample(&self, u: f64, v: f64, toward: Option<DVec3>) -> LightSample {
        match self.shape {
            LightShape::Point => LightSample { point: self.position, normal: None, weight: 1.0 },
            LightShape::Sphere { radius } => {
                let (z, weight) = if toward.is_some() { (u, 0.5) } else { (1.0 - 2.0 * u, 1.0) };
                let r = (1.0 - z * z).max(0.0).sqrt();
                let (sin, cos) = (TAU * v).sin_cos();
                let local = DVec3::new(r * cos, r * sin, z);
                let normal = match toward {
                    Some(p) => {
                        let axis = (p - self.position).normalize_or(DVec3::Z);
                        let (t, b) = axis.any_orthonormal_pair();
                        t * local.x + b * local.y + axis * local.z
                    }
                    None => local,
                };
                LightSample { point: self.position + normal * radius, normal: Some(normal), weight }
            }
            LightShape::Rect { half_u, half_v, normal } => LightSample {
                point: self.position + half_u * (2.0 * u - 1.0) + half_v * (2.0 * v - 1.0),
                normal: Some(normal),
                weight: 1.0,
            },
        }
    }

    /// Emission per unit power per steradian leaving `sample` along `dir`.
    pub fn emission(&self, sample: &LightSample, dir: DVec3) -> f64 {
        match sample.normal {
            None => 1.0 / (4.0 * PI),
            Some(n) => n.dot(dir).max(0.0) / PI * sample.weight,
        }
    }

    /// The largest emission any direction gets: what a Phong-shaded scene
    /// treats as full brightness.
    pub fn peak_emission(&self) -> f64 {
        match self.shape {
            LightShape::Point | LightShape::Sphere { .. } => 1.0 / (4.0 * PI),
            LightShape::Rect { .. } => 1.0 / PI,
        }
    }

    /// Distance along `ray` to the visible surface of an area light. Panels
    /// are only visible from the front.
    pub fn intersect(&self, ray: &Ray, t_max: f64) -> Option<f64> {
        match self.shape {
            LightShape::Point => None,
            LightShape::Sphere { radius } => {
                let oc = ray.origin - self.position;
                let b = oc.dot(ray.dir);
                let disc = b * b - (oc.length_squared() - radius * radius);
                if disc < 0.0 {
                    return None;
                }
                let sq = disc.sqrt();
                [-b - sq, -b + sq].into_iter().find(|&t| t > EPSILON && t < t_max)
            }
            LightShape::Rect { half_u, half_v, normal } => {
                let denom = ray.dir.dot(normal);
                if denom >= 0.0 {
                    return None;
                }
                let t = (self.position - ray.origin).dot(normal) / denom;
                if t <= EPSILON || t >= t_max {
                    return None;
                }
                let local = ray.at(t) - self.position;
                let inside = |half: DVec3| local.dot(half).abs() <= half.length_squared();
                (inside(half_u) && inside(half_v)).then_some(t)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> Light {
        Light {
            position: DVec3::new(0.0, 5.0, 0.0),
            diffuse: DVec3::ONE,
            specular: DVec3::ONE,
            shape: LightShape::Rect { half_u: DVec3::X, half_v: DVec3::Z * 0.5, normal: DVec3::NEG_Y },
        }
    }

    /// Averaged over its surface and all directions, every shape emits its
    /// whole power exactly once.
    #[test]
    fn emission_integrates_to_one() {
        let sphere = Light { shape: LightShape::Sphere { radius: 0.7 }, ..panel() };
        for light in [panel(), sphere] {
            let n = 10;
            let mut total = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let s = light.sample((i as f64 + 0.5) / n as f64, (j as f64 + 0.37) / n as f64, None);
                    // Integrate over directions on a fine equal-area grid.
                    let m = 64;
                    for a in 0..m {
                        for b in 0..m {
                            let z = 1.0 - 2.0 * (a as f64 + 0.5) / m as f64;
                            let phi = TAU * (b as f64 + 0.5) / m as f64;
                            let r = (1.0 - z * z).sqrt();
                            let dir = DVec3::new(r * phi.cos(), r * phi.sin(), z);
                            total += light.emission(&s, dir) * 4.0 * PI / (m * m) as f64;
                        }
                    }
                }
            }
            let mean = total / (n * n) as f64;
            assert!((mean - 1.0).abs() < 0.01, "{:?}: {mean}", light.shape);
        }
    }

    /// Seen from afar, a sphere light sampled on its near half shines like a
    /// point light of the same power.
    #[test]
    fn near_half_sampling_matches_a_point_from_afar() {
        let light = Light { shape: LightShape::Sphere { radius: 0.5 }, ..panel() };
        let p = light.position + DVec3::new(30.0, -40.0, 0.0);
        let n = 64;
        let mut sum = 0.0;
        for i in 0..n * n {
            let s = light.sample(((i % n) as f64 + 0.5) / n as f64, ((i / n) as f64 + 0.5) / n as f64, Some(p));
            let to = p - s.point;
            sum += light.emission(&s, to.normalize()) / to.length_squared();
        }
        let expected = 1.0 / (4.0 * PI * (p - light.position).length_squared());
        let got = sum / (n * n) as f64;
        assert!((got / expected - 1.0).abs() < 0.02, "{got} vs {expected}");
    }

    #[test]
    fn panel_is_hit_only_from_the_front_and_within_its_edges() {
        let light = panel();
        let down = |x: f64, z: f64| Ray::new(DVec3::new(x, 0.0, z), DVec3::Y);
        assert!((light.intersect(&down(0.9, 0.4), f64::INFINITY).unwrap() - 5.0).abs() < 1e-12);
        assert!(light.intersect(&down(1.1, 0.0), f64::INFINITY).is_none());
        assert!(light.intersect(&down(0.0, 0.6), f64::INFINITY).is_none());
        let from_above = Ray::new(DVec3::new(0.0, 9.0, 0.0), DVec3::NEG_Y);
        assert!(light.intersect(&from_above, f64::INFINITY).is_none());
        assert!((light.area() - 2.0).abs() < 1e-12);
    }
}
