use glam::DVec3;

use crate::geometry::{Hit, Ray};

pub fn reflect_dir(d: DVec3, n: DVec3) -> DVec3 {
    d - n * (2.0 * d.dot(n))
}

pub fn reflect(ray: &Ray, hit: &Hit) -> Ray {
    Ray::new(hit.point, reflect_dir(ray.dir, hit.normal))
}

/// Refracts through the surface. `internal` says whether the ray is currently
/// inside the object. Returns the new ray and whether it is inside afterward.
///
/// Like the original, inside/outside is tracked by flag rather than by the
/// normal's orientation. On total internal reflection (which produced NaNs in
/// the original) the ray reflects and stays inside.
pub fn refract(ray: &Ray, hit: &Hit, internal: bool, ior: f64) -> (Ray, bool) {
    let (n, eta) = if internal { (-hit.normal, ior) } else { (hit.normal, 1.0 / ior) };
    let d = ray.dir;
    let dn = d.dot(n);
    let k = 1.0 - eta * eta * (1.0 - dn * dn);
    if k < 0.0 {
        return (Ray::new(hit.point, reflect_dir(d, n)), internal);
    }
    let t = (d - n * dn) * eta - n * k.sqrt();
    (Ray::new(hit.point, t), !internal)
}
