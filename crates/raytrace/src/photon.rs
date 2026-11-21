//! Photon mapping: emission from point lights, and three kd-trees (direct,
//! indirect, caustic) searched with a fixed radius.

use glam::DVec3;
use rayon::prelude::*;

use crate::geometry::{Hit, Ray, World};
use crate::optics::{reflect, refract};
use crate::rng::Rng;
use crate::scene::PhotonSettings;

#[derive(Clone, Copy, Debug)]
pub struct Photon {
    pub pos: DVec3,
    pub power: DVec3,
    /// Split axis when this photon is an interior kd-tree node.
    axis: u8,
}

/// Balanced kd-tree stored implicitly: for any range `lo..hi`, the node is at
/// the midpoint and its subtrees are the two halves on either side.
#[derive(Default)]
pub struct KdTree {
    photons: Vec<Photon>,
}

impl KdTree {
    pub fn build(mut photons: Vec<Photon>) -> Self {
        build_range(&mut photons);
        Self { photons }
    }

    pub fn len(&self) -> usize {
        self.photons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.photons.is_empty()
    }

    /// Calls `f(photon, squared_distance)` for every photon within `radius`
    /// of `point`.
    pub fn for_each_within(&self, point: DVec3, radius: f64, mut f: impl FnMut(&Photon, f64)) {
        let r2 = radius * radius;
        let mut stack = vec![(0usize, self.photons.len())];
        while let Some((lo, hi)) = stack.pop() {
            if lo >= hi {
                continue;
            }
            let mid = (lo + hi) / 2;
            let p = &self.photons[mid];
            let d2 = p.pos.distance_squared(point);
            if d2 < r2 {
                f(p, d2);
            }
            let delta = point[p.axis as usize] - p.pos[p.axis as usize];
            let (near, far) = if delta < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
            if delta * delta < r2 {
                stack.push(far);
            }
            stack.push(near);
        }
    }
}

fn build_range(photons: &mut [Photon]) {
    if photons.len() <= 1 {
        return;
    }
    let (min, max) =
        photons.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), p| (lo.min(p.pos), hi.max(p.pos)));
    let extent = max - min;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let mid = photons.len() / 2;
    photons.select_nth_unstable_by(mid, |a, b| a.pos[axis].total_cmp(&b.pos[axis]));
    photons[mid].axis = axis as u8;
    let (left, rest) = photons.split_at_mut(mid);
    rayon::join(|| build_range(left), || build_range(&mut rest[1..]));
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Direct,
    Indirect,
    Caustic,
}

#[derive(Default)]
struct Buckets {
    direct: Vec<Photon>,
    indirect: Vec<Photon>,
    caustic: Vec<Photon>,
}

impl Buckets {
    fn merge(mut self, mut o: Self) -> Self {
        self.direct.append(&mut o.direct);
        self.indirect.append(&mut o.indirect);
        self.caustic.append(&mut o.caustic);
        self
    }
}

/// How far a photon sits off the surface plane and still counts, as a fraction
/// of the search radius. Flattens the search sphere into a disc so photons on
/// nearby perpendicular surfaces don't bleed in.
const SEARCH_COMPRESSION: f64 = 0.22;

pub struct PhotonMaps {
    pub direct: KdTree,
    pub indirect: KdTree,
    pub caustic: KdTree,
    settings: PhotonSettings,
}

impl PhotonMaps {
    pub fn build(world: &World, settings: &PhotonSettings, seed: u64) -> Self {
        let per_light = settings.photons_per_light as usize;
        let total = per_light * world.lights.len();
        let power = DVec3::splat(settings.light_power / total.max(1) as f64);

        let buckets = (0..total)
            .into_par_iter()
            .fold(Buckets::default, |mut out, i| {
                let light = &world.lights[i / per_light];
                let mut rng = Rng::new(seed.wrapping_mul(0x1000_0000_01B3) ^ i as u64);
                // The original biased photons downward, but the flip was a
                // no-op (`dir.sMult(-1)` discarded its result), so emission was
                // effectively uniform; keep it uniform.
                let ray = Ray::new(light.position, rng.unit_vector());
                trace_photon(world, settings, &ray, 0, false, power, Kind::Direct, &mut rng, &mut out);
                out
            })
            .reduce(Buckets::default, Buckets::merge);

        Self {
            direct: KdTree::build(buckets.direct),
            indirect: KdTree::build(buckets.indirect),
            caustic: KdTree::build(buckets.caustic),
            settings: settings.clone(),
        }
    }

    /// Sum of the three maps' radiance estimates, times the diffuse color.
    pub fn irradiance(&self, hit: &Hit, kd: DVec3) -> DVec3 {
        let r = &self.settings.radius;
        let e = &self.settings.exposure;
        let total = estimate(&self.direct, hit, r.direct, e.direct)
            + estimate(&self.indirect, hit, r.indirect, e.indirect)
            + estimate(&self.caustic, hit, r.caustic, e.caustic);
        total * kd
    }
}

/// Density estimate with a cone (linear in d²) filter over a compressed disc.
fn estimate(map: &KdTree, hit: &Hit, radius: f64, exposure: f64) -> DVec3 {
    if map.is_empty() {
        return DVec3::ZERO;
    }
    let r2 = radius * radius;
    let slab = radius * SEARCH_COMPRESSION;
    let mut sum = DVec3::ZERO;
    map.for_each_within(hit.point, radius, |p, d2| {
        if (p.pos - hit.point).dot(hit.normal).abs() < slab {
            sum += p.power * (1.0 - d2 / r2);
        }
    });
    sum * (exposure / (std::f64::consts::PI * r2))
}

#[allow(clippy::too_many_arguments)]
fn trace_photon(
    world: &World,
    settings: &PhotonSettings,
    ray: &Ray,
    depth: u32,
    internal: bool,
    power: DVec3,
    kind: Kind,
    rng: &mut Rng,
    out: &mut Buckets,
) {
    let Some(hit) = world.intersect(ray, f64::INFINITY) else { return };
    let m = &world.materials[hit.material];
    let pr = m.reflectivity;
    let pt = m.transmissivity;
    let pd = 1.0 - pr - pt;
    let r = rng.next_f64();

    if depth < settings.max_depth {
        if r < 0.5 {
            // Diffuse bounce into the hemisphere around the normal. (The
            // original's hemisphere flip was another discarded `sMult`.)
            let mut dir = rng.unit_vector();
            if dir.dot(hit.normal) < 0.0 {
                dir = -dir;
            }
            let bounced = Ray::new(hit.point, dir);
            trace_photon(world, settings, &bounced, depth + 1, false, power * m.kd * pd, Kind::Indirect, rng, out);
        }
        if pt > 0.0 {
            let (refracted, now_internal) = refract(ray, &hit, internal, m.effective_ior());
            trace_photon(world, settings, &refracted, depth + 1, now_internal, power * pt, Kind::Caustic, rng, out);
        }
        if pr > 0.0 && !internal {
            let reflected = reflect(ray, &hit);
            trace_photon(world, settings, &reflected, depth + 1, false, power * pr, Kind::Caustic, rng, out);
        }
    }

    if r >= 0.5 {
        let photon = Photon { pos: hit.point, power: power * pd, axis: 0 };
        match kind {
            Kind::Direct => out.direct.push(photon),
            Kind::Indirect => out.indirect.push(photon),
            Kind::Caustic => out.caustic.push(photon),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kd_radius_search_matches_brute_force() {
        let mut rng = Rng::new(7);
        let photons: Vec<Photon> = (0..5000)
            .map(|_| Photon {
                pos: DVec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()) * 10.0,
                power: DVec3::ONE,
                axis: 0,
            })
            .collect();
        let tree = KdTree::build(photons.clone());
        for _ in 0..50 {
            let q = DVec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()) * 10.0;
            let radius = 0.2 + rng.next_f64();
            let expected = photons.iter().filter(|p| p.pos.distance_squared(q) < radius * radius).count();
            let mut got = 0;
            tree.for_each_within(q, radius, |_, _| got += 1);
            assert_eq!(expected, got);
        }
    }
}
