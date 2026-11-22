//! Photon mapping: emission from point lights, and three kd-trees (direct,
//! indirect, caustic) read with nearest-neighbor density estimates.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use glam::DVec3;
use rayon::prelude::*;

use crate::geometry::{Hit, Ray, World};
use crate::optics::{reflect, refract};
use crate::rng::Rng;
use crate::scene::{DirectLighting, PhotonSettings};

#[derive(Clone, Copy, Debug)]
pub struct Photon {
    pub pos: DVec3,
    pub power: DVec3,
    /// Split axis when this photon is an interior kd-tree node.
    axis: u8,
}

/// A photon found by [`KdTree::nearest`]: its index into
/// [`KdTree::photons`] and squared distance. Ordered by distance.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub d2: f64,
    pub index: u32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.d2.total_cmp(&other.d2)
    }
}

/// Pending subtrees during a search: deeper than any balanced tree that fits
/// in memory.
const STACK_DEPTH: usize = 64;

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

    /// All photons, in kd-tree order.
    pub fn photons(&self) -> &[Photon] {
        &self.photons
    }

    /// Calls `f(photon, squared_distance)` for every photon within `radius`
    /// of `point`.
    pub fn for_each_within(&self, point: DVec3, radius: f64, mut f: impl FnMut(&Photon, f64)) {
        let r2 = radius * radius;
        let mut stack = [(0, 0); STACK_DEPTH];
        stack[0] = (0, self.photons.len());
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let (lo, hi) = stack[top];
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
                stack[top] = far;
                top += 1;
            }
            stack[top] = near;
            top += 1;
        }
    }

    /// Finds the `k` photons nearest `point` within `max_radius` that pass
    /// `accept`, leaving them in `heap` (farthest on top). Returns the squared
    /// radius they cover: the farthest one's distance once `k` are found,
    /// otherwise `max_radius` squared.
    pub fn nearest(
        &self,
        point: DVec3,
        k: usize,
        max_radius: f64,
        mut accept: impl FnMut(&Photon) -> bool,
        heap: &mut BinaryHeap<Neighbor>,
    ) -> f64 {
        heap.clear();
        let mut r2 = max_radius * max_radius;
        if k == 0 {
            return r2;
        }
        // Each entry carries the squared distance to its splitting plane, so
        // far subtrees are skipped if the radius has shrunk by the time they
        // are popped.
        let mut stack = [(0, 0, 0.0); STACK_DEPTH];
        stack[0] = (0, self.photons.len(), 0.0);
        let mut top = 1;
        while top > 0 {
            top -= 1;
            let (lo, hi, plane_d2) = stack[top];
            if lo >= hi || plane_d2 >= r2 {
                continue;
            }
            let mid = (lo + hi) / 2;
            let p = &self.photons[mid];
            let d2 = p.pos.distance_squared(point);
            if d2 < r2 && accept(p) {
                if heap.len() == k {
                    heap.pop();
                }
                heap.push(Neighbor { d2, index: mid as u32 });
                if heap.len() == k {
                    r2 = heap.peek().map_or(r2, |n| n.d2);
                }
            }
            let delta = point[p.axis as usize] - p.pos[p.axis as usize];
            let (near, far) = if delta < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
            stack[top] = (far.0, far.1, delta * delta);
            stack[top + 1] = (near.0, near.1, 0.0);
            top += 2;
        }
        r2
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

/// Matches light computed with shadow rays to the photon estimates' scale,
/// which is half the true density twice over: photons are stored at only half
/// of their surface hits (the other half bounce), and the cone filter's
/// weights average 1/2 over the disc.
pub const ESTIMATE_SCALE: f64 = 0.25;

pub struct PhotonMaps {
    pub direct: KdTree,
    pub indirect: KdTree,
    pub caustic: KdTree,
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
        }
    }

    /// The maps' radiance estimates, times the diffuse color. The direct map
    /// is only read when it, not shadow rays, provides direct light.
    pub fn irradiance(&self, hit: &Hit, kd: DVec3, settings: &PhotonSettings) -> DVec3 {
        let (r, e, k) = (&settings.radius, &settings.exposure, &settings.gather);
        let mut total = estimate(&self.indirect, hit, r.indirect, k.indirect) * e.indirect
            + estimate(&self.caustic, hit, r.caustic, k.caustic) * e.caustic;
        if settings.direct_lighting == DirectLighting::Map {
            total += estimate(&self.direct, hit, r.direct, k.direct) * e.direct;
        }
        total * kd
    }
}

thread_local! {
    static NEIGHBORS: RefCell<BinaryHeap<Neighbor>> = RefCell::default();
}

/// Density estimate with a cone (linear in d²) filter over a compressed disc,
/// from the `count` nearest photons within `radius`, or all of them if
/// `count` is 0.
fn estimate(map: &KdTree, hit: &Hit, radius: f64, count: u32) -> DVec3 {
    if map.is_empty() {
        return DVec3::ZERO;
    }
    let slab = radius * SEARCH_COMPRESSION;
    let on_disc = |p: &Photon| (p.pos - hit.point).dot(hit.normal).abs() < slab;
    let mut sum = DVec3::ZERO;
    let r2 = if count == 0 {
        let r2 = radius * radius;
        map.for_each_within(hit.point, radius, |p, d2| {
            if on_disc(p) {
                sum += p.power * (1.0 - d2 / r2);
            }
        });
        r2
    } else {
        NEIGHBORS.with_borrow_mut(|heap| {
            let r2 = map.nearest(hit.point, count as usize, radius, on_disc, heap);
            for n in heap.iter() {
                sum += map.photons[n.index as usize].power * (1.0 - n.d2 / r2);
            }
            r2
        })
    };
    sum / (std::f64::consts::PI * r2)
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

    #[test]
    fn kd_nearest_matches_brute_force() {
        let mut rng = Rng::new(11);
        let photons: Vec<Photon> = (0..5000)
            .map(|_| Photon {
                pos: DVec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()) * 10.0,
                power: DVec3::ONE,
                axis: 0,
            })
            .collect();
        let tree = KdTree::build(photons.clone());
        let mut heap = BinaryHeap::new();
        for k in [1, 7, 50] {
            for _ in 0..30 {
                let q = DVec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()) * 10.0;
                let max_radius = 0.3 + rng.next_f64() * 2.0;
                let mut expected: Vec<f64> = photons
                    .iter()
                    .map(|p| p.pos.distance_squared(q))
                    .filter(|&d2| d2 < max_radius * max_radius)
                    .collect();
                expected.sort_by(f64::total_cmp);
                expected.truncate(k);

                let r2 = tree.nearest(q, k, max_radius, |_| true, &mut heap);
                let mut got: Vec<f64> = heap.iter().map(|n| n.d2).collect();
                got.sort_by(f64::total_cmp);
                assert_eq!(expected, got);
                let expected_r2 = if expected.len() == k { expected[k - 1] } else { max_radius * max_radius };
                assert_eq!(r2, expected_r2);
            }
        }
    }
}
