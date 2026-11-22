//! Photon mapping: emission from point lights, and three kd-trees (direct,
//! indirect, caustic) read with nearest-neighbor density estimates.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use glam::{DVec3, Vec3};
use rayon::prelude::*;

use crate::geometry::{Hit, Ray, World};
use crate::light::Light;
use crate::optics::{reflect, refract};
use crate::rng::Rng;
use crate::scene::{DirectLighting, PhotonSettings, PhotonTriple};

#[derive(Clone, Copy, Debug)]
pub struct Photon {
    pub pos: DVec3,
    pub power: DVec3,
    /// Surface normal on the side the photon arrived from.
    pub normal: Vec3,
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

    /// For a tree of precomputed irradiance points: the irradiance of the
    /// nearest point within `max_radius` facing about the same way as
    /// `facing`, or zero if there is none.
    pub fn nearest_irradiance(&self, point: DVec3, facing: DVec3, max_radius: f64) -> DVec3 {
        NEIGHBORS.with_borrow_mut(|heap| {
            let facing = facing.as_vec3();
            self.nearest(point, 1, max_radius, |p| p.normal.dot(facing) > MIN_NORMAL_COS, heap);
            heap.peek().map_or(DVec3::ZERO, |n| self.photons[n.index as usize].power)
        })
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

/// Which photons a tracing pass keeps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pass {
    /// Every photon.
    All,
    /// Direct and indirect only; a separate pass supplies the caustics.
    Global,
    /// Caustic paths only, from photons aimed at specular objects.
    Caustic,
}

/// Cells of a projection map: equal-area bands of cos θ by slices of φ.
const PROJECTION_ROWS: usize = 256;
const PROJECTION_COLS: usize = 512;
/// Rays per cell side when deciding whether a cell sees a specular object.
const PROJECTION_SAMPLES: usize = 2;

/// The directions from a light that reach glass or mirrors first, as a set of
/// equal-area cells (Jensen's projection map). Caustic photons are emitted
/// only into these, so they aren't wasted on diffuse surfaces.
struct ProjectionMap {
    cells: Vec<u32>,
}

impl ProjectionMap {
    /// Rays leave from scattered points on an area light, so a cell is marked
    /// if glass or a mirror is first in its direction from anywhere on it.
    fn new(world: &World, light: &Light) -> Self {
        let specular: Vec<bool> = (0..PROJECTION_ROWS * PROJECTION_COLS)
            .into_par_iter()
            .map(|cell| {
                (0..PROJECTION_SAMPLES * PROJECTION_SAMPLES).any(|k| {
                    let u = (k % PROJECTION_SAMPLES) as f64 + 0.5;
                    let v = (k / PROJECTION_SAMPLES) as f64 + 0.5;
                    let dir = cell_direction(cell, u / PROJECTION_SAMPLES as f64, v / PROJECTION_SAMPLES as f64);
                    let mut rng = Rng::new((cell * PROJECTION_SAMPLES * PROJECTION_SAMPLES + k) as u64);
                    let origin = light.sample(rng.next_f64(), rng.next_f64(), None).point;
                    world.intersect(&Ray::new(origin, dir), f64::INFINITY).is_some_and(|h| {
                        let m = &world.materials[h.material];
                        m.reflectivity + m.transmissivity > 0.0
                    })
                })
            })
            .collect();
        // Widen by a cell so objects narrower than the sampling still count.
        let cells = (0..PROJECTION_ROWS * PROJECTION_COLS)
            .filter(|&cell| {
                let (row, col) = ((cell / PROJECTION_COLS) as isize, (cell % PROJECTION_COLS) as isize);
                (-1..=1).any(|dr| {
                    let r = row + dr;
                    (0..PROJECTION_ROWS as isize).contains(&r)
                        && (-1..=1).any(|dc| {
                            let c = (col + dc).rem_euclid(PROJECTION_COLS as isize);
                            specular[r as usize * PROJECTION_COLS + c as usize]
                        })
                })
            })
            .map(|cell| cell as u32)
            .collect();
        Self { cells }
    }

    /// Fraction of all directions the marked cells cover.
    fn coverage(&self) -> f64 {
        self.cells.len() as f64 / (PROJECTION_ROWS * PROJECTION_COLS) as f64
    }

    /// A uniformly distributed direction within the marked cells.
    fn sample(&self, rng: &mut Rng) -> DVec3 {
        let cell = self.cells[rng.below(self.cells.len())] as usize;
        cell_direction(cell, rng.next_f64(), rng.next_f64())
    }
}

/// The direction at (u, v) within a projection map cell. Rows are uniform in
/// cos θ, so every cell covers the same solid angle.
fn cell_direction(cell: usize, u: f64, v: f64) -> DVec3 {
    let (row, col) = (cell / PROJECTION_COLS, cell % PROJECTION_COLS);
    let z = 1.0 - 2.0 * (row as f64 + u) / PROJECTION_ROWS as f64;
    let phi = std::f64::consts::TAU * (col as f64 + v) / PROJECTION_COLS as f64;
    let r = (1.0 - z * z).max(0.0).sqrt();
    DVec3::new(r * phi.cos(), r * phi.sin(), z)
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

/// The cone filter's weights (1 - d²/r²) average 1/2 over the disc, so
/// estimates are scaled back up by this.
const CONE_NORMALIZATION: f64 = 2.0;

/// Irradiance is precomputed at every this many photons for final gathering.
const PRECOMPUTE_STRIDE: usize = 4;

/// A precomputed irradiance point only serves surfaces whose normal is within
/// about 25° of its own.
const MIN_NORMAL_COS: f32 = 0.9;

/// Settings the precomputed irradiance depends on.
type PrecomputeKey = (f64, u32, PhotonTriple);

pub struct PhotonMaps {
    pub direct: KdTree,
    pub indirect: KdTree,
    pub caustic: KdTree,
    /// Irradiance at a subset of the photons, for final gathering. Built on
    /// first use and rebuilt when the estimate settings change.
    precomputed: Mutex<Option<(PrecomputeKey, Arc<KdTree>)>>,
}

impl PhotonMaps {
    /// Emits `photons_per_light` photons uniformly from each light. A light
    /// emits `light_power` times its diffuse color in total. With
    /// `caustic_photons`, a second pass aims that many more per light at
    /// glass and mirrors through projection maps and supplies the caustic map.
    pub fn build(world: &World, settings: &PhotonSettings, seed: u64) -> Self {
        let targeted = settings.caustic_photons > 0;
        let per_light = settings.photons_per_light.max(1) as usize;
        let total = per_light * world.lights.len();
        let global_pass = if targeted { Pass::Global } else { Pass::All };

        let mut buckets = (0..total)
            .into_par_iter()
            .fold(Buckets::default, |mut out, i| {
                let light = &world.lights[i / per_light];
                let power = light.diffuse * (settings.light_power / per_light as f64);
                let mut rng = Rng::new(seed.wrapping_mul(0x1000_0000_01B3) ^ i as u64);
                let ray = match light.is_point() {
                    true => Ray::new(light.position, rng.unit_vector()),
                    // Area lights emit from anywhere on their surface, like a
                    // matte emitter.
                    false => {
                        let s = light.sample(rng.next_f64(), rng.next_f64(), None);
                        let normal = s.normal.unwrap_or(DVec3::Z);
                        Ray::new(s.point, cosine_direction(normal, &mut rng))
                    }
                };
                trace_photon(world, settings, global_pass, ray, power, &mut rng, &mut out);
                out
            })
            .reduce(Buckets::default, Buckets::merge);

        if targeted {
            let per_light = settings.caustic_photons as usize;
            let maps: Vec<ProjectionMap> = world.lights.iter().map(|l| ProjectionMap::new(world, l)).collect();
            let caustics = (0..per_light * world.lights.len())
                .into_par_iter()
                .fold(Buckets::default, |mut out, i| {
                    let (light, map) = (&world.lights[i / per_light], &maps[i / per_light]);
                    if map.cells.is_empty() {
                        return out;
                    }
                    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64);
                    let origin = light.sample(rng.next_f64(), rng.next_f64(), None);
                    let dir = map.sample(&mut rng);
                    // Directions are drawn evenly over the covered cells, so
                    // weight by how strongly the light emits that way: for a
                    // point light, just the covered share of its power.
                    let emission = light.emission(&origin, dir) * 4.0 * std::f64::consts::PI;
                    if emission <= 0.0 {
                        return out;
                    }
                    let power = light.diffuse * (settings.light_power * map.coverage() * emission / per_light as f64);
                    trace_photon(
                        world,
                        settings,
                        Pass::Caustic,
                        Ray::new(origin.point, dir),
                        power,
                        &mut rng,
                        &mut out,
                    );
                    out
                })
                .reduce(Buckets::default, Buckets::merge);
            buckets.caustic = caustics.caustic;
        }

        Self {
            direct: KdTree::build(buckets.direct),
            indirect: KdTree::build(buckets.indirect),
            caustic: KdTree::build(buckets.caustic),
            precomputed: Mutex::default(),
        }
    }

    /// Irradiance at `hit` read straight from the maps: caustics, the direct
    /// map when it (not shadow rays) provides direct light, and the indirect
    /// map unless final gathering replaces it.
    pub fn irradiance(&self, hit: &Hit, settings: &PhotonSettings, indirect: bool) -> DVec3 {
        let (r, e, k) = (&settings.radius, &settings.exposure, &settings.gather);
        let mut total = estimate(&self.caustic, hit.point, hit.normal, None, r.caustic, k.caustic) * e.caustic;
        if indirect {
            total += estimate(&self.indirect, hit.point, hit.normal, None, r.indirect, k.indirect) * e.indirect;
        }
        if settings.direct_lighting == DirectLighting::Map {
            total += estimate(&self.direct, hit.point, hit.normal, None, r.direct, k.direct) * e.direct;
        }
        total
    }

    /// Total irradiance from all three maps at every [`PRECOMPUTE_STRIDE`]th
    /// photon, as in Christensen's "Faster Photon Map Global Illumination".
    /// Final gather rays then need one nearest-point lookup each instead of
    /// three density estimates.
    ///
    /// Every map is read with the indirect radius and count, treating them as
    /// one global map: caustics' tight estimates would turn stray caustic
    /// photons into bright specks that gather rays pick up as noise.
    pub fn precomputed_irradiance(&self, settings: &PhotonSettings) -> Arc<KdTree> {
        let key = (settings.radius.indirect, settings.gather.indirect, settings.exposure);
        let mut cache = self.precomputed.lock().unwrap();
        if let Some((k, tree)) = &*cache
            && *k == key
        {
            return tree.clone();
        }
        let (radius, count, e) = (settings.radius.indirect, settings.gather.indirect, &settings.exposure);
        let photons = [&self.direct, &self.indirect, &self.caustic].into_iter().flat_map(|m| m.photons());
        let samples: Vec<&Photon> = photons.step_by(PRECOMPUTE_STRIDE).collect();
        let points = samples
            .into_par_iter()
            .map(|p| {
                let n = p.normal.as_dvec3();
                let side = Some(n);
                let power = estimate(&self.direct, p.pos, n, side, radius, count) * e.direct
                    + estimate(&self.indirect, p.pos, n, side, radius, count) * e.indirect
                    + estimate(&self.caustic, p.pos, n, side, radius, count) * e.caustic;
                Photon { power, ..*p }
            })
            .collect();
        let tree = Arc::new(KdTree::build(points));
        *cache = Some((key, tree.clone()));
        tree
    }
}

thread_local! {
    static NEIGHBORS: RefCell<BinaryHeap<Neighbor>> = RefCell::default();
}

/// Density estimate with a cone (linear in d²) filter over a compressed disc,
/// from the `count` nearest photons within `radius`, or all of them if
/// `count` is 0. With `side`, only photons that arrived on that side of the
/// surface count.
fn estimate(map: &KdTree, point: DVec3, normal: DVec3, side: Option<DVec3>, radius: f64, count: u32) -> DVec3 {
    if map.is_empty() {
        return DVec3::ZERO;
    }
    let slab = radius * SEARCH_COMPRESSION;
    let side = side.map(|s| s.as_vec3());
    let on_disc = |p: &Photon| (p.pos - point).dot(normal).abs() < slab && side.is_none_or(|s| p.normal.dot(s) > 0.0);
    let mut sum = DVec3::ZERO;
    let r2 = if count == 0 {
        let r2 = radius * radius;
        map.for_each_within(point, radius, |p, d2| {
            if on_disc(p) {
                sum += p.power * (1.0 - d2 / r2);
            }
        });
        r2
    } else {
        NEIGHBORS.with_borrow_mut(|heap| {
            let r2 = map.nearest(point, count as usize, radius, on_disc, heap);
            for n in heap.iter() {
                sum += map.photons[n.index as usize].power * (1.0 - n.d2 / r2);
            }
            r2
        })
    };
    sum * (CONE_NORMALIZATION / (std::f64::consts::PI * r2))
}

/// Follows one photon, storing it at every surface with a diffuse part, and
/// choosing what happens next by Russian roulette: diffuse bounce, mirror
/// reflection, refraction or absorption, in proportion to the material's
/// reflectances. A diffuse bounce rescales the power by reflectance over
/// probability, so the expected power is right while color can shift.
///
/// Photons are sorted by path: straight from a light (direct), through glass
/// or mirrors only (caustic), or after any diffuse bounce (indirect).
fn trace_photon(
    world: &World,
    settings: &PhotonSettings,
    pass: Pass,
    mut ray: Ray,
    mut power: DVec3,
    rng: &mut Rng,
    out: &mut Buckets,
) {
    let mut kind = Kind::Direct;
    let mut internal = false;
    for _ in 0..=settings.max_depth {
        let Some(hit) = world.intersect(&ray, f64::INFINITY) else { return };
        let m = &world.materials[hit.material];
        if internal {
            // Inside glass: carry on to the far surface (or reflect inside).
            let (refracted, now_internal) = refract(&ray, &hit, true, m.effective_ior());
            (ray, internal) = (refracted, now_internal);
            continue;
        }

        let facing = if hit.normal.dot(ray.dir) > 0.0 { -hit.normal } else { hit.normal };
        let diffuse_part = (1.0 - m.reflectivity - m.transmissivity).max(0.0);
        if diffuse_part > 0.0 {
            let photon = Photon { pos: hit.point, power, normal: facing.as_vec3(), axis: 0 };
            match (kind, pass) {
                (Kind::Direct, Pass::All | Pass::Global) => out.direct.push(photon),
                (Kind::Indirect, Pass::All | Pass::Global) => out.indirect.push(photon),
                (Kind::Caustic, Pass::All | Pass::Caustic) => out.caustic.push(photon),
                _ => {}
            }
        }

        let diffuse = m.kd * diffuse_part;
        let p_diffuse = diffuse.element_sum() / 3.0;
        let xi = rng.next_f64();
        if xi < p_diffuse {
            if pass == Pass::Caustic {
                // Past a diffuse bounce the path is no longer a caustic.
                return;
            }
            power *= diffuse / p_diffuse;
            ray = Ray::new(hit.point, cosine_direction(facing, rng));
            kind = Kind::Indirect;
        } else if xi < p_diffuse + m.reflectivity {
            ray = reflect(&ray, &hit);
            kind = after_specular(kind);
        } else if xi < p_diffuse + m.reflectivity + m.transmissivity {
            (ray, internal) = refract(&ray, &hit, false, m.effective_ior());
            kind = after_specular(kind);
        } else {
            return;
        }
    }
}

/// Light that has bounced diffusely stays indirect after glass or mirrors;
/// only light → specular → diffuse paths are caustics.
fn after_specular(kind: Kind) -> Kind {
    if kind == Kind::Indirect { Kind::Indirect } else { Kind::Caustic }
}

/// A direction in the hemisphere around `n`, with density proportional to
/// the cosine, as light leaves a Lambertian surface.
fn cosine_direction(n: DVec3, rng: &mut Rng) -> DVec3 {
    let (t, b) = n.any_orthonormal_pair();
    let u = rng.next_f64();
    let r = u.sqrt();
    let (sin, cos) = (std::f64::consts::TAU * rng.next_f64()).sin_cos();
    t * (r * cos) + b * (r * sin) + n * (1.0 - u).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;
    use crate::material::Material;

    /// A point light at the center of a sphere of radius `R` whose walls have
    /// diffuse reflectance `kd`.
    const R: f64 = 5.0;
    const LIGHT_POWER: f64 = 100.0;

    fn light_in_sphere(kd: f64) -> (World, PhotonSettings) {
        let mut world = World::empty();
        world.materials.push(Material {
            kd: DVec3::splat(kd),
            reflectivity: 0.0,
            transmissivity: 0.0,
            ..Material::default()
        });
        world.spheres.push(Sphere { center: DVec3::ZERO, radius: R, material: 0 });
        world.lights.push(Light::point(DVec3::ZERO, DVec3::ONE, DVec3::ONE));
        let settings = PhotonSettings {
            enabled: true,
            photons_per_light: 200_000,
            max_depth: 60,
            light_power: LIGHT_POWER,
            ..PhotonSettings::default()
        };
        (world, settings)
    }

    /// Average estimate over points spread around the sphere.
    fn mean_estimate(map: &KdTree) -> f64 {
        let mut rng = Rng::new(3);
        let n = 40;
        let total: f64 = (0..n)
            .map(|_| {
                let normal = rng.unit_vector();
                estimate(map, normal * R, normal, None, 1.5, 400).x
            })
            .sum();
        total / n as f64
    }

    #[test]
    fn direct_estimate_matches_inverse_square_irradiance() {
        let (world, settings) = light_in_sphere(0.0);
        let maps = PhotonMaps::build(&world, &settings, 1);
        let expected = LIGHT_POWER / (4.0 * std::f64::consts::PI * R * R);
        let got = mean_estimate(&maps.direct);
        assert!((got / expected - 1.0).abs() < 0.03, "direct {got} vs {expected}");
        assert!(maps.indirect.is_empty(), "black walls absorb everything");
    }

    /// A small panel facing down sends Φ cos θ / π per steradian, so straight
    /// below it at distance R the irradiance is Φ / (π R²).
    #[test]
    fn panel_light_photons_follow_its_cosine_lobe() {
        let (mut world, settings) = light_in_sphere(0.0);
        world.lights[0].shape =
            crate::light::LightShape::Rect { half_u: DVec3::X * 0.1, half_v: DVec3::Z * 0.1, normal: DVec3::NEG_Y };
        let maps = PhotonMaps::build(&world, &settings, 1);
        let mut rng = Rng::new(5);
        let n = 20;
        let got = (0..n)
            .map(|_| {
                // Points near the bottom, where the panel faces.
                let normal = (DVec3::NEG_Y + rng.unit_vector() * 0.05).normalize();
                estimate(&maps.direct, normal * R, normal, None, 1.5, 400).x
                    / (normal.dot(DVec3::NEG_Y) * normal.dot(DVec3::NEG_Y))
            })
            .sum::<f64>()
            / n as f64;
        let expected = LIGHT_POWER / (std::f64::consts::PI * R * R);
        assert!((got / expected - 1.0).abs() < 0.03, "panel {got} vs {expected}");
        assert!(maps.direct.photons().iter().all(|p| p.pos.y < 0.1), "a panel facing down lights nothing above it");
    }

    /// A clear glass ball beside the light: every photon that reaches it ends
    /// as a caustic on the black walls, so the caustic map's total power is the
    /// light's power times the share of directions the ball covers, whether
    /// caustics come from the main photons or a targeted pass.
    #[test]
    fn targeted_caustics_carry_the_same_power() {
        let (mut world, settings) = light_in_sphere(0.0);
        world.materials.push(Material { transmissivity: 1.0, reflectivity: 0.0, ..Material::default() });
        let (distance, radius) = (2.0, 0.8);
        world.spheres.push(Sphere { center: DVec3::new(distance, 0.0, 0.0), radius, material: 1 });
        let cos_half_angle = (1.0 - (radius / distance).powi(2)).sqrt();
        let expected = LIGHT_POWER * (1.0 - cos_half_angle) / 2.0;
        let total = |maps: &PhotonMaps| maps.caustic.photons().iter().map(|p| p.power.x).sum::<f64>();

        let untargeted = PhotonMaps::build(&world, &PhotonSettings { caustic_photons: 0, ..settings.clone() }, 1);
        let targeted = PhotonMaps::build(&world, &PhotonSettings { caustic_photons: 50_000, ..settings }, 1);
        for (name, maps) in [("untargeted", &untargeted), ("targeted", &targeted)] {
            let got = total(maps);
            assert!((got / expected - 1.0).abs() < 0.03, "{name} caustic power {got} vs {expected}");
        }
        // The targeted pass spends its photons on the ball, not the walls.
        assert!(targeted.caustic.len() > 4 * untargeted.caustic.len());
    }

    /// Inside a closed diffuse sphere every bounce spreads evenly, so the
    /// indirect irradiance is E0 (ρ + ρ² + …) = E0 ρ / (1 - ρ).
    #[test]
    fn roulette_conserves_energy_over_bounces() {
        let rho = 0.5;
        let (world, settings) = light_in_sphere(rho);
        let maps = PhotonMaps::build(&world, &settings, 1);
        let e0 = LIGHT_POWER / (4.0 * std::f64::consts::PI * R * R);
        let expected = e0 * rho / (1.0 - rho);
        let got = mean_estimate(&maps.indirect);
        assert!((got / expected - 1.0).abs() < 0.04, "indirect {got} vs {expected}");
    }

    #[test]
    fn kd_radius_search_matches_brute_force() {
        let mut rng = Rng::new(7);
        let photons: Vec<Photon> = (0..5000)
            .map(|_| Photon {
                pos: DVec3::new(rng.next_f64(), rng.next_f64(), rng.next_f64()) * 10.0,
                power: DVec3::ONE,
                normal: Vec3::Z,
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
                normal: Vec3::Z,
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
