//! Render-ready geometry. As in the original renderer, everything is
//! transformed into camera space up front, so the eye sits at the origin
//! looking down -Z.

use glam::{DMat3, DVec3};

use crate::material::Material;
use crate::scene::Scene;

/// Minimum hit distance, to keep secondary rays from re-hitting their origin.
pub const EPSILON: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: DVec3,
    /// Always unit length.
    pub dir: DVec3,
}

impl Ray {
    pub fn new(origin: DVec3, dir: DVec3) -> Self {
        Self { origin, dir: dir.normalize() }
    }

    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f64,
    pub point: DVec3,
    /// Shading normal (interpolated for triangles), unit length. Not flipped
    /// toward the ray; refraction uses its orientation to tell inside from
    /// outside.
    pub normal: DVec3,
    pub material: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: DVec3,
    pub e1: DVec3,
    pub e2: DVec3,
    pub normals: [DVec3; 3],
    pub material: u32,
}

impl Triangle {
    fn centroid(&self) -> DVec3 {
        self.v0 + (self.e1 + self.e2) / 3.0
    }

    fn bounds(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        b.grow(self.v0);
        b.grow(self.v0 + self.e1);
        b.grow(self.v0 + self.e2);
        b
    }

    /// Möller–Trumbore. Returns (t, u, v).
    fn intersect(&self, ray: &Ray, t_max: f64) -> Option<(f64, f64, f64)> {
        let p = ray.dir.cross(self.e2);
        let det = p.dot(self.e1);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let s = ray.origin - self.v0;
        let u = p.dot(s) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(self.e1);
        let v = q.dot(ray.dir) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = q.dot(self.e2) * inv;
        (t > EPSILON && t < t_max).then_some((t, u, v))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
    pub material: u32,
}

impl Sphere {
    /// Nearest root beyond EPSILON, like the original.
    fn intersect(&self, ray: &Ray, t_max: f64) -> Option<f64> {
        let oc = ray.origin - self.center;
        let b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let disc = b * b - c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        [-b - sq, -b + sq].into_iter().find(|&t| t > EPSILON && t < t_max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    /// Camera space.
    pub position: DVec3,
    pub diffuse: DVec3,
    pub specular: DVec3,
}

pub struct World {
    pub triangles: Vec<Triangle>,
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
    pub lights: Vec<PointLight>,
    bvh: Vec<BvhNode>,
}

impl World {
    pub fn from_scene(scene: &Scene) -> Self {
        let mut triangles = Vec::new();
        let mut spheres = Vec::new();
        let mut materials = Vec::new();

        for (mesh, model) in &scene.objects {
            let model_view = scene.view * *model;
            let normal_mat = DMat3::from_mat4(model_view).inverse().transpose();
            let mat_base = materials.len() as u32;
            materials.extend(mesh.materials.iter().cloned());

            if mesh.is_sphere {
                let m3 = DMat3::from_mat4(model_view);
                let scale = m3.x_axis.length().max(m3.y_axis.length()).max(m3.z_axis.length());
                spheres.push(Sphere {
                    center: model_view.transform_point3(mesh.center()),
                    radius: mesh.bounding_radius() * scale,
                    material: mat_base + mesh.primary_material() as u32,
                });
                continue;
            }

            let positions: Vec<DVec3> = mesh.positions.iter().map(|&p| model_view.transform_point3(p)).collect();
            for t in &mesh.triangles {
                let [a, b, c] = t.verts.map(|i| positions[i]);
                triangles.push(Triangle {
                    v0: a,
                    e1: b - a,
                    e2: c - a,
                    normals: t.normals.map(|n| (normal_mat * n).normalize_or_zero()),
                    material: mat_base + t.material as u32,
                });
            }
        }

        let lights = scene
            .file
            .lights
            .iter()
            .map(|l| PointLight {
                position: scene.view.transform_point3(l.position.into()),
                diffuse: l.diffuse.into(),
                specular: l.specular.into(),
            })
            .collect();

        let bvh = build_bvh(&mut triangles);
        Self { triangles, spheres, materials, lights, bvh }
    }

    /// Nearest intersection closer than `t_max`.
    pub fn intersect(&self, ray: &Ray, t_max: f64) -> Option<Hit> {
        let mut best_t = t_max;
        let mut best: Option<(usize, f64, f64, bool)> = None; // (index, u, v, is_sphere)

        for (i, s) in self.spheres.iter().enumerate() {
            if let Some(t) = s.intersect(ray, best_t) {
                best_t = t;
                best = Some((i, 0.0, 0.0, true));
            }
        }

        if !self.bvh.is_empty() {
            let inv_dir = ray.dir.recip();
            let mut stack = [0u32; 64];
            let mut sp = 1;
            while sp > 0 {
                sp -= 1;
                let node = &self.bvh[stack[sp] as usize];
                if !node.bounds.hit(ray.origin, inv_dir, best_t) {
                    continue;
                }
                if node.count > 0 {
                    for i in node.first..node.first + node.count {
                        if let Some((t, u, v)) = self.triangles[i as usize].intersect(ray, best_t) {
                            best_t = t;
                            best = Some((i as usize, u, v, false));
                        }
                    }
                } else {
                    // Visit the nearer child first so best_t shrinks sooner.
                    let (near, far) = if ray.dir[node.axis as usize] >= 0.0 {
                        (node.first, node.first + 1)
                    } else {
                        (node.first + 1, node.first)
                    };
                    stack[sp] = far;
                    stack[sp + 1] = near;
                    sp += 2;
                }
            }
        }

        let (i, u, v, is_sphere) = best?;
        let point = ray.at(best_t);
        Some(if is_sphere {
            let s = &self.spheres[i];
            Hit { t: best_t, point, normal: (point - s.center).normalize(), material: s.material as usize }
        } else {
            let tri = &self.triangles[i];
            let n = tri.normals[0] * (1.0 - u - v) + tri.normals[1] * u + tri.normals[2] * v;
            Hit {
                t: best_t,
                point,
                normal: n.normalize_or(tri.e1.cross(tri.e2).normalize()),
                material: tri.material as usize,
            }
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: DVec3,
    max: DVec3,
}

impl Aabb {
    const EMPTY: Self = Self { min: DVec3::splat(f64::INFINITY), max: DVec3::splat(f64::NEG_INFINITY) };

    fn grow(&mut self, p: DVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn union(&self, o: &Self) -> Self {
        Self { min: self.min.min(o.min), max: self.max.max(o.max) }
    }

    fn area(&self) -> f64 {
        let d = (self.max - self.min).max(DVec3::ZERO);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    fn hit(&self, origin: DVec3, inv_dir: DVec3, t_max: f64) -> bool {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let t_near = t0.min(t1).max_element().max(0.0);
        // Widen slightly so rounding can't make us miss flat boxes or hits
        // exactly on an edge (see pbrt's robust ray-bounds test).
        let t_far = t0.max(t1).min_element().min(t_max) * (1.0 + 1e-12);
        t_near <= t_far
    }
}

/// Interior nodes have `count == 0` and children at `first`, `first + 1`.
/// Leaves hold triangles `first..first + count`.
#[derive(Clone, Copy, Debug)]
struct BvhNode {
    bounds: Aabb,
    first: u32,
    count: u32,
    axis: u8,
}

const LEAF_SIZE: usize = 4;
const SAH_BINS: usize = 12;

/// Builds a binned-SAH BVH, reordering `tris` so each leaf is contiguous.
fn build_bvh(tris: &mut [Triangle]) -> Vec<BvhNode> {
    if tris.is_empty() {
        return Vec::new();
    }
    let mut nodes = vec![BvhNode { bounds: Aabb::EMPTY, first: 0, count: tris.len() as u32, axis: 0 }];
    let mut work = vec![0usize];
    while let Some(ni) = work.pop() {
        let (first, count) = (nodes[ni].first as usize, nodes[ni].count as usize);
        let slice = &mut tris[first..first + count];
        let bounds = slice.iter().fold(Aabb::EMPTY, |b, t| b.union(&t.bounds()));
        nodes[ni].bounds = bounds;
        if count <= LEAF_SIZE {
            continue;
        }
        let Some((axis, split_pos)) = sah_split(slice) else { continue };

        let mut mid = partition_in_place(slice, |t| t.centroid()[axis] < split_pos);
        if mid == 0 || mid == count {
            mid = count / 2;
            slice.select_nth_unstable_by(mid, |a, b| a.centroid()[axis].total_cmp(&b.centroid()[axis]));
        }

        let left = nodes.len();
        nodes.push(BvhNode { bounds: Aabb::EMPTY, first: first as u32, count: mid as u32, axis: 0 });
        nodes.push(BvhNode { bounds: Aabb::EMPTY, first: (first + mid) as u32, count: (count - mid) as u32, axis: 0 });
        nodes[ni] = BvhNode { bounds, first: left as u32, count: 0, axis: axis as u8 };
        work.push(left);
        work.push(left + 1);
    }
    nodes
}

/// Best (axis, split position) by the surface area heuristic, or None if a
/// leaf is cheaper.
fn sah_split(tris: &[Triangle]) -> Option<(usize, f64)> {
    let mut cb = Aabb::EMPTY;
    for t in tris {
        cb.grow(t.centroid());
    }
    let mut best: Option<(usize, f64, f64)> = None;
    for axis in 0..3 {
        let (lo, hi) = (cb.min[axis], cb.max[axis]);
        if hi - lo < 1e-12 {
            continue;
        }
        let scale = SAH_BINS as f64 / (hi - lo);
        let mut bins = [(Aabb::EMPTY, 0usize); SAH_BINS];
        for t in tris {
            let b = (((t.centroid()[axis] - lo) * scale) as usize).min(SAH_BINS - 1);
            bins[b].0 = bins[b].0.union(&t.bounds());
            bins[b].1 += 1;
        }
        for split in 1..SAH_BINS {
            let (l, r) = bins.split_at(split);
            let fold = |s: &[(Aabb, usize)]| s.iter().fold((Aabb::EMPTY, 0), |(a, n), (b, m)| (a.union(b), n + m));
            let ((la, ln), (ra, rn)) = (fold(l), fold(r));
            if ln == 0 || rn == 0 {
                continue;
            }
            let cost = la.area() * ln as f64 + ra.area() * rn as f64;
            if best.is_none_or(|(_, _, c)| cost < c) {
                best = Some((axis, lo + split as f64 / scale, cost));
            }
        }
    }
    let (axis, pos, cost) = best?;
    let leaf_cost = tris.iter().fold(Aabb::EMPTY, |b, t| b.union(&t.bounds())).area() * tris.len() as f64;
    (cost < leaf_cost).then_some((axis, pos))
}

/// In-place partition; returns the number of elements satisfying `pred`.
fn partition_in_place<T>(s: &mut [T], pred: impl Fn(&T) -> bool) -> usize {
    let mut i = 0;
    for j in 0..s.len() {
        if pred(&s[j]) {
            s.swap(i, j);
            i += 1;
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(a: DVec3, b: DVec3, c: DVec3) -> Triangle {
        Triangle { v0: a, e1: b - a, e2: c - a, normals: [DVec3::Z; 3], material: 0 }
    }

    #[test]
    fn bvh_matches_brute_force() {
        // A grid of small triangles at varying depths.
        let mut tris = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let o = DVec3::new(i as f64, j as f64, -5.0 - ((i * 7 + j * 3) % 5) as f64);
                tris.push(tri(o, o + DVec3::X, o + DVec3::Y));
            }
        }
        let brute = tris.clone();
        let bvh = build_bvh(&mut tris);
        let world = World { triangles: tris, spheres: vec![], materials: vec![], lights: vec![], bvh };

        for k in 0..200 {
            let target = DVec3::new((k % 23) as f64 * 0.9, (k % 19) as f64 * 1.05, -6.0);
            let ray = Ray::new(DVec3::new(10.0, 10.0, 5.0), target - DVec3::new(10.0, 10.0, 5.0));
            let expected = brute
                .iter()
                .filter_map(|t| t.intersect(&ray, f64::INFINITY))
                .map(|h| h.0)
                .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.min(t))));
            let got = world.intersect(&ray, f64::INFINITY).map(|h| h.t);
            assert_eq!(expected.is_some(), got.is_some(), "ray {k}");
            if let (Some(e), Some(g)) = (expected, got) {
                assert!((e - g).abs() < 1e-9, "ray {k}: {e} vs {g}");
            }
        }
    }

    #[test]
    fn sphere_hit_from_inside_uses_far_root() {
        let s = Sphere { center: DVec3::ZERO, radius: 2.0, material: 0 };
        let ray = Ray::new(DVec3::ZERO, DVec3::X);
        assert!((s.intersect(&ray, f64::INFINITY).unwrap() - 2.0).abs() < 1e-12);
    }
}
