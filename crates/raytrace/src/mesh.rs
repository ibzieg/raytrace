//! Wavefront OBJ/MTL loading.
//!
//! A small hand-written parser instead of a crate, because the old scenes rely
//! on a few non-standard extensions:
//!   - an `@sphere` line marks the mesh as a perfect analytic sphere, whose
//!     center is the vertex mean and radius the farthest vertex from it;
//!   - `Ir` / `It` material keys (see [`Material`]).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::DVec3;

use crate::material::Material;

#[derive(Clone, Debug)]
pub struct MeshTriangle {
    pub verts: [usize; 3],
    pub normals: [DVec3; 3],
    pub material: usize,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<MeshTriangle>,
    pub materials: Vec<Material>,
    /// Treat as an analytic sphere (`@sphere` flag) instead of triangles.
    pub is_sphere: bool,
}

impl Mesh {
    /// Mean of all vertex positions, which the old renderer used as the
    /// object's center.
    pub fn center(&self) -> DVec3 {
        if self.positions.is_empty() {
            return DVec3::ZERO;
        }
        self.positions.iter().copied().sum::<DVec3>() / self.positions.len() as f64
    }

    /// Distance from [`Mesh::center`] to the farthest vertex.
    pub fn bounding_radius(&self) -> f64 {
        let c = self.center();
        self.positions.iter().map(|p| p.distance(c)).fold(0.0, f64::max)
    }

    /// Material of the first face, used for `@sphere` meshes.
    pub fn primary_material(&self) -> usize {
        self.triangles.first().map_or(0, |t| t.material)
    }
}

struct Polygon {
    verts: Vec<usize>,
    material: usize,
    smoothing_group: u32,
}

pub fn load_obj(path: &Path) -> Result<Mesh> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));

    let mut positions = Vec::new();
    let mut polygons = Vec::new();
    // Index 0 is the fallback material for faces before any `usemtl`.
    let mut materials = vec![Material::default()];
    let mut material_index: HashMap<String, usize> = HashMap::new();
    let mut current_material = 0;
    let mut smoothing_group = 0;
    let mut saw_smoothing = false;
    let mut is_sphere = false;

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("@sphere") {
            is_sphere = true;
            continue;
        }
        let mut tokens = line.split_whitespace();
        let Some(keyword) = tokens.next() else { continue };
        let ctx = || format!("{}:{}", path.display(), line_no + 1);
        match keyword {
            "v" => positions.push(parse_vec3(&mut tokens).with_context(ctx)?),
            "f" => {
                let mut verts = Vec::new();
                for tok in tokens {
                    let idx_str = tok.split('/').next().unwrap_or("");
                    let idx: i64 = idx_str.parse().with_context(ctx)?;
                    let resolved = if idx < 0 { positions.len() as i64 + idx } else { idx - 1 };
                    if resolved < 0 || resolved as usize >= positions.len() {
                        bail!("{}: vertex index {idx} out of range", ctx());
                    }
                    verts.push(resolved as usize);
                }
                if verts.len() >= 3 {
                    polygons.push(Polygon { verts, material: current_material, smoothing_group });
                }
            }
            "s" => {
                saw_smoothing = true;
                smoothing_group = match tokens.next() {
                    Some("off") | None => 0,
                    Some(g) => g.parse().unwrap_or(0),
                };
            }
            "mtllib" => {
                for name in tokens {
                    let mtl_path = dir.join(name);
                    match load_mtl(&mtl_path) {
                        Ok(mats) => {
                            for m in mats {
                                material_index.insert(m.name.clone(), materials.len());
                                materials.push(m);
                            }
                        }
                        Err(e) => eprintln!("warning: {e:#}"),
                    }
                }
            }
            "usemtl" => {
                let name = tokens.next().unwrap_or("");
                current_material = match material_index.get(name) {
                    Some(&i) => i,
                    None => {
                        eprintln!("warning: {}: unknown material '{name}'", ctx());
                        0
                    }
                };
            }
            _ => {} // vn/vt are ignored: normals are always recomputed, as before.
        }
    }

    // The old loader smoothed every vertex regardless of `s` lines. Keep that
    // for files without any smoothing info, but respect groups when present
    // so e.g. box walls stay flat.
    if !saw_smoothing {
        for p in &mut polygons {
            p.smoothing_group = 1;
        }
    }

    let face_normals: Vec<DVec3> = polygons.iter().map(|p| newell_normal(&positions, &p.verts)).collect();

    let mut vertex_normals: HashMap<(usize, u32), DVec3> = HashMap::new();
    for (poly, n) in polygons.iter().zip(&face_normals) {
        if poly.smoothing_group != 0 {
            for &v in &poly.verts {
                *vertex_normals.entry((v, poly.smoothing_group)).or_default() += *n;
            }
        }
    }

    let mut triangles = Vec::new();
    for (poly, &face_n) in polygons.iter().zip(&face_normals) {
        let normal_of = |v: usize| {
            if poly.smoothing_group == 0 {
                face_n
            } else {
                vertex_normals[&(v, poly.smoothing_group)].normalize_or(face_n)
            }
        };
        // Fan triangulation. (The old code silently used only the first
        // three vertices of every polygon.)
        for i in 1..poly.verts.len() - 1 {
            let verts = [poly.verts[0], poly.verts[i], poly.verts[i + 1]];
            triangles.push(MeshTriangle { verts, normals: verts.map(normal_of), material: poly.material });
        }
    }

    Ok(Mesh { positions, triangles, materials, is_sphere })
}

pub fn load_mtl(path: &PathBuf) -> Result<Vec<Material>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut mats: Vec<Material> = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let mut tokens = raw.split_whitespace();
        let Some(keyword) = tokens.next() else { continue };
        let ctx = || format!("{}:{}", path.display(), line_no + 1);
        if keyword == "newmtl" {
            mats.push(Material { name: tokens.next().unwrap_or("").to_string(), ..Default::default() });
            continue;
        }
        let Some(m) = mats.last_mut() else { continue };
        match keyword {
            "Ka" => m.ka = parse_vec3(&mut tokens).with_context(ctx)?,
            "Kd" => m.kd = parse_vec3(&mut tokens).with_context(ctx)?,
            "Ks" => m.ks = parse_vec3(&mut tokens).with_context(ctx)?,
            "Ns" => m.shininess = parse_f64(&mut tokens).with_context(ctx)?,
            "Ni" => m.ior = parse_f64(&mut tokens).with_context(ctx)?,
            // Only the first component was ever used.
            "Ir" => m.reflectivity = parse_f64(&mut tokens).with_context(ctx)?,
            "It" => m.transmissivity = parse_f64(&mut tokens).with_context(ctx)?,
            _ => {}
        }
    }
    Ok(mats)
}

fn parse_f64<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Result<f64> {
    let tok = tokens.next().context("missing number")?;
    tok.parse().with_context(|| format!("bad number '{tok}'"))
}

fn parse_vec3<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Result<DVec3> {
    Ok(DVec3::new(parse_f64(tokens)?, parse_f64(tokens)?, parse_f64(tokens)?))
}

/// Polygon normal via Newell's method (robust for non-planar polygons).
fn newell_normal(positions: &[DVec3], verts: &[usize]) -> DVec3 {
    let mut n = DVec3::ZERO;
    for (i, &a) in verts.iter().enumerate() {
        let p = positions[a];
        let q = positions[verts[(i + 1) % verts.len()]];
        n += DVec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
    }
    n.normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let p = dir.join(name);
        fs::File::create(&p).unwrap().write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_quads_materials_and_sphere_flag() {
        let dir = std::env::temp_dir().join(format!("raytrace-mesh-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        write_tmp(&dir, "m.mtl", "newmtl glass\nKd 0.2 0.2 0.8\nIr 0.1\nIt 0.9\nNi 1.5\n");
        let obj = write_tmp(
            &dir,
            "q.obj",
            "mtllib m.mtl\n@sphere\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\ns 0\nusemtl glass\nf 1//1 2//1 3//1 4//1\n",
        );
        let mesh = load_obj(&obj).unwrap();
        assert!(mesh.is_sphere);
        assert_eq!(mesh.triangles.len(), 2);
        let m = &mesh.materials[mesh.primary_material()];
        assert_eq!(m.name, "glass");
        assert_eq!(m.transmissivity, 0.9);
        assert!((mesh.triangles[0].normals[0] - DVec3::Z).length() < 1e-12);
        fs::remove_dir_all(dir).ok();
    }
}
