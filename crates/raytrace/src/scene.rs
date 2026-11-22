//! Scene description (TOML on disk).
//!
//! All positions in the file are in world space. Matrices are written as four
//! rows, so `transform[r][c]` is row `r`, column `c`, and translation lives in
//! the last column.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{DMat4, DVec3, DVec4};
use serde::{Deserialize, Serialize};

use crate::material::Material;
use crate::mesh::{Mesh, load_obj};

pub type Rows4 = [[f64; 4]; 4];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneFile {
    pub camera: CameraDef,
    #[serde(default, rename = "light", skip_serializing_if = "Vec::is_empty")]
    pub lights: Vec<LightDef>,
    #[serde(default, rename = "object", skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<ObjectDef>,
    /// Materials that replace same-named MTL materials in every mesh, or that
    /// objects select with `material = "name"`.
    #[serde(default, rename = "material", skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialDef>,
    #[serde(default)]
    pub render: RenderSettings,
    #[serde(default)]
    pub photon: PhotonSettings,
}

/// The camera looks down its local -Z axis. The image covers `window` on the
/// near plane, as with OpenGL's `glFrustum`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraDef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up: Option<[f64; 3]>,
    /// World-to-camera matrix; an alternative to eye/target/up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_matrix: Option<Rows4>,
    pub near: f64,
    pub far: f64,
    pub window: Window,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Window {
    pub left: f64,
    pub right: f64,
    pub bottom: f64,
    pub top: f64,
}

impl Window {
    pub fn aspect(&self) -> f64 {
        (self.right - self.left) / (self.top - self.bottom)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightDef {
    pub position: [f64; 3],
    #[serde(default = "one3")]
    pub diffuse: [f64; 3],
    #[serde(default = "one3")]
    pub specular: [f64; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectDef {
    /// OBJ path, relative to the scene file.
    pub mesh: PathBuf,
    /// Object-to-world matrix (rows). Identity when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Rows4>,
    /// Convenience placement, applied after `transform` as
    /// translate · rotate · scale. Rotation is in degrees about X, then Y,
    /// then Z.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translate: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Scale>,
    /// Name of a scene `[[material]]` to use for every face.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Scale {
    Uniform(f64),
    Axes([f64; 3]),
}

impl ObjectDef {
    pub fn new(mesh: impl Into<PathBuf>) -> Self {
        Self { mesh: mesh.into(), transform: None, translate: None, rotate: None, scale: None, material: None }
    }

    /// Object-to-world matrix.
    pub fn model_matrix(&self) -> DMat4 {
        let base = self.transform.as_ref().map_or(DMat4::IDENTITY, rows_to_mat);
        let t = DMat4::from_translation(self.translate.map_or(DVec3::ZERO, DVec3::from));
        let [rx, ry, rz] = self.rotate.unwrap_or_default().map(f64::to_radians);
        let r = DMat4::from_rotation_z(rz) * DMat4::from_rotation_y(ry) * DMat4::from_rotation_x(rx);
        let s = DMat4::from_scale(match self.scale {
            None => DVec3::ONE,
            Some(Scale::Uniform(k)) => DVec3::splat(k),
            Some(Scale::Axes(v)) => v.into(),
        });
        t * r * s * base
    }
}

/// A material defined in the scene file. Unlike MTL `newmtl` blocks (which
/// keep the original renderer's defaults), unspecified fields here default
/// to a plain matte gray.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialDef {
    pub name: String,
    #[serde(default)]
    pub ka: [f64; 3],
    #[serde(default = "gray3")]
    pub kd: [f64; 3],
    #[serde(default = "spec3")]
    pub ks: [f64; 3],
    #[serde(default = "default_shininess")]
    pub shininess: f64,
    #[serde(default)]
    pub reflectivity: f64,
    #[serde(default)]
    pub transmissivity: f64,
    #[serde(default = "default_ior")]
    pub ior: f64,
}

impl MaterialDef {
    pub fn to_material(&self) -> Material {
        Material {
            name: self.name.clone(),
            ka: self.ka.into(),
            kd: self.kd.into(),
            ks: self.ks.into(),
            shininess: self.shininess,
            reflectivity: self.reflectivity,
            transmissivity: self.transmissivity,
            ior: self.ior,
        }
    }
}

fn gray3() -> [f64; 3] {
    [0.8; 3]
}

fn spec3() -> [f64; 3] {
    [0.3; 3]
}

fn default_shininess() -> f64 {
    32.0
}

fn default_ior() -> f64 {
    1.5
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderSettings {
    /// Image width in pixels. 0 = derive from `height` and the window aspect.
    pub width: u32,
    pub height: u32,
    /// Samples per pixel (n-rooks distributed). 1 = a single ray through the
    /// pixel center.
    pub samples: u32,
    /// Maximum recursion depth for reflection / refraction rays.
    pub max_depth: u32,
    pub shadows: bool,
    pub soft_shadows: bool,
    /// Soft shadows cast `shadow_rays`² rays.
    pub shadow_rays: u32,
    /// Radius of the soft shadow sample disc.
    pub shadow_jitter: f64,
    pub reflections: bool,
    pub refractions: bool,
    pub background: [f64; 3],
    /// Checkerboard on the far plane instead of a flat background.
    pub checker_background: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 0,
            height: 600,
            samples: 4,
            max_depth: 5,
            shadows: true,
            soft_shadows: false,
            shadow_rays: 4,
            shadow_jitter: 0.25,
            reflections: true,
            refractions: true,
            background: [0.0; 3],
            checker_background: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhotonSettings {
    /// Shade with photon maps instead of Phong lighting.
    pub enabled: bool,
    pub photons_per_light: u32,
    pub max_depth: u32,
    /// Total power shared by all emitted photons.
    pub light_power: f64,
    pub direct_lighting: DirectLighting,
    /// Maximum search radius of each map's estimate.
    pub radius: PhotonTriple,
    pub exposure: PhotonTriple,
    pub gather: GatherCounts,
}

impl Default for PhotonSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            photons_per_light: 100_000,
            max_depth: 5,
            light_power: 3300.0,
            direct_lighting: DirectLighting::default(),
            radius: PhotonTriple { direct: 1.0, indirect: 1.0, caustic: 1.0 },
            exposure: PhotonTriple { direct: 1.0, indirect: 1.0, caustic: 1.0 },
            gather: GatherCounts::default(),
        }
    }
}

impl PhotonSettings {
    /// Whether maps built with `other` can be reused with these settings;
    /// the rest only affects how they are read.
    pub fn same_emission(&self, other: &Self) -> bool {
        self.photons_per_light == other.photons_per_light
            && self.max_depth == other.max_depth
            && self.light_power == other.light_power
    }
}

/// How photon-mapped renders light surfaces directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DirectLighting {
    /// Shadow rays to each light, as in Jensen's method.
    #[default]
    Rays,
    /// The direct photon map's estimate, as the 2011 version did. Faster but
    /// blotchy.
    Map,
}

/// Photons gathered per estimate: the nearest this many within the map's
/// radius. 0 takes every photon within the radius.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GatherCounts {
    pub direct: u32,
    pub indirect: u32,
    pub caustic: u32,
}

impl Default for GatherCounts {
    fn default() -> Self {
        Self { direct: 200, indirect: 200, caustic: 80 }
    }
}

/// One value for each of the three photon maps.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhotonTriple {
    pub direct: f64,
    pub indirect: f64,
    pub caustic: f64,
}

fn one3() -> [f64; 3] {
    [1.0; 3]
}

pub fn rows_to_mat(rows: &Rows4) -> DMat4 {
    DMat4::from_cols_array_2d(rows).transpose()
}

pub fn mat_to_rows(m: &DMat4) -> Rows4 {
    m.transpose().to_cols_array_2d()
}

impl CameraDef {
    pub fn view_matrix(&self) -> Result<DMat4> {
        if let Some(rows) = &self.view_matrix {
            return Ok(rows_to_mat(rows));
        }
        match (self.eye, self.target) {
            (Some(eye), Some(target)) => Ok(look_at(eye.into(), target.into(), self.up.map_or(DVec3::Y, DVec3::from))),
            _ => bail!("camera needs either `view_matrix` or `eye` + `target`"),
        }
    }
}

/// Applies scene materials to a mesh: an object's `material` replaces every
/// face's material, and scene materials replace same-named MTL ones.
fn apply_materials(mesh: &mut Mesh, object: &ObjectDef, scene: &HashMap<&str, Material>) -> Result<()> {
    if let Some(name) = &object.material {
        let m = scene
            .get(name.as_str())
            .with_context(|| format!("{}: no [[material]] named '{name}'", object.mesh.display()))?;
        mesh.materials = vec![m.clone()];
        for t in &mut mesh.triangles {
            t.material = 0;
        }
        return Ok(());
    }
    for (i, m) in mesh.materials.iter_mut().enumerate() {
        if let Some(over) = scene.get(m.name.as_str()) {
            *m = over.clone();
            mesh.unresolved_materials.retain(|&u| u != i);
        }
    }
    if !mesh.unresolved_materials.is_empty() {
        for w in &mesh.warnings {
            eprintln!("warning: {w}");
        }
        for &i in &mesh.unresolved_materials {
            eprintln!(
                "warning: {}: material '{}' not found; using defaults",
                object.mesh.display(),
                mesh.materials[i].name
            );
        }
    }
    Ok(())
}

/// Right-handed world-to-camera matrix, like `gluLookAt`.
pub fn look_at(eye: DVec3, target: DVec3, up: DVec3) -> DMat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    DMat4::from_cols(
        DVec4::new(s.x, u.x, -f.x, 0.0),
        DVec4::new(s.y, u.y, -f.y, 0.0),
        DVec4::new(s.z, u.z, -f.z, 0.0),
        DVec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// A scene file with its meshes loaded.
pub struct Scene {
    pub file: SceneFile,
    pub view: DMat4,
    pub objects: Vec<(Mesh, DMat4)>,
}

impl Scene {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: SceneFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        Self::from_file(file, base)
    }

    pub fn from_file(file: SceneFile, base_dir: &Path) -> Result<Self> {
        let view = file.camera.view_matrix()?;
        let scene_materials: HashMap<&str, Material> =
            file.materials.iter().map(|m| (m.name.as_str(), m.to_material())).collect();
        let objects = file
            .objects
            .iter()
            .map(|o| {
                let mut mesh = load_obj(&base_dir.join(&o.mesh))?;
                apply_materials(&mut mesh, o, &scene_materials)?;
                Ok((mesh, o.model_matrix()))
            })
            .collect::<Result<_>>()?;
        Ok(Self { file, view, objects })
    }

    /// Output size, deriving a missing dimension from the window aspect so
    /// pixels stay square.
    pub fn image_size(&self) -> (u32, u32) {
        self.image_size_for(&self.file.render)
    }

    /// [`Scene::image_size`] with settings other than the scene's own.
    pub fn image_size_for(&self, r: &RenderSettings) -> (u32, u32) {
        let aspect = self.file.camera.window.aspect();
        match (r.width, r.height) {
            (0, 0) => ((600.0 * aspect).round() as u32, 600),
            (0, h) => (((h as f64) * aspect).round() as u32, h),
            (w, 0) => (w, ((w as f64) / aspect).round() as u32),
            (w, h) => (w, h),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_composes_translate_rotate_scale() {
        let o = ObjectDef {
            translate: Some([1.0, 2.0, 3.0]),
            rotate: Some([0.0, 90.0, 0.0]),
            scale: Some(Scale::Uniform(2.0)),
            ..ObjectDef::new("x.obj")
        };
        // Scale, then rotate +X onto -Z, then translate.
        let p = o.model_matrix().transform_point3(DVec3::X);
        assert!(p.abs_diff_eq(DVec3::new(1.0, 2.0, 1.0), 1e-12), "{p}");
    }

    #[test]
    fn parses_scene_materials_and_scale_forms() {
        let file: SceneFile = toml::from_str(
            r#"
            [camera]
            eye = [0, 0, 5]
            target = [0, 0, 0]
            near = 1
            far = 100
            window = { left = -1, right = 1, bottom = -1, top = 1 }
            [[material]]
            name = "glass"
            transmissivity = 0.9
            [[object]]
            mesh = "a.obj"
            scale = [1, 2, 3]
            material = "glass"
            [[object]]
            mesh = "b.obj"
            scale = 0.5
            "#,
        )
        .unwrap();
        assert!(matches!(file.objects[0].scale, Some(Scale::Axes([1.0, 2.0, 3.0]))));
        assert!(matches!(file.objects[1].scale, Some(Scale::Uniform(0.5))));
        let m = file.materials[0].to_material();
        assert_eq!((m.transmissivity, m.reflectivity, m.ior), (0.9, 0.0, 1.5));
    }
}
