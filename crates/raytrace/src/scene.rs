//! Scene description (TOML on disk).
//!
//! All positions in the file are in world space. Matrices are written as four
//! rows, so `transform[r][c]` is row `r`, column `c`, and translation lives in
//! the last column.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{DMat4, DVec3, DVec4};
use serde::{Deserialize, Serialize};

use crate::mesh::{Mesh, load_obj};

pub type Rows4 = [[f64; 4]; 4];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneFile {
    pub camera: CameraDef,
    #[serde(default, rename = "light", skip_serializing_if = "Vec::is_empty")]
    pub lights: Vec<LightDef>,
    #[serde(default, rename = "object", skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<ObjectDef>,
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
    /// Replace direct Phong shading with photon map radiance estimates.
    pub enabled: bool,
    pub photons_per_light: u32,
    pub max_depth: u32,
    /// Total power shared by all emitted photons.
    pub light_power: f64,
    pub radius: PhotonTriple,
    pub exposure: PhotonTriple,
}

impl Default for PhotonSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            photons_per_light: 100_000,
            max_depth: 5,
            light_power: 3300.0,
            radius: PhotonTriple { direct: 1.0, indirect: 1.0, caustic: 1.0 },
            exposure: PhotonTriple { direct: 1.0, indirect: 1.0, caustic: 1.0 },
        }
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
        let objects = file
            .objects
            .iter()
            .map(|o| {
                let mesh = load_obj(&base_dir.join(&o.mesh))?;
                let model = o.transform.as_ref().map_or(DMat4::IDENTITY, rows_to_mat);
                Ok((mesh, model))
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
