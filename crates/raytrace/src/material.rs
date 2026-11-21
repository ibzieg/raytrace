use glam::DVec3;

/// Surface material, read from an MTL file.
///
/// Besides the standard Phong terms this supports two non-standard MTL keys
/// from the original 2011 renderer:
///   `Ir` - fraction of light carried by the mirror reflection ray
///   `It` - fraction of light carried by the refracted ray
#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub ka: DVec3,
    pub kd: DVec3,
    pub ks: DVec3,
    /// Phong exponent (`Ns`).
    pub shininess: f64,
    /// `Ir`
    pub reflectivity: f64,
    /// `It`
    pub transmissivity: f64,
    /// Index of refraction (`Ni`). Values <= 1 mean "use the default glass IOR".
    pub ior: f64,
}

impl Default for Material {
    /// Matches the defaults of the original `MaterialCell`, which a `newmtl`
    /// block starts from. Note the 0.3 reflectivity: a material without an
    /// `Ir` line is still somewhat mirror-like.
    fn default() -> Self {
        Self {
            name: "default".into(),
            ka: DVec3::splat(0.2),
            kd: DVec3::new(0.8, 0.0, 0.0),
            ks: DVec3::ONE,
            shininess: 90.0,
            reflectivity: 0.3,
            transmissivity: 0.0,
            ior: 0.0,
        }
    }
}

pub const DEFAULT_GLASS_IOR: f64 = 1.5;

impl Material {
    pub fn effective_ior(&self) -> f64 {
        if self.ior > 1.0 { self.ior } else { DEFAULT_GLASS_IOR }
    }
}
