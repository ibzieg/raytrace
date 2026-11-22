//! Photon mapping ray tracer, ported from a 2011 Java course project.
//!
//! Typical use: [`Scene::load`] → [`World::from_scene`] → optionally
//! [`PhotonMaps::build`] → [`Renderer::render`].

pub mod geometry;
pub mod irradiance_cache;
pub mod legacy;
pub mod light;
pub mod material;
pub mod mesh;
pub mod optics;
pub mod photon;
pub mod render;
pub mod rng;
pub mod scene;

pub use geometry::World;
pub use photon::PhotonMaps;
pub use render::{Display, Image, Renderer};
pub use scene::{Scene, SceneFile};
