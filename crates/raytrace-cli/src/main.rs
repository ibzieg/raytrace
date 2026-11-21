use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use raytrace::{PhotonMaps, Renderer, Scene, World, legacy};

#[derive(Parser)]
#[command(version, about = "Photon mapping ray tracer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Render a .toml scene to a PNG.
    Render {
        scene: PathBuf,
        /// Output image path.
        #[arg(short, long, default_value = "render.png")]
        output: PathBuf,
        /// Image height (width follows the camera aspect unless set).
        #[arg(long)]
        height: Option<u32>,
        #[arg(long)]
        width: Option<u32>,
        /// Samples per pixel.
        #[arg(short, long)]
        samples: Option<u32>,
        /// Shade with photon maps instead of direct lighting.
        #[arg(short, long)]
        photons: bool,
        /// Photons emitted per light (implies --photons).
        #[arg(long)]
        photon_count: Option<u32>,
        #[arg(long)]
        soft_shadows: bool,
        /// Seed for photon emission.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Convert a 2011 Java-app XML .scene file to TOML.
    Convert {
        input: PathBuf,
        /// Output .toml path (defaults to the input name with .toml).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Directory holding the OBJ files, relative to the output file.
        #[arg(long, default_value = "../objects")]
        mesh_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Render { scene, output, height, width, samples, photons, photon_count, soft_shadows, seed } => {
            let t0 = Instant::now();
            let mut scene = Scene::load(&scene)?;
            let r = &mut scene.file.render;
            if let Some(h) = height {
                r.height = h;
                r.width = 0;
            }
            if let Some(w) = width {
                r.width = w;
            }
            if let Some(s) = samples {
                r.samples = s;
            }
            r.soft_shadows |= soft_shadows;
            let p = &mut scene.file.photon;
            if let Some(n) = photon_count {
                p.photons_per_light = n;
            }
            p.enabled |= photons || photon_count.is_some();

            let world = World::from_scene(&scene);
            eprintln!(
                "loaded {} objects ({} triangles, {} spheres), {} lights in {:.2?}",
                scene.objects.len(),
                world.triangles.len(),
                world.spheres.len(),
                world.lights.len(),
                t0.elapsed()
            );

            let maps = scene.file.photon.enabled.then(|| {
                let t = Instant::now();
                let maps = PhotonMaps::build(&world, &scene.file.photon, seed);
                eprintln!(
                    "photon maps: {} direct, {} indirect, {} caustic in {:.2?}",
                    maps.direct.len(),
                    maps.indirect.len(),
                    maps.caustic.len(),
                    t.elapsed()
                );
                maps
            });

            let (w, h) = scene.image_size();
            let t = Instant::now();
            let image = Renderer::new(&scene, &world, maps.as_ref()).render(w, h, &|done, total| {
                if done % 8 == 0 || done == total {
                    eprint!("\rrendering {w}x{h}: {:3}%", done * 100 / total);
                    std::io::stderr().flush().ok();
                }
            });
            eprintln!(" in {:.2?}", t.elapsed());
            image.save_png(&output)?;
            eprintln!("wrote {}", output.display());
        }
        Command::Convert { input, output, mesh_dir } => {
            let xml = std::fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?;
            let scene = legacy::convert(&xml, &mesh_dir).with_context(|| format!("converting {}", input.display()))?;
            let output = output.unwrap_or_else(|| input.with_extension("toml"));
            let name = input.file_name().map_or("".into(), |n| n.to_string_lossy());
            let header = format!("# Converted from legacy scene '{name}'.\n\n");
            std::fs::write(&output, header + &toml::to_string(&scene)?)?;
            warn_missing_meshes(&output, &scene);
            eprintln!("wrote {}", output.display());
        }
    }
    Ok(())
}

fn warn_missing_meshes(scene_path: &Path, scene: &raytrace::SceneFile) {
    let base = scene_path.parent().unwrap_or(Path::new("."));
    for o in &scene.objects {
        if !base.join(&o.mesh).exists() {
            eprintln!("warning: mesh not found: {}", o.mesh.display());
        }
    }
}
