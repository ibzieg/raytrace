mod orbit;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use raytrace::scene::DirectLighting;
use raytrace::{Display, PhotonMaps, Renderer, Scene, World, legacy};

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
        #[command(flatten)]
        scene: SceneArgs,
        /// Output image path.
        #[arg(short, long, default_value = "render.png")]
        output: PathBuf,
    },
    /// Render frames with the camera orbiting the scene's target, and
    /// optionally assemble them into a video or GIF with ffmpeg.
    Animate {
        #[command(flatten)]
        scene: SceneArgs,
        /// Directory for the frames (frame_0000.png, ...).
        #[arg(short, long, default_value = "frames")]
        output: PathBuf,
        #[arg(long, default_value_t = 120)]
        frames: usize,
        #[arg(long, default_value_t = 30)]
        fps: u32,
        /// Degrees the camera sweeps (default 36, or 360 with --turntable). By
        /// default it swings across the scene's own view, easing in and out.
        #[arg(long)]
        orbit: Option<f64>,
        /// Turn at a steady rate instead of swinging, looping seamlessly; the
        /// sweep defaults to 360°.
        #[arg(long)]
        turntable: bool,
        /// Also write an MP4 (H.264) here.
        #[arg(long)]
        video: Option<PathBuf>,
        /// Also write a looping GIF here. A swing plays forward then back.
        #[arg(long)]
        gif: Option<PathBuf>,
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

/// Scene file and the settings that can override it.
#[derive(Args)]
struct SceneArgs {
    scene: PathBuf,
    /// Image height (width follows the camera aspect unless set).
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    width: Option<u32>,
    /// Samples per pixel.
    #[arg(short, long)]
    samples: Option<u32>,
    /// Shade with photon mapping instead of Phong lighting.
    #[arg(short, long)]
    photons: bool,
    /// Photons emitted per light (implies --photons).
    #[arg(long)]
    photon_count: Option<u32>,
    /// Extra photons per light aimed at glass and mirrors for the caustic
    /// map; 0 takes caustics from the main photons.
    #[arg(long)]
    caustic_photons: Option<u32>,
    /// Take direct light from the photon map instead of shadow rays, as the
    /// 2011 version did.
    #[arg(long)]
    direct_map: bool,
    /// Final gather rays per shading point; 0 reads the indirect map
    /// directly.
    #[arg(long)]
    final_gather: Option<u32>,
    /// Gather at every shading point instead of interpolating cached gathers.
    #[arg(long)]
    no_irradiance_cache: bool,
    #[arg(long)]
    soft_shadows: bool,
    /// Exposure in stops (2^EV scale) applied when writing the PNG.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    exposure: f64,
    /// Display gamma; 1.0 matches the original renderer, 2.2 ≈ sRGB.
    #[arg(long, default_value_t = 1.0)]
    gamma: f64,
    /// Seed for photon emission.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// A scene ready to render: loaded, overridden and with its photon maps.
struct Loaded {
    scene: Scene,
    world: World,
    maps: Option<PhotonMaps>,
    display: Display,
}

impl SceneArgs {
    fn load(&self) -> Result<Loaded> {
        let t0 = Instant::now();
        let mut scene = Scene::load(&self.scene)?;
        let r = &mut scene.file.render;
        if let Some(h) = self.height {
            r.height = h;
            r.width = 0;
        }
        if let Some(w) = self.width {
            r.width = w;
        }
        if let Some(s) = self.samples {
            r.samples = s;
        }
        r.soft_shadows |= self.soft_shadows;
        let p = &mut scene.file.photon;
        if let Some(n) = self.photon_count {
            p.photons_per_light = n;
        }
        if let Some(n) = self.caustic_photons {
            p.caustic_photons = n;
        }
        p.enabled |= self.photons || self.photon_count.is_some();
        if self.direct_map {
            p.direct_lighting = DirectLighting::Map;
        }
        if let Some(n) = self.final_gather {
            p.final_gather = n;
        }
        p.irradiance_cache &= !self.no_irradiance_cache;

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
            let maps = PhotonMaps::build(&world, &scene.file.photon, self.seed);
            eprintln!(
                "photon maps: {} direct, {} indirect, {} caustic in {:.2?}",
                maps.direct.len(),
                maps.indirect.len(),
                maps.caustic.len(),
                t.elapsed()
            );
            maps
        });
        Ok(Loaded { scene, world, maps, display: Display { exposure: self.exposure, gamma: self.gamma } })
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Render { scene, output } => {
            let l = scene.load()?;
            let (w, h) = l.scene.image_size();
            let t = Instant::now();
            let image = Renderer::new(&l.scene, &l.world, l.maps.as_ref()).render(w, h, &|done, total| {
                if done % 8 == 0 || done == total {
                    eprint!("\rrendering {w}x{h}: {:3}%", done * 100 / total);
                    std::io::stderr().flush().ok();
                }
            });
            eprintln!(" in {:.2?}", t.elapsed());
            image.save_png_with(&output, l.display)?;
            eprintln!("wrote {}", output.display());
        }
        Command::Animate { scene, output, frames, fps, orbit, turntable, video, gif } => {
            if frames == 0 {
                bail!("--frames must be at least 1");
            }
            let degrees = orbit.unwrap_or(if turntable { 360.0 } else { 36.0 });
            let l = scene.load()?;
            std::fs::create_dir_all(&output).with_context(|| format!("creating {}", output.display()))?;
            let path = orbit::Orbit::new(&l.scene, &l.world);
            let mut renderer = Renderer::new(&l.scene, &l.world, l.maps.as_ref());
            let (w, h) = l.scene.image_size();
            let t = Instant::now();
            for (i, angle) in orbit::angles(frames, degrees, turntable).into_iter().enumerate() {
                renderer.view = path.view(angle);
                let image = renderer.render(w, h, &|done, total| {
                    if done % 8 == 0 || done == total {
                        eprint!("\rframe {}/{frames} ({w}x{h}): {:3}%", i + 1, done * 100 / total);
                        std::io::stderr().flush().ok();
                    }
                });
                image.save_png_with(&output.join(format!("frame_{i:04}.png")), l.display)?;
            }
            let elapsed = t.elapsed();
            eprintln!("\nrendered {frames} frames in {elapsed:.2?} ({:.2?} each)", elapsed / frames as u32);
            let pattern = output.join("frame_%04d.png");
            if let Some(video) = video {
                encode(&mp4_args(&pattern, fps, &video), &video)?;
            }
            if let Some(gif) = gif {
                encode(&gif_args(&pattern, fps, w, !turntable && frames > 1, &gif), &gif)?;
            }
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

/// H.264 in yuv420p (which QuickTime and browsers need, along with even
/// dimensions), high quality, streamable.
fn mp4_args(pattern: &Path, fps: u32, out: &Path) -> Vec<String> {
    let mut args = input_args(pattern, fps);
    args.extend(
        [
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
        ]
        .map(String::from),
    );
    args.push(out.display().to_string());
    args
}

/// At most 15 fps and 400 px wide, one palette for the whole clip, and
/// ordered dithering, which compresses far better than error diffusion. A
/// swing plays forward then back so the loop doesn't jump.
fn gif_args(pattern: &Path, fps: u32, width: u32, ping_pong: bool, out: &Path) -> Vec<String> {
    let mut filter = format!("[0]fps={},scale={}:-1:flags=lanczos", fps.min(15), width.min(400));
    if ping_pong {
        filter += ",split[f][r];[r]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[b];[f][b]concat=n=2:v=1";
    }
    filter +=
        ",split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle";
    let mut args = input_args(pattern, fps);
    args.extend(["-filter_complex".into(), filter, "-loop".into(), "0".into(), out.display().to_string()]);
    args
}

fn input_args(pattern: &Path, fps: u32) -> Vec<String> {
    ["-hide_banner", "-loglevel", "error", "-y", "-framerate"]
        .map(String::from)
        .into_iter()
        .chain([fps.to_string(), "-i".into(), pattern.display().to_string()])
        .collect()
}

/// Runs ffmpeg, or if it isn't installed, prints the command to run later.
fn encode(args: &[String], out: &Path) -> Result<()> {
    if Process::new("ffmpeg").arg("-version").output().is_err() {
        let quoted: Vec<String> = args.iter().map(|a| format!("'{a}'")).collect();
        eprintln!("ffmpeg not found; to make {} later, run:\n  ffmpeg {}", out.display(), quoted.join(" "));
        return Ok(());
    }
    let status = Process::new("ffmpeg").args(args).status().context("running ffmpeg")?;
    if !status.success() {
        bail!("ffmpeg failed making {}", out.display());
    }
    let size = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    eprintln!("wrote {} ({:.1} MB)", out.display(), size as f64 / 1e6);
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
