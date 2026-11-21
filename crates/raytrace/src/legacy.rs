//! Converts the 2011 Java app's XML `.scene` files to [`SceneFile`].
//!
//! Quirks of the old format:
//!   - matrices are OpenGL column-major arrays, written as x00..x33 in array
//!     order (so x30, x31, x32 hold the translation);
//!   - only the mesh file name is stored, not its path;
//!   - the ray tracer used light positions as camera-space coordinates (the
//!     OpenGL preview treated them as world space). The converter keeps the
//!     ray traced look by moving them into world space.

use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::{DMat4, DVec3};
use roxmltree::{Document, Node};

use crate::scene::{
    CameraDef, LightDef, ObjectDef, PhotonSettings, RenderSettings, SceneFile, Window, look_at, mat_to_rows,
};

pub fn convert(xml: &str, mesh_dir: &Path) -> Result<SceneFile> {
    // The files reference an external DTD that we don't need.
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = Document::parse_with_options(xml, opts).context("parsing XML")?;
    let root = doc.root_element();
    if root.tag_name().name() != "Scene" {
        bail!("root element is <{}>, expected <Scene>", root.tag_name().name());
    }

    let cam = child(root, "Camera")?;
    let view = matrix(child(cam, "viewMat")?)?;
    let camera = camera_def(cam, view)?;
    let camera_to_world = view.inverse();

    let mut objects = Vec::new();
    for mesh in children(child(root, "Objects")?, "PMesh") {
        let file_type = text(mesh, "fileType")?;
        if !file_type.eq_ignore_ascii_case("OBJ") {
            bail!("unsupported mesh type {file_type}");
        }
        let model = matrix(child(mesh, "modelMat")?)?;
        objects.push(ObjectDef {
            mesh: mesh_dir.join(text(mesh, "objName")?),
            transform: (!model.abs_diff_eq(DMat4::IDENTITY, 1e-12)).then(|| mat_to_rows(&model)),
        });
    }

    let mut lights = Vec::new();
    for light in children(child(root, "Lights")?, "Light") {
        if number(light, "lightSwitch")? < 0.5 {
            continue;
        }
        let pos = child(light, "position")?;
        let camera_pos = DVec3::new(number(pos, "X")?, number(pos, "Y")?, number(pos, "Z")?);
        lights.push(LightDef {
            position: round3(camera_to_world.transform_point3(camera_pos)),
            diffuse: rgb(child(light, "diffuse")?)?,
            specular: rgb(child(light, "specular")?)?,
        });
    }

    Ok(SceneFile { camera, lights, objects, render: RenderSettings::default(), photon: PhotonSettings::default() })
}

fn camera_def(cam: Node, view: DMat4) -> Result<CameraDef> {
    let v3 = |a: &str, b: &str, c: &str| -> Result<DVec3> {
        Ok(DVec3::new(number(cam, a)?, number(cam, b)?, number(cam, c)?))
    };
    let eye = v3("eyeX", "eyeY", "eyeZ")?;
    let target = v3("ctrX", "ctrY", "ctrZ")?;
    let up = v3("upX", "upY", "upZ")?;
    let window = Window {
        left: number(cam, "wLeft")?,
        right: number(cam, "wRight")?,
        bottom: number(cam, "wBottom")?,
        top: number(cam, "wTop")?,
    };
    let mut def = CameraDef {
        eye: None,
        target: None,
        up: None,
        view_matrix: None,
        near: number(cam, "near")?,
        far: number(cam, "far")?,
        window,
    };
    // Prefer the readable eye/target/up form when it reproduces the stored
    // view matrix (they can drift apart after mouse navigation).
    if look_at(eye, target, up).abs_diff_eq(view, 1e-6) {
        def.eye = Some(eye.into());
        def.target = Some(target.into());
        def.up = Some(up.into());
    } else {
        def.view_matrix = Some(mat_to_rows(&view));
    }
    Ok(def)
}

fn children<'a, 'i>(node: Node<'a, 'i>, name: &'static str) -> impl Iterator<Item = Node<'a, 'i>> {
    node.children().filter(move |n| n.has_tag_name(name))
}

fn child<'a, 'i>(node: Node<'a, 'i>, name: &str) -> Result<Node<'a, 'i>> {
    node.children()
        .find(|n| n.has_tag_name(name))
        .with_context(|| format!("<{}> has no <{name}>", node.tag_name().name()))
}

fn text<'a>(node: Node<'a, '_>, name: &str) -> Result<&'a str> {
    Ok(child(node, name)?.text().unwrap_or("").trim())
}

fn number(node: Node, name: &str) -> Result<f64> {
    let t = text(node, name)?;
    t.parse().with_context(|| format!("<{name}> is not a number: '{t}'"))
}

fn rgb(node: Node) -> Result<[f64; 3]> {
    Ok([number(node, "R")?, number(node, "G")?, number(node, "B")?])
}

fn matrix(node: Node) -> Result<DMat4> {
    let mut m = [0.0; 16];
    for (i, v) in m.iter_mut().enumerate() {
        *v = number(node, &format!("x{}{}", i / 4, i % 4))?;
    }
    Ok(DMat4::from_cols_array(&m))
}

/// Trims float noise like 6.740000009536743 (the old app stored f32s).
fn round3(v: DVec3) -> [f64; 3] {
    v.to_array().map(|x| (x * 1e6).round() / 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0" standalone="no"?>
<!DOCTYPE Scene SYSTEM "scene.dtd">
<Scene><Objects><PMesh><fileType>OBJ</fileType><objName>a.obj</objName>
<modelMat><x00>1</x00><x01>0</x01><x02>0</x02><x03>0</x03><x10>0</x10><x11>1</x11><x12>0</x12><x13>0</x13>
<x20>0</x20><x21>0</x21><x22>1</x22><x23>0</x23><x30>2</x30><x31>3</x31><x32>4</x32><x33>1</x33></modelMat>
<center><X>0</X><Y>0</Y><Z>0</Z></center></PMesh></Objects>
<Lights><Light><lightSwitch>1</lightSwitch><diffuse><R>.5</R><G>.5</G><B>.5</B><A>1</A></diffuse>
<specular><R>1</R><G>1</G><B>1</B><A>1</A></specular><position><X>0</X><Y>1</Y><Z>-5</Z><W>1</W></position></Light>
<Light><lightSwitch>0</lightSwitch></Light></Lights>
<Camera><viewMat><x00>1</x00><x01>0</x01><x02>0</x02><x03>0</x03><x10>0</x10><x11>1</x11><x12>0</x12><x13>0</x13>
<x20>0</x20><x21>0</x21><x22>1</x22><x23>0</x23><x30>0</x30><x31>0</x31><x32>-10</x32><x33>1</x33></viewMat>
<near>9</near><far>1000</far><upX>0</upX><upY>1</upY><upZ>0</upZ><eyeX>0</eyeX><eyeY>0</eyeY><eyeZ>10</eyeZ>
<ctrX>0</ctrX><ctrY>0</ctrY><ctrZ>0</ctrZ><wLeft>-5</wLeft><wRight>5</wRight><wBottom>-5</wBottom><wTop>5</wTop></Camera></Scene>"#;

    #[test]
    fn converts_matrices_lights_and_camera() {
        let s = convert(XML, Path::new("objects")).unwrap();
        assert_eq!(s.objects[0].mesh, Path::new("objects/a.obj"));
        // Column-major x30..x32 is translation, i.e. the last column in rows.
        let rows = s.objects[0].transform.unwrap();
        assert_eq!([rows[0][3], rows[1][3], rows[2][3]], [2.0, 3.0, 4.0]);
        // Disabled light dropped; camera-space light moved to world space.
        assert_eq!(s.lights.len(), 1);
        assert_eq!(s.lights[0].position, [0.0, 1.0, 5.0]);
        assert_eq!(s.camera.eye, Some([0.0, 0.0, 10.0]));
        assert!(s.camera.view_matrix.is_none());
    }
}
