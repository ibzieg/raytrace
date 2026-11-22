//! Ward's irradiance cache: final gathers computed at sparse points and
//! interpolated in between, spaced by the distance to nearby geometry so
//! records crowd into corners and spread out over open surfaces.

use std::collections::HashMap;

use glam::DVec3;

#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub pos: DVec3,
    /// Facing normal of the surface the gather was made from.
    pub normal: DVec3,
    pub irradiance: DVec3,
    /// Harmonic mean distance to the surfaces the gather rays hit, clamped.
    pub radius: f64,
}

/// Records indexed by a uniform grid with cells as wide as the largest
/// record's reach, so a lookup only checks its own and neighboring cells.
pub struct IrradianceCache {
    records: Vec<Record>,
    cells: HashMap<[i64; 3], Vec<u32>>,
    cell_size: f64,
}

/// How far behind a record's tangent plane (as a fraction of its radius) a
/// point may lie and still use it. A record in front of a point, as in a
/// crease, may see light the point can't.
const IN_FRONT_TOLERANCE: f64 = 0.05;

impl IrradianceCache {
    /// `max_reach` bounds how far any record's influence extends: accuracy
    /// times the largest radius.
    pub fn new(max_reach: f64) -> Self {
        Self { records: Vec::new(), cells: HashMap::new(), cell_size: max_reach.max(1e-9) }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn cell(&self, p: DVec3) -> [i64; 3] {
        let c = (p / self.cell_size).floor();
        [c.x as i64, c.y as i64, c.z as i64]
    }

    pub fn extend(&mut self, records: impl IntoIterator<Item = Record>) {
        for r in records {
            let cell = self.cell(r.pos);
            self.cells.entry(cell).or_default().push(self.records.len() as u32);
            self.records.push(r);
        }
    }

    /// Weighted average of the records valid at `p` with facing normal `n`,
    /// using Ward's weight 1 / (distance / radius + sqrt(1 - cos)). A record
    /// is valid where its weight exceeds 1 / `accuracy`.
    pub fn lookup(&self, p: DVec3, n: DVec3, accuracy: f64) -> Option<DVec3> {
        let [cx, cy, cz] = self.cell(p);
        let mut sum = DVec3::ZERO;
        let mut weights = 0.0;
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for z in cz - 1..=cz + 1 {
                    let Some(ids) = self.cells.get(&[x, y, z]) else { continue };
                    for &i in ids {
                        let r = &self.records[i as usize];
                        let offset = p - r.pos;
                        if offset.dot(n + r.normal) * 0.5 < -IN_FRONT_TOLERANCE * r.radius {
                            continue;
                        }
                        let error = offset.length() / r.radius + (1.0 - n.dot(r.normal)).max(0.0).sqrt();
                        if error < accuracy {
                            let w = 1.0 / error.max(1e-6);
                            sum += r.irradiance * w;
                            weights += w;
                        }
                    }
                }
            }
        }
        (weights > 0.0).then(|| sum / weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(pos: DVec3, irradiance: f64) -> Record {
        Record { pos, normal: DVec3::Y, irradiance: DVec3::splat(irradiance), radius: 1.0 }
    }

    #[test]
    fn interpolates_nearby_records_and_rejects_far_or_turned_ones() {
        let mut cache = IrradianceCache::new(0.5);
        cache.extend([record(DVec3::ZERO, 1.0), record(DVec3::new(0.3, 0.0, 0.0), 3.0)]);

        // Halfway between: equal weights.
        let mid = cache.lookup(DVec3::new(0.15, 0.0, 0.0), DVec3::Y, 0.5).unwrap();
        assert!((mid.x - 2.0).abs() < 1e-9, "{mid}");
        // On top of the first record: it dominates.
        let near = cache.lookup(DVec3::new(0.001, 0.0, 0.0), DVec3::Y, 0.5).unwrap();
        assert!(near.x < 1.1, "{near}");
        // Beyond every record's reach.
        assert!(cache.lookup(DVec3::new(2.0, 0.0, 0.0), DVec3::Y, 0.5).is_none());
        // A surface facing another way.
        assert!(cache.lookup(DVec3::ZERO, DVec3::X, 0.5).is_none());
        // A point below the records' tangent plane, as in a crease.
        assert!(cache.lookup(DVec3::new(0.1, -0.2, 0.0), DVec3::Y, 0.5).is_none());
        assert!(cache.lookup(DVec3::new(0.1, 0.2, 0.0), DVec3::Y, 0.5).is_some());
    }
}
