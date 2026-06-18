// Rust port of the SignWriting similarity-v2 metric (single-sign full score), tuned for speed.
// Ported from signwriting_evaluation/metrics/similarity_v2/similarity_v2.py. Mirror + multi-sign
// sequence handling stay in Python (cached, exact); Rust does the heavy single-sign scoring.
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
struct Sym { shape: u32, facing: u8, angle: u8, parallel: bool, size_index: u8,
             class: i32, nidx: i32, head_rim: bool, sw: f64, sh: f64, x: f64, y: f64 }

struct Data {
    // ERROR_WEIGHT
    w_shape: f64, w_facing: f64, w_angle: f64, w_parallel: f64, position_scale: f64,
    norm_factor: f64, exp_factor: f64, w_class: f64, w_size: f64, facial_scale: f64,
    // factor / implicit params
    reordering_weight: f64, overlap_weight: f64, touch_penalty: f64, movement_weight: f64,
    overlap_fraction: f64, touch_tolerance: f64,
    // materialized implicit face (conditional): a head circle shifted in, rest of sign pushed down
    face_sym: Sym, face_shift: f64,
    classes: Vec<(u32, u32)>,
    hand_class: i32, head_class: i32, contact_class: i32,
    face_classes: Vec<i32>,
    head_rim: (u32, u32),
    max_distance: f64,
    size_canon: FxHashMap<u32, (u32, u8)>,
    plane_canon: FxHashMap<u32, u32>,
    heel_to_top: FxHashMap<(u32, u8, u8), (u32, u8, u8)>,
    variant_canonical: FxHashMap<(u8, u8), (u8, u8)>,
    plane_angles: [bool; 16],
    shape_idx: FxHashMap<u32, u32>,
    name_n: usize,
    name_dist: Vec<f64>,
    sizes: FxHashMap<u32, (f64, f64)>,
}

static DATA: OnceLock<Data> = OnceLock::new();

fn tokenize(name: &str) -> Vec<String> {
    name.split_whitespace().map(|t| t.trim_matches(',').to_lowercase()).filter(|t| !t.is_empty()).collect()
}
fn word_lev(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() { return if a != b { 1.0 } else { 0.0 }; }
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let sub = prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] as f64 / n.max(m) as f64
}

fn load() -> Data {
    let v: serde_json::Value = serde_json::from_str(include_str!("../data/rust_data.json")).unwrap();
    let ew = &v["error_weight"]; let g = |k: &str| ew[k].as_f64().unwrap();
    let p = &v["params"]; let gp = |k: &str| p[k].as_f64().unwrap();
    let classes: Vec<(u32, u32)> = v["classes"].as_array().unwrap().iter()
        .map(|c| (c[0].as_u64().unwrap() as u32, c[1].as_u64().unwrap() as u32)).collect();
    let names = v["names"].as_object().unwrap();
    let mut shapes: Vec<u32> = names.keys().map(|k| u32::from_str_radix(k, 16).unwrap()).collect();
    shapes.sort_unstable();
    let mut shape_idx = FxHashMap::default();
    let mut toks: Vec<Vec<String>> = Vec::new();
    for (i, &s) in shapes.iter().enumerate() {
        shape_idx.insert(s, i as u32);
        toks.push(tokenize(names[&format!("{:x}", s)].as_str().unwrap()));
    }
    let name_n = shapes.len();
    let mut name_dist = vec![0.0; name_n * name_n];
    for i in 0..name_n { for j in (i + 1)..name_n {
        let d = word_lev(&toks[i], &toks[j]); name_dist[i * name_n + j] = d; name_dist[j * name_n + i] = d;
    }}
    let mut size_canon = FxHashMap::default();
    for (k, val) in v["size_canon"].as_object().unwrap() {
        let a = val.as_array().unwrap();
        size_canon.insert(u32::from_str_radix(k, 16).unwrap(), (a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u8));
    }
    let mut plane_canon = FxHashMap::default();
    for (k, val) in v["plane_canon"].as_object().unwrap() {
        plane_canon.insert(u32::from_str_radix(k, 16).unwrap(), val.as_u64().unwrap() as u32);
    }
    let mut heel_to_top = FxHashMap::default();
    for (k, val) in v["heel_to_top"].as_object().unwrap() {
        let q: Vec<u32> = k.split(',').map(|x| u32::from_str_radix(x, 16).unwrap()).collect();
        let a = val.as_array().unwrap();
        heel_to_top.insert((q[0], q[1] as u8, q[2] as u8), (a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u8, a[2].as_u64().unwrap() as u8));
    }
    let mut variant_canonical = FxHashMap::default();
    for pair in v["wall_floor"].as_array().unwrap() {
        let pp = pair.as_array().unwrap();
        let a = (pp[0][0].as_u64().unwrap() as u8, pp[0][1].as_u64().unwrap() as u8);
        let b = (pp[1][0].as_u64().unwrap() as u8, pp[1][1].as_u64().unwrap() as u8);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        variant_canonical.insert(hi, lo);
    }
    let mut plane_angles = [false; 16];
    for x in v["plane_intersection_angles"].as_array().unwrap() { plane_angles[x.as_u64().unwrap() as usize] = true; }
    let sizes_v: serde_json::Value = serde_json::from_str(include_str!("../data/sizes.json")).unwrap();
    let mut sizes = FxHashMap::default();
    for (k, val) in sizes_v.as_object().unwrap() {
        let a = val.as_array().unwrap();
        sizes.insert(u32::from_str_radix(k, 16).unwrap(), (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()));
    }
    let head_class = v["head_class"].as_i64().unwrap() as i32;
    let face_pos = p["implicit_face_position"].as_array().unwrap();
    let (fx, fy) = (face_pos[0].as_f64().unwrap(), face_pos[1].as_f64().unwrap());
    let face_shape = 0x2ffu32;  // S2ff00, the canonical head circle
    let (fsw_w, fsw_h) = sizes.get(&0x2ff00).copied().unwrap_or((0.0, 0.0));
    let face_sym = Sym {
        shape: face_shape, facing: 0, angle: 0, parallel: false, size_index: 0, class: head_class,
        nidx: shape_idx.get(&face_shape).map(|&i| i as i32).unwrap_or(-1),
        head_rim: false, sw: fsw_w, sh: fsw_h, x: fx, y: fy,
    };
    Data {
        w_shape: g("shape"), w_facing: g("facing"), w_angle: g("angle"), w_parallel: g("parallel"),
        position_scale: g("position_scale"), norm_factor: g("normalized_factor"), exp_factor: g("exp_factor"),
        w_class: g("class_penalty"), w_size: g("size"), facial_scale: g("facial_scale"),
        reordering_weight: gp("reordering_weight"), overlap_weight: gp("overlap_weight"),
        touch_penalty: gp("touch_penalty"), movement_weight: gp("movement_weight"),
        overlap_fraction: gp("overlap_fraction"), touch_tolerance: gp("touch_tolerance"),
        face_sym, face_shift: p["implicit_face_shift"].as_f64().unwrap(),
        classes, hand_class: v["hand_class"].as_i64().unwrap() as i32, head_class,
        contact_class: v["contact_class"].as_i64().unwrap() as i32,
        face_classes: v["face_classes"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect(),
        head_rim: (v["head_rim"][0].as_u64().unwrap() as u32, v["head_rim"][1].as_u64().unwrap() as u32),
        max_distance: v["max_distance"].as_f64().unwrap(),
        size_canon, plane_canon, heel_to_top, variant_canonical, plane_angles, shape_idx, name_n, name_dist, sizes,
    }
}
fn data() -> &'static Data { DATA.get_or_init(load) }

#[inline]
fn class_of(d: &Data, shape: u32) -> i32 {
    for (i, &(s, e)) in d.classes.iter().enumerate() { if shape >= s && shape < e { return i as i32; } }
    -1
}

fn parse_sign(d: &Data, fsw: &str) -> Vec<Sym> {
    let b = fsw.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'S' && i + 6 <= b.len() && b[i + 1..i + 6].iter().all(|c| c.is_ascii_hexdigit()) {
            let mut j = i + 6; let x0 = j;
            while j < b.len() && b[j].is_ascii_digit() { j += 1; }
            if j < b.len() && b[j] == b'x' && j > x0 {
                let x: f64 = fsw[x0..j].parse().unwrap_or(0.0);
                j += 1; let y0 = j;
                while j < b.len() && b[j].is_ascii_digit() { j += 1; }
                let y: f64 = fsw[y0..j].parse().unwrap_or(0.0);
                let raw5 = u32::from_str_radix(&fsw[i + 1..i + 6], 16).unwrap();
                let key = &fsw[i..i + 6];
                let kb = key.as_bytes();
                let mut shape = u32::from_str_radix(&key[1..4], 16).unwrap();
                let mut facing = (kb[4] as char).to_digit(16).unwrap() as u8;
                let mut angle = (kb[5] as char).to_digit(16).unwrap() as u8;
                if let Some(&(s, f, a)) = d.heel_to_top.get(&(shape, facing, angle)) { shape = s; facing = f; angle = a; }
                if class_of(d, shape) == d.hand_class {
                    if let Some(&(f, a)) = d.variant_canonical.get(&(facing, angle)) { facing = f; angle = a; }
                }
                let mut size_index = 0u8;
                if let Some(&(rep, si)) = d.size_canon.get(&shape) { shape = rep; size_index = si; }
                if d.plane_angles[angle as usize] { if let Some(&rep) = d.plane_canon.get(&shape) { shape = rep; } }
                let (sw, sh) = d.sizes.get(&raw5).copied().unwrap_or((0.0, 0.0));
                out.push(Sym {
                    shape, facing, angle, parallel: facing > 2, size_index,
                    class: class_of(d, shape), nidx: d.shape_idx.get(&shape).map(|&i| i as i32).unwrap_or(-1),
                    head_rim: shape >= d.head_rim.0 && shape < d.head_rim.1, sw, sh, x, y,
                });
                i = j; continue;
            }
        }
        i += 1;
    }
    out
}

#[inline]
fn name_dist(d: &Data, a: &Sym, b: &Sym) -> f64 {
    let raw = if a.shape == b.shape { 0.0 }
        else if a.nidx >= 0 && b.nidx >= 0 { d.name_dist[a.nidx as usize * d.name_n + b.nidx as usize] }
        else { 1.0 };
    // two head/face symbols are the same class -> never the full hand-vs-arrow distance
    if a.class == d.head_class && b.class == d.head_class { raw * d.facial_scale } else { raw }
}
#[inline]
fn pair_cost(d: &Data, a: &Sym, ax: f64, ay: f64, b: &Sym, bx: f64, by: f64) -> f64 {
    // identity (concave root, preserves spelling) + position (convex squared, jitter ~0); see symbols_score
    if a.class < 0 || b.class < 0 { return 1.0; }
    let ds = name_dist(d, a, b) * d.w_shape;
    let df = (a.facing as f64 - b.facing as f64) * d.w_facing;
    let dr = (a.angle as f64 - b.angle as f64) * d.w_angle;
    let dp = if a.parallel != b.parallel { d.w_parallel } else { 0.0 };
    let dz = if a.shape == b.shape { (a.size_index as f64 - b.size_index as f64).abs() * d.w_size } else { 0.0 };
    let identity = (ds * ds + df * df + dr * dr + dp * dp + dz * dz).sqrt()
        + (a.class - b.class).abs() as f64 * d.w_class;
    let identity_cost = (identity / d.max_distance).powf(d.norm_factor);
    let position_cost = (((ax - bx).powi(2) + (ay - by).powi(2)).sqrt() / d.position_scale).powi(2);
    (identity_cost + position_cost).min(1.0)
}

fn hungarian(cost: &[f64], n: usize) -> Vec<usize> {
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let mut p = vec![0usize; n + 1]; let mut way = vec![0usize; n + 1];
    let mut minv = vec![inf; n + 1]; let mut used = vec![false; n + 1];
    for i in 1..=n {
        p[0] = i; let mut j0 = 0;
        minv.iter_mut().for_each(|x| *x = inf); used.iter_mut().for_each(|x| *x = false);
        loop {
            used[j0] = true; let i0 = p[j0]; let mut delta = inf; let mut j1 = 0; let row = (i0 - 1) * n;
            for j in 1..=n { if !used[j] {
                let cur = cost[row + j - 1] - u[i0] - v[j];
                if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                if minv[j] < delta { delta = minv[j]; j1 = j; }
            }}
            for j in 0..=n { if used[j] { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; } }
            j0 = j1; if p[j0] == 0 { break; }
        }
        loop { let j1 = way[j0]; p[j0] = p[j1]; j0 = j1; if j0 == 0 { break; } }
    }
    let mut rc = vec![0usize; n];
    for j in 1..=n { rc[p[j] - 1] = j - 1; }
    rc
}

fn centroid(s: &[Sym]) -> (f64, f64) {
    let n = s.len() as f64;
    (s.iter().map(|p| p.x).sum::<f64>() / n, s.iter().map(|p| p.y).sum::<f64>() / n)
}

// principal axis (top eigenvector of the 2x2 scatter); None if all positions are zero.
fn principal_axis(pos: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pos.iter().all(|&(x, y)| x == 0.0 && y == 0.0) { return None; }
    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
    for &(x, y) in pos { a += x * x; b += x * y; c += y * y; }
    let lambda = (a + c) / 2.0 + (((a - c) / 2.0).powi(2) + b * b).sqrt();
    let (vx, vy) = if b.abs() > 1e-12 { (lambda - c, b) } else if a >= c { (1.0, 0.0) } else { (0.0, 1.0) };
    let n = (vx * vx + vy * vy).sqrt();
    Some((vx / n, vy / n))
}

fn discordant_fraction(hp: &[(f64, f64)], rp: &[(f64, f64)], axis: (f64, f64), w: &[f64]) -> f64 {
    let proj = |p: &[(f64, f64)]| -> Vec<f64> { p.iter().map(|&(x, y)| x * axis.0 + y * axis.1).collect() };
    let (hpr, rpr) = (proj(hp), proj(rp));
    let n = hp.len();
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n { for j in (i + 1)..n {
        let ho = hpr[i] - hpr[j]; let ro = rpr[i] - rpr[j];
        if ho.abs() > 1e-6 && ro.abs() > 1e-6 {
            let pw = w[i] * w[j];
            den += pw;
            if ho.signum() != ro.signum() { num += pw; }
        }
    }}
    if den == 0.0 { 0.0 } else { num / den }
}

fn hand_contact_count(d: &Data, s: &[Sym]) -> i64 {
    let n = s.len(); let mut cnt = 0i64;
    let is_hand = |c: i32| c == d.hand_class;
    let is_hof = |c: i32| c == d.hand_class || d.face_classes.contains(&c);
    for i in 0..n { for j in (i + 1)..n {
        let (a, b) = (&s[i], &s[j]);
        let gap_x = (a.x.max(b.x)) - ((a.x + a.sw).min(b.x + b.sw));
        let gap_y = (a.y.max(b.y)) - ((a.y + a.sh).min(b.y + b.sh));
        let touching = gap_x.max(0.0).max(gap_y.max(0.0)) <= d.touch_tolerance;
        let involved = (is_hand(a.class) && is_hof(b.class)) || (is_hand(b.class) && is_hof(a.class));
        if touching && involved { cnt += 1; }
    }}
    cnt
}

// available implicit symbols of a sign: (head_face count, contact count)
fn implicit_classes(d: &Data, s: &[Sym]) -> (i64, i64) {
    let head = if s.iter().any(|x| d.face_classes.contains(&x.class)) { 0 } else { 1 };
    let contact = s.iter().filter(|x| x.head_rim).count() as i64 + hand_contact_count(d, s);
    (head, contact)
}

fn unexplained_penalty(d: &Data, leftover: &[usize], s: &[Sym], mut head: i64, mut contact: i64) -> f64 {
    let mut penalty = 0.0;
    for &idx in leftover {
        let k = s[idx].class;
        if k == d.head_class && head > 0 { head -= 1; }
        else if k == d.contact_class && contact > 0 { contact -= 1; }
        else { penalty += if k == d.contact_class { d.touch_penalty } else { 1.0 }; }
    }
    penalty
}

fn implicit_length_acc(d: &Data, hyp: &[Sym], ref_: &[Sym], matched: &[(usize, usize)]) -> f64 {
    let mh: std::collections::HashSet<usize> = matched.iter().map(|&(r, _)| r).collect();
    let mr: std::collections::HashSet<usize> = matched.iter().map(|&(_, c)| c).collect();
    let lh: Vec<usize> = (0..hyp.len()).filter(|i| !mh.contains(i)).collect();
    let lr: Vec<usize> = (0..ref_.len()).filter(|j| !mr.contains(j)).collect();
    let (hh, hc) = implicit_classes(d, hyp);
    let (rh, rc) = implicit_classes(d, ref_);
    let unexplained = unexplained_penalty(d, &lr, ref_, hh, hc) + unexplained_penalty(d, &lh, hyp, rh, rc);
    (unexplained / (hyp.len().max(ref_.len()) as f64 + 1.0)).min(1.0)
}

fn overlap(d: &Data, a: &Sym, b: &Sym) -> bool {
    let ox = ((a.x + a.sw).min(b.x + b.sw) - a.x.max(b.x)).max(0.0);
    let oy = ((a.y + a.sh).min(b.y + b.sh) - a.y.max(b.y)).max(0.0);
    ox * oy >= d.overlap_fraction * (a.sw * a.sh).min(b.sw * b.sh)
}

fn score_parsed(d: &Data, hyp: &[Sym], ref_: &[Sym]) -> f64 {
    if hyp.is_empty() || ref_.is_empty() { return 0.0; }
    let (m, n) = (hyp.len(), ref_.len());
    let hc = centroid(hyp); let rc = centroid(ref_);
    let k = m.max(n);
    let mut cost = vec![1e6; k * k];
    for i in 0..m { for j in 0..n {
        cost[i * k + j] = pair_cost(d, &hyp[i], hyp[i].x - hc.0, hyp[i].y - hc.1, &ref_[j], ref_[j].x - rc.0, ref_[j].y - rc.1);
    }}
    let rcv = hungarian(&cost, k);
    let matched: Vec<(usize, usize)> = (0..k).map(|r| (r, rcv[r])).filter(|&(r, c)| r < m && c < n).collect();
    let cnt = matched.len() as f64;
    // mean cost (re-align by matched-pair offset)
    let (mut hx, mut hy, mut rx, mut ry) = (0.0, 0.0, 0.0, 0.0);
    for &(r, c) in &matched { hx += hyp[r].x; hy += hyp[r].y; rx += ref_[c].x; ry += ref_[c].y; }
    let off = (hx / cnt - rx / cnt, hy / cnt - ry / cnt);
    let mean_cost: f64 = matched.iter().map(|&(r, c)|
        pair_cost(d, &hyp[r], hyp[r].x - off.0, hyp[r].y - off.1, &ref_[c], ref_[c].x, ref_[c].y)).sum::<f64>() / cnt;
    let length_acc = implicit_length_acc(d, hyp, ref_, &matched);
    let length_weight = length_acc.powf(d.exp_factor);
    let error_rate = length_weight + mean_cost * (1.0 - length_weight);
    let mut score = (1.0 - error_rate).powi(2);

    // --- reordering factor ---
    if d.reordering_weight > 0.0 && m >= 2 && n >= 2 && matched.len() >= 2 {
        let w: Vec<f64> = matched.iter().map(|&(r, c)| 1.0 - cost[r * k + c]).collect();
        let mut hp: Vec<(f64, f64)> = matched.iter().map(|&(r, _)| (hyp[r].x, hyp[r].y)).collect();
        let mut rp: Vec<(f64, f64)> = matched.iter().map(|&(_, c)| (ref_[c].x, ref_[c].y)).collect();
        let hm = (hp.iter().map(|p| p.0).sum::<f64>() / cnt, hp.iter().map(|p| p.1).sum::<f64>() / cnt);
        let rm = (rp.iter().map(|p| p.0).sum::<f64>() / cnt, rp.iter().map(|p| p.1).sum::<f64>() / cnt);
        for p in hp.iter_mut() { p.0 -= hm.0; p.1 -= hm.1; }
        for p in rp.iter_mut() { p.0 -= rm.0; p.1 -= rm.1; }
        let axes: Vec<(f64, f64)> = [principal_axis(&hp), principal_axis(&rp)].into_iter().flatten().collect();
        if !axes.is_empty() {
            let disc: f64 = axes.iter().map(|&ax| discordant_fraction(&hp, &rp, ax, &w)).sum::<f64>() / axes.len() as f64;
            score *= (-d.reordering_weight * disc).exp();
        }
    }
    // --- overlap-order factor ---
    if d.overlap_weight > 0.0 && m >= 2 && n >= 2 && matched.len() >= 2 {
        let (mut inv_ov, mut ov) = (0.0, 0.0);
        for a in 0..matched.len() { for b in (a + 1)..matched.len() {
            let (ra, ca) = matched[a]; let (rb, cb) = matched[b];
            let overlapping = overlap(d, &hyp[ra], &hyp[rb]) || overlap(d, &ref_[ca], &ref_[cb]);
            if overlapping {
                ov += 1.0;
                // ra<rb always (ascending); inverted iff ca>cb
                if ca > cb { inv_ov += 1.0; }
            }
        }}
        if ov > 0.0 { score *= (-d.overlap_weight * inv_ov / ov).exp(); }
    }
    // --- direction factor ---
    if d.movement_weight > 0.0 {
        let mut total = 0.0;
        for &(r, c) in &matched {
            if hyp[r].shape != ref_[c].shape { continue; }
            let da = (hyp[r].angle as i32 - ref_[c].angle as i32).abs();
            let angle = da.min(16 - da) as f64 / 8.0;
            let facing = (hyp[r].facing as f64 - ref_[c].facing as f64).abs() / 5.0;
            total += (angle + facing) / 2.0;
        }
        score *= (-d.movement_weight * total).exp();
    }
    score
}

#[inline]
fn has_face(d: &Data, s: &[Sym]) -> bool { s.iter().any(|x| d.face_classes.contains(&x.class)) }

// A face-less sign compared against a faced one gets a head circle at the canonical position with the
// rest shifted down, so the matcher has a shared face anchor (mirrors add_implicit_face in Python).
fn add_face(d: &Data, s: &[Sym]) -> Vec<Sym> {
    let mut out = Vec::with_capacity(s.len() + 1);
    out.push(d.face_sym);
    for x in s { let mut y = *x; y.y += d.face_shift; out.push(y); }
    out
}

fn score_materialized(d: &Data, mut h: Vec<Sym>, mut r: Vec<Sym>) -> f64 {
    match (has_face(d, &h), has_face(d, &r)) {
        (true, false) => r = add_face(d, &r),
        (false, true) => h = add_face(d, &h),
        _ => {}
    }
    score_parsed(d, &h, &r)
}

#[pyfunction]
fn score_single(hyp: &str, ref_: &str) -> f64 {
    let d = data();
    score_materialized(d, parse_sign(d, hyp), parse_sign(d, ref_))
}

// Batch: many single-sign (hyp, ref) FSW pairs, scored in parallel with the GIL released.
#[pyfunction]
fn score_single_many(py: Python, pairs: Vec<(String, String)>) -> Vec<f64> {
    let d = data();
    py.allow_threads(|| pairs.par_iter()
        .map(|(h, r)| score_materialized(d, parse_sign(d, h), parse_sign(d, r))).collect())
}

#[pymodule]
fn _similarity_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_single, m)?)?;
    m.add_function(wrap_pyfunction!(score_single_many, m)?)?;
    Ok(())
}
