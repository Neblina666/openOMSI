//! The driver of the player's bus: a person on the bus's `[drivpos]` with both hands on the
//! steering wheel, turning it as the wheel turns - seen from outside, from the passengers'
//! places and in the mirrors, left out of the driver's own view (the cab view shows him
//! only in the mirrors, as OMSI does).

use glam::{Mat4, Vec3};
use omsi_render::{AlphaMode, MeshId, Renderer, Scene};
use omsi_sim::human::{curl_hands, grip_centres, hand_slot, skin_from, Activity, HumanType, Pose, PoseInput};
use omsi_sim::VehicleInstance;
use std::sync::Arc;

/// Where the hands rest on the rim, from the top, clockwise seen by the driver (degrees)
const REST: [f32; 2] = [-70.0, 70.0];
const RANGE: [(f32, f32); 2] = [(-160.0, -15.0), (15.0, 160.0)];
const SLIP: f32 = 30.0;
const ROLL: (f32, f32) = (-35.0, 130.0);
const ROLL_PLAIN: f32 = 30.0;
const ROLL_EASE: f32 = 0.12;
const FIX_EASE: f32 = 0.25;
const FIX_STEP: f32 = 0.002;
const FRAME_EASE: f32 = 0.07;
const REGRIP_BACK: f32 = 45.0;
const LIFT: f32 = 0.07;
const SETTLE_AFTER: f32 = 0.6;
const DIAGONAL: f32 = 40.0;
const SEAT_FRONT: f32 = 0.34;
const GRIP_RADIUS: f32 = 0.026;
const FINGER_HALF: f32 = 0.009;
const SLIDE_MAX: f32 = 0.10;
const ARM_RATIO: f32 = 0.98;
const LEAN_COMFORT: f32 = 2.0;
const UPPER_ARM: f32 = 0.30;
const REGRIP_LEAD: f32 = 0.08;
const LEAD_MAX: f32 = 14.0;
const ONE_HAND_OVER: f32 = 35.0; // Permite alcance maior dirindo com uma mão só

const REACH_TIME: f32 = 0.25;
const REACH_FAST: f32 = 0.12;
const BACK_TIME: f32 = 0.30;

const STOP_SPEED: f32 = 0.25;
const GO_SPEED: f32 = 0.8;
const STOP_AFTER: f32 = 0.6;
const LEVER_MOVING: f32 = 0.010;
const SHIFT_WAIT: f32 = 2.0;
const LEVER_REACH: f32 = 1.25;

const GEAR_VARS: &[&str] = &[
    "gear", "gearbox_gear", "gear_selected", "gang", "antrieb", 
    "ki_gear", "engine_gear", "cockpit_gang", "cockpit_gear"
];
const CLUTCH_VARS: &[&str] = &["clutch", "clutch_pedal", "kupplung", "cockpit_clutch"];

struct Shifter {
    mesh: usize,
    vars: Vec<String>,
    clutch: Option<String>,
    grab: Vec3,
    axis: Vec3,
    hand: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ShiftPhase {
    Away,
    Reach,
    Hold,
    Back,
}

struct ShiftState {
    phase: ShiftPhase,
    w: f32,
    reach: f32,
    stopped: bool,
    still_for: f32,
    cool: f32,
    last_pos: Option<glam::DVec3>,
    last_knob: Option<Vec3>,
    last_vars: Vec<f32>,
    last_clutch: f32,
    was_active: bool,
    idle: f32,
    pending: bool,
    waiting: f32,
    stroke: f32,
    event_from: Vec3,
    event_travel: f32,
    // Variações aleatórias dinâmicas por troca de marcha
    hold_target_time: f32,
    one_hand_drive_time: f32,
}

impl Default for ShiftState {
    fn default() -> Self {
        ShiftState {
            phase: ShiftPhase::Away,
            w: 0.0,
            reach: REACH_TIME,
            stopped: false,
            still_for: 0.0,
            cool: 0.0,
            last_pos: None,
            last_knob: None,
            last_vars: Vec::new(),
            last_clutch: 0.0,
            was_active: false,
            idle: 10.0,
            pending: false,
            waiting: 0.0,
            stroke: 1.0,
            event_from: Vec3::ZERO,
            event_travel: 0.0,
            hold_target_time: 0.15,
            one_hand_drive_time: 0.0,
        }
    }
}

struct Wheel {
    mesh: usize,
    var: String,
    factor: f32,
    axis: Vec3,
    centre: Vec3,
    up: Vec3,
    right: Vec3,
    radius: f32,
    tube: f32,
}

pub struct DriverFigure {
    ty: Arc<HumanType>,
    curled: Vec<(Vec<Vec3>, Vec<Vec3>)>,
    knuckles: f32,
    grip_rest: [Option<Vec3>; 2],
    grip_fix: [Vec3; 2],
    grip_radius: f32,
    pose: Pose,
    meshes: Vec<(MeshId, usize)>,
    skins: Vec<(Vec<Vec3>, Vec<Vec3>)>,
    hip: Vec3,
    floor: Vec3,
    heading: f32,
    lamps: [i32; 4],
    wheel: Option<Wheel>,
    sign: f32,
    lean: f32,
    base_lean: f32,
    pub show_hands_in_cab: bool,
    shown: bool,
    settled: bool,
    slide: f32,
    hands: [Hand; 2],
    hands_placed: bool,
    theta: f32,
    last_theta: f32,
    still: f32,
    rate: f32,
    elbows: Option<[Vec3; 2]>,
    frames: [Option<(Vec3, Vec3)>; 2],
    rolls: [Option<f32>; 2],
    hand_of: Vec<Vec<i8>>,
    arm_of: Vec<Vec<i8>>,
    blend: (Vec<Vec3>, Vec<Vec3>),
    shifter: Option<Shifter>,
    shift: ShiftState,
    rand_seed: u32,
}

const UPPER_ARM_SLOT: [usize; 2] = [4, 5];
const FORE_ARM_SLOT: [usize; 2] = [6, 7];

#[derive(Clone, Copy, Default)]
struct Hand {
    on_rim: f32,
    mv: Option<Regrip>,
}

#[derive(Clone, Copy)]
struct Regrip {
    from: f32,
    to: f32,
    t: f32,
    dur: f32,
    v0: f32,
    v1: f32,
}

impl Regrip {
    fn new(from: f32, to: f32, dur: f32, rate: f32) -> Regrip {
        let v = (rate * dur).clamp(-60.0, 60.0);
        Regrip { from, to, t: 0.0, dur, v0: v, v1: v }
    }
}

impl Hand {
    fn seen(&self, theta: f32) -> (f32, f32) {
        match self.mv {
            Some(m) => {
                let t = m.t.clamp(0.0, 1.0);
                let (t2, t3) = (t * t, t * t * t);
                let a = m.from * (2.0 * t3 - 3.0 * t2 + 1.0) + m.v0 * (t3 - 2.0 * t2 + t) + m.to * (3.0 * t2 - 2.0 * t3) + m.v1 * (t3 - t2);
                (a, (t * std::f32::consts::PI).sin().powi(2))
            }
            None => (self.on_rim + theta, 0.0),
        }
    }

    fn open(&self) -> f32 {
        self.mv.map(|m| (m.t.clamp(0.0, 1.0) * std::f32::consts::PI).sin().powi(2) * 0.45).unwrap_or(0.0)
    }
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

struct Targets {
    grips: [Vec3; 2],
    frames: [(Vec3, Vec3); 2],
    tubes: [Vec3; 2],
}

impl DriverFigure {
    pub fn new(world: &crate::scene::World, renderer: &Renderer, scene: &mut Scene, v: &VehicleInstance, pick: u64) -> Option<DriverFigure> {
        let ty = driver_type(world, pick)?;
        Self::new_with(world, renderer, scene, v, ty)
    }

    pub fn new_named(world: &crate::scene::World, renderer: &Renderer, scene: &mut Scene, v: &VehicleInstance, hum: &str, pick: u64) -> Option<DriverFigure> {
        let named = omsi_net::human_path(hum)
            .map(|rel| omsi_cfg::resolve_path(&world.root, &rel))
            .filter(|p| omsi_cfg::vfs::exists(p))
            .and_then(|p| cached_type(&p));
        let ty = named.or_else(|| driver_type(world, pick))?;
        Self::new_with(world, renderer, scene, v, ty)
    }

    fn new_with(world: &crate::scene::World, renderer: &Renderer, scene: &mut Scene, v: &VehicleInstance, ty: Arc<HumanType>) -> Option<DriverFigure> {
        let seat = seat_of(v)?;
        let mut meshes = Vec::new();
        let dirs = ty.texture_dirs(&world.root);
        for hm in &ty.meshes {
            let mut mats = Vec::new();
            for (k, m) in hm.materials.iter().enumerate() {
                let look: Vec<&std::path::Path> = dirs.iter().map(|p| p.as_path()).collect();
                let tex = omsi_texture::find_texture(&m.texture, &look).and_then(|path| {
                    let t = world.textures.get_gpu_fast(&path).map(|(img, _)| renderer.add_texture_data(scene, &img));
                    world.textures.release(&path);
                    t
                });
                let alpha = match hm.alpha.get(k).copied().unwrap_or(0) {
                    1 => AlphaMode::Test,
                    2 => AlphaMode::Blend,
                    _ => AlphaMode::Opaque,
                };
                mats.push(renderer.add_material(scene, tex, alpha, [1.0; 4], false));
            }
            let id = renderer.add_mesh(scene, &hm.data);
            let inst = renderer.add_instance(scene, id, v.position, Mat4::IDENTITY, mats);
            renderer.set_params(scene, inst, &[], false, &[]);
            meshes.push((id, inst));
        }
        let curled = curl_hands(&ty, GRIP_RADIUS);
        let knuckles = (ty.joints.finger - ty.joints.hand).length().clamp(0.12, 0.3) * 0.58;
        let hand_of = ty.meshes.iter().map(|m| {
            m.skin.iter().map(|inf| {
                (0..2).find(|&side| (0..inf.n as usize).any(|j| inf.slot[j] as usize == hand_slot(side) && inf.weight[j] > 0.5))
                    .map(|side| side as i8).unwrap_or(-1)
            }).collect()
        }).collect();
        let arm_of: Vec<Vec<i8>> = ty.meshes.iter().map(|m| {
            m.skin.iter().map(|inf| {
                (0..2).find(|&side| {
                    let w: f32 = (0..inf.n.max(1) as usize)
                        .filter(|&j| {
                            let s = inf.slot[j] as usize;
                            s == UPPER_ARM_SLOT[side] || s == FORE_ARM_SLOT[side]
                        })
                        .map(|j| if inf.n <= 1 { 1.0 } else { inf.weight[j] })
                        .sum();
                    w > 0.5
                }).map(|side| side as i8).unwrap_or(-1)
            }).collect()
        }).collect();
        let grip_rest = grip_centres(&ty, GRIP_RADIUS);
        let mut f = DriverFigure {
            ty,
            curled,
            knuckles,
            grip_rest,
            grip_fix: [Vec3::ZERO; 2],
            grip_radius: GRIP_RADIUS,
            pose: Pose::new(0x5eed_d71e),
            meshes,
            skins: Vec::new(),
            hip: Vec3::ZERO,
            floor: Vec3::ZERO,
            heading: 0.0,
            lamps: [-1; 4],
            wheel: None,
            sign: 0.0,
            lean: 0.0,
            base_lean: 0.0,
            show_hands_in_cab: false,
            shown: false,
            settled: false,
            slide: 0.0,
            hands: [Hand::default(); 2],
            hands_placed: false,
            theta: 0.0,
            last_theta: 0.0,
            still: 0.0,
            rate: 0.0,
            elbows: None,
            frames: [None; 2],
            rolls: [None; 2],
            hand_of,
            arm_of,
            blend: Default::default(),
            shifter: None,
            shift: ShiftState::default(),
            rand_seed: 12345,
        };
        f.seat_in(v, seat);
        Some(f)
    }

    fn next_rand(&mut self) -> f32 {
        self.rand_seed = self.rand_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rand_seed >> 8) as f32 / 16777216.0
    }

    pub fn attach(&mut self, v: &VehicleInstance) -> bool {
        match seat_of(v) {
            Some(seat) => {
                self.seat_in(v, seat);
                true
            }
            None => false,
        }
    }

    fn seat_in(&mut self, v: &VehicleInstance, seat: omsi_vehicle::cabin::PassPos) {
        let hip = Vec3::from(seat.pos);
        let r = seat.rot.to_radians();
        self.hip = hip;
        self.floor = Vec3::new(
            hip.x + r.sin() * SEAT_FRONT,
            hip.y + r.cos() * SEAT_FRONT,
            hip.z - seat.height.max(0.3),
        );
        self.heading = seat.rot;
        self.lamps = seat.illumination;
        self.wheel = find_wheel(v, hip);
        self.shifter = find_shifter(v, hip, seat.rot);
        self.shift = ShiftState::default();
        let r = self.wheel.as_ref().map(|w| w.tube + FINGER_HALF).unwrap_or(GRIP_RADIUS);
        if (r - self.grip_radius).abs() > 1e-4 {
            self.curled = curl_hands(&self.ty, r);
            self.grip_rest = grip_centres(&self.ty, r);
            self.grip_radius = r;
        }
        self.sign = 0.0;
        self.grip_fix = [Vec3::ZERO; 2];
        self.pose = Pose::new(0x5eed_d71e);
        self.settled = false;
        self.slide = 0.0;
        self.lean = 0.0;
        self.base_lean = 0.0;
        self.hands_placed = false;
        self.elbows = None;
        self.frames = [None; 2];
        self.rolls = [None; 2];
    }

    pub fn hide(&mut self, renderer: &Renderer, scene: &mut Scene) {
        for (_, inst) in &self.meshes {
            renderer.set_params(scene, *inst, &[], false, &[]);
        }
        self.shown = false;
    }

    pub fn update(&mut self, renderer: &Renderer, scene: &mut Scene, v: &VehicleInstance, render: &crate::scene::VehicleRender, dt: f32, show: bool, mirror_only: bool) {
        if show != self.shown {
            for (_, inst) in &self.meshes {
                renderer.set_params(scene, *inst, &[], show, &[]);
            }
            self.shown = show;
        }

        let force_visible_in_cab = mirror_only && self.show_hands_in_cab;
        for (_, inst) in &self.meshes {
            renderer.set_mirror_only(scene, *inst, if force_visible_in_cab { false } else { mirror_only });
        }

        if !show {
            return;
        }
        if self.settled {
            self.update_shift(v, dt);
        }
        if let Some(theta) = self.wheel_angle(v) {
            self.steer_hands(theta, if self.settled { dt } else { 0.0 });
        }

        let h = self.heading.to_radians();
        let fwd = Vec3::new(h.sin(), h.cos(), 0.0);
        if !self.settled {
            let mut comfort_extra = 0.0f32;
            for round in 0..28 {
                let mut p = Pose::new(0x5eed_d71e);
                let targets = self.hand_targets(v, 0.0);
                let try_input = self.pose_input(targets.as_ref(), fwd);
                for _ in 0..90 {
                    p.advance(&self.ty.rig, &try_input, 1.0 / 30.0);
                }
                let posed = p.bones(&self.ty.rig);
                let miss = match (try_input.grips, posed.ok) {
                    (Some(g), true) => (0..2).map(|k| (posed.wrist[k] - g[k]).length()).fold(0.0f32, f32::max),
                    _ => 0.0,
                };
                let excess = if posed.ok { self.arm_excess(&posed.elbow, &posed.wrist) } else { 0.0 };
                let moved = if posed.ok { self.keep_elbows(&posed.elbow) } else { 0.0 };
                let off = match (&targets, posed.ok) {
                    (Some(t), true) => {
                        let tubes = t.tubes.map(|q| self.to_person(q));
                        self.correct_grips(&posed.bones, tubes, 1.0, [true; 2])
                    }
                    _ => 0.0,
                };
                let comfort_left = self.slide < SLIDE_MAX || comfort_extra < LEAN_COMFORT - 0.01;
                let still_posing = miss < 0.02 && off < 0.01 && moved < 0.01;
                if (still_posing && (excess < 0.02 || !comfort_left)) || (self.slide >= SLIDE_MAX && self.lean >= 30.0) || round == 27 {
                    self.pose = p;
                    break;
                }
                if miss >= 0.02 {
                    if self.slide < SLIDE_MAX {
                        self.slide = (self.slide + miss * 0.8).min(SLIDE_MAX);
                    } else {
                        self.lean = (self.lean + (miss / 0.011).max(2.0)).min(30.0);
                    }
                } else if excess >= 0.02 && comfort_left {
                    if self.slide < SLIDE_MAX {
                        self.slide = (self.slide + excess * 0.8).min(SLIDE_MAX);
                    } else {
                        let add = (excess / 0.011).clamp(1.0, 3.0).min(LEAN_COMFORT - comfort_extra);
                        self.lean += add;
                        comfort_extra += add;
                    }
                }
            }
            self.settled = true;
            self.base_lean = self.lean;
            return self.update(renderer, scene, v, render, dt, show, mirror_only);
        }

        let targets = self.hand_targets(v, dt);
        let input = self.pose_input(targets.as_ref(), fwd);
        let floor = self.floor + fwd * self.slide;
        self.pose.advance(&self.ty.rig, &input, dt);
        let posed = self.pose.bones(&self.ty.rig);
        if posed.ok {
            self.keep_elbows(&posed.elbow);
        }
        if let (Some(t), true) = (&targets, posed.ok) {
            let tubes = t.tubes.map(|q| self.to_person(q));
            let holding = [0, 1].map(|k| self.hands[k].mv.is_none());
            self.correct_grips(&posed.bones, tubes, 1.0 - (-dt / FIX_EASE).exp(), holding);
        }

        if let (Some(g), true) = (input.grips, posed.ok && dt > 0.0) {
            let miss = (0..2).map(|k| (posed.wrist[k] - g[k]).length()).fold(0.0f32, f32::max);
            if miss > 0.015 {
                self.lean = (self.lean + (miss / 0.011) * dt * 4.0).min(34.0).min(self.base_lean + 6.0);
            } else if miss < 0.006 {
                self.lean = (self.lean - 4.0 * dt).max(self.base_lean);
            }
        }

        if !posed.ok && !self.skins.is_empty() {
            return;
        }
        self.skins.resize_with(self.ty.meshes.len(), Default::default);
        let open = [0, 1].map(|k| self.hands[k].open().max(self.shift_open(k)));
        
        for (k, m) in self.ty.meshes.iter().enumerate() {
            let (pos, nrm) = &mut self.skins[k];
            if open.iter().any(|&o| o > 0.01) {
                let blend = &mut self.blend;
                blend.0.clone_from(&self.curled[k].0);
                blend.1.clone_from(&self.curled[k].1);
                for (i, side) in self.hand_of[k].iter().enumerate() {
                    if *side >= 0 && open[*side as usize] > 0.01 {
                        let o = open[*side as usize];
                        blend.0[i] = blend.0[i].lerp(m.data.positions[i], o);
                        blend.1[i] = blend.1[i].lerp(m.data.normals[i], o).normalize_or_zero();
                    }
                }
                skin_from(m, blend, &posed.bones, pos, nrm);
            } else {
                skin_from(m, &self.curled[k], &posed.bones, pos, nrm);
            }

            // CORREÇÃO CRÍTICA DO CÂMERA/OMBROS TAMPANDO A VISÃO EM CAB VIEW
            if mirror_only && self.show_hands_in_cab {
                let (hand_of, arm_of) = (&self.hand_of[k], &self.arm_of[k]);
                let count = pos.len().min(hand_of.len()).min(arm_of.len());
                
                let mut hd_sum = [Vec3::ZERO; 2];
                let mut hd_n = [0.0f32; 2];
                for i in 0..count {
                    for side in 0..2 {
                        if hand_of[i] == side as i8 {
                            hd_sum[side] += pos[i];
                            hd_n[side] += 1.0;
                        }
                    }
                }
                
                // Ponto de recolhimento baseado puramente nos pulsos/mãos para evitar ombros esticados na tela
                let wrists: Vec<Vec3> = (0..2)
                    .filter_map(|h| if hd_n[h] > 0.0 { Some(hd_sum[h] / hd_n[h]) } else { None })
                    .collect();
                
                let fold = wrists.first().copied().unwrap_or(Vec3::ZERO);
                for i in 0..count {
                    // Se não for vértice pertencente à mão nem ao antebraço baixo, dobra para o pulso
                    if hand_of[i] < 0 && arm_of[i] < 0 {
                        let here = pos[i];
                        pos[i] = wrists.iter().copied()
                            .min_by(|a, b| a.distance_squared(here).total_cmp(&b.distance_squared(here)))
                            .unwrap_or(fold);
                    }
                }
            }

            renderer.update_mesh(scene, self.meshes[k].0, pos, nrm, &m.data.uvs);
        }
        let body = v.body_rotation();
        let at = v.position + body.transform_point3(floor).as_dvec3();
        let xf = body * Mat4::from_rotation_z(-h);
        let (first, count) = render.seat_lamps(v.ty.model.interior_lights.len(), &self.lamps).unwrap_or((0, 0));
        for (_, inst) in &self.meshes {
            renderer.set_transform(scene, *inst, at, xf);
            renderer.set_interior(scene, *inst, 0.0);
            renderer.set_interior_lamps(scene, *inst, first, count);
        }
    }
}

impl DriverFigure {
    fn wheel_frame(w: &Wheel, v: &VehicleInstance) -> (Mat4, Vec3, Vec3, Vec3, Vec3) {
        let turn = v.mesh_transforms.get(w.mesh).copied().unwrap_or(Mat4::IDENTITY);
        let centre = turn.transform_point3(w.centre);
        let axis = turn.transform_vector3(w.axis).normalize_or(w.axis);
        let mut up = (Vec3::Z - axis * axis.dot(Vec3::Z)).normalize_or_zero();
        if up.length_squared() < 0.5 {
            up = w.up;
        }
        let right = up.cross(axis).normalize_or(w.right);
        let right = if right.dot(w.right) < 0.0 { -right } else { right };
        (turn, centre, axis, up, right)
    }

    fn wheel_angle(&mut self, v: &VehicleInstance) -> Option<f32> {
        let w = self.wheel.as_ref()?;
        let (turn, _, _, up, right) = Self::wheel_frame(w, v);
        let up_now = turn.transform_vector3(w.up).normalize_or(w.up);
        let seen = up_now.dot(right).atan2(up_now.dot(up)).to_degrees();
        let by_var = v.var(&w.var).unwrap_or(0.0) * w.factor;
        if seen.abs() > 10.0 && seen.abs() < 170.0 {
            self.sign = if wrap(by_var - seen).abs() <= wrap(-by_var - seen).abs() { 1.0 } else { -1.0 };
        }
        let theta = if self.sign != 0.0 { by_var * self.sign } else { seen };
        self.theta = theta.clamp(-3600.0, 3600.0);
        Some(self.theta)
    }

    fn steer_hands(&mut self, theta: f32, dt: f32) {
        if !self.hands_placed {
            for k in 0..2 {
                self.hands[k] = Hand { on_rim: REST[k] - theta, mv: None };
            }
            self.hands_placed = true;
            self.last_theta = theta;
            self.still = 0.0;
            return;
        }
        let turned = theta - self.last_theta;
        self.last_theta = theta;
        if dt > 0.0 {
            self.rate += (turned / dt - self.rate) * (dt / 0.15).min(1.0);
        }
        if turned.abs() < 12.0 * dt.max(1e-3) {
            self.still += dt;
        } else {
            self.still = 0.0;
        }
        for k in 0..2 {
            if let Some(m) = &mut self.hands[k].mv {
                m.t = (m.t + dt / m.dur).clamp(0.0, 1.0);
                if m.t >= 1.0 {
                    self.hands[k] = Hand { on_rim: m.to - theta, mv: None };
                }
            }
        }
        let inside = |k: usize, a: f32| a >= RANGE[k].0 && a <= RANGE[k].1;
        let away = self.away();
        let freeze = self.shift.pending;

        let mut order = [0usize, 1];
        let out_by = |h: &Hand, k: usize| {
            let a = h.on_rim + theta;
            (RANGE[k].0 - a).max(a - RANGE[k].1)
        };
        if out_by(&self.hands[1], 1) > out_by(&self.hands[0], 0) {
            order = [1, 0];
        }
        let lead = (self.rate * REGRIP_LEAD).clamp(-LEAD_MAX, LEAD_MAX);
        for k in order {
            if self.hands[k].mv.is_some() || away[k] {
                continue;
            }
            let a = self.hands[k].on_rim + theta;
            let ahead = a + lead;
            if inside(k, a) && inside(k, ahead) {
                continue;
            }
            let (lo, hi) = RANGE[k];
            let lone = away[1 - k] || self.shift.one_hand_drive_time > 0.0;
            
            // Permite uma pegada mais elástica quando dirigindo com 1 mão só
            let max_over = if lone { ONE_HAND_OVER * 2.5 } else { ONE_HAND_OVER };
            let can_regrip = self.hands[1 - k].mv.is_none() && !freeze && (!lone || a > hi + max_over || a < lo - max_over);
            
            if can_regrip {
                let up = if a > hi { true } else if a < lo { false } else { ahead > hi };
                let to = if up { lo + REGRIP_BACK } else { hi - REGRIP_BACK };
                let to = if (to - REST[k]).abs() > 90.0 { REST[k] } else { to };
                let from = a;
                let dur = (0.38 + (to - from).abs() / 350.0) / (1.0 + self.rate.abs() / 2000.0);
                let dur = dur.max(0.28);
                self.hands[k] = Hand { on_rim: self.hands[k].on_rim, mv: Some(Regrip::new(from, to, dur, self.rate)) };
            } else {
                let (lo, hi) = RANGE[k];
                let before = self.hands[k].on_rim + theta - turned;
                let over = (lo - before).max(before - hi).max(0.0);
                let outward = (before > hi && turned > 0.0) || (before < lo && turned < 0.0);
                let follow = if outward { 1.0 - smooth(over / SLIP) } else { 1.0 };
                let now = (before + turned * follow).clamp(lo - SLIP * 2.0, hi + SLIP * 2.0);
                self.hands[k].on_rim = now - theta;
            }
        }
        if self.still > SETTLE_AFTER && self.hands.iter().all(|h| h.mv.is_none()) && !away.iter().any(|&a| a) {
            let far = |k: usize| (self.hands[k].on_rim + theta - REST[k]).abs();
            let k = if far(0) >= far(1) { 0 } else { 1 };
            if far(k) > 22.0 {
                let from = self.hands[k].on_rim + theta;
                self.hands[k].mv = Some(Regrip::new(from, REST[k], 0.45 + (REST[k] - from).abs() / 300.0, 0.0));
                self.still = 0.0;
            }
        }
    }

    fn hand_targets(&mut self, v: &VehicleInstance, dt: f32) -> Option<Targets> {
        let w = self.wheel.as_ref()?;
        let (_, centre, axis, up, right) = Self::wheel_frame(w, v);
        let h = self.heading.to_radians();
        let fwd = Vec3::new(h.sin(), h.cos(), 0.0);
        let mut t = Targets { grips: [Vec3::ZERO; 2], frames: [(Vec3::ZERO, Vec3::ZERO); 2], tubes: [Vec3::ZERO; 2] };
        let lever = self.lever_target(v, fwd);
        for k in 0..2 {
            let (seen, lift) = self.hands[k].seen(self.theta);
            let a = seen.to_radians();
            let radial = (up * a.cos() + right * a.sin()).normalize_or(up);
            let along = (right * a.cos() - up * a.sin()).normalize_or(right);
            let tube = centre + radial * w.radius + (axis * 0.85 + radial * 0.3) * (LIFT * lift);
            let elbow = self.elbows.map(|e| e[k]).unwrap_or(self.hip + Vec3::Z * 0.2 - fwd * 0.05 + (tube - centre).with_z(0.0) * 0.5);
            let fore = (tube - elbow).normalize_or(fwd);

            let (e1, e2) = (radial, -axis);
            let proj = fore - along * along.dot(fore);
            let want = proj.dot(e2).atan2(proj.dot(e1)).to_degrees().clamp(ROLL.0, ROLL.1);
            let want = ROLL_PLAIN + (want - ROLL_PLAIN) * smooth((proj.length() - 0.2) / 0.4);
            let roll = match self.rolls[k] {
                Some(r) if dt > 0.0 => r + (want - r) * (1.0 - (-dt / ROLL_EASE).exp()),
                _ => want,
            };
            self.rolls[k] = Some(roll);
            let (sr, cr) = roll.to_radians().sin_cos();
            let across = (e1 * cr + e2 * sr).normalize_or(e1);
            let dir = turn_towards(across, fore, DIAGONAL.to_radians());
            let palm = dir.cross(along).normalize_or(-axis);
            let palm = (palm - dir * dir.dot(palm)).normalize_or(palm);

            let mut tube = tube;
            let (mut dir, mut palm) = (dir, palm);
            if let Some((lh, e, knob, ldir, lpalm)) = lever {
                if lh == k {
                    tube = tube.lerp(knob, e) + Vec3::Z * (0.04 * (e * std::f32::consts::PI).sin());
                    let q = frame_quat(dir, palm).slerp(frame_quat(ldir, lpalm), e).normalize();
                    dir = q * Vec3::X;
                    palm = q * Vec3::Y;
                }
            }

            let (dir, palm) = match self.frames[k] {
                Some((d0, p0)) if dt > 0.0 => {
                    let q0 = glam::Quat::from_mat3(&glam::Mat3::from_cols(d0, p0, d0.cross(p0))).normalize();
                    let q1 = glam::Quat::from_mat3(&glam::Mat3::from_cols(dir, palm, dir.cross(palm))).normalize();
                    let q = q0.slerp(q1, 1.0 - (-dt / FRAME_EASE).exp()).normalize();
                    (q * Vec3::X, q * Vec3::Y)
                }
                _ => (dir, palm),
            };
            self.frames[k] = Some((dir, palm));
            let knuckle = tube - palm * self.grip_radius;
            t.grips[k] = knuckle - dir * self.knuckles;
            t.frames[k] = (dir, palm);
            t.tubes[k] = tube;
        }
        Some(t)
    }

    fn pose_input(&self, targets: Option<&Targets>, fwd: Vec3) -> PoseInput<'static> {
        let h = self.heading.to_radians();
        let turn_person = move |d: Vec3| Vec3::new(d.x * h.cos() - d.y * h.sin(), d.x * h.sin() + d.y * h.cos(), d.z);
        let floor = self.floor + fwd * self.slide;
        let hip = self.hip + fwd * self.slide;
        PoseInput {
            activity: Activity::Sit,
            origin: floor.as_dvec3(),
            heading: self.heading as f64,
            frame: 1,
            seat: Some(self.to_person(hip)),
            look: Some(self.to_person(hip + fwd * 20.0 + Vec3::Z * 0.4)),
            grips: targets.map(|t| [0, 1].map(|k| self.to_person(t.grips[k]) + self.grip_fix[k])),
            grip_frames: targets.map(|t| t.frames.map(|(d, p)| (turn_person(d), turn_person(p)))),
            grip_lean: self.lean,
            ..Default::default()
        }
    }

    fn to_person(&self, q: Vec3) -> Vec3 {
        let h = self.heading.to_radians();
        let d = q - (self.floor + Vec3::new(h.sin(), h.cos(), 0.0) * self.slide);
        Vec3::new(d.x * h.cos() - d.y * h.sin(), d.x * h.sin() + d.y * h.cos(), d.z)
    }

    fn from_person(&self, d: Vec3) -> Vec3 {
        let h = self.heading.to_radians();
        let floor = self.floor + Vec3::new(h.sin(), h.cos(), 0.0) * self.slide;
        floor + Vec3::new(d.x * h.cos() + d.y * h.sin(), -d.x * h.sin() + d.y * h.cos(), d.z)
    }

    fn keep_elbows(&mut self, posed: &[Vec3; 2]) -> f32 {
        let now = posed.map(|e| self.from_person(e));
        let moved = self.elbows.map(|e| (0..2).map(|k| (e[k] - now[k]).length()).fold(0.0, f32::max)).unwrap_or(1.0);
        self.elbows = Some(now);
        moved
    }

    fn correct_grips(&mut self, bones: &[glam::Affine3A], tubes: [Vec3; 2], gain: f32, which: [bool; 2]) -> f32 {
        let mut worst = 0.0f32;
        for k in (0..2).filter(|&k| which[k]) {
            let Some(rest) = self.grip_rest[k] else { continue };
            let Some(b) = bones.get(hand_slot(k)) else { continue };
            let held = Vec3::from(b.transform_point3a(rest.into()));
            let err = tubes[k] - held;
            if !err.is_finite() {
                continue;
            }
            worst = worst.max(err.length());
            let step = err * gain;
            let step = if gain < 1.0 { step.clamp_length_max(FIX_STEP) } else { step };
            let fix = self.grip_fix[k] + step;
            self.grip_fix[k] = fix.clamp_length_max(0.15);
        }
        worst
    }

    fn arm_excess(&self, elbow: &[Vec3], wrist: &[Vec3]) -> f32 {
        let h = self.heading.to_radians();
        let fwd = Vec3::new(h.sin(), h.cos(), 0.0);
        let hip = self.to_person(self.hip + fwd * self.slide);
        let l = self.lean.to_radians();
        let up = Vec3::new(0.0, l.sin(), l.cos());
        let mut worst = 0.0f32;
        for k in 0..2 {
            if k >= elbow.len() || k >= wrist.len() { continue; }
            let side = if k == 1 { 1.0 } else { -1.0 };
            let shoulder = hip + up * 0.55 + Vec3::new(0.14 * side, 0.0, 0.0);
            let arm = UPPER_ARM + (wrist[k] - elbow[k]).length();
            let excess = (wrist[k] - shoulder).length() - ARM_RATIO * arm;
            if excess.is_finite() {
                worst = worst.max(excess);
            }
        }
        worst
    }

    fn away(&self) -> [bool; 2] {
        let mut a = [false; 2];
        if let Some(sh) = &self.shifter {
            a[sh.hand] = self.shift.w > 0.0 || self.shift.one_hand_drive_time > 0.0;
        }
        a
    }

    fn shift_open(&self, k: usize) -> f32 {
        match &self.shifter {
            Some(sh) if sh.hand == k && self.shift.w > 0.0 && self.shift.w < 1.0 => {
                0.45 * (smooth(self.shift.w) * std::f32::consts::PI).sin().powi(2)
            }
            _ => 0.0,
        }
    }

    fn update_shift(&mut self, v: &VehicleInstance, dt: f32) {
        let Some(sh) = self.shifter.as_ref() else { return };
        if !(dt > 0.0) {
            return;
        }
        let dt = dt.min(0.1);
        let (hand, grab) = (sh.hand, sh.grab);
        let turn = v.mesh_transforms.get(sh.mesh).copied().unwrap_or(Mat4::IDENTITY);
        let knob = turn.transform_point3(grab);
        let vars: Vec<f32> = sh.vars.iter().map(|n| v.var(n).unwrap_or(0.0)).collect();
        let clutch = sh.clutch.as_ref().and_then(|n| v.var(n)).unwrap_or(0.0);

        let other = 1 - hand;
        let a = self.hands[other].on_rim + self.theta;
        let out = (RANGE[other].0 - a).max(a - RANGE[other].1);
        let busy = self.rate.abs() > 80.0 || out > 10.0;
        let calm = self.rate.abs() < 60.0 && out <= 0.0;
        let hands_free = self.hands[0].mv.is_none() && self.hands[1].mv.is_none();
        let other_free = self.hands[other].mv.is_none();

        // Atualiza timer de condução com uma mão só
        if self.shift.one_hand_drive_time > 0.0 {
            self.shift.one_hand_drive_time = (self.shift.one_hand_drive_time - dt).max(0.0);
        }

        let st = &mut self.shift;

        let speed = st.last_pos.map(|p| ((v.position - p).length() / dt as f64) as f32).unwrap_or(0.0);
        st.last_pos = Some(v.position);
        if speed < STOP_SPEED {
            st.still_for += dt;
        } else if speed > GO_SPEED {
            st.still_for = 0.0;
            st.stopped = false;
        }
        if st.still_for > STOP_AFTER {
            st.stopped = true;
        }
        st.cool = (st.cool - dt).max(0.0);

        let knob_speed = st.last_knob.map(|k| (knob - k).length() / dt).unwrap_or(0.0);
        st.last_knob = Some(knob);
        let var_moved = st.last_vars.len() == vars.len() && st.last_vars.iter().zip(&vars).any(|(a, b)| (a - b).abs() > 1e-3);
        st.last_vars = vars;
        let clutch_edge = clutch > 0.5 && st.last_clutch <= 0.5;
        st.last_clutch = clutch;
        let active = knob_speed > LEVER_MOVING || var_moved;
        if active || clutch_edge {
            st.idle = 0.0;
            if matches!(st.phase, ShiftPhase::Away | ShiftPhase::Back) {
                st.pending = true;
            }
        } else {
            st.idle += dt;
        }

        if active && !st.was_active && st.stroke >= 1.0 {
            st.stroke = 0.0;
            st.event_from = knob;
            st.event_travel = 0.0;
        }
        st.was_active = active;
        if st.stroke < 1.0 {
            st.event_travel = st.event_travel.max((knob - st.event_from).length());
            if st.phase == ShiftPhase::Hold {
                st.stroke += dt / STROKE_TIME;
            }
        }

        let want = (st.stopped && st.cool <= 0.0 && calm) || st.pending;
        let mut start: Option<bool> = None;
        
        match st.phase {
            ShiftPhase::Away => {
                if want {
                    let go = if st.pending { other_free } else { hands_free };
                    if go {
                        start = Some(st.pending);
                        st.phase = ShiftPhase::Reach;
                        st.pending = false;
                        st.waiting = 0.0;
                        
                        // Sorteia tempo dinâmico de permanecer com a mão na alavanca
                        st.hold_target_time = 0.2 + self.next_rand() * 1.8; 
                        
                        // Sorteia a possibilidade de dirigir com uma mão só por um tempo
                        if self.next_rand() < 0.4 {
                            st.one_hand_drive_time = 2.0 + self.next_rand() * 4.0;
                        }
                    } else {
                        st.waiting += dt;
                        if st.waiting > SHIFT_WAIT {
                            st.pending = false;
                            st.waiting = 0.0;
                        }
                    }
                } else {
                    st.waiting = 0.0;
                }
            }
            ShiftPhase::Reach => {
                st.w = (st.w + dt / st.reach).min(1.0);
                if st.w >= 1.0 {
                    st.phase = ShiftPhase::Hold;
                } else if busy && st.idle > 0.8 {
                    st.phase = ShiftPhase::Back;
                    st.cool = 1.5;
                }
            }
            ShiftPhase::Hold => {
                st.w = 1.0;
                if busy && st.idle > 0.8 {
                    st.phase = ShiftPhase::Back;
                    st.cool = 1.5;
                } else if !st.stopped && st.idle > (st.hold_target_time) && st.stroke >= 1.0 {
                    st.phase = ShiftPhase::Back;
                }
            }
            ShiftPhase::Back => {
                if want {
                    start = Some(st.pending);
                    st.phase = ShiftPhase::Reach;
                    st.pending = false;
                } else {
                    st.w = (st.w - dt / BACK_TIME).max(0.0);
                    if st.w <= 0.0 {
                        st.phase = ShiftPhase::Away;
                        st.stroke = 1.0;
                    }
                }
            }
        }
        if let Some(event) = start {
            st.reach = if event { REACH_FAST } else { REACH_TIME };
            st.w = st.w.max(0.001);
            if self.hands[hand].mv.is_some() {
                let (at, _) = self.hands[hand].seen(self.theta);
                self.hands[hand] = Hand { on_rim: at - self.theta, mv: None };
            }
        }
    }

    fn lever_target(&self, v: &VehicleInstance, fwd: Vec3) -> Option<(usize, f32, Vec3, Vec3, Vec3)> {
        let sh = self.shifter.as_ref()?;
        let st = &self.shift;
        if st.w <= 0.0 && st.one_hand_drive_time <= 0.0 {
            return None;
        }
        
        let weight = if st.w > 0.0 { smooth(st.w) } else { 1.0 };
        let turn = v.mesh_transforms.get(sh.mesh).copied().unwrap_or(Mat4::IDENTITY);
        let mut knob = turn.transform_point3(sh.grab);
        let axis = turn.transform_vector3(sh.axis).normalize_or(sh.axis);
        
        if st.stroke < 1.0 {
            let amp = 1.0 - smooth(st.event_travel / STROKE_FADE);
            knob += fwd * (STROKE * amp * (st.stroke.clamp(0.0, 1.0) * std::f32::consts::PI).sin());
        }
        
        let h = self.heading.to_radians();
        let right = Vec3::new(h.cos(), -h.sin(), 0.0);
        let side = if sh.hand == 1 { 1.0 } else { -1.0 };
        let shoulder = self.hip + fwd * self.slide + Vec3::Z * 0.55 + right * (0.14 * side);
        let palm = -axis;
        let reach = knob - shoulder;
        let mut dir = reach - palm * reach.dot(palm);
        if dir.length_squared() < 1e-4 {
            dir = fwd - palm * fwd.dot(palm);
        }
        let dir = dir.normalize_or(fwd);
        Some((sh.hand, weight, knob, dir, palm))
    }
}

fn frame_quat(d: Vec3, p: Vec3) -> glam::Quat {
    glam::Quat::from_mat3(&glam::Mat3::from_cols(d, p, d.cross(p))).normalize()
}

fn seat_of(v: &VehicleInstance) -> Option<omsi_vehicle::cabin::PassPos> {
    cabin_of(&v.ty.def)?.driver_positions.first().cloned()
}

pub fn cabin_of(def: &omsi_vehicle::Vehicle) -> Option<Arc<omsi_vehicle::PassengerCabin>> {
    static CABINS: std::sync::Mutex<Option<std::collections::HashMap<std::path::PathBuf, Option<Arc<omsi_vehicle::PassengerCabin>>>>> =
        std::sync::Mutex::new(None);
    let mut cabins = CABINS.lock().unwrap_or_else(|e| e.into_inner());
    cabins
        .get_or_insert_with(Default::default)
        .entry(def.path.clone())
        .or_insert_with(|| {
            let rel = def.passenger_cabin.as_ref()?;
            omsi_vehicle::PassengerCabin::load(&omsi_cfg::resolve_path(def.dir(), rel))
                .map_err(|e| log::warn!("driver: {e}"))
                .ok()
                .map(Arc::new)
        })
        .clone()
}

fn turn_towards(a: Vec3, b: Vec3, max: f32) -> Vec3 {
    let angle = a.angle_between(b);
    if angle <= max || !angle.is_finite() {
        return b;
    }
    let axis = a.cross(b).normalize_or_zero();
    if axis == Vec3::ZERO {
        return a;
    }
    glam::Quat::from_axis_angle(axis, max) * a
}

impl DriverFigure {
    pub fn human_type(&self) -> Arc<HumanType> {
        self.ty.clone()
    }
}

fn wrap(a: f32) -> f32 {
    (a + 540.0).rem_euclid(360.0) - 180.0
}

fn lever_knob(positions: &[Vec3], pivot: Option<Vec3>) -> Option<(Vec3, Vec3)> {
    if positions.len() < 4 {
        return None;
    }
    let base = pivot.unwrap_or_else(|| {
        let mut zs: Vec<f32> = positions.iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.total_cmp(b));
        let z = zs[zs.len() / 10];
        let low: Vec<&Vec3> = positions.iter().filter(|p| p.z <= z).collect();
        low.iter().fold(Vec3::ZERO, |s, p| s + **p) / low.len().max(1) as f32
    });
    let max = positions.iter().map(|p| (*p - base).length()).fold(0.0f32, f32::max);
    let (tip, axis) = if max >= 0.05 {
        let far: Vec<&Vec3> = positions.iter().filter(|p| (**p - base).length() >= max * 0.85).collect();
        let tip = far.iter().fold(Vec3::ZERO, |s, p| s + **p) / far.len().max(1) as f32;
        (tip, (tip - base).normalize_or(Vec3::Z))
    } else {
        let mut zs: Vec<f32> = positions.iter().map(|p| p.z).collect();
        zs.sort_by(|a, b| a.total_cmp(b));
        let z = zs[(zs.len() - 1) * 9 / 10];
        let top: Vec<&Vec3> = positions.iter().filter(|p| p.z >= z).collect();
        (top.iter().fold(Vec3::ZERO, |s, p| s + **p) / top.len().max(1) as f32, Vec3::Z)
    };
    if !tip.is_finite() || !axis.is_finite() {
        return None;
    }
    Some((tip - axis * 0.02, axis))
}

fn find_shifter(v: &VehicleInstance, hip: Vec3, heading: f32) -> Option<Shifter> {
    const STRONG: &[&str] = &[
        "gearlever", "gear_lever", "gearshift", "gear_shift", "shiftlever", "shift_lever",
        "shifter", "gearstick", "gear_stick", "schalthebel", "schaltknueppel", "schaltung",
        "antriebshebel", "antriebhebel", "antrieb_hebel", "cambio", "alavanca"
    ];
    const WEAK: &[&str] = &["antrieb", "gear", "shift", "schalt", "getriebe", "gang"];
    const NOT: &[&str] = &["light", "lamp", "display", "indic", "sound", "retard", "park", "door", "wiper", "text", "warn", "oil", "temp", "rpm", "tacho", "taster", "button", "btn"];
    
    let forced = omsi_cfg::env::var("OMSI_DRIVER_SHIFTER")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());

    if matches!(forced.as_deref(), Some("off") | Some("none") | Some("0")) {
        return None;
    }

    let grade = |name: &str| -> u8 {
        let n = name.to_ascii_lowercase();
        if let Some(f) = &forced {
            return if n.contains(f.as_str()) { 3 } else { 0 };
        }
        if NOT.iter().any(|w| n.contains(w)) {
            0
        } else if STRONG.iter().any(|w| n.contains(w)) {
            2
        } else if WEAK.iter().any(|w| n.contains(w)) {
            1
        } else {
            0
        }
    };

    let positions_of = |i: usize| -> Vec<Vec3> {
        let vm = &v.ty.meshes[i];
        let have: &[Vec3] = &vm.data.positions;
        if have.len() >= 6 {
            have.to_vec()
        } else {
            omsi_o3d::load_mesh(&vm.file)
                .ok()
                .map(|m| omsi_geometry::mesh_from_o3d(&m).positions)
                .unwrap_or_default()
        }
    };

    let shoulder = hip + Vec3::Z * 0.5;
    let mut best: Option<(f32, usize, Vec<String>, Vec3, Vec3)> = None;

    for (i, vm) in v.ty.meshes.iter().enumerate() {
        let def = &v.ty.model.meshes[vm.def_index];
        let mut grade_of_mesh = 0u8;
        let mut pivot: Option<Vec3> = None;
        let mut vars: Vec<String> = Vec::new();

        for a in &def.animations {
            let g = grade(&a.variable);
            if g == 0 {
                continue;
            }
            grade_of_mesh = grade_of_mesh.max(g);
            if pivot.is_none() {
                let origin = omsi_sim::anim::origin_matrix(&a.origins, vm.pivot);
                pivot = Some(origin.transform_point3(Vec3::ZERO));
            }
            if !vars.contains(&a.variable) {
                vars.push(a.variable.clone());
            }
        }

        if grade_of_mesh == 0 {
            let file = format!("{:?}", vm.file).to_ascii_lowercase();
            let by_file = match &forced {
                Some(f) => file.contains(f.as_str()),
                None => STRONG.iter().any(|w| file.contains(w)) || file.contains("antrieb"),
            };
            if !by_file {
                continue;
            }
            grade_of_mesh = 1;
        }

        let positions = positions_of(i);
        if positions.len() < 4 {
            continue;
        }

        let centroid = positions.iter().fold(Vec3::ZERO, |s, p| s + *p) / positions.len() as f32;
        let dist = (centroid - hip).length();
        if dist > 1.6 {
            continue;
        }

        let Some((grab, axis)) = lever_knob(&positions, pivot) else { continue };
        if (grab - shoulder).length() > LEVER_REACH {
            continue;
        }

        let score = grade_of_mesh as f32 * 2.0 - dist;
        if best.as_ref().map(|b| score > b.0).unwrap_or(true) {
            best = Some((score, i, vars, grab, axis));
        }
    }

    let (_, mesh, mut vars, grab, axis) = best?;
    for n in GEAR_VARS {
        if v.var(n).is_some() && !vars.iter().any(|x| x == n) {
            vars.push(n.to_string());
        }
    }
    let clutch = CLUTCH_VARS.iter().find(|n| v.var(n).is_some()).map(|n| n.to_string());
    let h = heading.to_radians();
    let d = grab - hip;
    let side = d.x * h.cos() - d.y * h.sin();
    let hand = if side >= 0.0 { 1 } else { 0 };
    Some(Shifter { mesh, vars, clutch, grab, axis, hand })
}

fn driver_type(world: &crate::scene::World, pick: u64) -> Option<Arc<HumanType>> {
    let listed: Vec<std::path::PathBuf> = omsi_map::ailists::load_list(&world.map_dir.join("drivers.txt"))
        .iter()
        .map(|l| omsi_cfg::resolve_path(&world.root, l))
        .collect();
    let path = if listed.is_empty() {
        let mut found: Vec<std::path::PathBuf> = Vec::new();
        for r in omsi_cfg::content_dirs("Humans") {
            for (group, is_dir) in omsi_cfg::vfs::list_dir(&r).unwrap_or_default() {
                if !is_dir {
                    continue;
                }
                let d = r.join(&group);
                for (n, _) in omsi_cfg::vfs::list_dir(&d).unwrap_or_default() {
                    let lower = n.to_string_lossy().to_ascii_lowercase();
                    if lower.ends_with(".hum") && lower.contains("driver") {
                        found.push(d.join(&n));
                    }
                }
            }
        }
        found.sort_by_key(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            (!n.starts_with("dbc_man04"), n)
        });
        found.into_iter().next()?
    } else {
        listed[(pick % listed.len() as u64) as usize].clone()
    };
    cached_type(&path)
}

pub fn cached_type(path: &std::path::Path) -> Option<Arc<HumanType>> {
    static TYPES: std::sync::Mutex<Option<std::collections::HashMap<std::path::PathBuf, Option<Arc<HumanType>>>>> =
        std::sync::Mutex::new(None);
    let path = path.to_path_buf();
    let mut types = TYPES.lock().unwrap_or_else(|e| e.into_inner());
    types
        .get_or_insert_with(Default::default)
        .entry(path.clone())
        .or_insert_with(|| {
            HumanType::load(&path)
                .map_err(|e| log::warn!("driver {}: {e:#}", path.display()))
                .ok()
                .map(Arc::new)
        })
        .clone()
}

fn find_wheel(v: &VehicleInstance, hip: Vec3) -> Option<Wheel> {
    let mut best: Option<(f32, usize, Mat4, String, f32)> = None;
    for (i, vm) in v.ty.meshes.iter().enumerate() {
        let def = &v.ty.model.meshes[vm.def_index];
        for a in &def.animations {
            let var_name = a.variable.to_ascii_lowercase();
            let is_steering_var = var_name.starts_with("axle_steering") 
                || var_name.contains("steering") 
                || var_name.contains("volante")
                || var_name.contains("stwheel");
                
            if !is_steering_var || a.kind != Some(omsi_model::AnimKind::Rot) {
                continue;
            }
            
            let factor = if a.factor.abs() < 1.0 { 1000.0 } else { a.factor };
            let origin = omsi_sim::anim::origin_matrix(&a.origins, vm.pivot);
            let c = origin.transform_point3(Vec3::ZERO);
            if (c - hip).length() > 1.5 {
                continue;
            }
            if best.as_ref().map(|b| factor.abs() > b.0).unwrap_or(true) {
                best = Some((factor.abs(), i, origin, a.variable.clone(), factor));
            }
        }
    }

    let (_, mesh, origin, var, factor) = best?;
    let origin_point = origin.transform_point3(Vec3::ZERO);
    let centre = origin_point;
    let mut axis = origin.transform_vector3(Vec3::X).normalize_or_zero();

    let shoulders = hip + Vec3::Z * 0.5;
    if (axis.z.abs() > 0.4 && axis.z < 0.0) || (axis.z.abs() <= 0.4 && axis.dot(shoulders - centre) < 0.0) {
        axis = -axis;
    }
    let mut up = (Vec3::Z - axis * axis.dot(Vec3::Z)).normalize_or_zero();
    if up.length_squared() < 0.5 {
        up = (Vec3::Y - axis * axis.dot(Vec3::Y)).normalize_or_zero();
    }
    let right = up.cross(axis).normalize_or_zero();
    let right = if right.x < 0.0 { -right } else { right };

    let vm = &v.ty.meshes[mesh];
    let loaded;
    let positions: &[Vec3] = if vm.data.positions.len() >= 12 {
        &vm.data.positions
    } else {
        loaded = omsi_o3d::load_mesh(&vm.file)
            .ok()
            .map(|m| omsi_geometry::mesh_from_o3d(&m).positions)
            .unwrap_or_default();
        &loaded
    };
    let radius_of = |p: &Vec3| {
        let d = *p - centre;
        (d - axis * axis.dot(d)).length()
    };
    let mut radii: Vec<f32> = positions.iter().map(radius_of).collect();
    radii.sort_by(|a, b| a.total_cmp(b));

    let rim = if radii.len() < 12 { 0.26 } else { radii[(radii.len() as f32 * 0.93) as usize] };
    let pct = |v: &mut Vec<f32>, f: f32| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f32 * f) as usize]
    };
    let ring: Vec<&Vec3> = positions.iter().filter(|p| radius_of(p) > rim * 0.8).collect();
    let (radius, along, tube) = if ring.len() >= 8 {
        let mut rs: Vec<f32> = ring.iter().map(|p| radius_of(p)).collect();
        let mut zs: Vec<f32> = ring.iter().map(|p| axis.dot(**p - origin_point)).collect();
        let (r0, r1) = (pct(&mut rs, 0.02), pct(&mut rs, 0.98));
        let (z0, z1) = (pct(&mut zs, 0.02), pct(&mut zs, 0.98));
        ((r0 + r1) * 0.5, (z0 + z1) * 0.5, ((r1 - r0).max(z1 - z0) * 0.5).clamp(0.01, 0.03))
    } else {
        (rim * 0.93, 0.0, 0.017)
    };
    let centre = origin_point + axis * along.clamp(-0.3, 0.3);
    let radius = radius.clamp(0.14, 0.32);
    Some(Wheel { mesh, var, factor, axis, centre, up, right, radius, tube })
}
