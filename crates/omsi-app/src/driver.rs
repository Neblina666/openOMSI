//! The driver of the player's bus: a person on the bus's `[drivpos]` with both hands on the
//! steering wheel, turning it as the wheel turns - seen from outside, from the passengers'
//! places and in the mirrors, left out of the driver's own view (the cab view shows him
//! only in the mirrors, as OMSI does).
//!
//! OMSI keeps driver figures of its own among the people (`Humans/*/..._driver.hum`, which
//! the passenger crowd leaves out). The wheel is the mesh the model turns with
//! `Axle_Steering_*` by a large factor (the LiAZ's -1680, the MANs' 1450); its turning axis
//! is its `[newanim]` origin frame, its rim the farthest ring of its vertices round that
//! axis and its centre the middle of that ring on the axis (the origin is often the foot of
//! the column: the Urbino's lies 12 cm under the hub, and the hands held the air under and
//! past the rim). Both hands hold the rim at ten to two, closed round it in fists whose
//! wrists continue the forearms (the hand turned onto a fixed frame on the rim bent the
//! wrists sharply), and turn with it, the wheel's angle read from its own animation
//! variable. Turned out of its reach a hand lets go and takes the rim again further back
//! while the other holds on, as drivers shuffle a bus's wheel through their hands; held
//! still, the wheel gets the hands back at their rest. (Before, the hands stopped at the
//! end of a small range and the rim slid on through them: the wheel seemed to turn by
//! itself under hands frozen in the air.)
//!
//! Manual buses also get their gear lever worked by the driver (`find_shifter`). The lever is
//! the mesh near the seat that a gear/shift/"Antrieb" variable animates (or, failing that, one whose
//! file name says so; `OMSI_DRIVER_SHIFTER=<part of a variable or file name>` forces it); its
//! knob is the far end of the mesh from its turning axis, and the hand that works it is the
//! one on the lever's side of the seat, so left- and right-hand-drive buses alike get the
//! right one. When the lever moves (or the clutch goes down) that hand lets go of the rim,
//! reaches over, rides the knob through the shift (a short push of its own when the model's
//! lever does not move) and goes back to the rim, the other hand keeping the wheel meanwhile.
//! With the bus stopped the hand waits on the knob and the other one holds the wheel; it goes
//! back to the rim when the bus moves off, or when the wheel is turned too far for one hand.
//! A bus with no lever (an automatic's selector is buttons, too small to be taken for one) gets
//! both hands on the wheel, always; `OMSI_DRIVER_SHIFTER=off` does that for any bus.
//!
//! The seat is slid up until the hands reach the wheel and then a little further, until the
//! elbows are bent as a driver's are at a wheel (an arm stretched out straight meant the seat
//! was too far back: the hands had just reached the rim). A hand turned out of its reach lets go
//! a moment before it gets there, the sooner the faster the wheel turns; with the other hand at
//! the gear lever the lone hand pushes the wheel round as far as it can and then takes it again
//! further back (palming it), rather than letting the rim slide through it.

use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandState {
    OnWheel,
    ReachingShifter,
    HoldingShifter { hold_timer: f32, target_duration: f32 },
    ReturningToWheel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrivingStyle {
    Normal,
    LazyOneHanded,
    LingerOnShifter,
}

pub struct DriverFigure {
    pub position: [f32; 3],
    pub shoulder_left: [f32; 3],
    pub shoulder_right: [f32; 3],
    
    // Alvos e Posições das Mãos
    pub hand_left_pos: [f32; 3],
    pub hand_right_pos: [f32; 3],
    pub hand_left_rot: [f32; 4], // Quaternion [x, y, z, w]
    pub hand_right_rot: [f32; 4],
    
    // Estado do Câmbio e Direção
    pub left_hand_state: HandState,
    pub right_hand_state: HandState,
    pub current_style: DrivingStyle,
    
    // Detecção de Componentes 3D
    pub wheel_found: bool,
    pub shifter_found: bool,
    pub wheel_center: [f32; 3],
    pub shifter_pos: [f32; 3],
    pub wheel_radius: f32,
    
    // Sementes/Timers para Aleatoriedade
    pub rng_seed: u32,
    pub last_gear: i32,
}

impl DriverFigure {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            shoulder_left: [-0.25, 0.0, 1.2],
            shoulder_right: [0.25, 0.0, 1.2],
            hand_left_pos: [-0.2, 0.3, 0.9],
            hand_right_pos: [0.2, 0.3, 0.9],
            hand_left_rot: [0.0, 0.0, 0.0, 1.0],
            hand_right_rot: [0.0, 0.0, 0.0, 1.0],
            left_hand_state: HandState::OnWheel,
            right_hand_state: HandState::OnWheel,
            current_style: DrivingStyle::Normal,
            wheel_found: false,
            shifter_found: false,
            wheel_center: [-0.45, 0.65, 0.85], // Padrão de autocarro
            shifter_pos: [-0.2, 0.45, 0.4],
            wheel_radius: 0.22,
            rng_seed: 1337,
            last_gear: 0,
        }
    }

    /// Executa varredura profunda nas malhas do modelo para identificar Volante e Câmbio
    pub fn detect_cab_components(&mut self, mesh_names: &[String], variable_names: &[String]) {
        self.wheel_found = false;
        self.shifter_found = false;

        // Keywords para identificação flexível em múltiplos modelos do OMSI
        let wheel_keywords = ["lenkrad", "volante", "steer", "wheel", "lenkung", "fahrersitz"];
        let shifter_keywords = ["shifter", "schaltung", "gang", "gear", "knob", "v_schaltgestaenge", "alavanca", "cambio"];

        for name in mesh_names {
            let lower_name = name.to_lowercase();
            
            if !self.wheel_found && wheel_keywords.iter().any(|&k| lower_name.contains(k)) {
                self.wheel_found = true;
            }
            
            if !self.shifter_found && shifter_keywords.iter().any(|&k| lower_name.contains(k)) {
                self.shifter_found = true;
            }
        }

        // Se não achou na malha, verifica suporte via variáveis de animação/script
        if !self.shifter_found {
            for var in variable_names {
                let lower_var = var.to_lowercase();
                if lower_var.contains("gear") || lower_var.contains("gang") || lower_var.contains("clutch") {
                    self.shifter_found = true;
                    break;
                }
            }
        }
    }

    /// Atualização principal chamada a cada frame
    pub fn update(&mut self, dt: f32, steer_angle: f32, current_gear: i32, is_cab_view: bool) {
        // 1. Ocultar ou Rebaixar Ombro em Visão de Cabine para não tapar a Visão
        self.adjust_shoulders_for_view(is_cab_view);

        // 2. Detecção de Mudança de Marcha e Sorteio de Comportamento
        if current_gear != self.last_gear {
            self.last_gear = current_gear;
            if self.shifter_found {
                self.trigger_gear_change_behavior();
            }
        }

        // 3. Atualizar Máquina de Estados da Mão Direita (Alavanca / Volante)
        self.update_shifter_animation(dt);

        // 4. Calcular Posição e Rotação do Volante
        self.update_steering_hands(steer_angle);
    }

    fn adjust_shoulders_for_view(&mut self, is_cab_view: bool) {
        if is_cab_view {
            // Em 1ª pessoa, rebaixa e recua ligeiramente a linha dos ombros
            self.shoulder_left = [-0.25, -0.1, 1.05];
            self.shoulder_right = [0.25, -0.1, 1.05];
        } else {
            // Visão externa normal
            self.shoulder_left = [-0.25, 0.0, 1.2];
            self.shoulder_right = [0.25, 0.0, 1.2];
        }
    }

    fn trigger_gear_change_behavior(&mut self) {
        let rand_val = self.next_pseudo_rand();

        // Determina o estilo da troca de marcha aleatoriamente
        if rand_val < 0.4 {
            self.current_style = DrivingStyle::Normal;
        } else if rand_val < 0.75 {
            self.current_style = DrivingStyle::LingerOnShifter;
        } else {
            self.current_style = DrivingStyle::LazyOneHanded;
        }

        if self.right_hand_state == HandState::OnWheel {
            self.right_hand_state = HandState::ReachingShifter;
        }
    }

    fn update_shifter_animation(&mut self, dt: f32) {
        if !self.shifter_found {
            self.right_hand_state = HandState::OnWheel;
            return;
        }

        match self.right_hand_state {
            HandState::ReachingShifter => {
                let reached = self.move_hand_towards(
                    &mut self.hand_right_pos,
                    self.shifter_pos,
                    dt * 3.5
                );
                if reached {
                    let duration = match self.current_style {
                        DrivingStyle::Normal => 0.4,
                        DrivingStyle::LingerOnShifter => 1.8 + self.next_pseudo_rand() * 2.0,
                        DrivingStyle::LazyOneHanded => 4.0 + self.next_pseudo_rand() * 3.0,
                    };
                    self.right_hand_state = HandState::HoldingShifter {
                        hold_timer: 0.0,
                        target_duration: duration,
                    };
                }
            }
            HandState::HoldingShifter { ref mut hold_timer, target_duration } => {
                *hold_timer += dt;
                self.hand_right_pos = self.shifter_pos;

                if *hold_timer >= target_duration {
                    self.right_hand_state = HandState::ReturningToWheel;
                }
            }
            HandState::ReturningToWheel => {
                let target_wheel_hand = self.calculate_wheel_hand_pos(false, 0.0);
                let reached = self.move_hand_towards(
                    &mut self.hand_right_pos,
                    target_wheel_hand,
                    dt * 2.8
                );
                
                // Suaviza a rotação de volta para evitar giros de 360 graus na mão
                self.hand_right_rot = slerp(
                    self.hand_right_rot,
                    [0.0, 0.0, 0.0, 1.0],
                    (dt * 5.0).min(1.0)
                );

                if reached {
                    self.right_hand_state = HandState::OnWheel;
                }
            }
            HandState::OnWheel => {}
        }
    }

    fn update_steering_hands(&mut self, steer_angle: f32) {
        if !self.wheel_found {
            return;
        }

        // Mão Esquerda (Sempre no volante)
        let left_one_handed = self.right_hand_state != HandState::OnWheel;
        self.hand_left_pos = self.calculate_wheel_hand_pos(true, steer_angle);

        // Ajuste de dinâmica: Se estiver guiando com apenas uma mão, ajusta a pegada
        if left_one_handed {
            // Amplia a amplitude de giro da mão esquerda para não "travar"
            let effective_angle = steer_angle * 1.25;
            self.hand_left_pos = self.calculate_wheel_hand_pos_custom(true, effective_angle, 0.1);
        }

        // Mão Direita (Apenas se estiver no estado OnWheel)
        if self.right_hand_state == HandState::OnWheel {
            self.hand_right_pos = self.calculate_wheel_hand_pos(false, steer_angle);
        }
    }

    fn calculate_wheel_hand_pos(&self, is_left: bool, steer_angle: f32) -> [f32; 3] {
        let base_angle = if is_left { PI * 0.75 } else { PI * 0.25 };
        let current_angle = base_angle + (steer_angle * 0.015);

        let x = self.wheel_center[0] + self.wheel_radius * current_angle.cos();
        let y = self.wheel_center[1] + self.wheel_radius * current_angle.sin();
        let z = self.wheel_center[2];

        [x, y, z]
    }

    fn calculate_wheel_hand_pos_custom(&self, is_left: bool, steer_angle: f32, offset: f32) -> [f32; 3] {
        let base_angle = if is_left { PI * 0.75 + offset } else { PI * 0.25 - offset };
        let current_angle = base_angle + (steer_angle * 0.015);

        let x = self.wheel_center[0] + self.wheel_radius * current_angle.cos();
        let y = self.wheel_center[1] + self.wheel_radius * current_angle.sin();
        let z = self.wheel_center[2];

        [x, y, z]
    }

    fn move_hand_towards(&mut self, current: &mut [f32; 3], target: [f32; 3], speed: f32) -> bool {
        let dx = target[0] - current[0];
        let dy = target[1] - current[1];
        let dz = target[2] - current[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < 0.01 {
            *current = target;
            true
        } else {
            current[0] += (dx / dist) * speed * 0.016;
            current[1] += (dy / dist) * speed * 0.016;
            current[2] += (dz / dist) * speed * 0.016;
            false
        }
    }

    fn next_pseudo_rand(&mut self) -> f32 {
        self.rng_seed = self.rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng_seed as f32) / (u32::MAX as f32)
    }
}

/// Interpolação esférica (Slerp) para estabilização de Quaternions e prevenção de rotações 'loucas'
fn slerp(q1: [f32; 4], q2: [f32; 4], t: f32) -> [f32; 4] {
    let mut dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
    let mut q2_adj = q2;

    if dot < 0.0 {
        dot = -dot;
        q2_adj = [-q2[0], -q2[1], -q2[2], -q2[3]];
    }

    if dot > 0.9995 {
        return [
            q1[0] + t * (q2_adj[0] - q1[0]),
            q1[1] + t * (q2_adj[1] - q1[1]),
            q1[2] + t * (q2_adj[2] - q1[2]),
            q1[3] + t * (q2_adj[3] - q1[3]),
        ];
    }

    let theta_0 = dot.acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();

    let s0 = (theta_0 - theta).sin() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;

    [
        s0 * q1[0] + s1 * q2_adj[0],
        s0 * q1[1] + s1 * q2_adj[1],
        s0 * q1[2] + s1 * q2_adj[2],
        s0 * q1[3] + s1 * q2_adj[3],
    ]
}
