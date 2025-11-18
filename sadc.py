#!/usr/bin/env python3
"""
LEO Nadir-Pointing Attitude Control via Receding-Horizon CLF–QP with Magnetorquers

- State: x = [q0, q1, q2, q3, wx, wy, wz] (scalar-first quaternion + body rates)
- Dynamics: I * w_dot = -w × (I w) + m × B_B + tau_dist
- Actuator: 3-axis magnetorquer, |m_i| <= 8.19 A·m^2 (Panyalert et al.)
- Control: Receding-horizon CLF–QP solved with CasADi (IPOPT)
- Objective: Track nadir-pointing reference in LEO, with CLF-based stability
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class HyperParams:
    """All tunable knobs for the CLF–QP MPC and simulation."""
    dt: float = 0.2
    horizon_steps: int = 40           # 8 s horizon
    # Simulated time window [s]; set long enough
    # to see significant variation of B_body over the orbit
    total_time: float = 5400        # one full 90-minute is one orbit revolution
    theta0: float = 0.0
    initial_attitude_offset_deg: float = 10.0
    magnetorquer_limit: float = 81.9   # 10x larger dipole limit
    control_weight_diag: tuple = (1e-4, 1e-4, 1e-4)
    slack_weight: float = 1e5        # CLF slack cost
    terminal_weight: float = 20.0
    kappa: float = 2.0               # CLF attitude weight
    alpha: float = 0.1               # CLF decay rate (per second)
    state_q_weight: float = 1e5      # strongly prioritize attitude error
    state_w_weight: float = 10.0     # de-emphasize rate tracking
    rw_max_torque: float = 5e-5      # reaction wheel max torque about body Y [N·m]
    rw_weight: float = 1.0           # cost weight for reaction wheel torque
    enable_clf: bool = True          # enable CLF constraints/penalties
    solver_tol: float = 1e-6
    solver_max_iter: int = 200
    disturbance_std: float = 1e-5
    # Orbit-tracking reference options
    reference_rate_scale: float = 1.0
    ref_enable_dither: bool = True
    ref_dither_harmonic: float = 3.0
    ref_roll_deg: float = 6.0
    ref_pitch_deg: float = 4.0
    ref_yaw_deg: float = 0.0
    ref_pitch_phase_deg: float = 90.0
    ref_yaw_phase_deg: float = 180.0
    # Optional sinusoidal reference for validation
    use_sinusoidal_reference: bool = False
    sin_ref_amp_deg: tuple = (10.0, 0.0, 5.0)  # (roll,pitch,yaw) amplitudes
    sin_ref_omega: float = 0.005               # [rad/s] sinusoid frequency
    # MPC visualization options
    mpc_snapshot_window: float = 20.0          # [s] history window before snapshot
    # mpc_snapshot_count <= 0  => snapshot at every step (full MPC video)
    # mpc_snapshot_count > 0   => that many snapshots spread across the run
    mpc_snapshot_count: int = 0
    animation_interval_ms: int = 50
    results_dir: str = "results"
    rng_seed: Optional[int] = None


HP = HyperParams()
if HP.rng_seed is not None:
    np.random.seed(HP.rng_seed)

# =========================
#  Utility: Quaternion math
# =========================

def quat_mul(q1, q2):
    """
    Hamilton product of two scalar-first quaternions q1, q2 (shape (4,)).
    Works for numpy arrays and CasADi vectors (SX/MX).
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return ca.vertcat(w, x, y, z) if isinstance(w, ca.MX) or isinstance(w, ca.SX) else np.array([w, x, y, z])


def quat_conj(q):
    """Quaternion conjugate, scalar-first."""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3]) if isinstance(q[0], (ca.SX, ca.MX)) else np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q):
    """
    Convert scalar-first quaternion q = [q0, q1, q2, q3] to rotation matrix R (body->inertial).
    This is numeric (numpy) only; CasADi version not needed here.
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
    ])
    return R


def dcm_to_quat(R):
    """
    Convert rotation matrix R (body->inertial) to scalar-first quaternion.
    Numeric (numpy) implementation.
    """
    tr = np.trace(R)
    if tr > 0:
        t = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * t
        q1 = (R[2, 1] - R[1, 2]) / t
        q2 = (R[0, 2] - R[2, 0]) / t
        q3 = (R[1, 0] - R[0, 1]) / t
    else:
        # Robust branches
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            t = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q0 = (R[2, 1] - R[1, 2]) / t
            q1 = 0.25 * t
            q2 = (R[0, 1] + R[1, 0]) / t
            q3 = (R[0, 2] + R[2, 0]) / t
        elif R[1, 1] > R[2, 2]:
            t = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q0 = (R[0, 2] - R[2, 0]) / t
            q1 = (R[0, 1] + R[1, 0]) / t
            q2 = 0.25 * t
            q3 = (R[1, 2] + R[2, 1]) / t
        else:
            t = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q0 = (R[1, 0] - R[0, 1]) / t
            q1 = (R[0, 2] + R[2, 0]) / t
            q2 = (R[1, 2] + R[2, 1]) / t
            q3 = 0.25 * t
    q = np.array([q0, q1, q2, q3])
    # Normalize and fix sign
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


# =========================
#  Satellite & Environment
# =========================

# Inertia matrix (kg·m^2) from characterization (ITC1)
Ixx, Iyy, Izz = 0.07264, 0.07258, 0.11644  # reduced by factor 10
I_cas = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
I_num = np.diag([Ixx, Iyy, Izz])
I_inv_num = np.linalg.inv(I_num)

# Magnetorquer limits (Panyalert et al.), scaled up for stronger actuation
m_max = HP.magnetorquer_limit  # [A·m^2]
tau_rw_max = HP.rw_max_torque  # [N·m] reaction wheel about body Y
R_rw = HP.rw_weight
# Control & CLF weights
R_c = ca.diag(ca.vertcat(*HP.control_weight_diag))
W_s = HP.slack_weight
kappa = HP.kappa   # CLF attitude weight
alpha = HP.alpha   # CLF decay rate
Q_w = HP.state_w_weight * ca.DM.eye(3)
Q_q = HP.state_q_weight * ca.DM.eye(3)
dt = HP.dt         # control step
N_horizon = HP.horizon_steps  # horizon steps
HORIZON_TIME = dt * N_horizon
terminal_weight = HP.terminal_weight

# Orbit and Earth magnetic model (very simple dipole)
R_earth = 6371000.0   # [m]
h_orbit = 500000.0    # [m] (500 km)
R_orbit = R_earth + h_orbit
T_orbit = 90.0*60.0   # 90 min
omega_orbit = 2.0*np.pi / T_orbit  # [rad/s]

# Earth magnetic dipole
M_earth = 7.96e22  # [A·m^2] approximate
mu0_over_4pi = 1e-7  # [H/m]/(4π) ~ 10^-7


def earth_B_inertial(r_I):
    """
    Simple dipole model for Earth's magnetic field in inertial coordinates.
    r_I : position vector in inertial frame (np.array, shape (3,))
    Returns B_I in Tesla.
    """
    r = r_I
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        return np.zeros(3)
    m_vec = np.array([0.0, 0.0, M_earth])  # dipole aligned with inertial Z
    term1 = 3.0 * np.dot(m_vec, r) * r / (r_norm**5)
    term2 = m_vec / (r_norm**3)
    B_I = mu0_over_4pi * (term1 - term2)
    return B_I


def reference_attitude(theta):
    """
    LVLH-like reference for equatorial orbit angle theta.
    - x_B points to nadir (-r_hat)  [dominant payload axis]
    - z_B points along velocity direction (tangential)
    - y_B completes right-handed triad
    Returns (q_ref, w_ref_body)
    """
    # Position and velocity in inertial frame for equatorial circular orbit
    r_I = np.array([R_orbit*np.cos(theta), R_orbit*np.sin(theta), 0.0])
    v_I = omega_orbit * np.array([-R_orbit*np.sin(theta), R_orbit*np.cos(theta), 0.0])

    r_hat = r_I / np.linalg.norm(r_I)
    v_hat = v_I / np.linalg.norm(v_I)

    x_B = -r_hat         # body x axis to nadir (dominant)
    z_B = v_hat          # body z axis along-track
    y_B = np.cross(z_B, x_B)
    y_B = y_B / np.linalg.norm(y_B)

    # Re-orthogonalize if needed
    z_B = np.cross(x_B, y_B)
    z_B = z_B / np.linalg.norm(z_B)

    # Rotation matrix from body to inertial: columns are body axes in inertial coords
    R_IB = np.column_stack((x_B, y_B, z_B))

    # Reference body angular velocity: roughly orbit rate about y_B
    w_ref_body = np.array([0.0, omega_orbit * HP.reference_rate_scale, 0.0])

    if HP.ref_enable_dither:
        R_IB, w_ref_body = apply_reference_dither(R_IB, theta, w_ref_body)

    q_ref = dcm_to_quat(R_IB)
    return q_ref, w_ref_body, r_I


def sinusoid_reference(t):
    """
    Simple body-fixed sinusoidal reference for validation.
    Euler angles (roll, pitch, yaw) follow sin(ω t) with configurable amplitudes.
    Returns (q_ref, w_ref_body).
    """
    if not HP.use_sinusoidal_reference:
        raise RuntimeError("sinusoid_reference called but use_sinusoidal_reference=False")
    amp_roll_deg, amp_pitch_deg, amp_yaw_deg = HP.sin_ref_amp_deg
    w = HP.sin_ref_omega
    roll = np.deg2rad(amp_roll_deg) * np.sin(w * t)
    pitch = np.deg2rad(amp_pitch_deg) * np.sin(w * t)
    yaw = np.deg2rad(amp_yaw_deg) * np.sin(w * t)
    R_extra = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    q_ref = dcm_to_quat(R_extra)
    # Approximate body rates from derivative of Euler angles (small-angle assumption)
    roll_rate = np.deg2rad(amp_roll_deg) * w * np.cos(w * t)
    pitch_rate = np.deg2rad(amp_pitch_deg) * w * np.cos(w * t)
    yaw_rate = np.deg2rad(amp_yaw_deg) * w * np.cos(w * t)
    w_ref_body = np.array([roll_rate, pitch_rate, yaw_rate])
    return q_ref, w_ref_body


def apply_reference_dither(R_IB, theta, base_w):
    """
    Inject small roll/pitch/yaw dithers so the reference trajectory remains dynamic.
    """
    harmonic = max(HP.ref_dither_harmonic, 0.0)
    if harmonic == 0.0:
        return R_IB, base_w

    roll_amp = np.deg2rad(HP.ref_roll_deg)
    pitch_amp = np.deg2rad(HP.ref_pitch_deg)
    yaw_amp = np.deg2rad(HP.ref_yaw_deg)
    pitch_phase = np.deg2rad(HP.ref_pitch_phase_deg)
    yaw_phase = np.deg2rad(HP.ref_yaw_phase_deg)

    roll_cmd = roll_amp * np.sin(harmonic * theta)
    pitch_cmd = pitch_amp * np.sin(harmonic * theta + pitch_phase)
    yaw_cmd = yaw_amp * np.sin(harmonic * theta + yaw_phase)

    R_extra = rot_z(yaw_cmd) @ rot_y(pitch_cmd) @ rot_x(roll_cmd)
    R_IB_new = R_IB @ R_extra

    theta_dot = omega_orbit
    roll_rate = roll_amp * harmonic * theta_dot * np.cos(harmonic * theta)
    pitch_rate = pitch_amp * harmonic * theta_dot * np.cos(harmonic * theta + pitch_phase)
    yaw_rate = yaw_amp * harmonic * theta_dot * np.cos(harmonic * theta + yaw_phase)

    w_ref_body = base_w + np.array([roll_rate, pitch_rate, yaw_rate])
    return R_IB_new, w_ref_body


# =========================
#  Build CLF–QP MPC in CasADi
# =========================

def build_clf_qp_mpc():
    """
    Build a receding-horizon CLF–QP MPC problem in CasADi Opti.

    Decision variables:
      X[:,0..N]     - state trajectory (q0,q1,q2,q3,wx,wy,wz)
      U[0:3,0..N-1] - magnetorquer dipole sequence m = [m_x,m_y,m_z]
      U[3,0..N-1]   - reaction wheel torque about body Y (tau_rw)
      Delta[0..N-1] - CLF slack sequence (optional)

    Parameters:
      X0         - initial state
      Q_ref_seq  - ref quaternion sequence (4x(N+1))
      W_ref_seq  - ref angular velocity sequence (3x(N+1))
      B_body     - magnetic field in body frame (assumed constant over short horizon)
    """
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(7, N_horizon+1)
    # U = [m_x, m_y, m_z, tau_rw_y]
    U = opti.variable(4, N_horizon)
    Delta = opti.variable(1, N_horizon)

    # Parameters
    X0 = opti.parameter(7)
    Q_ref_seq = opti.parameter(4, N_horizon+1)
    W_ref_seq = opti.parameter(3, N_horizon+1)
    B_body = opti.parameter(3)

    # Initial condition
    opti.subject_to(X[:, 0] == X0)

    # Build dynamics and optional CLF constraints over horizon
    for k in range(N_horizon):
        xk = X[:, k]
        qk = xk[0:4]
        wk = xk[4:7]
        m_k = U[0:3, k]
        tau_rw_k = U[3, k]

        # Discrete dynamics (Euler) with renormalization
        wx, wy, wz = wk[0], wk[1], wk[2]
        Omega = ca.vertcat(
            ca.horzcat(0, -wx, -wy, -wz),
            ca.horzcat(wx,  0,  wz, -wy),
            ca.horzcat(wy, -wz,  0,  wx),
            ca.horzcat(wz,  wy, -wx,  0)
        )
        q_dot = 0.5 * Omega @ qk
        # Hybrid actuation: magnetorquer torque + reaction wheel torque about Y
        tau_mag = ca.cross(m_k, B_body)
        tau_rw_vec = ca.vertcat(0, tau_rw_k, 0)
        tau_total = tau_mag + tau_rw_vec
        w_dot = ca.solve(I_cas, (tau_total - ca.cross(wk, I_cas @ wk)))
        q_next = qk + dt * q_dot
        # normalize quaternion
        q_next = q_next / ca.norm_2(q_next)
        w_next = wk + dt * w_dot
        x_next = ca.vertcat(q_next, w_next)

        opti.subject_to(X[:, k+1] == x_next)

        if HP.enable_clf and alpha > 0.0:
            # CLF at step k (discrete-time decrease)
            q_ref_k = Q_ref_seq[:, k]
            w_ref_k = W_ref_seq[:, k]

            # Error quaternion: q_tilde = q_ref^{-1} ⊗ q
            q_ref_conj = quat_conj(q_ref_k)
            q_tilde = quat_mul(q_ref_conj, qk)
            q0_t = q_tilde[0]
            qv_t = q_tilde[1:4]

            # Angular velocity error
            w_err = wk - w_ref_k

            # CLF V(x_k)
            V_k = 0.5 * ca.dot(w_err, I_cas @ w_err) + 0.5 * kappa * ca.dot(qv_t, qv_t)
            # V(x_{k+1}) with reference at k+1
            q_ref_kp1 = Q_ref_seq[:, k+1]
            w_ref_kp1 = W_ref_seq[:, k+1]
            q_ref_kp1_conj = quat_conj(q_ref_kp1)
            q_tilde_next = quat_mul(q_ref_kp1_conj, q_next)
            qv_t_next = q_tilde_next[1:4]
            w_err_next = w_next - w_ref_kp1
            V_next = 0.5 * ca.dot(w_err_next, I_cas @ w_err_next) + 0.5 * kappa * ca.dot(qv_t_next, qv_t_next)

            # Discrete-time CLF inequality:
            # V_{k+1} - V_k <= -alpha*dt * V_k + Delta_k
            opti.subject_to(V_next - V_k <= -alpha * dt * V_k + Delta[0, k])

            # Slack non-negative
            opti.subject_to(Delta[0, k] >= 0)

        # Input saturation: magnetorquer dipoles and reaction wheel torque
        opti.subject_to(opti.bounded(-m_max, U[0, k], m_max))   # m_x
        opti.subject_to(opti.bounded(0.0,   U[1, k], 0.0))      # m_y = 0 (no Y magnetorquer)
        opti.subject_to(opti.bounded(-m_max, U[2, k], m_max))   # m_z
        opti.subject_to(opti.bounded(-tau_rw_max, U[3, k], tau_rw_max))  # tau_rw_y

    # Objective: minimize sum of control and tracking penalties
    J = 0
    for k in range(N_horizon):
        m_k = U[0:3, k]
        tau_rw_k = U[3, k]
        xk = X[:, k]
        qk = xk[0:4]
        wk = xk[4:7]
        q_ref_k = Q_ref_seq[:, k]
        w_ref_k = W_ref_seq[:, k]
        q_tilde_k = quat_mul(quat_conj(q_ref_k), qk)
        qv_t_k = q_tilde_k[1:4]
        w_err_k = wk - w_ref_k
        # state tracking cost
        J += ca.dot(w_err_k, Q_w @ w_err_k) + ca.dot(qv_t_k, Q_q @ qv_t_k)
        # control cost: magnetorquer dipole and reaction wheel torque
        J += ca.mtimes([m_k.T, R_c, m_k]) + R_rw * (tau_rw_k ** 2)
        if HP.enable_clf and W_s > 0.0:
            J += W_s * (Delta[0, k] ** 2)
    # Optionally: add terminal CLF cost (only if enabled)
    if HP.enable_clf and terminal_weight > 0.0:
        xN = X[:, -1]
        qN = xN[0:4]
        wN = xN[4:7]
        q_ref_N = Q_ref_seq[:, -1]
        w_ref_N = W_ref_seq[:, -1]
        q_tilde_N = quat_mul(quat_conj(q_ref_N), qN)
        qv_t_N = q_tilde_N[1:4]
        w_err_N = wN - w_ref_N
        V_N = 0.5 * ca.dot(w_err_N, I_cas @ w_err_N) + 0.5 * kappa * ca.dot(qv_t_N, qv_t_N)
        J += terminal_weight * V_N  # terminal weight

    opti.minimize(J)

    # Solver options
    opts = {
        "print_time": False,
        "ipopt": {
            "print_level": 0,
            "tol": HP.solver_tol,
            "max_iter": HP.solver_max_iter
        }
    }
    opti.solver("ipopt", opts)

    # Initial guesses (helps IPOPT)
    opti.set_initial(X, np.tile(np.array([1, 0, 0, 0, 0, 0, 0]).reshape(7, 1), (1, N_horizon+1)))
    opti.set_initial(U, 0)
    opti.set_initial(Delta, 0)

    # Pack everything needed to use this MPC
    mpc = {
        "opti": opti,
        "X": X,
        "U": U,
        "Delta": Delta,
        "X0": X0,
        "Q_ref_seq": Q_ref_seq,
        "W_ref_seq": W_ref_seq,
        "B_body": B_body
    }
    return mpc


# =========================
#  Simulation
# =========================

def run_simulation():
    # Build CLF–QP MPC
    mpc = build_clf_qp_mpc()
    opti = mpc["opti"]
    X_var = mpc["X"]
    U_var = mpc["U"]  # actuator sequence [m_x,m_y,m_z,tau_rw_y]
    Delta_var = mpc["Delta"]
    X0_param = mpc["X0"]
    Qref_param = mpc["Q_ref_seq"]
    Wref_param = mpc["W_ref_seq"]
    B_body_param = mpc["B_body"]

    # Results directory (needed for MPC snapshot and final outputs)
    RESULTS_DIR = HP.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Simulation parameters
    T_total = HP.total_time  # [s]
    steps = int(T_total / dt)

    # Initial state: 10 deg offset about body Y from initial reference
    theta0 = HP.theta0
    q_ref0, w_ref0, r_I0 = reference_attitude(theta0)
    angle0 = np.deg2rad(HP.initial_attitude_offset_deg)
    axis_y = np.array([0.0, 1.0, 0.0])
    axis_y = axis_y / np.linalg.norm(axis_y)
    dq0 = np.array([
        np.cos(angle0/2.0),
        axis_y[0]*np.sin(angle0/2.0),
        axis_y[1]*np.sin(angle0/2.0),
        axis_y[2]*np.sin(angle0/2.0),
    ])
    q0 = quat_mul(q_ref0, dq0)
    q0 = np.array(q0).astype(float).flatten()
    q0 = q0 / np.linalg.norm(q0)
    w0 = np.array([0.0, 0.0, 0.0])
    x0 = np.hstack((q0, w0))

    # Logs
    log_time = []
    log_q = []
    log_q_ref = []
    log_w = []
    log_w_ref = []
    log_m = []
    log_tau_rw = []
    log_V = []
    log_delta = []
    log_angle_err = []
    log_rate_err = []

    # For controllability/observability commentary: log B directions
    log_Bb = []

    solver_success = 0
    solver_fail = 0
    solver_iterations = []
    solver_statuses = []

    # Pre-compute MPC snapshot steps (indices) for visualization
    snapshot_indices = []
    if steps > 0:
        if HP.mpc_snapshot_count <= 0:
            # Snapshot at every MPC step (full video)
            snapshot_indices = list(range(steps))
        else:
            import numpy as _np
            # Spread snapshots roughly between 20% and 80% of the simulation
            start_idx = int(0.2 * steps)
            end_idx = int(0.8 * steps)
            snapshot_indices = _np.linspace(
                start_idx,
                max(start_idx, end_idx),
                num=HP.mpc_snapshot_count,
                dtype=int,
            )
            snapshot_indices = sorted(set(int(i) for i in snapshot_indices))
    mpc_snapshots = []

    theta = theta0

    for k in range(steps):
        t = k * dt

        # Reference generation over horizon
        theta_seq = theta + omega_orbit * dt * np.arange(N_horizon+1)
        q_ref_seq_val = np.zeros((4, N_horizon+1))
        w_ref_seq_val = np.zeros((3, N_horizon+1))
        r_I_last = None
        for j in range(N_horizon+1):
            # Always use orbital position for magnetic field
            q_orbit_j, w_orbit_j, r_I_j = reference_attitude(theta_seq[j])
            r_I_last = r_I_j
            if HP.use_sinusoidal_reference:
                t_ref = t + j * dt
                q_ref_j, w_ref_j = sinusoid_reference(t_ref)
            else:
                q_ref_j, w_ref_j = q_orbit_j, w_orbit_j
            q_ref_seq_val[:, j] = q_ref_j
            w_ref_seq_val[:, j] = w_ref_j

        # Magnetic field at current state (in inertial → body)
        B_I = earth_B_inertial(r_I_last)
        R_IB = quat_to_dcm(x0[0:4])
        C_BI = R_IB.T
        B_body = C_BI @ B_I

        # Set parameters
        opti.set_value(X0_param, x0)
        opti.set_value(Qref_param, q_ref_seq_val)
        opti.set_value(Wref_param, w_ref_seq_val)
        opti.set_value(B_body_param, B_body)

        # Warm-start using previous solution if available
        try:
            sol = opti.solve()
        except RuntimeError as e:
            solver_fail += 1
            solver_statuses.append(f"FAIL: {e}")
            print(f"[WARN] IPOPT failed at step {k}, using previous control or zero. Error: {e}")
            if k == 0:
                m_opt = np.zeros(3)
                tau_rw_opt = 0.0
            else:
                m_opt = log_m[-1]
                tau_rw_opt = log_tau_rw[-1]
        else:
            solver_success += 1
            stats = opti.stats()
            solver_statuses.append(stats.get("return_status", "Solve_Succeeded"))
            iter_field = stats.get("iter_count", stats.get("iterations"))
            if isinstance(iter_field, dict):
                iter_value = iter_field.get("iter_count")
            else:
                iter_value = iter_field
            if iter_value is not None:
                try:
                    solver_iterations.append(float(iter_value))
                except (TypeError, ValueError):
                    pass
            # First control in horizon: physical actuators
            u0 = sol.value(U_var[:, 0]).flatten()
            m_opt = u0[0:3]
            tau_rw_opt = float(u0[3])

        # Integrate actual dynamics one step with disturbance and noise
        q_curr = x0[0:4]
        w_curr = x0[4:7]
        w_err_curr = w_curr - w_ref_seq_val[:, 0]

        # Compute torque from actuators: magnetorquer (x,z) + reaction wheel (y)
        tau_mag = np.cross(m_opt, B_body)
        tau_rw_vec = np.array([0.0, tau_rw_opt, 0.0])
        tau_dist = np.random.normal(0.0, HP.disturbance_std, size=3)  # small Gaussian disturbance
        tau_total = tau_mag + tau_rw_vec + tau_dist

        # Angular acceleration
        w_dot_val = I_inv_num @ (tau_total - np.cross(w_curr, I_num @ w_curr))

        # Quaternion derivative
        wx, wy, wz = w_curr
        Omega_num = np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0,  wx],
            [wz,  wy, -wx,  0.0],
        ])
        q_dot_val = 0.5 * Omega_num @ q_curr

        # Integrate
        q_next = q_curr + dt * q_dot_val
        q_next = q_next / np.linalg.norm(q_next)
        w_next = w_curr + dt * w_dot_val
        x0 = np.hstack((q_next, w_next))

        # Log values
        V_k = 0.5 * w_curr @ I_num @ w_curr
        # approximate orientation error for logging
        q_err0 = np.dot(q_ref_seq_val[:, 0], q_curr)
        q_err0 = np.clip(q_err0, -1.0, 1.0)
        angle_err = 2.0 * np.arccos(np.abs(q_err0))
        rate_err_norm = np.linalg.norm(w_err_curr)
        V_k += 0.5 * kappa * (1 - q_err0)  # approximate scalar-based term

        log_time.append(t)
        log_q.append(q_curr.copy())
        log_q_ref.append(q_ref_seq_val[:, 0].copy())
        log_w.append(w_curr.copy())
        log_w_ref.append(w_ref_seq_val[:, 0].copy())
        log_m.append(m_opt.copy())
        log_tau_rw.append(tau_rw_opt)
        log_V.append(V_k)
        log_Bb.append(B_body.copy())
        log_angle_err.append(np.rad2deg(angle_err))
        log_rate_err.append(rate_err_norm)
        if HP.enable_clf and 'sol' in locals():
            try:
                delta0 = float(sol.value(Delta_var[0, 0]))
            except Exception:
                delta0 = 0.0
        else:
            delta0 = 0.0
        log_delta.append(delta0)

        # Advance orbital angle
        theta += omega_orbit * dt

        # MPC snapshot logging: store history + predicted horizon at selected steps
        if k in snapshot_indices and 'sol' in locals():
            hist_len = int(HP.mpc_snapshot_window / dt)
            hist_start = max(0, len(log_time) - hist_len)
            snapshot = {
                "step": k,
                "t": t,
                "time_hist": np.array(log_time[hist_start:], copy=True),
                "q_hist": np.array(log_q[hist_start:], copy=True),
                "q_ref_hist": np.array(log_q_ref[hist_start:], copy=True),
                "w_hist": np.array(log_w[hist_start:], copy=True),
                "w_ref_hist": np.array(log_w_ref[hist_start:], copy=True),
                "m_hist": np.array(log_m[hist_start:], copy=True),
                "tau_rw_hist": np.array(log_tau_rw[hist_start:], copy=True),
                "time_pred": t + dt * np.arange(N_horizon + 1),
                "X_pred": np.array(sol.value(X_var), dtype=float),
                "Q_ref_pred": np.array(q_ref_seq_val, dtype=float),
                "W_ref_pred": np.array(w_ref_seq_val, dtype=float),
            }
            mpc_snapshots.append(snapshot)

        # Update initial guess for next solve (warm start)
        if 'sol' in locals():
            opti.set_initial(X_var, sol.value(X_var))
            opti.set_initial(U_var, sol.value(U_var))
            if HP.enable_clf:
                opti.set_initial(Delta_var, sol.value(Delta_var))

    # Convert logs to numpy arrays
    log_time = np.array(log_time)
    log_q = np.array(log_q)
    log_q_ref = np.array(log_q_ref)
    log_w = np.array(log_w)
    log_w_ref = np.array(log_w_ref)
    log_m = np.array(log_m)
    log_V = np.array(log_V)
    log_delta = np.array(log_delta)
    log_Bb = np.array(log_Bb)
    log_tau_rw = np.array(log_tau_rw)
    log_angle_err = np.array(log_angle_err)
    log_rate_err = np.array(log_rate_err)
    dipole_norm = np.linalg.norm(log_m, axis=1)
    # Torque contributions from magnetorquers and reaction wheel
    tau_mag_hist = np.cross(log_m, log_Bb)
    tau_mag_norm = np.linalg.norm(tau_mag_hist, axis=1)
    tau_rw_norm = np.abs(log_tau_rw)
    # Time-integrated torque magnitudes (approximate impulse)
    total_tau_mag = float(np.sum(tau_mag_norm) * dt)
    total_tau_rw = float(np.sum(tau_rw_norm) * dt)
    rms_tau_mag = float(np.sqrt(np.mean(tau_mag_norm ** 2))) if tau_mag_norm.size else 0.0
    rms_tau_rw = float(np.sqrt(np.mean(tau_rw_norm ** 2))) if tau_rw_norm.size else 0.0
    max_tau_mag = float(np.max(tau_mag_norm)) if tau_mag_norm.size else 0.0
    max_tau_rw = float(np.max(tau_rw_norm)) if tau_rw_norm.size else 0.0
    avg_angle = float(np.mean(log_angle_err)) if log_angle_err.size else 0.0
    peak_angle = float(np.max(log_angle_err)) if log_angle_err.size else 0.0
    rms_rate = float(np.sqrt(np.mean(log_rate_err ** 2))) if log_rate_err.size else 0.0
    max_dipole = float(np.max(dipole_norm)) if dipole_norm.size else 0.0
    max_slack = float(np.max(log_delta)) if log_delta.size else 0.0
    success_total = max(solver_success + solver_fail, 1)
    success_pct = 100.0 * solver_success / success_total
    avg_ipopt_iter = float(np.mean(solver_iterations)) if solver_iterations else float('nan')

    # =========================
    #  Plots
    # =========================

    fig0, (ax0a, ax0b) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax0a.plot(log_time, log_angle_err, label='Pointing error')
    ax0a.set_ylabel('Angle error [deg]')
    ax0a.set_title('Pointing & Rate Tracking Errors')
    ax0a.grid(True)
    ax0a.legend(loc='upper right')
    ax0b.plot(log_time, log_rate_err, color='tab:orange', label='||w - w^d||')
    ax0b.set_ylabel('Angular-rate error [rad/s]')
    ax0b.set_xlabel('Time [s]')
    ax0b.grid(True)
    ax0b.legend(loc='upper right')
    fig0.tight_layout()

    # Quaternion tracking
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    q_labels = [r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$']
    q_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i in range(4):
        ax1.plot(
            log_time,
            log_q[:, i],
            color=q_colors[i],
            linewidth=1.5,
            label=f'{q_labels[i]} actual',
        )
        ax1.plot(
            log_time,
            log_q_ref[:, i],
            linestyle='--',
            linewidth=2.0,
            color=q_colors[i],
            label=f'{q_labels[i]} ref',
        )
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Quaternion components')
    ax1.set_title('Attitude Tracking (Quaternion)')
    ax1.grid(True)
    ax1.legend(loc='best')
    fig1.tight_layout()
    # fig1.savefig('quat_tracking.png', dpi=200)

    # Angular velocity tracking
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    w_labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    w_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(3):
        ax2.plot(
            log_time,
            log_w[:, i],
            color=w_colors[i],
            linewidth=1.5,
            label=f'{w_labels[i]} actual',
        )
        ax2.plot(
            log_time,
            log_w_ref[:, i],
            linestyle='--',
            linewidth=2.0,
            color=w_colors[i],
            label=f'{w_labels[i]}^d',
        )
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular velocity [rad/s]')
    ax2.set_title('Body Angular Velocity vs Reference')
    ax2.grid(True)
    ax2.legend(loc='best')
    fig2.tight_layout()
    # fig2.savefig('omega_tracking.png', dpi=200)

    # CLF and slack
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(log_time, log_V, label='V(x) (CLF)')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('V(x)')
    ax32 = ax3.twinx()
    ax32.plot(log_time, log_delta, 'r', label='Slack δ')
    ax32.set_ylabel('δ')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax32.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.set_title('CLF and Slack over Time')
    ax3.grid(True)
    fig3.tight_layout()
    # fig3.savefig('clf_and_slack.png', dpi=200)

    # Control dipole
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(log_time, log_m[:, 0], label='$m_x$')
    ax4.plot(log_time, log_m[:, 1], label='$m_y$')
    ax4.plot(log_time, log_m[:, 2], label='$m_z$')
    ax4.axhline(m_max, color='k', linestyle='--', linewidth=0.8)
    ax4.axhline(-m_max, color='k', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Dipole [A·m²]')
    ax4.set_title('Magnetorquer Dipole Commands')
    ax4.grid(True)
    ax4.legend(loc='best')
    fig4.tight_layout()
    # fig4.savefig('dipole_commands.png', dpi=200)

    # =========================
    #  MPC snapshot animation (history + prediction)
    # =========================
    if mpc_snapshots:
        fig_s, (ax_s1, ax_s2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

        def update_mpc_frame(idx):
            snap = mpc_snapshots[idx]
            step = snap["step"]
            t0 = snap["t"]
            time_hist = snap["time_hist"]
            q_hist = snap["q_hist"]
            q_ref_hist = snap["q_ref_hist"]
            time_pred = snap["time_pred"]
            X_pred = snap["X_pred"]
            Q_ref_pred = snap["Q_ref_pred"]

            ax_s1.clear()
            ax_s2.clear()

            # Attitude (q0) history and predicted horizon
            ax_s1.plot(time_hist, q_hist[:, 0], color='tab:blue', label=r'$q_0$ actual (history)')
            ax_s1.plot(time_hist, q_ref_hist[:, 0], '--', color='tab:blue', alpha=0.7, label=r'$q_0$ ref (history)')
            ax_s1.plot(time_pred, X_pred[0, :], '-.', color='tab:orange', label=r'$q_0$ predicted')
            ax_s1.plot(time_pred, Q_ref_pred[0, :], ':', color='tab:green', label=r'$q_0$ ref (horizon)')
            ax_s1.axvline(t0, color='k', linestyle='--', linewidth=0.8)
            ax_s1.set_ylabel('q0')
            ax_s1.set_title(f'MPC Snapshot at t = {t0:.1f} s (step {step})')
            ax_s1.grid(True)
            ax_s1.legend(loc='best', fontsize=8)

            # Magnetorquer dipole magnitude history
            m_hist = snap["m_hist"]
            m_norm_hist = np.linalg.norm(m_hist, axis=1)
            ax_s2.plot(time_hist, m_norm_hist, color='tab:red', label=r'$||m||$ history')
            ax_s2.axvline(t0, color='k', linestyle='--', linewidth=0.8)
            ax_s2.set_xlabel('Time [s]')
            ax_s2.set_ylabel('Dipole norm [A·m²]')
            ax_s2.grid(True)
            ax_s2.legend(loc='best', fontsize=8)

            fig_s.tight_layout()
            return ax_s1, ax_s2

        mpc_mp4 = os.path.join(RESULTS_DIR, "mpc_snapshot_animation.mp4")
        mpc_gif = os.path.join(RESULTS_DIR, "mpc_snapshot_animation.gif")

        try:
            from matplotlib.animation import FFMpegWriter, PillowWriter
            ani_mpc = animation.FuncAnimation(
                fig_s,
                update_mpc_frame,
                frames=len(mpc_snapshots),
                interval=500,
                blit=False,
            )
            writer = FFMpegWriter(
                fps=2,
                metadata=dict(artist="AE642 CLF-QP"),
                bitrate=1800,
                codec="libx264",
            )
            ani_mpc.save(mpc_mp4, writer=writer)
            print(f"Saved MPC snapshot animation to {mpc_mp4}")
        except Exception as e:
            print(f"[WARN] Failed to write MPC MP4 via ffmpeg ({e}), falling back to GIF.")
            try:
                ani_mpc = animation.FuncAnimation(
                    fig_s,
                    update_mpc_frame,
                    frames=len(mpc_snapshots),
                    interval=500,
                    blit=False,
                )
                from matplotlib.animation import PillowWriter
                writer_gif = PillowWriter(fps=2)
                ani_mpc.save(mpc_gif, writer=writer_gif)
                print(f"Saved MPC snapshot animation GIF to {mpc_gif}")
            except Exception as e2:
                print(f"[WARN] Failed to write MPC GIF as well ({e2}). Skipping MPC animation.")

    # =========================
    #  Attitude animation
    # =========================
    fig5 = plt.figure(figsize=(5, 5))
    ax5 = fig5.add_subplot(111, projection='3d')
    ax5.set_xlim([-1.2, 1.2])
    ax5.set_ylim([-1.2, 1.2])
    ax5.set_zlim([-1.2, 1.2])
    ax5.set_xlabel('Inertial X')
    ax5.set_ylabel('Inertial Y')
    ax5.set_zlabel('Inertial Z')
    ax5.set_title('Attitude Tracking (Body Axes)')

    # Nadir direction (for initial position)
    ax5.scatter(0, 0, -1, color='r', s=40, label='Nadir dir.')
    line_x, = ax5.plot([], [], [], 'b', lw=2, label='Body X')
    line_y, = ax5.plot([], [], [], 'g', lw=2, label='Body Y')
    line_z, = ax5.plot([], [], [], 'm', lw=2, label='Body Z')
    ax5.legend(loc='best')



    # =========================
    #  Controllability & Observability comments
    # =========================
    # Approximate controllability: check how B_body direction spans over time
    B_norms = np.linalg.norm(log_Bb, axis=1)
    B_dirs = log_Bb / (B_norms.reshape(-1, 1) + 1e-9)
    # Compute covariance of B directions; if full rank ~3, field spans 3D space
    cov_B = B_dirs.T @ B_dirs / len(B_dirs)
    eigvals_B, _ = np.linalg.eig(cov_B)
    print("Approx. B-body direction covariance eigenvalues (controllability indicator):", eigvals_B)

    # Observability: with gyro + magnetometer, time-varying B gives attitude observability over orbit
    print("Gyro + magnetometer with time-varying B provide nonlinear attitude observability over the orbit.")
    print("Check eigenvalues above: if they are nonzero and diverse, B explores multiple directions,")
    print("which supports time-varying controllability and observability assumptions for magnetorquer-only control.")

    def update_frame(idx):
        qk = log_q[idx]
        R_bi = quat_to_dcm(qk)  # body->inertial
        origin = np.array([0.0, 0.0, 0.0])
        X_axis = R_bi[:, 0]
        Y_axis = R_bi[:, 1]
        Z_axis = R_bi[:, 2]
        line_x.set_data([origin[0], X_axis[0]], [origin[1], X_axis[1]])
        line_x.set_3d_properties([origin[2], X_axis[2]])
        line_y.set_data([origin[0], Y_axis[0]], [origin[1], Y_axis[1]])
        line_y.set_3d_properties([origin[2], Y_axis[2]])
        line_z.set_data([origin[0], Z_axis[0]], [origin[1], Z_axis[1]])
        line_z.set_3d_properties([origin[2], Z_axis[2]])
        return line_x, line_y, line_z

    # IMPORTANT: blit=False for 3D animations
    ani = animation.FuncAnimation(
        fig5,
        update_frame,
        frames=len(log_time),
        interval=HP.animation_interval_ms,
        blit=False
    )

    # =========================
    #  SAVE ALL OUTPUTS TO /results
    # =========================
    RESULTS_DIR = HP.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save raw logs
    np.save(os.path.join(RESULTS_DIR, "time.npy"), log_time)
    np.save(os.path.join(RESULTS_DIR, "q.npy"), log_q)
    np.save(os.path.join(RESULTS_DIR, "q_ref.npy"), log_q_ref)
    np.save(os.path.join(RESULTS_DIR, "w.npy"), log_w)
    np.save(os.path.join(RESULTS_DIR, "w_ref.npy"), log_w_ref)
    np.save(os.path.join(RESULTS_DIR, "m.npy"), log_m)
    np.save(os.path.join(RESULTS_DIR, "V.npy"), log_V)
    np.save(os.path.join(RESULTS_DIR, "delta.npy"), log_delta)
    np.save(os.path.join(RESULTS_DIR, "B_body.npy"), log_Bb)
    np.save(os.path.join(RESULTS_DIR, "tau_rw.npy"), log_tau_rw)

    # Save figures (PNG)
    fig0.savefig(os.path.join(RESULTS_DIR, "tracking_errors.png"), dpi=200)
    fig1.savefig(os.path.join(RESULTS_DIR, "quat_tracking.png"), dpi=200)
    fig2.savefig(os.path.join(RESULTS_DIR, "omega_tracking.png"), dpi=200)
    fig3.savefig(os.path.join(RESULTS_DIR, "clf_and_slack.png"), dpi=200)
    fig4.savefig(os.path.join(RESULTS_DIR, "dipole_commands.png"), dpi=200)

    # Save animation: try MP4 via ffmpeg, fall back to GIF
    from matplotlib.animation import FFMpegWriter, PillowWriter

    mp4_path = os.path.join(RESULTS_DIR, "attitude_tracking.mp4")
    gif_path = os.path.join(RESULTS_DIR, "attitude_tracking.gif")

    # try:
    #     writer = FFMpegWriter(
    #         fps=20,
    #         metadata=dict(artist="AE642 CLF-QP"),
    #         bitrate=1800,
    #         codec="libx264"
    #     )
    #     ani.save(mp4_path, writer=writer)
    #     print(f"Saved MP4 animation to {mp4_path}")
    # except Exception as e:
    #     print(f"[WARN] Failed to write MP4 via ffmpeg ({e}), falling back to GIF.")
    #     writer_gif = PillowWriter(fps=15)
    #     ani.save(gif_path, writer=writer_gif)
    #     print(f"Saved GIF animation to {gif_path}")

    # Text summary
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("CLF–QP Simulation Results\n")
        f.write("=========================\n\n")
        f.write(f"Total time simulated: {T_total:.2f} s\n")
        f.write(f"Control frequency: {1/dt:.2f} Hz\n")
        f.write(f"Horizon length: {N_horizon} steps ({HORIZON_TIME:.2f} s)\n")
        f.write(f"Magnetorquer max dipole: {m_max} A·m²\n\n")
        f.write(f"Average pointing error: {avg_angle:.3f} deg (peak {peak_angle:.3f} deg)\n")
        f.write(f"RMS angular-rate error: {rms_rate:.5f} rad/s\n")
        f.write(f"Max dipole magnitude: {max_dipole:.2f} A·m²\n")
        f.write(f"Max slack δ: {max_slack:.3e}\n")
        f.write("\nTorque usage:\n")
        f.write(f"  Magnetorquer: RMS |τ_mag| = {rms_tau_mag:.4e} N·m, max = {max_tau_mag:.4e} N·m,\n")
        f.write(f"                ∑|τ_mag| dt ≈ {total_tau_mag:.4e} N·m·s\n")
        f.write(f"  Reaction wheel: RMS |τ_rw| = {rms_tau_rw:.4e} N·m, max = {max_tau_rw:.4e} N·m,\n")
        f.write(f"                  ∑|τ_rw| dt ≈ {total_tau_rw:.4e} N·m·s\n\n")
        f.write(f"IPOPT success: {solver_success}/{steps} ({success_pct:.1f}%) with {solver_fail} failures\n")
        f.write(f"Average IPOPT iterations: {avg_ipopt_iter:.1f}\n\n")
        f.write("Controllability Indicator (Cov(B_body) eigenvalues):\n")
        f.write(str(eigvals_B) + "\n\n")
        f.write("Observability:\n")
        f.write("Gyro + magnetometer ensure nonlinear attitude observability over orbit.\n")
        f.write("Time-varying magnetic field excites rotational degrees of freedom.\n")

    print(f"\nHorizon: {N_horizon} steps ({HORIZON_TIME:.2f} s) at dt = {dt:.3f} s")
    print(f"IPOPT success {solver_success}/{steps} ({success_pct:.1f}%), average iterations {avg_ipopt_iter:.1f}, failures {solver_fail}")
    print(f"Pointing error avg {avg_angle:.2f} deg (peak {peak_angle:.2f} deg); RMS rate error {rms_rate:.4f} rad/s")
    print(f"Max |m| {max_dipole:.2f} A·m²; Max slack δ {max_slack:.3e}")
    print(f"Magnetorquer torque: RMS {rms_tau_mag:.4e} N·m, max {max_tau_mag:.4e} N·m, integral ∑|τ_mag|dt ≈ {total_tau_mag:.4e} N·m·s")
    print(f"Reaction wheel torque: RMS {rms_tau_rw:.4e} N·m, max {max_tau_rw:.4e} N·m, integral ∑|τ_rw|dt ≈ {total_tau_rw:.4e} N·m·s")

    print(f"\nAll logs, plots, and animation saved under ./{RESULTS_DIR}/")

 

if __name__ == "__main__":
    run_simulation()
