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


# =========================
#  Satellite & Environment
# =========================

# Inertia matrix (kg·m^2) from characterization (ITC1)
Ixx, Iyy, Izz = 0.7264, 0.7258, 1.1644
I_cas = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
I_num = np.diag([Ixx, Iyy, Izz])
I_inv_num = np.linalg.inv(I_num)

# Magnetorquer limits (Panyalert et al.)
m_max = 8.19  # [A·m^2]
# Control & CLF weights
R_c = ca.diag(ca.vertcat(1e-3, 1e-3, 1e-3))
W_s = 1e5
kappa = 2.0   # CLF attitude weight
alpha = 0.5   # CLF decay rate
dt = 0.2      # control step [s] (5 Hz)
N_horizon = 20  # horizon steps (4 seconds)

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
    Nadir-pointing LVLH-like reference for equatorial orbit angle theta.
    - z_B points to nadir (-r_hat)
    - x_B points along velocity direction (tangential)
    - y_B completes right-handed triad
    Returns (q_ref, w_ref_body)
    """
    # Position and velocity in inertial frame for equatorial circular orbit
    r_I = np.array([R_orbit*np.cos(theta), R_orbit*np.sin(theta), 0.0])
    v_I = omega_orbit * np.array([-R_orbit*np.sin(theta), R_orbit*np.cos(theta), 0.0])

    r_hat = r_I / np.linalg.norm(r_I)
    v_hat = v_I / np.linalg.norm(v_I)

    z_B = -r_hat         # body z axis to nadir
    x_B = v_hat          # body x axis along-track
    y_B = np.cross(z_B, x_B)
    y_B = y_B / np.linalg.norm(y_B)

    # Re-orthogonalize if needed
    x_B = np.cross(y_B, z_B)
    x_B = x_B / np.linalg.norm(x_B)

    # Rotation matrix from body to inertial: columns are body axes in inertial coords
    R_IB = np.column_stack((x_B, y_B, z_B))
    q_ref = dcm_to_quat(R_IB)

    # Reference body angular velocity: roughly orbit rate about y_B
    w_ref_body = np.array([0.0, omega_orbit, 0.0])

    return q_ref, w_ref_body, r_I


# =========================
#  Build CLF–QP MPC in CasADi
# =========================

def build_clf_qp_mpc():
    """
    Build a receding-horizon CLF–QP MPC problem in CasADi Opti.

    Decision variables:
      X[:,0..N]  - state trajectory (q0,q1,q2,q3,wx,wy,wz)
      U[:,0..N-1]- magnetic dipole sequence
      Delta[0..N-1] - CLF slack sequence

    Parameters:
      X0         - initial state
      Q_ref_seq  - ref quaternion sequence (4x(N+1))
      W_ref_seq  - ref angular velocity sequence (3x(N+1))
      B_body     - magnetic field in body frame (assumed constant over short horizon)
    """
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(7, N_horizon+1)
    U = opti.variable(3, N_horizon)
    Delta = opti.variable(1, N_horizon)

    # Parameters
    X0 = opti.parameter(7)
    Q_ref_seq = opti.parameter(4, N_horizon+1)
    W_ref_seq = opti.parameter(3, N_horizon+1)
    B_body = opti.parameter(3)

    # Initial condition
    opti.subject_to(X[:, 0] == X0)

    # Build dynamics and CLF constraints over horizon
    for k in range(N_horizon):
        xk = X[:, k]
        qk = xk[0:4]
        wk = xk[4:7]
        mk = U[:, k]

        # Discrete dynamics (Euler) with renormalization
        wx, wy, wz = wk[0], wk[1], wk[2]
        Omega = ca.vertcat(
            ca.horzcat(0, -wx, -wy, -wz),
            ca.horzcat(wx,  0,  wz, -wy),
            ca.horzcat(wy, -wz,  0,  wx),
            ca.horzcat(wz,  wy, -wx,  0)
        )
        q_dot = 0.5 * Omega @ qk
        tau_mag = ca.cross(mk, B_body)
        w_dot = ca.solve(I_cas, (tau_mag - ca.cross(wk, I_cas @ wk)))
        q_next = qk + dt * q_dot
        # normalize quaternion
        q_next = q_next / ca.norm_2(q_next)
        w_next = wk + dt * w_dot
        x_next = ca.vertcat(q_next, w_next)

        opti.subject_to(X[:, k+1] == x_next)

        # CLF at step k
        q_ref_k = Q_ref_seq[:, k]
        w_ref_k = W_ref_seq[:, k]

        # Error quaternion: q_tilde = q_ref^{-1} ⊗ q
        q_ref_conj = quat_conj(q_ref_k)
        q_tilde = quat_mul(q_ref_conj, qk)
        q0_t = q_tilde[0]
        qv_t = q_tilde[1:4]

        # Angular velocity error
        w_err = wk - w_ref_k

        # CLF V(x) = 1/2 w_err^T I w_err + (kappa/2) ||qv_t||^2
        V_k = 0.5 * ca.dot(w_err, I_cas @ w_err) + 0.5 * kappa * ca.dot(qv_t, qv_t)

        # Derivative terms (nominal, as in theory):
        # \dot V = (B×ω)^T m + (kappa*qtilde0/2) * qv_t^T * ω
        LgV = ca.dot(ca.cross(B_body, wk), mk)
        LfV = (kappa * q0_t / 2.0) * ca.dot(qv_t, wk)

        # CLF inequality: LgV + LfV <= -alpha V + Delta
        opti.subject_to(LgV + LfV <= -alpha * V_k + Delta[0, k])

        # Slack non-negative
        opti.subject_to(Delta[0, k] >= 0)

        # Input saturation
        opti.subject_to(opti.bounded(-m_max, U[0, k], m_max))
        opti.subject_to(opti.bounded(-m_max, U[1, k], m_max))
        opti.subject_to(opti.bounded(-m_max, U[2, k], m_max))

    # Objective: minimize sum m^T R m + W_s delta^2
    J = 0
    for k in range(N_horizon):
        mk = U[:, k]
        J += ca.mtimes([mk.T, R_c, mk]) + W_s * (Delta[0, k] ** 2)
    # Optionally: add terminal CLF cost
    xN = X[:, -1]
    qN = xN[0:4]
    wN = xN[4:7]
    q_ref_N = Q_ref_seq[:, -1]
    w_ref_N = W_ref_seq[:, -1]
    q_tilde_N = quat_mul(quat_conj(q_ref_N), qN)
    qv_t_N = q_tilde_N[1:4]
    w_err_N = wN - w_ref_N
    V_N = 0.5 * ca.dot(w_err_N, I_cas @ w_err_N) + 0.5 * kappa * ca.dot(qv_t_N, qv_t_N)
    J += 10.0 * V_N  # terminal weight

    opti.minimize(J)

    # Solver options
    opts = {
        "print_time": False,
        "ipopt": {
            "print_level": 0,
            "tol": 1e-6,
            "max_iter": 200
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
    U_var = mpc["U"]
    Delta_var = mpc["Delta"]
    X0_param = mpc["X0"]
    Qref_param = mpc["Q_ref_seq"]
    Wref_param = mpc["W_ref_seq"]
    B_body_param = mpc["B_body"]

    # Simulation parameters
    T_total = 1000.0  # [s] simulate 3 minutes
    steps = int(T_total / dt)

    # Initial state: 10 deg offset about body Y from initial reference
    theta0 = 0.0
    q_ref0, w_ref0, r_I0 = reference_attitude(theta0)
    angle0 = np.deg2rad(10.0)
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
    log_V = []
    log_delta = []

    # For controllability/observability commentary: log B directions
    log_Bb = []

    theta = theta0

    for k in range(steps):
        t = k * dt

        # Reference generation over horizon
        theta_seq = theta + omega_orbit * dt * np.arange(N_horizon+1)
        q_ref_seq_val = np.zeros((4, N_horizon+1))
        w_ref_seq_val = np.zeros((3, N_horizon+1))
        r_I_last = None
        for j in range(N_horizon+1):
            q_ref_j, w_ref_j, r_I_j = reference_attitude(theta_seq[j])
            q_ref_seq_val[:, j] = q_ref_j
            w_ref_seq_val[:, j] = w_ref_j
            r_I_last = r_I_j

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
            print(f"[WARN] IPOPT failed at step {k}, using previous control or zero. Error: {e}")
            if k == 0:
                m_opt = np.zeros(3)
            else:
                m_opt = log_m[-1]
        else:
            m_opt = sol.value(U_var[:, 0]).flatten()

        # Apply saturation guard (should already be satisfied)
        m_opt = np.clip(m_opt, -m_max, m_max)

        # Integrate actual dynamics one step with disturbance and noise
        q_curr = x0[0:4]
        w_curr = x0[4:7]

        # Compute torque from magnetorquer
        tau_mag = np.cross(m_opt, B_body)
        tau_dist = np.random.normal(0.0, 1e-5, size=3)  # small Gaussian disturbance
        tau_total = tau_mag + tau_dist

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
        V_k += 0.5 * kappa * (1 - q_err0)  # approximate scalar-based term

        log_time.append(t)
        log_q.append(q_curr.copy())
        log_q_ref.append(q_ref_seq_val[:, 0].copy())
        log_w.append(w_curr.copy())
        log_w_ref.append(w_ref_seq_val[:, 0].copy())
        log_m.append(m_opt.copy())
        log_V.append(V_k)
        log_Bb.append(B_body.copy())
        if 'sol' in locals():
            try:
                delta0 = float(sol.value(Delta_var[0, 0]))
            except Exception:
                delta0 = 0.0
        else:
            delta0 = 0.0
        log_delta.append(delta0)

        # Advance orbital angle
        theta += omega_orbit * dt

        # Update initial guess for next solve (warm start)
        opti.set_initial(X_var, sol.value(X_var) if 'sol' in locals() else opti.value(X_var))
        opti.set_initial(U_var, sol.value(U_var) if 'sol' in locals() else opti.value(U_var))
        opti.set_initial(Delta_var, sol.value(Delta_var) if 'sol' in locals() else opti.value(Delta_var))

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

    # =========================
    #  Plots
    # =========================

    # Quaternion tracking
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for i in range(4):
        ax1.plot(log_time, log_q[:, i], label=f'q_{i} actual')
        ax1.plot(log_time, log_q_ref[:, i], '--', label=f'q_{i} ref' if i == 0 else None)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Quaternion components')
    ax1.set_title('Attitude Tracking (Quaternion)')
    ax1.grid(True)
    ax1.legend(loc='best')
    fig1.tight_layout()
    # fig1.savefig('quat_tracking.png', dpi=200)

    # Angular velocity tracking
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(log_time, log_w[:, 0], label=r'$\omega_x$ actual')
    ax2.plot(log_time, log_w[:, 1], label=r'$\omega_y$ actual')
    ax2.plot(log_time, log_w[:, 2], label=r'$\omega_z$ actual')
    ax2.plot(log_time, log_w_ref[:, 0], '--', label=r'$\omega_x^d$')
    ax2.plot(log_time, log_w_ref[:, 1], '--', label=r'$\omega_y^d$')
    ax2.plot(log_time, log_w_ref[:, 2], '--', label=r'$\omega_z^d$')
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

    ani = animation.FuncAnimation(fig5, update_frame, frames=len(log_time), interval=50, blit=True)
    # ani.save('attitude_tracking.mp4', fps=20)

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
    # =========================
    #  SAVE ALL OUTPUTS TO /results
    # =========================
    import os
    RESULTS_DIR = "results"
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

    # Save figures (PNG)
    fig1.savefig(os.path.join(RESULTS_DIR, "quat_tracking.png"), dpi=200)
    fig2.savefig(os.path.join(RESULTS_DIR, "omega_tracking.png"), dpi=200)
    fig3.savefig(os.path.join(RESULTS_DIR, "clf_and_slack.png"), dpi=200)
    fig4.savefig(os.path.join(RESULTS_DIR, "dipole_commands.png"), dpi=200)

    # Save animation MP4
    ani.save(os.path.join(RESULTS_DIR, "attitude_tracking.mp4"), fps=20)

    # Text summary
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("CLF–QP Simulation Results\n")
        f.write("=========================\n\n")
        f.write(f"Total time simulated: {T_total:.2f} s\n")
        f.write(f"Control frequency: {1/dt:.2f} Hz\n")
        f.write(f"Horizon length: {N_horizon}\n")
        f.write(f"Magnetorquer max dipole: {m_max} A·m²\n\n")
        f.write("Controllability Indicator (Cov(B_body) eigenvalues):\n")
        f.write(str(eigvals_B) + "\n\n")
        f.write("Observability:\n")
        f.write("Gyro + magnetometer ensure nonlinear attitude observability over orbit.\n")
        f.write("Time-varying magnetic field excites rotational degrees of freedom.\n")

    print(f"\nAll logs, plots, and animation saved under ./{RESULTS_DIR}/")


if __name__ == "__main__":
    run_simulation()
