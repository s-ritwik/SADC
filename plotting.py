#!/usr/bin/env python3
"""
Post-processing plots for SADC CLF–QP simulations.

Loads .npy files from the 'results' directory and generates:
1. Individual quaternion tracking (q0–q3).
2. Individual angular-velocity tracking (ωx, ωy, ωz), with optional initial skip.
3. Pointing-error and rate-error histories.
4. Magnetorquer dipole commands.
5. CLF and slack over time.
6. Torque usage (magnetorquer vs reaction wheel).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")  # change if needed
OMEGA_SKIP_SECONDS = 1000.0    # skip this many seconds at start in ω plots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_array(name: str) -> np.ndarray:
    path = RESULTS_DIR / f"{name}.npy"
    return np.load(path)


def compute_pointing_error_deg(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    q, q_ref shape: (N, 4), scalar-first unit quaternions.
    Returns θ in degrees, where θ is angle of shortest rotation q_err.
    """
    dots = np.sum(q * q_ref, axis=1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return np.rad2deg(angles)


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def main():
    # Load data
    t = load_array("time")              # (N,)
    q = load_array("q")                 # (N,4)
    q_ref = load_array("q_ref")         # (N,4)
    w = load_array("w")                 # (N,3)
    w_ref = load_array("w_ref")         # (N,3)
    m = load_array("m")                 # (N,3)
    V = load_array("V")                 # (N,)
    delta = load_array("delta")         # (N,)
    B_body = load_array("B_body")       # (N,3)
    tau_rw = load_array("tau_rw")       # (N,)

    # Derived quantities
    pointing_err_deg = compute_pointing_error_deg(q, q_ref)
    w_err = w - w_ref
    w_err_norm = np.linalg.norm(w_err, axis=1)
    m_norm = np.linalg.norm(m, axis=1)
    tau_mag = np.cross(m, B_body)
    tau_mag_norm = np.linalg.norm(tau_mag, axis=1)

    # Index to skip initial window in ω and torque plots
    if OMEGA_SKIP_SECONDS > 0.0:
        skip_idx = np.searchsorted(t, OMEGA_SKIP_SECONDS)
    else:
        skip_idx = 0

    # -----------------------------------------------------------------------
    # 1. Individual quaternion tracking (4 subplots)
    # -----------------------------------------------------------------------
    fig_q, axs_q = plt.subplots(4, 1, sharex=True, figsize=(7, 8))
    q_labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i in range(4):
        ax = axs_q[i]
        ax.plot(t, q[:, i], color=colors[i], label=f"{q_labels[i]} actual")
        ax.plot(t, q_ref[:, i], "--", color=colors[i], linewidth=2.0,
                label=f"{q_labels[i]} ref")
        ax.set_ylabel(q_labels[i])
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    axs_q[-1].set_xlabel("Time [s]")
    fig_q.suptitle("Quaternion Tracking (All Components)")
    fig_q.tight_layout(rect=(0, 0, 1, 0.96))
    fig_q.savefig(RESULTS_DIR / "report_quat_tracking.png", dpi=300)

    # -----------------------------------------------------------------------
    # 2. Individual angular-velocity tracking (skip initial window)
    # -----------------------------------------------------------------------
    t_w = t[skip_idx:]
    w_plot = w[skip_idx:, :]
    w_ref_plot = w_ref[skip_idx:, :]

    fig_w, axs_w = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    w_labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    for i in range(3):
        ax = axs_w[i]
        ax.plot(t_w, w_plot[:, i], color=colors[i], label=f"{w_labels[i]} actual")
        # ax.plot(t_w, w_ref_plot[:, i], "--", color=colors[i], linewidth=2.0,
        #         label=f"{w_labels[i]} ref (colored)")
        # Overlay a thick black reference line to make it stand out
        ax.plot(t_w, w_ref_plot[:, i], "-", color="k", linewidth=3.0,
                alpha=0.6, label=f"{w_labels[i]} ref (black)")
        ax.set_ylabel("[rad/s]")
        ax.set_title(w_labels[i])
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    axs_w[-1].set_xlabel("Time [s]")
    fig_w.suptitle("Body Angular Velocity vs Reference")
    fig_w.tight_layout(rect=(0, 0, 1, 0.96))
    fig_w.savefig(RESULTS_DIR / "report_omega_tracking.png", dpi=300)

    # -----------------------------------------------------------------------
    # 3. Pointing error & rate-error norms (start at 500 s)
    # -----------------------------------------------------------------------
    pe_start_idx = np.searchsorted(t, 500.0)
    t_pe = t[pe_start_idx:]
    pointing_pe = pointing_err_deg[pe_start_idx:]
    w_err_norm_pe = w_err_norm[pe_start_idx:]

    fig_err, (ax_e1, ax_e2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax_e1.plot(t_pe, pointing_pe, color="tab:blue")
    ax_e1.set_ylabel("Pointing error [deg]")
    ax_e1.set_title("Pointing & Rate Tracking Errors")
    ax_e1.grid(True)

    ax_e2.plot(t_pe, w_err_norm_pe, color="tab:orange")
    ax_e2.set_ylabel(r"$\|\omega - \omega^d\|$ [rad/s]")
    ax_e2.set_xlabel("Time [s]")
    ax_e2.grid(True)

    fig_err.tight_layout()
    fig_err.savefig(RESULTS_DIR / "report_tracking_errors.png", dpi=300)

    # -----------------------------------------------------------------------
    # 4. Magnetorquer dipole commands (per axis + norm)
    # -----------------------------------------------------------------------
    fig_m, axs_m = plt.subplots(4, 1, sharex=True, figsize=(7, 7))
    m_labels = [r"$m_x$", r"$m_y$", r"$m_z$"]

    for i in range(3):
        ax = axs_m[i]
        ax.plot(t, m[:, i], color=colors[i], label=m_labels[i])
        ax.set_ylabel("[A·m$^2$]")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    axs_m[3].plot(t, m_norm, color="tab:red", label=r"$\|m\|$")
    axs_m[3].set_ylabel("[A·m$^2$]")
    axs_m[3].set_xlabel("Time [s]")
    axs_m[3].grid(True)
    axs_m[3].legend(loc="best", fontsize=8)

    fig_m.suptitle("Magnetorquer Dipole Commands")
    fig_m.tight_layout(rect=(0, 0, 1, 0.96))
    fig_m.savefig(RESULTS_DIR / "report_dipole_commands.png", dpi=300)

    # -----------------------------------------------------------------------
    # 5. CLF and slack over time
    # -----------------------------------------------------------------------
    # Optionally skip last part of the trajectory for CLF plots
    clf_end_idx = max(0, len(t) - 500)
    t_clf = t[:clf_end_idx] if clf_end_idx > 0 else t
    V_clf = V[:clf_end_idx] if clf_end_idx > 0 else V
    delta_clf = delta[:clf_end_idx] if clf_end_idx > 0 else delta

    fig_clf, ax_c1 = plt.subplots(figsize=(7, 4))
    ax_c1.plot(t_clf, V_clf, color="tab:blue", label="V(x) (CLF)")
    ax_c1.set_xlabel("Time [s]")
    ax_c1.set_ylabel("V(x)")
    ax_c1.grid(True)

    ax_c2 = ax_c1.twinx()
    ax_c2.plot(t_clf, delta_clf, color="tab:red", label="Slack δ")
    ax_c2.set_ylabel("δ")

    lines1, labels1 = ax_c1.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig_clf.tight_layout()
    fig_clf.savefig(RESULTS_DIR / "report_clf_and_slack.png", dpi=300)

    # -----------------------------------------------------------------------
    # 6. Torque usage: magnetorquer vs reaction wheel (skip initial window)
    # -----------------------------------------------------------------------
    t_tau = t[skip_idx:]
    tau_mag_norm_plot = tau_mag_norm[skip_idx:]
    tau_rw_plot = np.abs(tau_rw[skip_idx:])

    fig_tau, axs_tau = plt.subplots(3, 1, sharex=True, figsize=(7, 7))

    axs_tau[0].plot(t_tau, tau_mag_norm_plot, color="tab:blue", label=r"$|\tau_{\mathrm{mag}}|$")
    axs_tau[0].set_ylabel("[N·m]")
    axs_tau[0].set_title("Magnetorquer Torque Norm")
    axs_tau[0].grid(True)
    axs_tau[0].legend(loc="best", fontsize=8)

    axs_tau[1].plot(t_tau, tau_rw_plot, color="tab:orange", label=r"$|\tau_{\mathrm{rw}}|$")
    axs_tau[1].set_ylabel("[N·m]")
    axs_tau[1].set_title("Reaction-Wheel Torque (Y-axis)")
    axs_tau[1].grid(True)
    axs_tau[1].legend(loc="best", fontsize=8)

    axs_tau[2].plot(t_tau, tau_mag_norm_plot, color="tab:blue", label=r"$|\tau_{\mathrm{mag}}|$")
    axs_tau[2].plot(t_tau, tau_rw_plot, color="tab:orange", label=r"$|\tau_{\mathrm{rw}}|$")
    axs_tau[2].set_ylabel("[N·m]")
    axs_tau[2].set_xlabel("Time [s]")
    axs_tau[2].set_title("Torque Comparison")
    axs_tau[2].grid(True)
    axs_tau[2].legend(loc="best", fontsize=8)

    fig_tau.tight_layout()
    fig_tau.savefig(RESULTS_DIR / "report_torque_usage.png", dpi=300)

    print(f"Report plots saved under {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
