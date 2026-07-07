"""
mock_sim.py
A deliberately minimal, transparent numpy-only physics stand-in, used ONLY
because this sandbox has no network access and cannot pip-install mujoco.
It is NOT the deliverable simulator -- mujoco_ctrl.py + model/scene.xml are.

Physics modeled here:
  - Board: 1-DOF hinge, true spring-damper ODE:
        I_h * phi_ddot + b_phi * phi_dot + k_phi * phi = tau_contact
  - Contact: simple compliant point-contact model,
        F_n = k_c * max(0, -v_actual)   (v_actual = signed normal penetration
                                          of the actual, IK-realized pen tip)
  - Arm: NOT integrated as a full rigid-body dynamic chain here (that's what
    MuJoCo is for) -- instead each tick the controller solves for the theta
    that (approximately) reaches p_target given the CURRENT phi, exactly as
    it would inside a real control loop. This is enough to validate the
    moving-frame coupling logic and produce honest tracking/force plots.

Run: python3 mock_sim.py --profile sine
"""
import argparse
import numpy as np
import sys
sys.path.insert(0, ".")
from kinematics import Arm3R, Board
from controller import MovingFrameController


def profile_u(s, kind, u0=0.35, u1=0.75, A=0.15):
    if kind == "hold":
        return np.full_like(s, u0)
    if kind == "sweep":
        return u0 + (u1 - u0) * s
    if kind == "sine":
        return u0 + A * np.sin(2 * np.pi * s)
    raise ValueError(kind)


def run(profile="sine", T=6.0, dt=2e-3, Fn_star=3.0,
        k_phi=8.0, b_phi=0.6, I_h=0.03, k_c=800.0,
        l1=0.30, l2=0.28, l3=0.18,
        H=(0.55, 0.10), phi0=np.deg2rad(100)):

    arm = Arm3R(l1, l2, l3)
    board = Board(H=H, phi0=phi0)
    joint_limits = [(-2.9, 2.9)] * 3
    ctrl = MovingFrameController(arm, board, joint_limits=joint_limits, Kf=3e-4)

    n = int(T / dt)
    t = np.arange(n) * dt
    s = t / T
    u_cmd_series = profile_u(s, profile)

    phi, phidot = 0.0, 0.0
    Fn = 0.0

    log = dict(t=t, u_cmd=np.zeros(n), u_actual=np.zeros(n), v_actual=np.zeros(n),
               Fn=np.zeros(n), phi=np.zeros(n), pos_err=np.zeros(n),
               theta=np.zeros((n, 3)))

    for k in range(n):
        u_cmd = u_cmd_series[k]
        # --- compose board FK into target using CURRENT measured phi & Fn ---
        p_target = ctrl.target_from_board(u_cmd, phi_meas=phi, Fn_star=Fn_star, Fn_meas=Fn)
        # --- small per-step optimization exploiting redundancy ---
        theta = ctrl.solve_step(p_target)
        p_actual = arm.fk(theta)

        t_hat, n_hat = board.frame(phi)
        r = p_actual - board.H
        u_actual = np.dot(r, t_hat)
        v_actual = np.dot(r, n_hat)

        Fn = k_c * max(0.0, -v_actual)
        tau = -u_actual * Fn  # moment arm x normal force about hinge

        # semi-implicit Euler integration of hinge ODE
        phi_ddot = (tau - b_phi * phidot - k_phi * phi) / I_h
        phidot += phi_ddot * dt
        phi += phidot * dt

        log["u_cmd"][k] = u_cmd
        log["u_actual"][k] = u_actual
        log["v_actual"][k] = v_actual
        log["Fn"][k] = Fn
        log["phi"][k] = phi
        log["pos_err"][k] = u_actual - u_cmd
        log["theta"][k] = theta

    return log, dict(Fn_star=Fn_star, k_phi=k_phi, b_phi=b_phi, I_h=I_h, k_c=k_c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="sine", choices=["hold", "sweep", "sine"])
    ap.add_argument("--out", default="../figures/log_{profile}.npz")
    args = ap.parse_args()
    log, params = run(profile=args.profile)
    outpath = args.out.format(profile=args.profile)
    np.savez(outpath, **log, **{f"param_{k}": v for k, v in params.items()})
    print(f"saved {outpath}")
    print(f"final tracking RMS error (u): {np.sqrt(np.mean(log['pos_err']**2)):.5f} m")
    print(f"mean Fn: {log['Fn'].mean():.3f} N, final phi: {np.rad2deg(log['phi'][-1]):.2f} deg")
