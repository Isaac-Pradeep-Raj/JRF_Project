# IIT Madras INTERFACE Lab JRF Task Review

Source of truth: `C:\Users\ISAAC\OneDrive\Desktop\JRF26_task.pdf`.

## Architecture Summary

The runnable implementation is centered on `model/scene.xml` and `src/simulation.py`.
The XML contains the MuJoCo model: a fixed-base 3R planar arm, contact geometry, and a passive hinged board.
`src/kinematics.py` provides analytical planar FK/Jacobian and board-frame transforms.
`src/controller.py` implements bounded optimization-based IK with a contact-force outer loop.
`src/board.py` contains trajectory, board-frame/contact helpers, and MuJoCo contact-force measurement.
`src/plots.py` generates result plots from the simulation log.
`model/robot.xml` and `model/board.xml` are component copies for inspection.
`src/mock_sim.py` is stale and marked as a non-deliverable fallback, not the main simulator.

## Requirement Checklist

| Status | Assignment requirement | Files responsible | Match to assignment | Improvement needed |
| --- | --- | --- | --- | --- |
| ✓ | Use MuJoCo with official Python bindings and author MJCF model | `model/scene.xml`, `src/simulation.py` | Main simulation uses `mujoco.MjModel` and custom MJCF | None major |
| ✓ | Everything lives in one vertical x-z plane | `model/scene.xml`, `src/kinematics.py` | Joint axes are normal to x-z plane; FK uses x-z | None major |
| ✓ | Fixed-base planar 3-DOF / 3-link arm | `model/scene.xml`, `model/robot.xml`, `src/kinematics.py` | Three revolute joints and three visual link geoms exist | Add visually distinct rigid pen after link 3 |
| ✓ | Arm holds a rigid pen; pen does not add a joint | `model/scene.xml`, `model/robot.xml` | Rigid shaft is attached to link 3 with no joint; analytical FK uses the tool-tip length | None major |
| ✓ | Only pen tip contacts board | `model/scene.xml`, `src/board.py` | Shaft has collisions disabled; only small tip sphere contacts board | Continue tuning contact smoothness |
| ✓ | Board is passive and tilts on a single hinge | `model/scene.xml`, `model/board.xml` | Board hinge has no actuator | Add clearer hinge/pivot visualization |
| ✓ | Board hinge is spring-loaded and damped | `model/scene.xml`, `model/board.xml` | `stiffness` and `damping` are set on board hinge and visual spring/damper cues are present | None major |
| ✓ | Board tilt responds to pen contact force | `model/scene.xml`, `src/board.py`, `src/simulation.py` | Contact between pen and board is measured and board is dynamic | Tune visibility after pen/contact changes |
| ✓ | Target trajectory is measured along board tangent `u(s)` | `src/board.py`, `src/controller.py` | `BoardTrajectory.u()` defines board-frame path | Add selectable static/sweep/sinusoidal profiles |
| ✓ | Profiles: static hold, sweep, sinusoidal sweep; pick one allowed | `src/board.py`, `src/simulation.py` | All three profiles are selectable from the CLI; sine is the default demo | None major |
| ✓ | Target is converted from board frame to world frame each timestep | `src/controller.py`, `src/kinematics.py`, `src/simulation.py` | `board.surface_point(u, offset, angle)` is called every step using current board angle | None major |
| ✓ | Moving-frame target changes as board tilts | `src/simulation.py`, `src/controller.py` | Board hinge angle is read every timestep before IK | None major |
| ✓ | Use optimization / small numerical IK, not black-box IK stack | `src/controller.py` | Uses SciPy L-BFGS-B with explicit FK/Jacobian | None major |
| ✓ | Exploit redundancy of 3 joints for 1-D path plus force target | `src/controller.py` | Optimizes 2-D target with posture/step regularization and force-to-offset loop | Could document objective better in report |
| ✓ | Derive/use correct FK | `src/kinematics.py` | Explicit 3R planar FK is implemented with effective third segment ending at the rigid pen tip | None major |
| ✓ | Derive/use correct Jacobian | `src/kinematics.py`, `src/controller.py` | Analytical 2x3 Jacobian is implemented and used in optimizer gradient | None major |
| ⚠ | Regulate contact normal force inside `[Fmin, Fmax]` | `src/controller.py`, `src/board.py`, `src/simulation.py` | Force band is implemented and logged; current run is mostly inside band | Retune after changing smaller pen tip/contact |
| ⚠ | Surface-normal offset / penetration near zero | `src/controller.py`, `src/kinematics.py`, `src/simulation.py` | Log now reports contact-point gap relative to the tip radius; after settling it stays near sub-mm positive separation | Continue improving force smoothness during moving profiles |
| ✓ | Maintain contact while tracing | `src/simulation.py`, `src/controller.py` | Current full run reports high contact fraction | Reverify after contact geometry changes |
| ✓ | Logging and metrics | `src/simulation.py` | Logs time, joints, targets, pen/contact points, force, board angle, IK status, gap | None major |
| ✓ | Tracking-error and contact-force plots | `src/plots.py` | Generates tracking/force/gap, board/joints, and world target/pen/contact path plots | None major |
| ✓ | Video should clearly show pen tracing and board visibly tilting | `model/scene.xml`, `src/simulation.py` | Viewer includes rigid pen, hinge support, visual spring/damper, trajectory dots, and moving markers | Record final video |
| ✓ | Display desired trajectory, actual traced trajectory, pen tip, contact point | `model/scene.xml`, `src/simulation.py`, `src/plots.py` | Viewer has desired/actual/contact markers; plots include desired, pen, and contact paths | Optional force-vector visualization |
| ✓ | Runnable code with short README: dependencies and how to run | `README.md` | README contains dependencies, run commands, profiles, viewer command, and model summary | None major |
| ✗ | 2-page report with formulation, FK/Jacobian, IK, board coupling, results/failure modes | `report/` | Report directory is empty | Future report writing required |

## Current Limitations

- Moving-profile contact force remains somewhat chattery with the very small 2 mm tip; static contact is solid and the default sine profile settles to small contact-point gap.
- `report/` is empty.
- `src/mock_sim.py` is stale relative to the current implementation and should either be repaired or explicitly ignored.

## Suggested Future Improvements

- Add automated smoke checks for FK/Jacobian consistency and no-actuator board passivity.
- Add force-vector and board-normal visual arrows if more video clarity is needed.
- Add report text and cite any reused libraries.
- Record a 30-60 s viewer video after final visual tuning.
