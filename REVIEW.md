# Engineering Review

This review reflects the submitted implementation in this repository. `model/scene.xml` and `src/simulation.py` are the runnable source of truth.

## Implemented architecture

- `model/scene.xml` defines a fixed-base 3R arm in the world x-z plane, a rigid pen with a 2 mm spherical contact tip, and a passive board with one spring-damped hinge. Only the pen tip and board have collision enabled.
- `src/kinematics.py` implements analytical forward kinematics and the 2x3 Jacobian for the pen-tip position.
- `src/controller.py` uses SciPy L-BFGS-B with explicit objective and gradient terms, joint bounds, posture regularization, and a force-to-normal-offset outer loop.
- `src/board.py` defines static, linear-sweep, and sinusoidal board-frame trajectories, obtains the board angle each timestep, and measures only pen-tip/board contact force.
- `src/simulation.py` converts each board-frame target to world coordinates at every timestep, runs MuJoCo, logs metrics, and optionally opens the passive viewer.
- `src/plots.py` generates the three figures listed in the README from a saved simulation log.

## Requirement checklist

| Status | Requirement | Evidence |
| --- | --- | --- |
| Complete | MuJoCo simulation using the official Python bindings and custom MJCF | `model/scene.xml`, `src/simulation.py` |
| Complete | Fixed-base 3R arm constrained to the x-z plane | `model/scene.xml`, `src/kinematics.py` |
| Complete | Rigid pen without an additional joint; only its tip contacts the board | `model/scene.xml` |
| Complete | Passive, single-hinge board with spring and damping and no actuator | `model/scene.xml` |
| Complete | Static, linear-sweep, and sinusoidal board-frame paths | `src/board.py`, `src/simulation.py` |
| Complete | Per-timestep moving-frame target conversion | `src/controller.py`, `src/kinematics.py`, `src/simulation.py` |
| Complete | Bounded optimization-based IK using explicit FK and Jacobian | `src/controller.py`, `src/kinematics.py` |
| Complete | Normal contact-force measurement, force-band control, and logging | `src/board.py`, `src/controller.py`, `src/simulation.py` |
| Complete | Tracking, force/gap, board/joint, and world-path plots | `src/plots.py` |
| Complete | Fresh-machine setup and execution instructions | `README.md`, `requirements.txt` |

## Remaining limitations

- The compliant contact model produces some force chatter during moving trajectories; the 2 mm tip makes this more visible.
- Force is controlled indirectly through the commanded normal offset, so it is regulated to a band rather than held at an exact value.
- The model is intentionally planar and omits sensor noise, actuator dynamics, and other nonideal hardware effects.
- No report or recorded viewer video is included in this repository.
