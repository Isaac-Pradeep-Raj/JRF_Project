# JRF Project: Chopstick Crane

MuJoCo simulation for the IIT Madras INTERFACE Lab JRF task.

## Setup

Use Python 3.10+ and install:

```powershell
pip install mujoco numpy scipy matplotlib pypdf
```

`pypdf` is only needed for extracting/reviewing the supplied assignment PDF.

## Run

From the project root:

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the simulation with visualization
python src/simulation.py --viewer --realtime

# Or run headless and generate plots
python src/simulation.py

This saves:

- `figures/simulation_log.npz`
- `figures/tracking_force.png`
- `figures/board_and_joints.png`
- `figures/world_path.png`

Available board-frame profiles:

```powershell
venv\Scripts\python.exe src\simulation.py --profile static --duration 8
venv\Scripts\python.exe src\simulation.py --profile sweep --duration 8
venv\Scripts\python.exe src\simulation.py --profile sine --duration 8
```

Show the MuJoCo viewer:

```powershell
venv\Scripts\python.exe src\simulation.py --viewer --realtime --duration 30
```

## Model

- Fixed-base 3R planar arm in the world x-z plane.
- Rigid pen shaft attached to link 3 with no extra joint.
- Only the small pen tip collides with the board.
- Passive board hinge with spring and damping; no board actuator.
- IK target is defined in the moving board frame and converted to world coordinates every timestep.
- Contact force is measured from MuJoCo pen-tip/board contacts and regulated by a normal-offset outer loop.
