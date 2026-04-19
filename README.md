# Launch Vehicle GNC Simulator

[![CI](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml)
[![Docker Build](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml/badge.svg)](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml)

![Banner](readme_imgs/earth.png)

*A complete Python-based 6-DoF simulation of a two-stage launch vehicle from liftoff to stable low Earth orbit.*

## Table of Contents

- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Technical Highlights](#technical-highlights)
- [Repository Structure](#repository-structure)
- [Results & Visualizations](#results--visualizations)
- [FastAPI Endpoints](#fastapi-endpoints)
- [Docker Containerization](#docker-containerization)
- [CI Pipeline](#ci-pipeline)
- [Future Extensions](#future-extensions)

## Overview

This project is a modular, physics-based 6-DoF rocket ascent simulator that models the full flight from vertical liftoff through staging, vacuum guidance, coast, and orbital insertion. The simulation successfully reaches user-defined orbits (e.g. 275 × 290 km) using realistic control and guidance laws. It also includes production-oriented interfaces and workflows through a FastAPI service layer, a Docker Compose deployment with PostgreSQL-backed Monte Carlo persistence, and CI validation pipeline.

## Key Achievements

- Implemented full 6-DoF mechanics with rigid-body dynamics, quaternion propagation, multi-engine gimballing, RCS thrusters, and dynamic center of mass  
- Built modular mission planner with clean phase transitions (Initial Ascent → Pitch Program → PEG → Coast → Circularization)  
- Achieved stable orbital insertion with tight apoapsis/periapsis tolerances under gravity, drag, and J2 perturbations  
- Created quality 3D trajectory visualizations segmented by mission phase  
- Maintained clean, object-oriented architecture suitable for extension (C++ port, Hardware-in-the-loop, etc.)
- Exposed simulation execution via FastAPI endpoints, including health and Monte Carlo batch interfaces for programmatic use  
- Added Docker and Docker Compose workflows for reproducible local and containerized deployment with a dedicated Postgres service  
- Established CI checks for formatting, linting, tests, and Docker build health to protect simulator reliability

## Technical Highlights

- **Dynamics**  
  - Quaternion-based attitude with angular velocity propagation  
  - Multi-engine thrust vector control and least-squares RCS allocation
  - Dynamic center-of-mass & gimbal arm length from propellant depletion 
  - Verlet integration with normalized quaternions  

- **Guidance & Control**  
  - Powered Explicit Guidance (PEG) solving real-time burn direction & throttle  
  - Programmed pitch maneuver for initial gravity turn 
  - PID attitude controller with quaternion error and gain scheduling  

- **Environment**  
  - Newtonian + J2 oblateness gravity  
  - Rotating Earth with coriolis wind in atmosphere  
  - Exponential atmosphere with angle-of-attack dependent drag 

- **Mission Sequencing**  
  - Phase objects (TimeBased, Kick, PEG, Coast, CircBurn, etc.)  
  - Automatic completion checks (time, apoapsis reached, eccentricity condition)  
  - Seamless hand-over between stages  

- **Platform & Reliability**  
  - FastAPI application layer with configuration-driven runtime settings and Monte Carlo execution endpoints  
  - Dockerfile + Compose support for consistent environment setup and one-command service startup  
  - PostgreSQL-backed Monte Carlo persistence using a dedicated container and Docker volume  
  - CI pipeline enforcing Black, Ruff, Pytest, and container build validation on each change

## Repository Structure

The repository is split between the simulation engine and the application layer.

- `src/simulator/` contains the core physics and mission code: dynamics, environment, guidance, controller, integrator, state, plotting, vehicle models, and the simulator driver.
- `src/app/paths/` contains FastAPI route definitions and API-facing orchestration.
- `src/app/models/` contains Pydantic request/response models.
- `src/app/runners/` contains application runners that translate API requests into simulator executions and Monte Carlo workflows.
- `src/app/storage/` contains persistence logic, currently PostgreSQL-backed Monte Carlo batch storage.
- `examples/` contains standalone scripts that exercise the simulator package directly.
- `tests/` validates the simulator package and API-adjacent behavior.

This separation keeps the simulation engine reusable while the FastAPI layer owns delivery concerns such as request handling, persistence, and batch orchestration.

## Results & Visualizations

### 3D Trajectory by Mission Phase with Earth Reference

![Orbit](readme_imgs/orbit.gif)

### Engine Gimbal Angles During Gravity Turn

![Engine_Gimbals](readme_imgs/gimbal.gif)

> Note: The gimbal angles above are highly exaggerated for effect. 

### Altitude Profile (Launch through Second Stage Cut-off)

![Altitude vs Time_1](readme_imgs/AltitudeVsTime_1.png)

### Altitude Profile (Coast through Orbit)

![Altitude vs Time_2](readme_imgs/AltitudeVsTime_2.png)

### Velocity Profile

![Velocity vs Time](readme_imgs/VelocityVsTime.png)

### Pitch Angle Evolution

![Pitch Angle vs Time](readme_imgs/PitchVsTime.png)

### Example Logs

![Logs](readme_imgs/logs.png)



## FastAPI Endpoints

The simulator includes a FastAPI service and Monte Carlo batch endpoints.

- Interactive docs (Swagger UI): `http://localhost:8000/docs`
- Alternative docs (ReDoc): `http://localhost:8000/redoc`

### Endpoint Summary

| Method | Path | Purpose | Typical Status Codes |
|---|---|---|---|
| GET | `/health` | Service health and runtime environment | `200` |
| POST | `/simulations/simulate` | Run one simulation request and return results | `200`, `422`, `500` |
| GET | `/simulations/live/view` | Minimal browser UI to start/connect and view live telemetry text | `200` |
| POST | `/simulations/live/start` | Kick off async simulation run with live telemetry stream | `200`, `422`, `500` |
| WS | `/simulations/live/{run_id}/ws` | WebSocket stream of telemetry frames and final status | `101`, `404` |
| POST | `/simulations/monte-carlo` | Kick off async Monte Carlo batch run | `200`, `422`, `500` |
| GET | `/simulations/monte-carlo` | List Monte Carlo batches | `200`, `500` |
| GET | `/simulations/monte-carlo/{batch_id}` | Poll a batch by id (returns `202` while in progress) | `200`, `202`, `404`, `500` |

### Local API Run

1. Install dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Start the API from the project root:

```bash
uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8000
```

3. Confirm health:

```bash
curl http://localhost:8000/health
```

### Monte Carlo Example

Kick off a batch (returns immediately with a `batch_id`):

```bash
curl -X POST http://localhost:8000/simulations/monte-carlo \
  -H "Content-Type: application/json" \
  -d '{
    "num_simulations": 20,
    "base_simulation": {},
    "dispersions": {
      "stage1_base_thrust_magnitude": {"mean": 7600000, "std_dev": 100000}
    }
  }'
```

Poll for completion:

```bash
curl http://localhost:8000/simulations/monte-carlo/<batch_id>
```

Note: Monte Carlo runs execute in the background via a thread pool. Polling can return `202` until the batch completes.

### Live Telemetry Example

Browser viewer page:

```text
http://localhost:8000/simulations/live/view
```

The page can start a new live run and stream telemetry in plain text directly in the browser.

Start a live run (returns immediately with a `run_id`):

```bash
curl -X POST "http://localhost:8000/simulations/live/start?telemetry_interval=0.5" \
  -H "Content-Type: application/json" \
  -d '{}'
```

WebSocket stream URL:

```text
ws://localhost:8000/simulations/live/<run_id>/ws
```

The socket sends telemetry messages with `type=telemetry` while running and one final `type=status` payload on completion or failure.

## Docker Containerization

The Docker image runs the FastAPI service with Uvicorn on port `8000` using Python `3.11-slim`.

### Environment Variables

Create a `.env` file (or set variables directly) to customize runtime behavior:

- `SIM_API_TITLE` (default: `6DOF Launch Simulator`)
- `SIM_API_DESCRIPTION` (default: Configurable launch-to-orbit simulation for mission software testing)
- `SIM_ENVIRONMENT` (default: `dev`)
- `SIM_DEBUG` (default: `false`)
- `SIM_EXECUTOR_MAX_WORKERS` (default: `4`)
- `SIM_DB_HOST` (default: `localhost`)
- `SIM_DB_PORT` (default: `5432`)
- `SIM_DB_NAME` (default: `launch_sim`)
- `SIM_DB_USER` (default: `sim_user`)
- `SIM_DB_PASSWORD` (default: `sim_password`)

All variables are optional; defaults are defined in application settings.

### Local Development (Compose)

Use Compose for local iteration with the API and PostgreSQL running as separate services:

```bash
docker compose up --build
```

Compose provisions a named Postgres volume for Monte Carlo persistence so results survive container restarts.

### Standalone Image Run

Build and run a standalone image:

```bash
docker build -t launch-gnc-sim .
docker run --rm -p 8000:8000 --env-file .env launch-gnc-sim
```

Use this mode for clean, reproducible runtime checks that mirror deployment behavior.

## CI Pipeline

Workflow pages:

- CI (format, lint, tests): https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml
- Docker Build (image build/publish): https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml

### Triggers

- CI workflow runs on every `push`, `pull_request`, and manual dispatch.
- Docker Build workflow runs on `push`/`pull_request` for Docker and source-related path changes, plus manual dispatch.

### What Gets Validated

- Black formatting check (`black --check src tests`)
- Ruff lint check (`ruff check src tests`)
- Unit tests with Pytest (`PYTHONPATH=src pytest -v`)
- Docker image build on GitHub Actions; image push to GHCR on `main`

### Merge Gate

Branch protection rule requires both workflows to pass before merge. This keeps style, correctness, and container buildability enforced on every change.


## Future Extensions

- Re-entry, descent, and landing with added control surfaces
- C++/Rust performance port of dynamics & integration core  
- Hardware-in-the-loop (HIL) interface layer  
- Real-time visualization dashboard