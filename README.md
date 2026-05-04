# Orbital Mechanics GNC Simulator

[![CI](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml)
[![Docker Build](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml/badge.svg)](https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml)

![Banner](readme_imgs/earth.png)

## Table of Contents

- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Technical Highlights](#technical-highlights)
- [Repository Structure](#repository-structure)
- [Results & Visualizations](#results--visualizations)
- [Mission Software Platform](#mission-software-platform)
- [API Reference](#api-reference)
- [Deployment & Operations](#deployment--operations)
- [Future Extensions](#future-extensions)

## Overview

This project is a modular, physics-based 6-DoF orbital mechanics simulator that models the full flight from vertical liftoff through staging, vacuum guidance, coast, orbital insertion, re-entry, and landing. The simulation successfully reaches user-defined orbits (e.g. 275 × 290 km) using realistic control and guidance laws. The simulation engine is backed by a production-style mission software platform: a FastAPI service with live WebSocket telemetry streaming, operator command uplink for constellation deorbit operations, async Monte Carlo dispersion analysis with PostgreSQL persistence, a browser-based dashboard, containerized deployment via Docker Compose, and CI validation on every change.  

<br>

![Telemetry](readme_imgs/constellation.gif)

## Key Achievements

- Implemented full 6-DoF mechanics with rigid-body dynamics, quaternion propagation, multi-engine gimballing, RCS thrusters, and dynamic center of mass  
- Built modular mission planner with clean phase transitions (Initial Ascent → Pitch Program → PEG → Coast → Circularization)  
- Achieved stable orbital insertion with tight apoapsis/periapsis tolerances under gravity, drag, and J2 perturbations  
- Created quality 3D trajectory visualizations segmented by mission phase  
- Maintained clean, object-oriented architecture suitable for extension (C++ port, Hardware-in-the-loop, etc.)
- Exposed simulation execution via a full FastAPI service layer: synchronous single-run endpoint, async live telemetry endpoint with WebSocket frame streaming and a browser dashboard, and async Monte Carlo batch endpoint with PostgreSQL-persisted results  
- Added Docker and Docker Compose workflows for reproducible local and containerized deployment with a dedicated Postgres service  
- Established CI checks for formatting, linting, tests, and Docker build health to protect simulator reliability
- Added command uplink and evented deorbit/re-entry workflow for constellation vehicles (Orbit Await Command -> Deorbit Burn -> Ballistic Reentry -> Parachute Descent -> Landed)
- Added an observability system with structured JSON logs, OpenTelemetry tracing to Jaeger, Prometheus metrics, alerting, and container health probes

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

- **Mission Software Platform**  
  - FastAPI application layer with environment-driven configuration and three route groups: single-run simulation, live telemetry, and Monte Carlo  
  - WebSocket telemetry stream with sequenced frames and a browser dashboard (Chart.js altitude/speed charts + Plotly 3D orbit view)  
  - Operator command uplink API for constellation deorbit requests, including accepted/rejected command outcomes and telemetry event emission
  - Constellation guidance phases that transition from passive orbit into deorbit burn and atmospheric recovery sequence
  - Async background execution using a shared thread pool; live telemetry tracked with a thread-safe in-memory state machine (`queued → running → completed/failed`)  
  - PostgreSQL-backed Monte Carlo batch persistence with schema auto-initialization and named Docker volume  
  - Observability stack with JSON logs, OTLP traces to Jaeger, Prometheus metrics, and alert rules for runtime health
  - CI pipeline enforcing Black, Ruff, Pytest, and container build validation on each change

## Repository Structure

The repository is split between the simulation engine and the application layer.

- `src/simulator/` contains the core physics and mission code: dynamics, environment, guidance, controller, integrator, state, plotting, vehicle models, and the simulator driver.
- `src/app/paths/` contains FastAPI route definitions and API-facing orchestration.
- `src/app/models/` contains Pydantic request/response models.
- `src/app/runners/` contains application runners that translate API requests into simulator executions and Monte Carlo workflows.
- `src/app/storage/` contains two persistence implementations: PostgreSQL-backed Monte Carlo batch storage and a thread-safe in-memory store for live telemetry runs.
- `src/app/frontend/live/` contains the browser-based live telemetry dashboard (HTML/CSS/JS using Chart.js and Plotly, served directly by FastAPI).
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



## Mission Software Platform

The physics core is packaged behind a production-style mission software interface built for repeatable operations and automation.

### Backend Highlights

- Exposes deterministic simulation execution through typed request/response contracts.
- Separates heavy compute from API request handling using a shared thread pool.
- Supports both human-in-the-loop workflows (live telemetry UI) and machine-driven workflows (batch Monte Carlo).
- Supports command-in-the-loop constellation operations where operators can uplink timed deorbit commands to specific vehicles.
- Persists batch analysis artifacts to PostgreSQL for traceability and post-run analysis.
- Runs consistently in local development, Docker Compose, and CI.

### Service architecture

- FastAPI app with three route groups under `/simulations`: single-run simulation, live telemetry, and Monte Carlo.
- Shared background executor configured by `SIM_SIMULATOR_EXECUTOR_MAX_WORKERS`.
- Live telemetry store is in-memory and thread-safe (`queued -> running -> completed/failed`).
- Constellation command uplink endpoint supports per-vehicle deorbit command queueing and validation (`accepted`/`rejected`).
- Constellation telemetry stream emits command lifecycle events (`command_accepted`, `command_armed`, `burn_started`, `reentry_started`, `parachute_deployed`, `landed`).
- Monte Carlo storage is PostgreSQL-backed and initializes schema on startup.
- Runtime configuration is environment-driven using `pydantic-settings` with `SIM_` prefix.

### Local API run

1. Install dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Start the API from the project root:

```bash
uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8000 --log-config log_config.json
```

3. Confirm health:

```bash
curl http://localhost:8000/health
```

4. Open docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Constellation Observability 

This repository includes a focused observability slice for constellation operations.

- Logging: Structured JSON logs for application events (`app.observability`) and Uvicorn access/error logs, unified through a shared JSON formatter in `log_config.json`.
- Tracing: OpenTelemetry spans exported over OTLP to Jaeger (`SIM_OTLP_ENDPOINT`). Request spans are short-lived; run lifecycle spans model background execution (`constellation.run` with `initialize`, `execute`, `finalize`, and per-satellite child spans).
- Metrics: Prometheus metrics exposed at `/metrics`: `constellation_run_outcome_total`, `app_process_cpu_seconds_total`, and `app_process_resident_memory_bytes`.
- Alerting: Prometheus rule `SimulatorApiHighCpu` warns when CPU stays above 80% of one core for 10s.

Quick verification flow:

1. Start a constellation run:

```bash
curl -X POST "http://localhost:8000/simulations/live/constellation/start"
```

2. Send one accepted command (replace `<run_id>`):

```bash
curl -X POST "http://localhost:8000/simulations/live/constellation/command" \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "<run_id>",
    "vehicle_id": "W-4",
    "action": "deorbit_burn",
    "execute_at_sim_time_s": 20,
    "target_perigee_alt_km": 120
  }'
```

3. Send one rejected command (invalid run id):

```bash
curl -X POST "http://localhost:8000/simulations/live/constellation/command" \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "missing-run",
    "vehicle_id": "W-4",
    "action": "deorbit_burn",
    "execute_at_sim_time_s": 20,
    "target_perigee_alt_km": 120
  }'
```

4. Inspect metrics:

```bash
curl http://localhost:8000/metrics
```

Look for:

- `constellation_run_outcome_total`
- `app_process_cpu_seconds_total`
- `app_process_resident_memory_bytes`

5. Inspect traces in Jaeger:

```text
http://localhost:16686
```

Expected span hierarchy for a run:

- `constellation.start` (short request span)
- `constellation.run` (long background lifecycle span)
  - `constellation.run.initialize`
  - `constellation.run.execute`
    - `constellation.run.satellite.W-4`
    - `constellation.run.satellite.W-6`
    - `constellation.run.satellite.W-7`
  - `constellation.run.finalize`

## API Reference

### Endpoint summary

| Method | Path | Purpose | Typical status codes |
|---|---|---|---|
| GET | `/health` | Health + runtime environment | `200` |
| POST | `/simulations/simulate` | Run one 6-DoF simulation request | `200`, `422`, `500` |
| GET | `/simulations/live/view` | Browser UI for live telemetry | `200` |
| GET | `/simulations/live/assets/styles.css` | Live UI stylesheet | `200` |
| GET | `/simulations/live/assets/app.js` | Live UI JavaScript | `200` |
| POST | `/simulations/live/start` | Start async simulation run and return `run_id` | `200`, `422`, `500` |
| POST | `/simulations/live/constellation/start` | Start async constellation run and return `run_id` | `200`, `422`, `500` |
| POST | `/simulations/live/constellation/command` | Queue a deorbit command for a constellation vehicle | `200`, `422` |
| WS | `/simulations/live/{run_id}/ws` | Stream telemetry frames + terminal status | `101`, `404` |
| POST | `/simulations/monte-carlo` | Start async Monte Carlo batch and return `batch_id` | `200`, `422`, `500` |
| GET | `/simulations/monte-carlo` | List Monte Carlo batches | `200`, `500` |
| GET | `/simulations/monte-carlo/{batch_id}` | Poll batch status/results | `200`, `202`, `404`, `500` |
| GET | `/metrics` | Prometheus metrics snapshot | `200` |

### Canonical response behavior

- `POST /simulations/simulate` returns:
  - `message`
  - `summary` containing key orbit metrics (`final_altitude_km`, `apoapsis_altitude_km`, `periapsis_altitude_km`, `eccentricity`, etc.)
  - `full_data` only when `sim_results=full`
- `POST /simulations/live/start` returns immediately and executes the simulation in background.
- Live WebSocket emits:
  - `{"type": "telemetry", "data": ...}` while running
  - one final `{"type": "status", "data": ...}` when finished or failed
- `GET /simulations/monte-carlo/{batch_id}` returns `202` while a batch is still running.

### Example: single simulation

```bash
curl -X POST http://localhost:8000/simulations/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "target_apo_alt_km": 300,
    "target_peri_alt_km": 200,
    "sim_results": "orbital_elements_only"
  }'
```

### Example: live telemetry flow

Start run:

```bash
curl -X POST "http://localhost:8000/simulations/live/start?telemetry_interval=0.5" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Connect to stream:

```text
ws://localhost:8000/simulations/live/<run_id>/ws
```

Browser telemetry dashboard:

```text
http://localhost:8000/simulations/live/view
```

### Example: Monte Carlo batch

Kick off:

```bash
curl -X POST http://localhost:8000/simulations/monte-carlo \
  -H "Content-Type: application/json" \
  -d '{
    "num_simulations": 20,
    "base_simulation": {},
    "dispersions": {
      "stage1_base_thrust_magnitude": {"mean": 7600000, "std_dev": 100000},
      "stage1_initial_prop_mass": {"mean": 395700, "std_dev": 5000}
    }
  }'
```

Poll batch:

```bash
curl http://localhost:8000/simulations/monte-carlo/<batch_id>
```

The returned statistics include distribution summaries (mean/std/min/max/p5/p95) for altitude/eccentricity plus failure-mode aggregation.

## Deployment & Operations

### Environment variables

All runtime settings are optional and read from `.env` (or process env) with prefix `SIM_`:

- `SIM_API_TITLE` (default: `6DOF Launch Simulator`)
- `SIM_API_DESCRIPTION` (default: `Configurable launch-to-orbit simulation for mission software testing`)
- `SIM_ENVIRONMENT` (default: `dev`)
- `SIM_DEBUG` (default: `false`)
- `SIM_SIMULATOR_EXECUTOR_MAX_WORKERS` (default: `4`)
- `SIM_DB_HOST` (default: `localhost`)
- `SIM_DB_PORT` (default: `5432`)
- `SIM_DB_NAME` (default: `launch_sim`)
- `SIM_DB_USER` (default: `sim_user`)
- `SIM_DB_PASSWORD` (default: `sim_password`)

### Docker and Compose

Compose runs API + PostgreSQL + Prometheus + Jaeger with dependency health checks and a persistent named volume:

```bash
docker compose up --build
```

Prometheus expression browser:

```text
http://localhost:9090/graph
```

Jaeger trace UI:

```text
http://localhost:16686
```

Prometheus target status page:

```text
http://localhost:9090/targets
```

Standalone container workflow:

```bash
docker build -t launch-gnc-sim .
docker run --rm -p 8000:8000 --env-file .env launch-gnc-sim
```

### CI pipeline

Workflow pages:

- CI (format, lint, tests): https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/ci.yml
- Docker Build (image build/publish): https://github.com/LindstromKyle/Launch-Vehicle-GNC-Simulator/actions/workflows/docker-build.yml

Validation gates:

- Black formatting check (`black --check src tests`)
- Ruff lint check (`ruff check src tests`)
- Unit tests with Pytest (`PYTHONPATH=src pytest -v`)
- Docker image build on GitHub Actions; image push to GHCR on `main`

Branch protection requires CI + Docker workflows to pass before merge.


## Future Extensions

- Higher-fidelity re-entry aerothermodynamics and guidance tuning (heating, load envelopes, and control-surface effects)
- C++/Rust performance port of dynamics & integration core  
- Hardware-in-the-loop (HIL) interface layer  