# Rocket Ascent Simulator

![Banner](readme_imgs/earth.png)

*A complete Python-based 6-DoF simulation of a two-stage launch vehicle from liftoff to stable low-Earth orbit.*

## Table of Contents

- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Features & Technical Highlights](#features--technical-highlights)
- [Results & Visualizations](#results--visualizations)
- [Future Extensions](#future-extensions)

## Overview

This project is a modular, physics-based 6-DoF rocket ascent simulator that models the full flight from vertical liftoff through staging, vacuum guidance, coast, and orbital insertion. It includes:

- Rigid-body dynamics with quaternion propagation  
- Multi-engine thrust vector control with torque allocation  
- Powered Explicit Guidance (PEG) for precise orbit targeting  
- Realistic environment (J2 gravity, rotating Earth, AoA-dependent drag)  
- Phase-based mission sequencing with automatic transitions  

The simulation successfully reaches user-defined orbits (e.g. 200 × 300 km) using realistic control and guidance laws.

## Key Achievements

- Implemented full 6-DoF mechanics with dynamic center of mass, multi-engine gimballing, and RCS thrusters  
- Built modular mission planner with clean phase transitions (Initial Ascent → Pitch Program → PEG → Coast → Circularization)  
- Achieved stable orbital insertion with tight apoapsis/periapsis tolerances under gravity, drag, and J2 perturbations  
- Created quality 3D trajectory visualizations segmented by mission phase  
- Maintained clean, object-oriented architecture suitable for extension (Monte Carlo, C++ port, Hardware-in-the-loop, etc.)

## Technical Highlights

- **Dynamics**  
  - Quaternion-based attitude with angular velocity propagation  
  - Least-squares gimbal allocation and RCS assist  
  - Dynamic center-of-mass & gimbal arm length from propellant depletion 
  - RK4 integration with normalized quaternions  

- **Guidance & Control**  
  - Powered Explicit Guidance (PEG) solving real-time burn direction & throttle  
  - Programmed pitch maneuver for initial gravity turn 
  - PID attitude controller with quaternion error, gain scheduling, anti-windup  

- **Environment**  
  - Newtonian + J2 oblateness gravity  
  - Rotating Earth with coriolis wind in atmosphere  
  - Exponential atmosphere with angle-of-attack dependent drag 

- **Mission Sequencing**  
  - Phase objects (TimeBased, Kick, PEG, Coast, CircBurn, etc.)  
  - Automatic completion checks (time, apoapsis reached, periapsis condition)  
  - Seamless hand-over between stages  

## Results & Visualizations

The following plots serve as strong visual proof of realism and performance.

### 3D Trajectory by Mission Phase + Earth Reference

![3D Trajectory with Phase Overlays and Earth](images/3d_trajectory_phases.png)

### Altitude, Velocity & Acceleration Profiles

![Altitude and Velocity vs Time](images/altitude_velocity_profiles.png)

### Pitch Angle & Guidance Mode Evolution

![Pitch Angle and Attitude Error vs Time](images/pitch_profile.png)

### Orbital Elements Convergence (Apoapsis / Periapsis Targeting)

![Orbital Elements Evolution](images/orbital_elements_convergence.png)

**Typical final orbit** (example targets 200 × 300 km):  
- Achieved apoapsis altitude: ~299–301 km  
- Achieved periapsis altitude: ~199–202 km  
- Final inclination close to launch latitude (small out-of-plane steering demonstrated)

## Future Extensions

- Monte Carlo framework for dispersion analysis  
- Boost-back, re-entry, and landing burns with added control surfaces
- C++/Rust performance port of dynamics & integration core  
- Hardware-in-the-loop (HIL) interface layer  
- Real-time visualization dashboard