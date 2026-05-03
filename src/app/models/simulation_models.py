from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class SimResults(str, Enum):
    full = "full"
    orbital_elements_only = "orbital_elements_only"


class SimulationRequest(BaseModel):
    """
    Full configuration for the two-stage orbital simulation.
    """

    # ==================== SIMULATION SETTINGS ====================
    delta_t_stage1: float = Field(
        0.1, gt=0, description="Time step for Stage 1 integration (s)"
    )
    delta_t_stage2: float = Field(
        0.2, gt=0, description="Time step for Stage 2 integration (s)"
    )
    log_interval: float = Field(5.0, gt=0, description="How often to log data (s)")
    downsample: int = Field(
        100, ge=1, description="Downsample factor for returned trajectory"
    )

    # ==================== LAUNCH SITE ====================
    launch_latitude_deg: float = Field(
        28.5, ge=-90, le=90, description="Launch latitude (deg)"
    )

    # ==================== STAGE 1 VEHICLE (Falcon9FirstStage) ====================
    stage1_dry_mass: float = Field(167100, gt=0)
    stage1_initial_prop_mass: float = Field(395700, gt=0)
    stage1_base_thrust_magnitude: float = Field(7600000, gt=0)
    stage1_average_isp: float = Field(300, gt=0)
    stage1_moment_of_inertia: List[float] = Field(
        [480297, 480297, 725445],
        min_length=3,
        max_length=3,
        description="Ixx, Iyy, Izz",
    )
    stage1_base_drag_coefficient: float = Field(0.3, ge=0)
    stage1_drag_scaling_coefficient: float = Field(0.2, ge=0)
    stage1_cross_sectional_area: float = Field(10.5, gt=0)
    stage1_engine_gimbal_limit_deg: float = Field(10.0, gt=0)
    stage1_engine_gimbal_arm_len: float = Field(20.0, gt=0)
    stage1_dry_com_z: float = Field(15.0, gt=0)
    stage1_prop_com_z: float = Field(20.0, gt=0)

    # ==================== STAGE 1 GUIDANCE & PHASES ====================
    stage1_pitch_start_time: float = Field(10.0, ge=0)
    stage1_burnout_time: float = Field(155.0, gt=0)
    stage1_initial_pitch_deg: float = Field(90.0)
    stage1_final_pitch_deg: float = Field(23.1)

    # ==================== STAGE 1 PID GAINS ====================
    stage1_kp: List[float] = Field([6000, 6000, 9000], min_length=3, max_length=3)
    stage1_ki: List[float] = Field([0.1, 0.1, 0.15], min_length=3, max_length=3)
    stage1_kd: List[float] = Field([200000, 200000, 300000], min_length=3, max_length=3)

    # ==================== STAGE 2 VEHICLE (Falcon9SecondStage) ====================
    stage2_dry_mass: float = Field(4000, gt=0)
    stage2_initial_prop_mass: float = Field(111500, gt=0)
    stage2_base_thrust_magnitude: float = Field(934000, gt=0)
    stage2_average_isp: float = Field(348, gt=0)
    stage2_moment_of_inertia: List[float] = Field(
        [10000, 10000, 20000], min_length=3, max_length=3
    )
    stage2_base_drag_coefficient: float = Field(0.3, ge=0)
    stage2_drag_scaling_coefficient: float = Field(2.0, ge=0)
    stage2_cross_sectional_area: float = Field(7.0, gt=0)
    stage2_engine_gimbal_limit_deg: float = Field(5.0, gt=0)
    stage2_engine_gimbal_arm_len: float = Field(2.0, gt=0)
    stage2_dry_com_z: float = Field(3.0, gt=0)
    stage2_prop_com_z: float = Field(6.0, gt=0)

    # ==================== STAGE 2 TARGET ORBIT ====================
    target_apo_alt_km: float = Field(300.0, gt=0)
    target_peri_alt_km: float = Field(200.0, gt=0)

    # ==================== STAGE 2 GUIDANCE PARAMETERS ====================
    peg_apo_tolerance: float = Field(20000.0, gt=0)
    peg_peri_tolerance: float = Field(20000.0, gt=0)
    circ_target_eccentricity: float = Field(0.0011, gt=0)

    # ==================== STAGE 2 PID GAINS ====================
    stage2_kp: List[float] = Field([3e3, 3e3, 6e3], min_length=3, max_length=3)
    stage2_ki: List[float] = Field([0, 0, 0], min_length=3, max_length=3)
    stage2_kd: List[float] = Field([2e4, 2e4, 4e4], min_length=3, max_length=3)

    # Simulation results
    sim_results: SimResults = SimResults.orbital_elements_only


class SimulationResponse(BaseModel):
    message: str
    summary: Dict[str, float]
    full_data: Dict[str, List[Any]]


class ParameterDispersion(BaseModel):
    """
    Specifies mean and standard deviation for a dispersed parameter.
    """

    mean: float
    std_dev: float


class MonteCarloRequest(BaseModel):
    """
    Configuration for Monte Carlo dispersion analysis.
    """

    num_simulations: int = Field(
        10,
        ge=1,
        le=500,
        description="Number of individual simulations in this Monte Carlo batch",
    )

    base_simulation: SimulationRequest = Field(description="Base simulation parameters")

    dispersions: Dict[str, ParameterDispersion] = Field(
        default_factory=dict,
        description="Parameter dispersions - empty dict means no dispersions",
    )

    def simulation_count(self) -> int:
        """Return requested simulation count"""
        return self.num_simulations


class MonteCarloBatchResponse(BaseModel):
    """
    Results summary from a Monte Carlo analysis batch.
    """

    batch_id: str
    created_at: str
    status: str
    total_simulations: int
    completed_simulations: int
    failed_simulations: int
    success_rate: float
    statistics: Dict[str, Any]


class MonteCarloKickoffResponse(BaseModel):
    """
    Confirmation response for a Monte Carlo batch that has just been started.
    """

    batch_id: str
    created_at: str
    status: str
    total_simulations: int


class LiveSimulationStartResponse(BaseModel):
    run_id: str
    status: str
    created_at: str


class DeorbitCommandRequest(BaseModel):
    run_id: str = Field(min_length=1)
    vehicle_id: str = Field(min_length=1)
    action: Literal["deorbit_burn"]
    execute_at_sim_time_s: float = Field(
        ge=0,
        description="Simulation time when command should execute",
    )
    target_perigee_alt_km: float = Field(
        gt=0,
        le=200,
        description="Requested target perigee altitude in kilometers",
    )


class CommandUploadResponse(BaseModel):
    command_id: str
    status: Literal["accepted", "rejected"]
    server_received_at: str
    reason: str | None = None
