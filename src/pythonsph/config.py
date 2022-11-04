"""Configuration file defining the simulation parameters."""

# Simulation parameters
N = 250  # Number of particles
SIM_W = 0.5  # Simulation space width
BOTTOM = 0  # Simulation space ground
DAM = -0.3  # Position of the dam, simulation space is between -0.5 and 0.5
DAM_BREAK = 200  # Number of frames before the dam breaks

# Physics parameters
G = 0.02 * 0.25  # Acceleration of gravity
SPACING = 0.08  # Spacing between particles, used to calculate pressure
K = SPACING / 1000.0  # Pressure factor
K_NEAR = K * 10  # Near pressure factor, pressure when particles are close to each other
# Default density, will be compared to local density to calculate pressure
REST_DENSITY = 3.0
# Neighbour radius, if the distance between two particles is less than R, they are neighbours
R = SPACING * 1.25
SIGMA = 0.2  # Viscosity factor
MAX_VEL = 2.0  # Maximum velocity of particles, used to avoid instability
# Wall constraints factor, how much the particle is pushed away from the simulation walls
WALL_DAMP = 0.05
VEL_DAMP = 0.5  # Velocity reduction factor when particles are going above MAX_VEL


class Config:
    """Contains the simulation parameters and the physics parameters."""

    def __init__(self):
        return None

    def return_config(self):
        """Returns the simulation parameters and the physics parameters."""
        return (
            N,
            SIM_W,
            BOTTOM,
            DAM,
            DAM_BREAK,
            G,
            SPACING,
            K,
            K_NEAR,
            REST_DENSITY,
            R,
            SIGMA,
            MAX_VEL,
            WALL_DAMP,
            VEL_DAMP,
        )
