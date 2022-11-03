"""
Python 2D SPH by Alexandre Sajus

This script generates a 2D animation of a dam break using the Smoothed Particle Hydrodynamics method

More information at:
https://github.com/AlexandreSajus
https://web.archive.org/web/20090722233436/http://blog.brandonpelfrey.com/?p=303
"""

from math import sqrt
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

# Simulation parameters
N = 200  # Number of particles
SIM_W = 0.5  # Simulation space width
BOTTOM = 0  # Simulation space ground
DAM = -0.2  # Position of the dam, simulation space is between -0.5 and 0.5

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


class Particle:
    """
    A single particle of the simulated fluid

    Attributes:
        x_pos: x position of the particle
        y_pos: y position of the particle
        previous_x_pos: x position of the particle in the previous frame
        previous_y_pos: y position of the particle in the previous frame
        visual_x_pos: x position of the particle that is shown on the screen
        visual_y_pos: y position of the particle that is shown on the screen
        rho: density of the particle
        rho_near: near density of the particle, used to avoid collisions between particles
        press: pressure of the particle
        press_near: near pressure of the particle, used to avoid collisions between particles
        neighbors: list of the particle's neighbors
        x_vel: x velocity of the particle
        y_vel: y velocity of the particle
        x_force: x force applied to the particle
        y_force: y force applied to the particle
    """

    def __init__(self, x_pos: float, y_pos: float):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.previous_x_pos = x_pos
        self.previous_y_pos = y_pos
        self.visual_x_pos = x_pos
        self.visual_y_pos = y_pos
        self.rho = 0.0
        self.rho_near = 0.0
        self.press = 0.0
        self.press_near = 0.0
        self.neighbors = []
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.x_force = 0.0
        self.y_force = -G

    def update_state(self, dam: bool = True):
        """
        Updates the state of the particle
        """
        # Reset previous position
        (self.previous_x_pos, self.previous_y_pos) = (self.x_pos, self.y_pos)

        # Apply force using Newton's second law and Euler integration with mass = 1 and dt = 1
        (self.x_vel, self.y_vel) = (
            self.x_vel + self.x_force,
            self.y_vel + self.y_force,
        )

        # Move particle according to its velocity using Euler integration with dt = 1
        (self.x_pos, self.y_pos) = (self.x_pos + self.x_vel, self.y_pos + self.y_vel)

        # Set visual position. Visual position is the one shown on the screen
        # It is used to avoid unstable particles to be shown
        (self.visual_x_pos, self.visual_y_pos) = (self.x_pos, self.y_pos)

        # Reset force
        (self.x_force, self.y_force) = (0.0, -G)

        # Define velocity using Euler integration with dt = 1
        (self.x_vel, self.y_vel) = (
            self.x_pos - self.previous_x_pos,
            self.y_pos - self.previous_y_pos,
        )

        # Calculate velocity
        velocity = sqrt(self.x_vel**2 + self.y_vel**2)

        # Reduces the velocity if it is too high
        if velocity > MAX_VEL:
            self.x_vel *= VEL_DAMP
            self.y_vel *= VEL_DAMP

        # Wall constraints, if a particle is out of bounds, create a spring force to bring it back
        if self.x_pos < -SIM_W:
            self.x_force -= (self.x_pos - -SIM_W) * WALL_DAMP
            self.visual_x_pos = -SIM_W

        # Same thing as a wall constraint but for the dam that will move from dam to SIM_W
        if dam is True and self.x_pos > DAM:
            self.x_force -= (self.x_pos - DAM) * WALL_DAMP
            self.visual_x_pos = DAM

        # Same thing for the right wall
        if self.x_pos > SIM_W:
            self.x_force -= (self.x_pos - SIM_W) * WALL_DAMP
            self.visual_x_pos = SIM_W

        # Same thing but for the floor
        if self.y_pos < BOTTOM:
            self.y_force -= (self.y_pos - BOTTOM) * WALL_DAMP
            self.visual_y_pos = BOTTOM

        # Reset density
        self.rho = 0.0
        self.rho_near = 0.0

        # Reset neighbors
        self.neighbors = []

    def calculate_pressure(self):
        """
        Calculates the pressure of the particle
        """
        self.press = K * (self.rho - REST_DENSITY)
        self.press_near = K_NEAR * self.rho_near


# This is used to create a rectangle of particles
def start(
    xmin: float, xmax: float, ymin: float, space: float, count: int
) -> list[Particle]:
    """
    Creates a rectangle of particles within xmin, xmax and ymin
    We start by creating a particle at (xmin, ymin)
    and then add particles until we reach count particles
    Particles are represented by their position [x, y]

    Args:
        xmin (float): x min bound of the rectangle
        xmax (float): x max bound of the rectangle
        ymin (float): y min bound of the rectangle
        space (float): space between particles
        count (int): number of particles

    Returns:
        list: list of Particle objects
    """
    result = []
    x_pos, y_pos = xmin, ymin
    for _ in range(count):
        result.append(Particle(x_pos, y_pos))
        x_pos += space
        if x_pos > xmax:
            x_pos = xmin
            y_pos += space
    return result


def calculate_density(particles: list[Particle]) -> None:
    """
    Calculates density of particles
        Density is calculated by summing the relative distance of neighboring particles
        We distinguish density and near density to avoid particles to collide with each other
        which creates instability

    Args:
        particles (list[Particle]): list of particles
    """
    for i, particle_1 in enumerate(particles):
        density = 0.0
        density_near = 0.0
        # Density is calculated by summing the relative distance of neighboring particles
        for particle_2 in particles[i + 1 :]:
            distance = sqrt(
                (particle_1.x_pos - particle_2.x_pos) ** 2
                + (particle_1.y_pos - particle_2.y_pos) ** 2
            )
            if distance < R:
                # normal distance is between 0 and 1
                normal_distance = 1 - distance / R
                density += normal_distance**2
                density_near += normal_distance**3
                particle_2.rho += normal_distance**2
                particle_2.rho_near += normal_distance**3
                particle_1.neighbors.append(particle_2)
        particle_1.rho += density
        particle_1.rho_near += density_near


def create_pressure(particles: list[Particle]) -> None:
    """
    Calculates pressure force of particles
        Neighbors list and pressure have already been calculated by calculate_density
        We calculate the pressure force by summing the pressure force of each neighbor
        and apply it in the direction of the neighbor

    Args:
        particles (list[Particle]): list of particles
    """
    for particle in particles:
        press_x = 0.0
        press_y = 0.0
        for neighbor in particle.neighbors:
            particle_to_neighbor = [
                neighbor.x_pos - particle.x_pos,
                neighbor.y_pos - particle.y_pos,
            ]
            distance = sqrt(particle_to_neighbor[0] ** 2 + particle_to_neighbor[1] ** 2)
            normal_distance = 1 - distance / R
            total_pressure = (
                particle.press + neighbor.press
            ) * normal_distance**2 + (
                particle.press_near + neighbor.press_near
            ) * normal_distance**3
            pressure_vector = [
                particle_to_neighbor[0] * total_pressure / distance,
                particle_to_neighbor[1] * total_pressure / distance,
            ]
            neighbor.x_force += pressure_vector[0]
            neighbor.y_force += pressure_vector[1]
            press_x += pressure_vector[0]
            press_y += pressure_vector[1]
        particle.x_force -= press_x
        particle.y_force -= press_y


def calculate_viscosity(particles: list[Particle]) -> None:
    """
    Calculates the viscosity force of particles
    Force = (relative distance of particles)*(viscosity weight)*(velocity difference of particles)
    Velocity difference is calculated on the vector between the particles

    Args:
        particles (list[Particle]): list of particles
    """

    for particle in particles:
        for neighbor in particle.neighbors:
            particle_to_neighbor = [
                neighbor.x_pos - particle.x_pos,
                neighbor.y_pos - particle.y_pos,
            ]
            distance = sqrt(particle_to_neighbor[0] ** 2 + particle_to_neighbor[1] ** 2)
            normal_p_to_n = [
                particle_to_neighbor[0] / distance,
                particle_to_neighbor[1] / distance,
            ]
            relative_distance = distance / R
            velocity_difference = (particle.x_vel - neighbor.x_vel) * normal_p_to_n[
                0
            ] + (particle.y_vel - neighbor.y_vel) * normal_p_to_n[1]
            if velocity_difference > 0:
                viscosity_force = [
                    (1 - relative_distance)
                    * SIGMA
                    * velocity_difference
                    * normal_p_to_n[0],
                    (1 - relative_distance)
                    * SIGMA
                    * velocity_difference
                    * normal_p_to_n[1],
                ]
                particle.x_vel -= viscosity_force[0] * 0.5
                particle.y_vel -= viscosity_force[1] * 0.5
                neighbor.x_vel += viscosity_force[0] * 0.5
                neighbor.y_vel += viscosity_force[1] * 0.5


def update(particles: list[Particle], dam: bool) -> list[Particle]:
    """
    Calculates a step of the simulation
    """
    # Update the state of the particles (apply forces, reset values, etc.)
    for particle in particles:
        particle.update_state(dam)

    # Calculate density
    calculate_density(particles)

    # Calculate pressure
    for particle in particles:
        particle.calculate_pressure()

    # Apply pressure force
    create_pressure(particles)

    # Apply viscosity force
    calculate_viscosity(particles)

    return particles


# Setup matplotlib
fig = plt.figure()
axes = fig.add_subplot(xlim=(-SIM_W, SIM_W), ylim=(BOTTOM, 2 * SIM_W))
(POINTS,) = axes.plot([], [], "bo", ms=20)

simulation_state = start(-SIM_W, DAM, BOTTOM, 0.03, N)

# Animation function
def animate(i: int):
    """
    Animates the simulation in matplotlib

    Args:
        i: frame number

    Returns:
        points: the points to be plotted
    """
    global simulation_state
    dam = True
    if i == 250:  # Break the dam at frame 250
        dam = False
    simulation_state = update(simulation_state, dam)
    # Create an array with the x and y coordinates of the particles
    visual = np.array(
        [[particle.x_pos, particle.y_pos] for particle in simulation_state]
    )
    POINTS.set_data(visual[:, 0], visual[:, 1])  # Updates the position of the particles
    i += 1
    return (POINTS,)


ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)
plt.show()
