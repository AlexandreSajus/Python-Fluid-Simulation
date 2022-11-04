"""Utilities and physics calculations"""

from math import sqrt

from pythonsph.config import Config
from pythonsph.particle import Particle


(
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
) = Config().return_config()


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
