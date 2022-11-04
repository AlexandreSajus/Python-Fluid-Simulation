"""Defines the Particle class."""

from math import sqrt
from pythonsph.config import Config


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

    def update_state(self, dam: bool):
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

        # Same thing for the right wall
        if self.x_pos > SIM_W:
            self.x_force -= (self.x_pos - SIM_W) * WALL_DAMP
            self.visual_x_pos = SIM_W

        # Same thing but for the floor
        if self.y_pos < BOTTOM:
            # We use SIM_W instead of BOTTOM here because otherwise particles are too low
            self.y_force -= (self.y_pos - SIM_W) * WALL_DAMP
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
