"""
Python 2D SPH by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus
https://web.archive.org/web/20090722233436/http://blog.brandonpelfrey.com/?p=303
"""

from math import sqrt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

g = 0.02 * 0.25  # Acceleration of gravity
N = 200  # Number of particles
spacing = 0.08  # Spacing of particles
k = spacing / 1000.  # Pressure weight
k_near = k * 10  # Near pressure weight
rest_density = 3.  # Rest density
r = spacing * 1.25  # Radius for neighboring particles
sigma = 0.2  # Viscosity weight
max_vel = 2.  # Maximum velocity
sim_w = 0.5  # Simulation width
bottom = 0  # Simulation ground
wall_damp = 0.05  # Wall constraints weight
vel_damp = 0.5  # Maximum velocity reduction weight
dam = -0.4  # Position of the dam


# This is used to create a rectangle of particles
def start(xlim, ylim, xspray, space, count):
    result = []
    x, y = xlim, ylim
    for k in range(count):
        result.append([x, y])
        x += space
        if x > xlim + xspray:
            x = xlim
            y += space
    return result


# Initial dam break conditions
initial = start(-0.5, 0, 0.15, 0.03, N) + start(-0.17, 0, 0.67, 0.03, 0)


# Initial conditions
def init():
    state = initial
    previous_state = initial
    visual = initial
    rho = []
    rho_near = []
    press = []
    press_near = []
    neighbor = []
    force = []
    vel = []
    for i in range(N):
        rho.append(0.)
        rho_near.append(0.)
        press.append(0.)
        press_near.append(0.)
        neighbor.append([])
        force.append([0., 0.])
        vel.append([0., 0.])
    return np.asarray(state), np.asarray(previous_state), np.asarray(
        visual), rho, rho_near, press, press_near, neighbor, force, vel


state, previous_state, visual, rho, rho_near, press, press_near, neighbor, force, vel = init()


# Calculates a step of the simulation
def update():
    global state, previous_state, rho, rho_near, press, press_near, neighbor, force, vel
    for i in range(N):
        previous_state[i] = state[i]  # Verlet integration with dt = 1
        state[i][0] += vel[i][0]
        state[i][1] += vel[i][1]
        state[i][0] += force[i][0]
        state[i][1] += force[i][1]
        visual[i] = state[i]
        force[i] = [0, -g]
        vel[i][0] = state[i][0] - previous_state[i][0]
        vel[i][1] = state[i][1] - previous_state[i][1]
        V = sqrt(vel[i][0] ** 2 + vel[i][1] ** 2)
        if V > max_vel:  # Reduces the velocity if it is too high
            vel[i][0] = vel[i][0] * vel_damp
            vel[i][1] = vel[i][1] * vel_damp
        if state[i][0] < -sim_w:  # Wall constraints (springs)
            force[i][0] -= (state[i][0] - -sim_w) * wall_damp
            visual[i][0] = -sim_w
        if state[i][0] > dam:
            force[i][0] -= (state[i][0] - dam) * wall_damp
        if state[i][0] > sim_w:  # This creates a wall that simulates a dam
            visual[i][0] = dam
        if state[i][1] < bottom:
            force[i][1] -= (state[i][1] - sim_w) * wall_damp
            visual[i][1] = bottom
        rho[i] = 0
        rho_near[i] = 0
        neighbor[i] = []

    """
    This calculates the density of particles by summing the relative distances to neighboring particles
    It also spots neighboring particles for later
    """
    for i in range(N):
        d = 0.
        dn = 0.
        for j in range(N):
            if i < j:
                dist = sqrt((state[i][0] - state[j][0]) ** 2 + (state[i][1] - state[j][1]) ** 2)
                if dist < r:  # if the particle is a neighbor
                    length = dist
                    q = 1 - length / r  # q = 1 if the neighbor is on the particle, 0 if it is r far
                    d += q ** 2
                    dn += q ** 3
                    rho[j] += q ** 2
                    rho_near[j] += q ** 3
                    neighbor[i].append([j, q])  # writes down which particles are neighbors
        rho[i] += d  # density used for later
        rho_near[i] += dn  # near density used for later

    # This calculates pressure, a spring force where the length of the spring is the difference between density and rest density
    for i in range(N):
        press[i] = k * (rho[i] - rest_density)
        press_near[i] = k_near * rho_near[i]

    # This creates the pressure force
    for i in range(N):
        dX = [0, 0]
        ncount = len(neighbor[i])
        for x in range(ncount):  # for each neighbor
            j = neighbor[i][x][0]
            q = neighbor[i][x][1]
            rij = [state[j][0] - state[i][0], state[j][1] - state[i][1]]  # rij is the vector between the two particles
            norm = sqrt((state[j][0] - state[i][0]) ** 2 + (state[j][1] - state[i][1]) ** 2)  # norm is the norm of rij
            dm = (press[i] + press[j]) * q + (press_near[i] + press_near[j]) * q ** 2  # dm is the total pressure weight
            D = [(rij[0] / norm) * dm, (rij[1] / norm) * dm]  # D is the vector of pressure caused by one particle
            dX = [dX[0] + D[0], dX[1] + D[1]]  # dX is the vector of total pressure
            force[j][0] += D[0]
            force[j][1] += D[1]
        force[i][0] -= dX[0]
        force[i][1] -= dX[1]

    """
    This calculates the viscosity force: F = (relative distance of particles (1 - l/r))*(viscosity weight sigma)*(the difference of velocities projected onto the vector between the two particles)
    """
    for i in range(N):
        ncount = len(neighbor[i])
        for x in range(ncount):
            j = neighbor[i][x][0]
            rij = [state[j][0] - state[i][0], state[j][1] - state[i][1]]
            l = sqrt((state[j][0] - state[i][0]) ** 2 + (state[j][1] - state[i][1]) ** 2)
            q = l / r
            rijn = [rij[0] / l, rij[1] / l]  # rijn is the same as before
            u = (vel[i][0] - vel[j][0]) * rijn[0] + (vel[i][1] - vel[j][1]) * rijn[
                1]  # u is the difference of the velocities of the two particles projected onto the vector between the two particles
            if u > 0:  # u > 0 to compute the forces once
                I = [(1 - q) * (sigma * u) * rijn[0],  # Calculates the force
                     (1 - q) * (sigma * u) * rijn[1]]
                vel[i][0] -= I[0] * 0.5
                vel[i][1] -= I[1] * 0.5
                vel[j][0] += I[0] * 0.5
                vel[j][1] += I[1] * 0.5


# Setup matplotlib
fig = plt.figure()
axes = fig.add_subplot(xlim=(-sim_w, sim_w), ylim=(bottom, 2 * sim_w))
points, = axes.plot([], [], 'bo', ms=20)
frame = 0


# Animation function
def animate(i):
    global state, frame, dam
    frame += 1
    if frame == 250:  # Break the dam at frame 250
        dam = 0.5
    update()
    points.set_data(visual[:, 0], visual[:, 1])  # Updates the position of the particles
    return points,


ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)
plt.show()
