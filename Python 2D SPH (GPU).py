"""
Python 2D SPH using CUDA GPU by Alexandre Sajus

The version without CUDA GPU is commented and is available on GitHub

More information at:
https://github.com/AlexandreSajus
https://web.archive.org/web/20090722233436/http://blog.brandonpelfrey.com/?p=303
"""

from math import sqrt, pow
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numba import vectorize

g = 0.02 * 0.25
N = 200
spacing = 0.08
k = spacing / 1000.
k_near = k * 10
rest_density = 3.
r = spacing * 1.25
sigma = 0.2
max_vel = 2.
sim_w = 0.5
bottom = 0
wall_damp = 0.05
vel_damp = 0.5
surplus = 0
bsurplus = 0
flat = 0
dam = -0.4


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


initial = start(-0.5, 0, 0.15, 0.03, N - flat) + start(-0.17, 0, 0.67, 0.03, flat)


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


# The functions that use CUDA GPU
@vectorize(["float64(float64,float64,float64,float64)"], target='cuda')
def vectq(X1, X2, Y1, Y2):
    return max(1 - sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2) / r, 0)


@vectorize(["float64(float64)"], target='cuda')
def pow2(X):
    return pow(X, 2)


@vectorize(["float64(float64)"], target='cuda')
def pow3(X):
    return pow(X, 3)


def update():
    global state, previous_state, rho, rho_near, press, press_near, neighbor, force, vel
    for i in range(N):
        previous_state[i] = state[i]
        state[i][0] += vel[i][0]
        state[i][1] += vel[i][1]
        state[i][0] += force[i][0]
        state[i][1] += force[i][1]
        visual[i] = state[i]
        force[i] = [0, -g]
        vel[i][0] = state[i][0] - previous_state[i][0]
        vel[i][1] = state[i][1] - previous_state[i][1]
        V = sqrt(vel[i][0] ** 2 + vel[i][1] ** 2)
        if V > max_vel:
            vel[i][0] = vel[i][0] * vel_damp
            vel[i][1] = vel[i][1] * vel_damp
        if state[i][0] < -sim_w:
            force[i][0] -= (state[i][0] - -sim_w) * wall_damp
            visual[i][0] = -sim_w
        if state[i][0] > dam:
            force[i][0] -= (state[i][0] - dam) * wall_damp
        if state[i][0] > sim_w:
            visual[i][0] = dam
        if state[i][1] < bottom:
            force[i][1] -= (state[i][1] - sim_w) * wall_damp
            visual[i][1] = bottom
        rho[i] = 0
        rho_near[i] = 0
        neighbor[i] = []

    # Using CUDA GPU functions to calculate lists that will be used later
    listX1 = np.asarray(np.ndarray.flatten(np.asarray([[x] * N for x in np.ndarray.tolist(state[:, 0])])))
    listX2 = np.asarray(np.ndarray.tolist(state[:, 0]) * N)
    listY1 = np.asarray(np.ndarray.flatten(np.asarray([[x] * N for x in np.ndarray.tolist(state[:, 1])])))
    listY2 = np.asarray(np.ndarray.tolist(state[:, 1]) * N)
    qlist = vectq(listX1, listX2, listY1, listY2)
    q2list = pow2(qlist)
    q3list = pow3(qlist)

    for i in range(N):
        for j in range(N):
            if i < j:
                q = qlist[i * N + j]
                if q > 0:
                    neighbor[i].append([j, q])
        rho[i] = sum(q2list[i * N:(i + 1) * N])
        rho_near[i] = sum(q3list[i * N:(i + 1) * N])

    for i in range(N):
        press[i] = k * (rho[i] - rest_density)
        press_near[i] = k_near * rho_near[i]

    for i in range(N):
        dX = [0, 0]
        ncount = len(neighbor[i])
        for x in range(ncount):
            j = neighbor[i][x][0]
            q = neighbor[i][x][1]
            rij = [state[j][0] - state[i][0], state[j][1] - state[i][1]]
            norm = sqrt((state[j][0] - state[i][0]) ** 2 + (state[j][1] - state[i][1]) ** 2)
            dm = (press[i] + press[j]) * q + (press_near[i] + press_near[j]) * q ** 2
            D = [(rij[0] / norm) * dm, (rij[1] / norm) * dm]
            dX = [dX[0] + D[0], dX[1] + D[1]]
            force[j][0] += D[0]
            force[j][1] += D[1]
        force[i][0] -= dX[0]
        force[i][1] -= dX[1]

    for i in range(N):
        ncount = len(neighbor[i])
        for x in range(ncount):
            j = neighbor[i][x][0]
            rij = [state[j][0] - state[i][0], state[j][1] - state[i][1]]
            l = sqrt((state[j][0] - state[i][0]) ** 2 + (state[j][1] - state[i][1]) ** 2)
            q = l / r
            rijn = [rij[0] / l, rij[1] / l]
            u = (vel[i][0] - vel[j][0]) * rijn[0] + (vel[i][1] - vel[j][1]) * rijn[1]
            if u > 0:
                I = [(1 - q) * (sigma * u) * rijn[0], (1 - q) * (sigma * u) * rijn[1]]
                vel[i][0] -= I[0] * 0.5
                vel[i][1] -= I[1] * 0.5
                vel[j][0] += I[0] * 0.5
                vel[j][1] += I[1] * 0.5


fig = plt.figure()
axes = fig.add_subplot(xlim=(-sim_w - surplus, sim_w + surplus), ylim=(bottom - bsurplus, 2 * sim_w))
points, = axes.plot([], [], 'bo', ms=20)
frame = 0


def animate(i):
    global state, frame, dam
    frame += 1
    if frame == 250:
        dam = 0.5
    update()
    points.set_data(visual[:, 0], visual[:, 1])
    return points,


ani = animation.FuncAnimation(fig, animate, interval=10, blit=True)
plt.show()
