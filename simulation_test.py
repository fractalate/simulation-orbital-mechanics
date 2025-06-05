import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Idea for project: do the moons of Jupiter with an animation

# In this test I want to create a bunch of point-bodies in 3d space and simulate their
# gravitational interactions. This should produce a cloud of points with some flying
# off outside of the simulation bounds.

# Masses of Solar Bodies
#
# Sun       1.989e30 kg
# Mercury   3.301e23 kg
# Venus     4.867e24 kg
# Earth     5.972e24 kg
# Mars      6.417e23 kg
# Jupiter   1.898e27 kg
# Saturn    5.683e26 kg
# Uranus    8.681e25 kg
# Neptune   1.024e26 kg

# Distance from Sun
#
# Earth     1.495979e11 m

# Steps
#
# * [X] Decide on units.
# * [X] Generate some points in 3D space and plot them.
# * [X] Give the points velocity and have them move in space.
# * [X] Give the points some mass.
# * [X] Simulate gravitational interaction between the bodies. (Very naively!)
# * [ ] Clean up the simulation code (remove Earth/Sun sanity check or move to another file).

AU_METERS=1.495979e+11
SOL_MASS_KG=1.989e30
EARTH_MASS_KG=5.972e24
GRAVITATIONAL_CONSTANT = 6.67430e-11 # m^3 / kg / s^2

SIMULATION_RADIUS = AU_METERS * 2
VELOCITY_MAX = SIMULATION_RADIUS / 1000000
MASS_MIN = SOL_MASS_KG / 100
MASS_MAX = SOL_MASS_KG * 10
NUM_BODIES = 2

TIME_STEP = 60.0 * 60.0 * 24.0 # seconds

body_locations = np.random.uniform(low=-SIMULATION_RADIUS, high=SIMULATION_RADIUS, size=(NUM_BODIES, 3))
body_locations[0] = [0.0, 0.0, 0.0]
body_locations[1] = [AU_METERS, 0.0, 0.0]
body_mass = np.random.uniform(low=MASS_MIN, high=MASS_MAX, size=NUM_BODIES)
body_mass[0] = SOL_MASS_KG
body_mass[1] = EARTH_MASS_KG
body_velocities = np.random.uniform(low=-VELOCITY_MAX, high=VELOCITY_MAX, size=(NUM_BODIES, 3))
body_velocities[0] = [0.0, 0.0, 0.0]
body_velocities[1] = [0.0, 29.78e3, 0.0] # earth's velocity

def calcuate_acceleration_vector(body_no):
    global body_locations
    global body_mass

    acceleration = np.array([0.0, 0.0, 0.0])
    location = body_locations[body_no]
    for i in range(NUM_BODIES):
        if i == body_no:
            continue
        location2 = body_locations[i]
        delta_location = location2 - location
        r = np.sqrt((delta_location * delta_location).sum())
        normal_vector = delta_location / r
        accel_scalar = GRAVITATIONAL_CONSTANT * body_mass[i] / r / r
        accel_vector = normal_vector * accel_scalar
        acceleration += accel_vector
    return acceleration

fig = plt.figure()

try:
    fig.canvas.manager.window.wm_geometry("+100+100")
    fig.canvas.manager.window.geometry("800x600")
except AttributeError:
    print("Your backend may not support window resizing")

ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-SIMULATION_RADIUS, SIMULATION_RADIUS)
ax.set_ylim(-SIMULATION_RADIUS, SIMULATION_RADIUS)
ax.set_zlim(-SIMULATION_RADIUS, SIMULATION_RADIUS)

sc = ax.scatter(body_locations[:, 0], body_locations[:, 1], body_locations[:, 2], alpha=0.6)

ax.set_title("3D Scatter Plot of Uniformly Distributed Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def update(frame):
    global body_locations
    global body_velocities

    accelerations = np.stack([calcuate_acceleration_vector(i) for i in range(body_locations.shape[0])])
    body_velocities += accelerations * TIME_STEP
    body_locations += body_velocities * TIME_STEP

    sc._offsets3d = (body_locations[:, 0], body_locations[:, 1], body_locations[:, 2])
    return sc,

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

plt.show()
