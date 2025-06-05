import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import measure

# Idea for project: do the moons of Jupiter with an animation

# In this test I want to create a bunch of point-bodies in 3d space and simulate their
# gravitational interactions. This should produce a cloud of points with some flying
# off outside of the simulation bounds.

# Steps
#
# * [X] Decide on units.
# * [X] Generate some points in 3D space and plot them.
# * [X] Give the points velocity and have them move in space.
# * [X] Give the points some mass.
# * [X] Simulate gravitational interaction between the bodies. (Very naively!)
# * [ ] Clean up the simulation code (remove Earth/Sun sanity check or move to another file).

SIMULATION_RADIUS = measure.astronomical_unit * 2
MASS_MIN = measure.mass_sun / 200
MASS_MAX = measure.mass_sun / 5
NUM_BODIES = 100

TIME_STEP = 60.0 * 60.0 * 24.0 # seconds

body_locations = np.random.uniform(low=-SIMULATION_RADIUS, high=SIMULATION_RADIUS, size=(NUM_BODIES, 3))
body_locations[:, 2] /= 50.0
body_mass = np.random.normal(loc=measure.mass_neptune, scale=measure.mass_mercury, size=NUM_BODIES)
body_velocities = np.zeros_like(body_locations) # np.random.uniform(low=-VELOCITY_MAX, high=VELOCITY_MAX, size=(NUM_BODIES, 3))


# Put the sun in the center of the simulation.
body_locations[0] = [0.0, 0.0, 0.0]
body_mass[0] = measure.mass_sun
body_velocities[0] = [0.0, 0.0, 0.0]

# This assumes body 0 is the sun and it finds the velocity for a stable enough orbit
def calculate_initial_velocity(body_no):
    global body_locations
    global body_mass

    if body_no == 0:
        return np.array([0.0, 0.0, 0.0])

    location = body_locations[0]
    location2 = body_locations[body_no]
    delta_location = location2 - location
    r = np.sqrt((delta_location * delta_location).sum())
    normal_vector = delta_location / r
    down_vector = np.array([0.0, 0.0, -1.0])
    r = np.sqrt((delta_location * delta_location).sum())
    velocity_to_preserve_orbit = np.sqrt(measure.gravitational_constant * body_mass[0] / r)
    orbit_preserving_velocity_vector = np.cross(normal_vector, down_vector) * velocity_to_preserve_orbit
    return orbit_preserving_velocity_vector

for i in range(NUM_BODIES):
    body_velocities[i] = calculate_initial_velocity(i)

# Not a great implementation for this.
def calculate_acceleration_vector(body_no):
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
        accel_scalar = measure.gravitational_constant * body_mass[i] / r / r
        accel_vector = normal_vector * accel_scalar
        acceleration += accel_vector
    return acceleration

fig = plt.figure()

try:
    fig.canvas.manager.window.wm_geometry("+100+100")
    fig.canvas.manager.window.geometry("640x480")
except AttributeError:
    print("Your backend may not support window resizing")

ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-SIMULATION_RADIUS, SIMULATION_RADIUS)
ax.set_ylim(-SIMULATION_RADIUS, SIMULATION_RADIUS)
ax.set_zlim(-SIMULATION_RADIUS, SIMULATION_RADIUS)

sc = ax.scatter(body_locations[:, 0], body_locations[:, 1], body_locations[:, 2], alpha=0.6)

ax.set_title("Sample Orbits Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def update(frame):
    global body_locations
    global body_velocities

    # Not a great simulation.
    accelerations = np.stack([calculate_acceleration_vector(i) for i in range(body_locations.shape[0])])
    body_velocities += accelerations * TIME_STEP
    body_locations += body_velocities * TIME_STEP

    sc._offsets3d = (body_locations[:, 0], body_locations[:, 1], body_locations[:, 2])
    return sc,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

ani.save("output_sample_orbits.gif", writer="pillow", fps=30)
