import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import measure

SIMULATION_RADIUS = measure.astronomical_unit * 2
NUM_BODIES = 100

TIME_STEP = 60.0 * 60.0 * 24.0 # seconds

# Disperse the bodies through the simulation space uniformly.
body_locations = np.random.uniform(low=-SIMULATION_RADIUS, high=SIMULATION_RADIUS, size=(NUM_BODIES, 3))
# However, we want a disc shape, so squash along the z-axis.
body_locations[:, 2] /= 50.0

# Create a normal distribution of masses around the mass of Uranus,
# varying on the order of the mass of Earth.
body_mass = np.random.normal(loc=measure.mass_uranus, scale=measure.mass_earth, size=NUM_BODIES)

# Put the sun in the center of the simulation.
body_locations[0] = [0.0, 0.0, 0.0]
body_mass[0] = measure.mass_sun

# This assumes body 0 is the sun and it finds the velocity for a stable enough orbit
def calculate_initial_velocity(body_no):
    global body_locations
    global body_mass

    # The sun is initially stationary.
    if body_no == 0:
        return np.array([0.0, 0.0, 0.0])

    # Find the normal vector pointing from the sun to the indicated body.
    delta_location = body_locations[body_no] - body_locations[0]
    r = np.sqrt((delta_location * delta_location).sum())
    normal_vector = delta_location / r

    # In our simulation, since our disc is squished in the z-axis, we consider
    # "down" to be in the negative z direction.
    down_vector = np.array([0.0, 0.0, -1.0])

    # Determine the velocity that would be required for a perfectly circular
    # orbit (we don't quite have that because of our spread in the z-axis and
    # choice of down vector).
    velocity_to_preserve_orbit = np.sqrt(measure.gravitational_constant * body_mass[0] / r)

    # The cross product of the normal vector and the down vector should be roughly a
    # unit vector and point in the direction that the body should be moving initially
    # to orbit, scale it by the previously determined velocity to start the orbit.
    return np.cross(normal_vector, down_vector) * velocity_to_preserve_orbit

# Give each body a velocity.
body_velocities = np.array([calculate_initial_velocity(i) for i in range(NUM_BODIES)])

# Not a great implementation for this.
def calculate_acceleration_vector(body_no):
    global body_locations
    global body_mass

    location = body_locations[body_no]
    acceleration = np.array([0.0, 0.0, 0.0])

    # Determine the acceleration due to all the other masses in the system.
    for i in range(NUM_BODIES):
        if i == body_no:
            continue

        # Find the normal vector pointing from the indicated body to each other body
        # (the direction it'll accelerate).
        delta_location = body_locations[i] - location
        r = np.sqrt((delta_location * delta_location).sum())
        normal_vector = delta_location / r

        # The acceleration is toward the other mass and is calculated by the gravity equation
        # a = GM/r^2.
        accel_scalar = measure.gravitational_constant * body_mass[i] / r / r
        accel_vector = normal_vector * accel_scalar

        # Accumulate all the accelerations.
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
