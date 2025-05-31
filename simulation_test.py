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
# * [ ] Give the points velocity and have them move in space.
# * [ ] Give the points some mass.
# * [ ] Simulate gravitational interaction between the bodies.

AU_METERS=495979e11
SOL_MASS_KG=1.989e30

SIMULATION_RADIUS = AU_METERS * 1000
NUM_BODIES = 10

body_locations = np.random.uniform(low=-SIMULATION_RADIUS, high=SIMULATION_RADIUS, size=(NUM_BODIES, 3))

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
    body_locations += np.random.normal(0, SIMULATION_RADIUS/100, body_locations.shape)
    sc._offsets3d = (body_locations[:, 0], body_locations[:, 1], body_locations[:, 2])
    return sc,

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

plt.show()
