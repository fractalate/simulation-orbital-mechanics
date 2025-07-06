import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import measure
from simulation import Simulation
from samples import setup_sample_orbits

SIMULATION_RADIUS = measure.astronomical_unit * 2

simulation = Simulation()
setup_sample_orbits(
    simulation,
    num_bodies=2,
    simulation_radius=SIMULATION_RADIUS,
)
simulation.body_mass[1] *= 1000
simulation.body_locations[1][0] = 0

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

sc = ax.scatter(simulation.body_locations[:, 0], simulation.body_locations[:, 1], simulation.body_locations[:, 2], alpha=0.6)

ax.set_title("Sample Orbits Simulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

TIME_STEP = 60.0 * 60.0 * 24.0  # seconds

def update(frame):
    global simulation

    # Not a great simulation.
    accelerations = np.stack([simulation.calculate_acceleration_vector(i) for i in range(simulation.body_locations.shape[0])])
    simulation.body_velocities += accelerations * TIME_STEP
    simulation.body_locations += simulation.body_velocities * TIME_STEP

    sc._offsets3d = (simulation.body_locations[:, 0], simulation.body_locations[:, 1], simulation.body_locations[:, 2])

    return sc,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

ani.save("output/simulation_sample_orbits.gif", writer="pillow", fps=30)
