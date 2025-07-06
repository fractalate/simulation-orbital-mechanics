import numpy as np

import measure


# TODO instead of injecting directly into the simulation, make some methods to add new bodies
def setup_sample_orbits(simulation, num_bodies=50, simulation_radius=measure.astronomical_unit * 2):
    # Disperse the bodies through the simulation space uniformly.
    simulation.body_locations = np.random.uniform(low=-simulation_radius, high=simulation_radius, size=(num_bodies, 3))
    # However, we want a disc shape, so squash along the z-axis.
    simulation.body_locations[:, 2] /= 50.0

    # Create a normal distribution of masses around the mass of Uranus,
    # varying on the order of the mass of Earth.
    simulation.body_mass = np.random.normal(loc=measure.mass_uranus, scale=measure.mass_earth, size=num_bodies)

    # Put the sun in the center of the simulation.
    simulation.body_locations[0] = [0.0, 0.0, 0.0]
    simulation.body_mass[0] = measure.mass_sun

    # This assumes body 0 is the sun and it finds the velocity for a stable enough orbit
    def calculate_initial_velocity(body_no):
        nonlocal simulation

        # The sun is initially stationary.
        if body_no == 0:
            return np.array([0.0, 0.0, 0.0])

        # Find the normal vector pointing from the sun to the indicated body.
        delta_location = simulation.body_locations[body_no] - simulation.body_locations[0]
        r = np.sqrt((delta_location * delta_location).sum())
        normal_vector = delta_location / r

        # In our simulation, since our disc is squished in the z-axis, we consider
        # "down" to be in the negative z direction.
        down_vector = np.array([0.0, 0.0, -1.0])

        # Determine the velocity that would be required for a perfectly circular
        # orbit (we don't quite have that because of our spread in the z-axis and
        # choice of down vector).
        velocity_to_preserve_orbit = np.sqrt(measure.gravitational_constant * simulation.body_mass[0] / r)

        # The cross product of the normal vector and the down vector should be roughly a
        # unit vector and point in the direction that the body should be moving initially
        # to orbit, scale it by the previously determined velocity to start the orbit.
        return np.cross(normal_vector, down_vector) * velocity_to_preserve_orbit

    # Give each body a velocity.
    simulation.body_velocities = np.array([calculate_initial_velocity(i) for i in range(num_bodies)])
