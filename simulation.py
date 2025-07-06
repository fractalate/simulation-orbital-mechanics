import numpy as np

import measure

class Simulation():
    def __init__(self):
        # Simulation Parameters
        self.time_step = 60.0 * 60.0 * 24.0  # seconds

        # Simulation State
        self.body_mass = np.empty((0, 3), dtype=float)
        self.body_locations = np.empty((0, 3), dtype=float)
        self.body_velocities = np.empty((0, 3), dtype=float)

    def get_number_of_bodies(self):
        return self.body_locations.shape[0]

    def add_body(self, mass, location, velocity):
        body_no = self.get_number_of_bodies()
        self.body_mass.append(mass)
        self.body_locations.append(location)
        self.body_velocities.append(velocity)
        return body_no

    def calculate_acceleration_vector(self, body_no):
        location = self.body_locations[body_no]
        acceleration = np.array([0.0, 0.0, 0.0])

        # Determine the acceleration due to all the other masses in the system.
        for i in range(self.get_number_of_bodies()):
            if i == body_no:
                continue

            # Find the normal vector pointing from the indicated body to each other body
            # (the direction it'll accelerate).
            delta_location = self.body_locations[i] - location
            r = np.linalg.norm(delta_location)
            if r != 0.0:
                normal_vector = delta_location / r

                # The acceleration is toward the other mass and is calculated by the gravity equation
                # a = GM/r^2.
                accel_scalar = measure.gravitational_constant * self.body_mass[i] / r / r
                accel_vector = normal_vector * accel_scalar

                # Accumulate all the accelerations.
                acceleration += accel_vector

        return acceleration

    def calculate_gravitational_vector(self, position):
        """
        Calculates the gravitational acceleration at a given position or array of positions.
        Supports both single 3D position (shape: (3,)) and meshgrid/array of positions (shape: (3, ...)).
        """
        position = np.asarray(position)
        acceleration = np.zeros_like(position, dtype=float)

        # Handle single position (shape: (3,))
        if position.ndim == 1:
            for i in range(self.get_number_of_bodies()):
                normal_vector = self.body_locations[i] - position
                r = np.linalg.norm(normal_vector)
                if r != 0.0:
                    normal_vector /= r
                    accel_scalar = self.body_mass[i] / r / r
                    accel_vector = normal_vector * accel_scalar
                    acceleration += accel_vector
            return measure.gravitational_constant * acceleration

        # Handle meshgrid/array of positions (shape: (3, ...))
        else:
            # acceleration shape: (3, ...)
            for i in range(self.get_number_of_bodies()):
                normal_vector = self.body_locations[i][:, np.newaxis, np.newaxis, np.newaxis] - position
                r = np.linalg.norm(normal_vector, axis=0, keepdims=True)
                mask = r != 0.0
                accel_vector = np.zeros_like(normal_vector)
                accel_scalar = np.zeros_like(r)
                accel_scalar[mask] = self.body_mass[i] / (r[mask] ** 2)
                accel_vector += np.where(mask, normal_vector / r * accel_scalar, 0)
                acceleration += accel_vector
            return measure.gravitational_constant * acceleration
