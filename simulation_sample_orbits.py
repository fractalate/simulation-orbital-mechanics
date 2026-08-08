import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib
import sys

import measure
from simulation import Simulation
from samples import setup_sample_orbits


parser = argparse.ArgumentParser(description="Simulation of Sample Orbits")  
parser.add_argument("-o", "--output", default="out", help="where to save simulation files")
parser.add_argument("-n", "--number_of_bodies", "--number-of-bodies", default=10, help="number of bodies in the simulation > 0")
args = parser.parse_args()


if args.number_of_bodies < 1:
    print(f"number of bodies {args.number_of_bodies} must be > 0", file=sys.stderr)
    sys.exit(1)


out_dir = pathlib.Path(args.output)
if out_dir.exists():
    print(f"output directory {args.output} already exists", file=sys.stderr)
    sys.exit(1)
out_dir.mkdir()


SIMULATION_RADIUS_METERS = measure.astronomical_unit * 2
NUMBER_OF_BODIES = args.number_of_bodies
TIME_STEP_SECONDS = 60.0 * 60.0 * 24.0  # seconds
NUMBER_OF_STEPS = 200

state_schema = pa.schema([
    ("time", pa.float64()),
]).with_metadata({
    b"parameters": json.dumps({
        "simulation_radius_meters": str(SIMULATION_RADIUS_METERS),
        "number_of_bodies": str(NUMBER_OF_BODIES),
        "time_step_seconds": str(TIME_STEP_SECONDS),
        "number_of_steps": str(NUMBER_OF_STEPS),
    }).encode(),
})

def make_body_schema(parameters):
    body_schema = pa.schema([
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("z", pa.float64()),
        ("vx", pa.float64()),
        ("vy", pa.float64()),
        ("vz", pa.float64()),
    ]).with_metadata({
        b"parameters": json.dumps(parameters).encode(),
    })
    return body_schema

simulation = Simulation()
setup_sample_orbits(
    simulation,
    num_bodies=NUMBER_OF_BODIES,
    simulation_radius=SIMULATION_RADIUS_METERS,
)


time_buffer = np.zeros((NUMBER_OF_STEPS, ))
velocity_buffer = np.zeros((NUMBER_OF_STEPS, NUMBER_OF_BODIES, 3))
position_buffer = np.zeros((NUMBER_OF_STEPS, NUMBER_OF_BODIES, 3))

i = 0
time = 0.0
while i < NUMBER_OF_STEPS:
    time_buffer[i] = time
    position_buffer[i, :, :] = simulation.body_locations
    velocity_buffer[i, :, :] = simulation.body_velocities

    if i >= NUMBER_OF_STEPS:
        break

    accelerations = np.stack([simulation.calculate_acceleration_vector(i) for i in range(simulation.body_locations.shape[0])])
    simulation.body_velocities += accelerations * TIME_STEP_SECONDS
    simulation.body_locations += simulation.body_velocities * TIME_STEP_SECONDS

    i += 1
    time += TIME_STEP_SECONDS


with pq.ParquetWriter(out_dir / "state.parquet", state_schema) as writer:
    table = pa.table({
        "time": time_buffer,
    }, schema=state_schema)
    writer.write_table(table)

for i in range(NUMBER_OF_BODIES):
    body_schema = make_body_schema({
        "body_mass_kg": simulation.body_mass[i],
    })
    with pq.ParquetWriter(out_dir / f"body_{i}.parquet", body_schema) as writer:
        table = pa.table({
            "x": position_buffer[i, :, 0],
            "y": position_buffer[i, :, 1],
            "z": position_buffer[i, :, 2],
            "vx": velocity_buffer[i, :, 0],
            "vy": velocity_buffer[i, :, 1],
            "vz": velocity_buffer[i, :, 2],
        }, schema=body_schema)
        writer.write_table(table)
