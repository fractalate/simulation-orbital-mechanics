import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pathlib
import sys

import measure
from simulation import Simulation
from samples import setup_sample_orbits


parser = argparse.ArgumentParser(description="Simulation of Sample Orbits")  
parser.add_argument("-o", "--output", default="out", help="output directory to save simulation files (will be created)")
parser.add_argument("-f", "--force", action="store_true", default=False, help="set if you want to generate output in an existing output directory")
parser.add_argument("-n", "--number-of-bodies", type=int, default=10, help="number of bodies in the simulation")
parser.add_argument("-t", "--time-step", type=float, default=60.0*60.0*24.0, help="time step for the simulation in seconds")
parser.add_argument("-l", "--number-of-steps", type=int, default=300, help="number of steps to simulate")
args = parser.parse_args()

if args.number_of_bodies <= 0:
    print(f"number of bodies {args.number_of_bodies} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.time_step <= 0.0:
    print(f"time step {args.time_step} must be > 0", file=sys.stderr)
    sys.exit(1)

if args.number_of_steps <= 0:
    print(f"number of steps {args.number_of_steps} must be > 0", file=sys.stderr)
    sys.exit(1)

out_dir = pathlib.Path(args.output)
if out_dir.exists() and not args.force:
    print(f"output directory {args.output} already exists", file=sys.stderr)
    sys.exit(1)
out_dir.mkdir(exist_ok=args.force)


SIMULATION_RADIUS_METERS = measure.astronomical_unit * 2
NUMBER_OF_BODIES = args.number_of_bodies
TIME_STEP_SECONDS = args.time_step
NUMBER_OF_STEPS = args.number_of_steps

state_schema = pa.schema([
    ("time", pa.float64()),
]).with_metadata({
    b"parameters": json.dumps({
        "simulation_radius_meters": SIMULATION_RADIUS_METERS,
        "number_of_bodies": NUMBER_OF_BODIES,
        "time_step_seconds": TIME_STEP_SECONDS,
        "number_of_steps": NUMBER_OF_STEPS,
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
while True:
    time_buffer[i] = time
    position_buffer[i, :, :] = simulation.body_locations
    velocity_buffer[i, :, :] = simulation.body_velocities

    i += 1
    if i >= NUMBER_OF_STEPS:
        break

    accelerations = np.stack([simulation.calculate_acceleration_vector(k) for k in range(simulation.body_locations.shape[0])])
    simulation.body_velocities += accelerations * TIME_STEP_SECONDS
    simulation.body_locations += simulation.body_velocities * TIME_STEP_SECONDS

    time += TIME_STEP_SECONDS


with pq.ParquetWriter(out_dir / "state.parquet", state_schema) as writer:
    table = pa.table({
        "time": time_buffer,
    }, schema=state_schema)
    writer.write_table(table)

for i in range(NUMBER_OF_BODIES):
    body_schema = make_body_schema({
        "mass_kg": simulation.body_mass[i],
    })
    with pq.ParquetWriter(out_dir / f"body_{i}.parquet", body_schema) as writer:
        table = pa.table({
            "x": position_buffer[:, i, 0],
            "y": position_buffer[:, i, 1],
            "z": position_buffer[:, i, 2],
            "vx": velocity_buffer[:, i, 0],
            "vy": velocity_buffer[:, i, 1],
            "vz": velocity_buffer[:, i, 2],
        }, schema=body_schema)
        writer.write_table(table)
