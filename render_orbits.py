import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyarrow.parquet as pq
import pathlib
import sys


parser = argparse.ArgumentParser(description="Render Orbits")
parser.add_argument("-o", "--output", help="save animation to file")
parser.add_argument("-f", "--force", action="store_true", default=False, help="set if you want to generate output in an existing file")
parser.add_argument("input", help="input directory containing simulation data")
args = parser.parse_args()

in_dir = pathlib.Path(args.input)
if not in_dir.is_dir():
    print(f"input {in_dir} is not a directory", file=sys.stderr)
    sys.exit(1)

out_file = None
if args.output:
    out_file = pathlib.Path(args.output)
    if out_file.exists() and not args.force:
        print(f"output {out_file} already exists", file=sys.stderr)
        sys.exit(1)

state_table = pq.ParquetFile(in_dir / "state.parquet")
state_table_metadata = json.loads(state_table.schema_arrow.metadata[b"parameters"].decode())
number_of_bodies = state_table_metadata["number_of_bodies"]


all_positions = []
for i in range(number_of_bodies):
    body_table = pq.ParquetFile(in_dir / f"body_{i}.parquet")
    xyz_data = body_table.read(["x", "y", "z"])
    positions = np.stack([
        xyz_data["x"].to_numpy(),
        xyz_data["y"].to_numpy(),
        xyz_data["z"].to_numpy(),
    ])
    all_positions.append(positions)
all_positions = np.stack(all_positions)
frames = all_positions.shape[2]
sample_frames = min(frames, 10)


fig = plt.figure()

try:
    fig.canvas.manager.window.wm_geometry("+100+100")
    fig.canvas.manager.window.geometry("640x480")
except AttributeError:
    print("Your backend may not support window resizing")

ax = fig.add_subplot(111, projection="3d")

# We want to render cube shaped space in space.
# XXX This assumes we're interested in seeing the origin in the center.
sample_min = all_positions[:, :, :sample_frames].min()
sample_max = all_positions[:, :, :sample_frames].max()
sample_radius = max(
    sample_min, -sample_min,
    sample_max, -sample_max,
)
ax.set_xlim(-sample_radius, sample_radius)
ax.set_ylim(-sample_radius, sample_radius)
ax.set_zlim(-sample_radius, sample_radius)

sc = ax.scatter(all_positions[:, 0, 0], all_positions[:, 1, 0], all_positions[:, 2, 0])

ax.set_title("Orbits")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def update(frame):
    global sc

    sc._offsets3d = (all_positions[:, 0, frame], all_positions[:, 1, frame], all_positions[:, 2, frame])

    return sc,

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

if out_file:
    print(f"writing animation to {out_file}...")
    ani.save(out_file, writer="pillow", fps=30)
    print("done!")
else:
    plt.show()
