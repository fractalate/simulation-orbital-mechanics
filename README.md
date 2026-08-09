# Simulation of Orbital Mechanics and Multi-body Systems

![Animated Simulation of Sample Orbits](./output/simulation_sample_orbits.gif)

## Setup

Required python libraries:

```bash
pip install numpy matplotlib pyarrow
```

## Examples

Try some of these examples:

* Produce sample orbit simulation data in the `out` directory:
  ```bash
  python3 simulation_sample_orbits.py -o out
  ```
* Show animation of simulation data from the `out` directory:
  ```bash
  python3 render_orbits.py out
  ```
* Render animated gif of simulation data from the `out` directory to `out.gif`:
  ```bash
  python3 render_orbits.py -o out.gif out
  ```

Some scripts have additional command line arguments. Use `--help` for more information.
