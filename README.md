# Simulation of Orbital Mechanics and Multi-body Systems

![Animated Simulation of Sample Orbits](./output/simulation_sample_orbits.gif)

## Setup

Required python libraries:

```bash
pip install numpy matplotlib pyarrow
```

## Examples

Try some of these examples:

* Produce an animation of sample orbits:
  ```bash
  python3 simulation_sample_orbits.py -o out
  ```
  which will create the `out` directory containing simulation data.
* Render orbits as an animation:
  ```bash
  python3 render_orbits.py out
  ```
  or
  ```bash
  python3 render_orbits.py -o out.gif out
  ```
  which will render `out.gif` from the simulation data in the `out` directory.

Some scripts have additional command line arguments. Use `--help` for more information.
