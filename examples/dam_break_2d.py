"""
2D dam-break demo. Seed a column of fluid on the left, let it collapse under gravity.
Writes PNG frames into out/dam_break_2d/.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make src importable when running from project root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gpufluid import FlipSolver2D


def main():
    nx = 96
    ny = 64
    dx = 1.0 / nx  # domain ~ 1.0 x 0.67

    solver = FlipSolver2D(nx=nx, ny=ny, dx=dx, gravity=-9.81, flip_blend=0.95, rho=1.0)
    # seed left column
    solver.seed_box(0.05, 0.05, 0.30, 0.55, ppc=4)
    print(f"particles: {solver.n_particles}")

    out_dir = os.path.join(HERE, "..", "out", "dam_break_2d")
    os.makedirs(out_dir, exist_ok=True)

    dt = 0.005
    n_steps = 400
    save_every = 4
    pressure_iters = 60

    t0 = time.time()
    for step in range(n_steps):
        solver.step(dt, pressure_iters=pressure_iters)
        if step % save_every == 0:
            pos, _ = solver.get_particles()
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.scatter(pos[:, 0], pos[:, 1], s=2, c="#3aa7ff")
            ax.set_xlim(0, nx * dx)
            ax.set_ylim(0, ny * dx)
            ax.set_aspect("equal")
            ax.set_title(f"step {step:04d}  t={step*dt:.3f}s  N={solver.n_particles}")
            ax.set_facecolor("#0c1a26")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"frame_{step:04d}.png"))
            plt.close(fig)
            print(f"  saved frame_{step:04d}.png")
    t1 = time.time()
    print(f"Done. {n_steps} steps in {t1 - t0:.2f}s  ({n_steps/(t1-t0):.1f} steps/s)")
    print(f"Frames in: {out_dir}")


if __name__ == "__main__":
    main()
