"""
3D dam-break. Seed a block of fluid in one corner of a box, let it collapse.
Saves PNG renders (matplotlib 3D scatter) and a .npy of particles per frame.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from gpufluid import FlipSolver3D


def main():
    nx, ny, nz = 48, 48, 48
    dx = 1.0 / 48
    solver = FlipSolver3D(nx=nx, ny=ny, nz=nz, dx=dx,
                          gravity=-9.81, flip_blend=0.95, rho=1.0)
    # column of water in the back-left corner, full height
    solver.seed_box(lo=(0.05, 0.05, 0.05), hi=(0.40, 0.70, 0.40), ppc=8)
    print(f"particles: {solver.n_particles}")

    out_dir = os.path.join(HERE, "..", "out", "dam_break_3d")
    os.makedirs(out_dir, exist_ok=True)
    npy_dir = os.path.join(out_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    dt = 0.005
    n_steps = 200
    save_every = 5
    pressure_iters = 50

    t0 = time.time()
    for step in range(n_steps):
        solver.step(dt, pressure_iters=pressure_iters)
        if step % save_every == 0:
            pos, _ = solver.get_particles()
            np.save(os.path.join(npy_dir, f"f{step:04d}.npy"), pos)
            fig = plt.figure(figsize=(6, 5), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=1.5, c="#3aa7ff", alpha=0.55)
            ax.set_xlim(0, nx * dx); ax.set_ylim(0, nz * dx); ax.set_zlim(0, ny * dx)
            ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
            ax.set_title(f"step {step:04d}  t={step*dt:.3f}s  N={solver.n_particles}")
            ax.view_init(elev=22, azim=-58)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"frame_{step:04d}.png"))
            plt.close(fig)
            print(f"  saved frame_{step:04d}.png")
    t1 = time.time()
    print(f"Done. {n_steps} steps in {t1-t0:.2f}s ({n_steps/(t1-t0):.2f} steps/s)")
    print(f"Out: {out_dir}")


if __name__ == "__main__":
    main()
