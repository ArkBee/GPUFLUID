"""Pure solver benchmark, no rendering."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from gpufluid import FlipSolver2D, FlipSolver3D


def bench_2d():
    s = FlipSolver2D(nx=128, ny=96, dx=1.0/128)
    s.seed_box(0.05, 0.05, 0.35, 0.70, ppc=4)
    print(f"[2D 128x96] N={s.n_particles}")
    # warmup
    for _ in range(3): s.step(0.005, pressure_iters=60)
    n = 200
    t = time.time()
    for _ in range(n): s.step(0.005, pressure_iters=60)
    dt = time.time() - t
    print(f"  {n} steps in {dt:.2f}s  ({n/dt:.1f} steps/s)")


def bench_3d():
    s = FlipSolver3D(nx=48, ny=48, nz=48, dx=1.0/48)
    s.seed_box(lo=(0.05, 0.05, 0.05), hi=(0.40, 0.70, 0.40), ppc=8)
    print(f"[3D 48^3] N={s.n_particles}")
    for _ in range(3): s.step(0.005, pressure_iters=50)
    n = 100
    t = time.time()
    for _ in range(n): s.step(0.005, pressure_iters=50)
    dt = time.time() - t
    print(f"  {n} steps in {dt:.2f}s  ({n/dt:.1f} steps/s)")

    s2 = FlipSolver3D(nx=64, ny=64, nz=64, dx=1.0/64)
    s2.seed_box(lo=(0.05, 0.05, 0.05), hi=(0.40, 0.70, 0.40), ppc=8)
    print(f"[3D 64^3] N={s2.n_particles}")
    for _ in range(3): s2.step(0.005, pressure_iters=50)
    t = time.time()
    for _ in range(n): s2.step(0.005, pressure_iters=50)
    dt = time.time() - t
    print(f"  {n} steps in {dt:.2f}s  ({n/dt:.1f} steps/s)")


if __name__ == "__main__":
    bench_2d()
    bench_3d()
