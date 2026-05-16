"""
3D dam-break with a cylindrical obstacle in the middle of the domain.
Outputs:
  - particles PNG previews
  - PLY mesh sequence in out/dam_break_obstacle_3d/ply/
  - mesh render PNG previews (matplotlib trisurf)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from gpufluid import FlipSolver3D, MeshExtractor, write_ply, sdf_cylinder_y
import warp as wp


def render_mesh_preview(verts, faces, dom, out_path, title):
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    if verts is not None and len(verts) > 0 and faces is not None and len(faces) > 0:
        tris = verts[faces]
        # swap Y/Z so Y points up in mpl
        tris_disp = tris.copy()
        tris_disp[..., [1, 2]] = tris_disp[..., [2, 1]]
        coll = Poly3DCollection(tris_disp, alpha=0.85, facecolor="#3aa7ff",
                                edgecolor="#1a5680", linewidth=0.05)
        ax.add_collection3d(coll)
    ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[2]); ax.set_zlim(0, dom[1])
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    nx, ny, nz = 64, 64, 64
    dx = 1.0 / 64
    solver = FlipSolver3D(nx=nx, ny=ny, nz=nz, dx=dx,
                          gravity=-9.81, flip_blend=0.95, rho=1.0)

    # --- obstacle: vertical cylinder in middle of floor ---
    grid_xyz = solver.cell_centers_np()
    dom_size = np.array([nx * dx, ny * dx, nz * dx])
    cyl_sdf = sdf_cylinder_y(grid_xyz,
                             center=(0.50, 0.30, 0.50),
                             radius=0.10,
                             half_height=0.30)
    solver.add_solid_from_sdf(cyl_sdf)

    # --- fluid: column on the left, full height ---
    solver.seed_box(lo=(0.06, 0.06, 0.06), hi=(0.30, 0.70, 0.95), ppc=8)
    print(f"particles: {solver.n_particles}")

    out_root = os.path.join(HERE, "..", "out", "dam_break_obstacle_3d")
    out_part = os.path.join(out_root, "particles"); os.makedirs(out_part, exist_ok=True)
    out_mesh = os.path.join(out_root, "mesh");      os.makedirs(out_mesh, exist_ok=True)
    out_ply  = os.path.join(out_root, "ply");       os.makedirs(out_ply,  exist_ok=True)

    dt = 0.005
    n_steps = 240
    save_every = 6
    pressure_iters = 60

    # reuse mesh extractor across frames (holds density buffers)
    extractor = MeshExtractor(nx, ny, nz, dx)

    t_sim = 0.0
    t_mesh = 0.0
    t_render = 0.0
    n_render = 0

    for step in range(n_steps):
        ts = time.time(); solver.step(dt, pressure_iters=pressure_iters); t_sim += time.time() - ts
        if step % save_every == 0:
            # ----- particles preview -----
            pos, _ = solver.get_particles()
            fig = plt.figure(figsize=(6, 5), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=1.0, c="#3aa7ff", alpha=0.5)
            # draw cylinder as wireframe for context
            theta = np.linspace(0, 2*np.pi, 40)
            xs = 0.50 + 0.10 * np.cos(theta)
            zs = 0.50 + 0.10 * np.sin(theta)
            for yh in [0.0, 0.30, 0.60]:
                ax.plot(xs, zs, yh, color="#cccccc", lw=0.8)
            ax.set_xlim(0, dom_size[0]); ax.set_ylim(0, dom_size[2]); ax.set_zlim(0, dom_size[1])
            ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
            ax.set_title(f"particles  step {step:04d}  t={step*dt:.3f}s")
            ax.view_init(elev=22, azim=-58)
            fig.tight_layout()
            fig.savefig(os.path.join(out_part, f"frame_{step:04d}.png"))
            plt.close(fig)

            # ----- mesh -----
            tm = time.time()
            verts, faces = extractor.extract(solver.pos, iso_level=0.6, smooth_passes=2)
            t_mesh += time.time() - tm

            if verts is not None:
                write_ply(os.path.join(out_ply, f"frame_{step:04d}.ply"), verts, faces)
                tr = time.time()
                render_mesh_preview(verts, faces, dom_size,
                                    os.path.join(out_mesh, f"frame_{step:04d}.png"),
                                    f"mesh  step {step:04d}  t={step*dt:.3f}s  V={len(verts)} F={len(faces)}")
                t_render += time.time() - tr
                n_render += 1
                print(f"  step {step:04d}  V={len(verts):>6}  F={len(faces):>6}")
            else:
                print(f"  step {step:04d}  no mesh")

    print(f"\nsim total:    {t_sim:.2f}s  ({n_steps/t_sim:.1f} steps/s)")
    print(f"mesh total:   {t_mesh:.2f}s  ({n_render/t_mesh:.1f} meshes/s)" if t_mesh > 0 else "")
    print(f"render total: {t_render:.2f}s")
    print(f"output: {out_root}")


if __name__ == "__main__":
    main()
