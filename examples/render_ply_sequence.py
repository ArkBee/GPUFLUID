"""Render a PLY mesh sequence to MP4 via matplotlib (Blender-free fallback)."""
import os, sys, glob, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from gpufluid.io.ply import read_ply


def render_mesh(verts, faces, ax, color="#3aa7ff", alpha=0.85):
    if verts is None or len(verts) == 0:
        return
    tris = verts[faces]
    # Y up convention: swap mpl's z and y for display
    tris_disp = tris.copy()
    tris_disp[..., [1, 2]] = tris_disp[..., [2, 1]]
    coll = Poly3DCollection(tris_disp, alpha=alpha, facecolor=color,
                             edgecolor="#1a4d70", linewidth=0.03)
    ax.add_collection3d(coll)


def render_sphere(ax, center, radius, color="#e0a050", n=14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    z = center[1] + radius * np.outer(np.sin(u), np.sin(v))  # swap y/z for display
    y = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.95, linewidth=0)


def render_torus(ax, center, major_r, minor_r, color="#e0a050", n_major=24, n_minor=12):
    # X-axis-aligned torus (ring opens along +X / -X)
    th = np.linspace(0, 2 * np.pi, n_major)
    ph = np.linspace(0, 2 * np.pi, n_minor)
    TH, PH = np.meshgrid(th, ph)
    rx = center[0] + (minor_r * np.cos(PH))
    ry = center[1] + (major_r + minor_r * np.cos(PH)) * np.cos(TH) * 0  # zero — collapses; use proper torus formula instead
    # Proper torus aligned with X axis: ring lies in YZ plane
    rx = center[0] + minor_r * np.cos(PH)
    ry = center[1] + (major_r + minor_r * np.sin(PH)) * np.cos(TH)
    rz = center[2] + (major_r + minor_r * np.sin(PH)) * np.sin(TH)
    # display swap y/z
    ax.plot_surface(rx, rz, ry, color=color, alpha=0.95, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache dir with mesh/frame_NNNN.ply")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cam-elev", type=float, default=22)
    ap.add_argument("--cam-azim", type=float, default=-58)
    ap.add_argument("--dom", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--obstacle", default=None,
                    help="optional 'sphere:cx,cy,cz,r' or 'torus_x:cx,cy,cz,R,r,start_x,vel_x'")
    ap.add_argument("--max-frames", type=int, default=10000)
    args = ap.parse_args()

    mesh_files = sorted(glob.glob(os.path.join(args.cache, "mesh", "frame_*.ply")))[: args.max_frames]
    if not mesh_files:
        print("no PLY files in", os.path.join(args.cache, "mesh"))
        return 2
    print(f"rendering {len(mesh_files)} frames -> {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=None) as writer:
        for f_idx, ply in enumerate(mesh_files):
            verts, faces = read_ply(ply)
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            render_mesh(verts, faces, ax)
            if args.obstacle:
                kind, payload = args.obstacle.split(":", 1)
                vals = [float(v) for v in payload.split(",")]
                if kind == "sphere":
                    render_sphere(ax, vals[:3], vals[3])
                elif kind == "torus_x":
                    cx, cy, cz, R, r, sx, vx = vals
                    # update centre per frame: c.x = sx + vx * (f/fps)
                    secs = f_idx / args.fps
                    render_torus(ax, (sx + vx * secs, cy, cz), R, r)
            ax.set_xlim(0, args.dom[0])
            ax.set_ylim(0, args.dom[2])   # mpl-y = sim-z
            ax.set_zlim(0, args.dom[1])   # mpl-z = sim-y
            ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
            ax.set_facecolor("#0c1a26")
            ax.set_title(f"frame {f_idx:04d}  V={len(verts) if verts is not None else 0}")
            ax.view_init(elev=args.cam_elev, azim=args.cam_azim)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf)
            plt.close(fig)
            if (f_idx + 1) % 10 == 0:
                print(f"  {f_idx+1}/{len(mesh_files)} ({(time.time()-t0):.1f}s elapsed)")
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
