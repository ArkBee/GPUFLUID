"""Side-by-side PLY-sequence MP4 renderer (matplotlib).
Useful for ablation: with-feature vs without-feature."""
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
    tris_disp = tris.copy()
    tris_disp[..., [1, 2]] = tris_disp[..., [2, 1]]
    coll = Poly3DCollection(tris_disp, alpha=alpha, facecolor=color,
                             edgecolor="#1a4d70", linewidth=0.03)
    ax.add_collection3d(coll)


def render_torus(ax, center, major_r, minor_r, color="#e0a050", n_major=24, n_minor=12):
    th = np.linspace(0, 2 * np.pi, n_major)
    ph = np.linspace(0, 2 * np.pi, n_minor)
    TH, PH = np.meshgrid(th, ph)
    rx = center[0] + minor_r * np.cos(PH)
    ry = center[1] + (major_r + minor_r * np.sin(PH)) * np.cos(TH)
    rz = center[2] + (major_r + minor_r * np.sin(PH)) * np.sin(TH)
    ax.plot_surface(rx, rz, ry, color=color, alpha=0.95, linewidth=0)


def setup_axis(ax, dom, title, elev, azim):
    ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[2]); ax.set_zlim(0, dom[1])
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
    ax.set_facecolor("#0c1a26")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-cache", required=True)
    ap.add_argument("--left-label", default="left")
    ap.add_argument("--right-cache", required=True)
    ap.add_argument("--right-label", default="right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-58)
    ap.add_argument("--dom", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--torus", type=str, default=None,
                    help="cx,cy,cz,R,r,start_x,vel_x — drawn on left only")
    args = ap.parse_args()

    L = sorted(glob.glob(os.path.join(args.left_cache, "mesh", "frame_*.ply")))
    R = sorted(glob.glob(os.path.join(args.right_cache, "mesh", "frame_*.ply")))
    n = min(len(L), len(R))
    print(f"rendering {n} frames -> {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                             quality=8, macro_block_size=None) as writer:
        for f_idx in range(n):
            va, fa = read_ply(L[f_idx])
            vb, fb = read_ply(R[f_idx])
            fig = plt.figure(figsize=(12, 5), dpi=100)
            ax_l = fig.add_subplot(121, projection="3d")
            ax_r = fig.add_subplot(122, projection="3d")
            render_mesh(va, fa, ax_l)
            render_mesh(vb, fb, ax_r)
            if args.torus:
                cx, cy, cz, R_, r_, sx, vx = [float(v) for v in args.torus.split(",")]
                secs = f_idx / args.fps
                render_torus(ax_l, (sx + vx * secs, cy, cz), R_, r_)
            setup_axis(ax_l, args.dom, f"{args.left_label}  V={len(va) if va is not None else 0}", args.elev, args.azim)
            setup_axis(ax_r, args.dom, f"{args.right_label}  V={len(vb) if vb is not None else 0}", args.elev, args.azim)
            fig.suptitle(f"frame {f_idx:04d}", color="white", fontsize=12)
            fig.set_facecolor("#0c1a26")
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf)
            plt.close(fig)
            if (f_idx + 1) % 10 == 0:
                print(f"  {f_idx+1}/{n} ({time.time()-t0:.1f}s)")
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
