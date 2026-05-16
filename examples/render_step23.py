"""step23 renderer — side-by-side legacy vs W7.7 potential whitewater A/B.

Reads two caches and renders a 2-panel mp4 with the mesh + colour-coded
whitewater (foam = white, spray = cyan, bubble = blue). Counts per class
are overlaid as a live text annotation on each panel so the viewer can
see the difference numerically without leaving the video.

Run:
    python examples/render_step23.py \
        --left  out/step23_legacy    \
        --right out/step23_potential \
        --out   out/videos/step23.mp4
"""
import os
import sys
import glob
import argparse
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


KIND_COLOR = {
    0: (0.92, 0.95, 1.00),    # foam — near-white
    1: (0.40, 0.95, 1.00),    # spray — cyan
    2: (0.10, 0.30, 0.95),    # bubble — saturated blue
}
KIND_SIZE = {0: 7, 1: 4, 2: 5}
KIND_NAME = {0: "foam", 1: "spray", 2: "bubble"}


def render_mesh(ax, verts, faces, alpha=0.45):
    if verts is None or len(verts) == 0:
        return
    tris = verts[faces].copy()
    tris[..., [1, 2]] = tris[..., [2, 1]]
    coll = Poly3DCollection(tris, alpha=alpha, facecolor="#80b8d8",
                            edgecolor="#234560", linewidth=0.02)
    ax.add_collection3d(coll)


def render_ww(ax, pos, kinds):
    if pos is None or pos.shape[0] == 0:
        return
    # swap Y/Z so up is up in matplotlib's view
    p = pos[:, [0, 2, 1]]
    for k in (2, 0, 1):  # bubble first (deepest), then foam, then spray on top
        mask = kinds == k
        if not mask.any():
            continue
        ax.scatter(p[mask, 0], p[mask, 1], p[mask, 2],
                   s=KIND_SIZE[k], c=[KIND_COLOR[k]],
                   edgecolors="none", depthshade=True)


def setup_axes(ax, dom, title):
    ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[2]); ax.set_zlim(0, dom[1])
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect((dom[0], dom[2], dom[1]))
    ax.set_title(title, fontsize=11, color="#dddddd", pad=4)


def _glob_frames(cache):
    mesh_files = sorted(glob.glob(os.path.join(cache, "mesh", "frame_*.ply")))
    ww_files = sorted(glob.glob(os.path.join(cache, "whitewater", "frame_*.npy")))
    kind_files = sorted(glob.glob(os.path.join(cache, "whitewater_kinds", "frame_*.npy")))
    return mesh_files, ww_files, kind_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="legacy cache dir")
    ap.add_argument("--right", required=True, help="potential cache dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dom", type=float, nargs=3, default=[1, 1, 1])
    args = ap.parse_args()

    lm, lw, lk = _glob_frames(args.left)
    rm, rw, rk = _glob_frames(args.right)
    n = min(len(lm), len(lw), len(lk), len(rm), len(rw), len(rk))
    assert n > 0, "no frames found in one of the caches"
    print(f"[step23] rendering {n} frames")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for i in range(n):
            fig = plt.figure(figsize=(14, 7), facecolor="#111111")
            for col, (mfiles, wfiles, kfiles, title) in enumerate([
                (lm, lw, lk, "Legacy  |v| > threshold"),
                (rm, rw, rk, "W7.7    Trapped-air potential"),
            ]):
                ax = fig.add_subplot(1, 2, col + 1, projection="3d",
                                     facecolor="#111111")
                verts, faces = read_ply(mfiles[i])
                pos = np.load(wfiles[i]).astype(np.float32)
                kinds = np.load(kfiles[i]).astype(np.int32)
                render_mesh(ax, verts, faces)
                render_ww(ax, pos, kinds)
                # per-class count overlay
                counts = {k: int((kinds == k).sum()) for k in (0, 1, 2)}
                label = (f"frame {i:02d}\n"
                         f"foam   {counts[0]:5d}\n"
                         f"spray  {counts[1]:5d}\n"
                         f"bubble {counts[2]:5d}\n"
                         f"total  {kinds.size:5d}")
                ax.text2D(0.02, 0.98, label, transform=ax.transAxes,
                          color="#dddddd", fontsize=9, va="top",
                          family="monospace")
                setup_axes(ax, args.dom, title)
            fig.suptitle("gpufluid v0.8 — whitewater selector A/B (step23)",
                         color="#eeeeee", fontsize=13, y=0.97)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02,
                                wspace=0.02)
            fig.canvas.draw()
            # tostring_rgb was removed in matplotlib 3.10; use the RGBA buffer
            # and drop alpha to get an Nx M x 3 array for imageio.
            buf = np.asarray(fig.canvas.buffer_rgba())
            img = buf[..., :3]
            writer.append_data(img)
            plt.close(fig)
            if i % 5 == 0 or i == n - 1:
                print(f"  frame {i+1}/{n}")
    finally:
        writer.close()
    print(f"[step23] wrote {args.out}")


if __name__ == "__main__":
    main()
