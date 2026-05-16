"""Render mesh (fluid surface) + whitewater overlay, colour-coded by kind.
Used for the W7 demo (step22). Reads mesh PLYs from cache_dir/mesh/ and
whitewater positions + kinds from cache_dir/{whitewater,whitewater_kinds}/.
"""
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


KIND_COLOR = {
    0: (0.92, 0.95, 1.00),   # foam — near-white
    1: (0.40, 0.95, 1.00),   # spray — cyan
    2: (0.10, 0.30, 0.95),   # bubble — saturated blue
}
KIND_SIZE = {0: 8, 1: 4, 2: 6}


def render_mesh(ax, verts, faces, alpha=0.45):
    if verts is None or len(verts) == 0:
        return
    tris = verts[faces].copy()
    tris[..., [1, 2]] = tris[..., [2, 1]]
    coll = Poly3DCollection(tris, alpha=alpha, facecolor="#80b8d8",
                            edgecolor="#234560", linewidth=0.02)
    ax.add_collection3d(coll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dom", type=float, nargs=3, default=[1, 1, 1])
    ap.add_argument("--azim", type=float, default=35)
    ap.add_argument("--elev", type=float, default=18)
    a = ap.parse_args()

    mesh_files = sorted(glob.glob(os.path.join(a.cache, "mesh/frame_*.ply")))
    ww_files = sorted(glob.glob(os.path.join(a.cache, "whitewater/frame_*.npy")))
    kind_files = sorted(glob.glob(os.path.join(a.cache, "whitewater_kinds/frame_*.npy")))
    assert mesh_files and ww_files and kind_files, "missing cache subdirs"
    n = min(len(mesh_files), len(ww_files), len(kind_files))
    print(f"rendering {n} frames -> {a.out}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264",
                                quality=8, macro_block_size=1)
    t0 = time.time()
    for i in range(n):
        verts, faces = read_ply(mesh_files[i])
        ww_pos = np.load(ww_files[i])
        ww_kind = np.load(kind_files[i])
        fig = plt.figure(figsize=(7, 7), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, a.dom[0]); ax.set_ylim(0, a.dom[2]); ax.set_zlim(0, a.dom[1])
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
        ax.set_facecolor("#08121a")
        render_mesh(ax, verts, faces, alpha=0.55)
        # Whitewater: scatter per class (separate calls so legend works clean)
        for k_id, lbl in [(2, "bubble"), (0, "foam"), (1, "spray")]:
            m = (ww_kind == k_id)
            if m.any():
                p = ww_pos[m]
                ax.scatter(p[:, 0], p[:, 2], p[:, 1],
                           c=[KIND_COLOR[k_id]], s=KIND_SIZE[k_id],
                           depthshade=False, marker=".", linewidths=0,
                           label=f"{lbl} ({m.sum()})")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.4)
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.set_title(f"W7 foam/spray/bubble — frame {i:03d}",
                     fontsize=11, color="white")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("white")
            for tick in axis.get_ticklabels():
                tick.set_color("#aaaaaa")
        fig.patch.set_facecolor("#08121a")
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3].copy()
        writer.append_data(img)
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} ({time.time()-t0:.1f}s)")
    writer.close()
    print(f"wrote {a.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
