"""Render a colored-particle sequence (positions + per-particle RGB) to MP4.
Pairs `particles/frame_NNNN.npy` (Nx3 float32) with `colors/frame_NNNN.npy`
(Nx3 float32). Used for S2.15 demos where mesh rendering alone can't show
the per-particle colour."""
import os, sys, glob, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="cache dir containing particles/ and colors/ subdirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--domain", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--size", type=float, default=4.0, help="point size")
    ap.add_argument("--azim", type=float, default=45)
    ap.add_argument("--elev", type=float, default=20)
    args = ap.parse_args()

    pfs = sorted(glob.glob(os.path.join(args.cache, "particles", "frame_*.npy")))
    cfs = sorted(glob.glob(os.path.join(args.cache, "colors", "frame_*.npy")))
    if not pfs:
        raise SystemExit(f"no particles found in {args.cache}/particles/")
    if len(cfs) != len(pfs):
        print(f"WARNING: {len(pfs)} positions but {len(cfs)} color files")
    print(f"rendering {len(pfs)} frames -> {args.out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dom = args.domain
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)
    t0 = time.time()
    for i, (pf, cf) in enumerate(zip(pfs, cfs)):
        pos = np.load(pf)
        col = np.load(cf)
        if len(pos) == 0:
            continue
        col = np.clip(col, 0.0, 1.0)
        fig = plt.figure(figsize=(7, 7), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[2]); ax.set_zlim(0, dom[1])
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
        ax.set_facecolor("#0c1a26")
        # display: swap y/z so gravity (-y) points down in the rendered frame
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1],
                   c=col, s=args.size, depthshade=False, marker=".",
                   linewidths=0)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"S2.15 two-colour mix — frame {i:03d}", fontsize=11,
                     color="white")
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
            print(f"  {i+1}/{len(pfs)} ({time.time()-t0:.1f}s)")
    writer.close()
    print(f"wrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
