"""Build a GIF from a folder of frame_XXXX.png files."""
import os, sys, glob
import imageio.v2 as imageio

def make(folder, out_path, fps=24):
    files = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    if not files:
        print(f"no frames in {folder}")
        return
    frames = [imageio.imread(f) for f in files]
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"wrote {out_path}  ({len(files)} frames, {fps} fps)")

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(HERE, "..", "out")
    targets = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    for t in targets:
        make(os.path.join(root, t), os.path.join(root, f"{t}.gif"), fps=24)
