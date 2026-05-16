"""Entry point: ``python -m gpufluid.cli`` or via the ``gpufluid`` console script."""
from .commands import main

if __name__ == "__main__":
    raise SystemExit(main())
