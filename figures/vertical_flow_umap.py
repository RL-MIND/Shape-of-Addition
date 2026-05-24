"""Reproduce vertical-flow UMAP visualizations."""

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("figures/vertical_flow_umap.py: Reproduce vertical-flow UMAP visualizations.")
    print("Common arguments: --data-path, --position, --layer, --marker-mode, --color-mode, --max-points, --save-dir")
    print("Run `python -m src.plotting.umap_plots --help` in an environment with compatible numpy/numba for the full CLI.")
    raise SystemExit(0)

from src.plotting.umap_plots import main


if __name__ == "__main__":
    main()
