"""Plot UMAP views for the error-decomposition analysis."""

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("figures/error_decomposition_umap.py: Plot UMAP views for the error-decomposition analysis.")
    print("Run without --help to execute. Configure defaults in src.plotting.error_decomposition_plots or edit this wrapper.")
    raise SystemExit(0)

from src.plotting.error_decomposition_plots import main


if __name__ == "__main__":
    main()
