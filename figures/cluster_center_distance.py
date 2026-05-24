"""Plot cluster-center distances in the vertical flow."""

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("figures/cluster_center_distance.py: Plot cluster-center distances in the vertical flow.")
    print("Run without --help to execute. Configure defaults in src.plotting.cluster_center_distance or edit this wrapper.")
    raise SystemExit(0)

from src.plotting.cluster_center_distance import main


if __name__ == "__main__":
    main()
