"""Loaded automatically via PYTHONPATH by vm_run_all.py.

Headless runs have no display, so turn `plt.show()` into "save every open
figure to $COLAB_FIG_DIR" instead of silently dropping the plots.
"""

import os

_fig_dir = os.environ.get("COLAB_FIG_DIR")

if _fig_dir:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _count = [0]

        _dir: str = _fig_dir

        def _save_and_close(*_args, **_kwargs):  # noqa: ARG001 - same signature as plt.show
            os.makedirs(_dir, exist_ok=True)
            for num in plt.get_fignums():
                _count[0] += 1
                plt.figure(num).savefig(os.path.join(_dir, f"fig{_count[0]:02d}.png"), dpi=110, bbox_inches="tight")
            plt.close("all")

        plt.show = _save_and_close
    except ImportError:  # matplotlib not installed: nothing to patch
        pass
