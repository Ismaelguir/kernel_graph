from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "src.analyze_taus est obsolète avec tau interne. "
        "Utilise uniquement: python -m src.summarize_results --results_dir results_final"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise