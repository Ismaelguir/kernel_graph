from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "src.mean_baseline est obsolète. Utilise: "
        "python -m src.baseline_train_eval --model mean ..."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise