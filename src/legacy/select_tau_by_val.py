from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "src.select_tau_by_val est obsolète avec tau interne. "
        "Le choix de tau est fait dans src.train_eval sur validation."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise