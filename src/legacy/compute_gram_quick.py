from __future__ import annotations

import sys


def main() -> None:
    raise RuntimeError(
        "src.compute_gram_quick est obsolète. "
        "Utilise le pipeline final et results_final uniquement."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise