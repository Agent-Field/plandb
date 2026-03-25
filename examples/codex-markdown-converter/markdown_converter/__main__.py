from __future__ import annotations

import argparse
import sys

from .converter import convert_markdown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a subset of Markdown to HTML."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Optional input file. Reads from stdin when omitted.",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            markdown = handle.read()
    else:
        markdown = sys.stdin.read()

    sys.stdout.write(convert_markdown(markdown))
    if markdown and not markdown.endswith("\n"):
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
