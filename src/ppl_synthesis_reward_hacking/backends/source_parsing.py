from __future__ import annotations

import re


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def extract_block_lines(source: str, block_name: str) -> list[str]:
    lines: list[str] = []
    in_block = False
    pending = False
    depth = 0
    block_pattern = re.compile(rf"^{re.escape(block_name)}\b")

    for line in source.splitlines():
        uncommented = strip_line_comment(line)
        stripped = uncommented.strip()

        if not in_block and not pending and block_pattern.match(stripped):
            pending = True

        if pending:
            if "{" not in uncommented:
                continue
            pending = False
            in_block = True
            depth = uncommented.count("{") - uncommented.count("}")

            # Preserve inline bodies like: data { int N; real C; }
            after_open = uncommented.split("{", 1)[1].strip()
            if after_open:
                lines.append(after_open)

            if depth <= 0:
                in_block = False
            continue

        if not in_block:
            continue

        lines.append(line)
        depth += uncommented.count("{") - uncommented.count("}")
        if depth <= 0:
            in_block = False
    return lines
