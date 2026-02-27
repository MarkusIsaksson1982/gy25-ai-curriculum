#!/usr/bin/env python3
"""
Improved consolidate.py – fixed imports, full paths, junk skipping, LLM-friendly markers
"""

import os
import sys
from pathlib import Path
from datetime import datetime  # ← correct import

# Delimiters unlikely to appear in Python code
BEGIN_MARKER = "╔═ BEGIN_FILE ═════════════════════════════════════════════╗"
END_MARKER   = "╚═ END_FILE ═══════════════════════════════════════════════╝"

def looks_like_junk_file(filename: str) -> bool:
    """Skip logs, state files, summaries, caches, binaries, tiny files"""
    name_lower = filename.lower()
    skip_ext = {'.log', '.jsonl', '.tmp', '.bak', '.pyc', '.pyo'}
    skip_names = {'state.json', 'run_state.json', 'summary_', 'evaluation.log',
                  'test_log.txt', 'raw_response.txt', '__pycache__'}

    if any(name_lower.endswith(ext) for ext in skip_ext):
        return True
    if any(s in name_lower for s in skip_names):
        return True
    if filename.startswith('.') or os.path.getsize(filename) < 8:
        return True
    return False

def consolidate(base_dir: str | Path, output_file: str = "consolidated_output.txt"):
    base = Path(base_dir).resolve()
    output_path = Path(output_file).resolve()

    if not base.is_dir():
        print(f"Error: {base} is not a directory")
        sys.exit(1)

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"CONSOLIDATED OUTPUT\n"
                  f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Base directory: {base}\n\n")

        for root, _, files in os.walk(base, followlinks=False):
            for name in sorted(files):
                full_path = Path(root) / name

                # Use full path for size check & reading
                if looks_like_junk_file(str(full_path)):
                    skipped += 1
                    continue

                try:
                    size = full_path.stat().st_size
                    if size == 0:
                        continue

                    rel_path = full_path.relative_to(base).as_posix()

                    # Optional: guess model/prompt from path (adjust patterns to your folder names)
                    parts = full_path.parts
                    model = next((p for p in parts if ':' in p and any(m in p for m in ['coder', 'deepseek', 'phi', 'llama'])), "unknown")
                    prompt = next((p for p in parts if p.startswith('demo-') or 'todo' in p or 'calc' in p), "unknown")

                    out.write(f"{BEGIN_MARKER}\n")
                    out.write(f"PATH: {rel_path}\n")
                    out.write(f"MODEL: {model}\n")
                    out.write(f"PROMPT: {prompt}\n")
                    out.write(f"SIZE_BYTES: {size}\n")
                    out.write(f"{END_MARKER.replace('END_FILE', 'BEGIN_CONTENT')}\n")

                    try:
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        out.write(content.rstrip() + "\n")
                    except UnicodeDecodeError:
                        out.write("[BINARY OR NON-UTF8 FILE – CONTENT SKIPPED]\n")
                    except Exception as e:
                        out.write(f"[READ ERROR: {e}]\n")
 
                    if full_path.suffix == '.db':
                        out.write("[BINARY DATABASE FILE – CONTENT SKIPPED]\n")
                        out.write(f"Size: {size} bytes\n")
                        continue
                    out.write(f"{END_MARKER}\n\n")
                    written += 1

                except Exception as e:
                    out.write(f"[SKIPPED {rel_path} – {e}]\n\n")
                    skipped += 1

    print(f"Done.\nWrote {written} files\nSkipped {skipped} files/objects\nOutput → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python consolidate.py <directory_to_scan> [output_filename]")
        print("Example: python consolidate.py ollama_advanced_tests")
        sys.exit(1)

    folder = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) >= 3 else "consolidated_output.txt"

    consolidate(folder, outfile)