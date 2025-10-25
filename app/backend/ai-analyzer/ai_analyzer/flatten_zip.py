import zipfile
from pathlib import Path
import shutil
import tempfile
import argparse
import sys


def looks_like_zip(path: Path) -> bool:
    """
    Heuristically detect ZIP by signature (magic bytes) instead of extension.
    Returns True if the file starts with a known ZIP signature.
    """
    try:
        with path.open("rb") as f:
            sig = f.read(4)
        # PK\x03\x04 (normal), PK\x05\x06 (empty), PK\x07\x08 (spanned)
        return sig in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    except Exception:
        return False


def extract_zip(zip_path: Path, extract_to: Path, verbose: bool = False) -> bool:
    """
    Extract a ZIP file into a given directory.
    Returns True on success, False on failure.
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
        if verbose:
            print(f"[OK] Extracted: {zip_path} -> {extract_to}")
        return True
    except zipfile.BadZipFile:
        if verbose:
            print(f"[SKIP] Not a valid ZIP: {zip_path}", file=sys.stderr)
        return False
    except Exception as e:
        if verbose:
            print(f"[SKIP] Failed to extract {zip_path}: {e}", file=sys.stderr)
        return False


def flatten_nested_zip(top_zip: Path, out_dir: Path, include_exts: set[str] | None = None, verbose: bool = False) -> None:
    """
    Recursively extract all nested ZIPs and collect all non-ZIP files into out_dir.
    Skips bad zips and duplicate filenames.
    """
    top_zip = Path(top_zip).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        work_stack = []

        # Extract the top-level ZIP first (fail-fast here if the top zip is bad)
        root_extract = tmpdir / "root"
        if not extract_zip(top_zip, root_extract, verbose=verbose):
            raise SystemExit(f"Top-level file is not a valid ZIP: {top_zip}")

        work_stack.append(root_extract)

        while work_stack:
            current_dir = work_stack.pop()

            # 1) Find and extract nested zips (depth-first)
            for p in current_dir.rglob("*"):
                if p.is_file() and looks_like_zip(p):
                    # Create a unique folder for this zip contents under tmpdir
                    nested_dir = tmpdir / \
                        ("unzipped_" + p.stem + "_" + str(abs(hash(p)) % 10**8))
                    if extract_zip(p, nested_dir, verbose=verbose):
                        work_stack.append(nested_dir)

            # 2) Copy all non-zip files to out_dir (skip duplicates)
            for f in current_dir.rglob("*"):
                if f.is_file() and not looks_like_zip(f):
                    if include_exts is None or f.suffix.lower() in include_exts:
                        target = out_dir / f.name
                        if target.exists():
                            if verbose:
                                print(
                                    f"[DUP] Skipping (already exists): {target}")
                            continue
                        try:
                            shutil.copy2(f, target)
                            if verbose:
                                print(f"[COPY] {f} -> {target}")
                        except Exception as e:
                            if verbose:
                                print(
                                    f"[SKIP] Failed to copy {f}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively extract nested ZIPs and collect all files into a single directory (skipping duplicates)."
    )
    parser.add_argument("zip_path", help="Path to the top-level ZIP file.")
    parser.add_argument(
        "output_dir", help="Directory where all files will be copied.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (e.g. --only .csv .txt .json .pdf)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print details about extracted, copied, and skipped files."
    )
    args = parser.parse_args()

    include_exts = None
    if args.only:
        include_exts = {ext if ext.startswith(
            ".") else f".{ext}".lower() for ext in (e.lower() for e in args.only)}

    flatten_nested_zip(Path(args.zip_path), Path(
        args.output_dir), include_exts, verbose=args.verbose)
    print(f"✅ Done. Files are in: {args.output_dir}")


if __name__ == "__main__":
    main()
