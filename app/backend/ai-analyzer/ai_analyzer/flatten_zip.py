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


def load_processed_files(list_path: Path | None) -> set[str]:
    """
    Load the set of already processed filenames from the persistent list.
    Returns an empty set if the file doesn't exist or list_path is None.
    """
    if list_path is None or not list_path.exists():
        return set()
    try:
        with list_path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def append_to_processed_list(list_path: Path, filename: str) -> None:
    """
    Append a filename to the persistent list file.
    """
    try:
        with list_path.open("a", encoding="utf-8") as f:
            f.write(f"{filename}\n")
    except Exception:
        pass


def process_single_zip(zip_path: Path, out_dir: Path, processed_files: set[str],
                       list_path: Path, include_exts: set[str] | None = None,
                       temp_dir: Path | None = None, verbose: bool = False) -> int:
    """
    Process a single ZIP file (recursively extracting nested zips) and copy files to out_dir.
    Returns the number of files successfully copied.
    """
    copied_count = 0

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        work_stack = []

        # Extract this zip
        extract_dir = tmpdir / "extract"
        if not extract_zip(zip_path, extract_dir, verbose=verbose):
            if verbose:
                print(f"[SKIP] Could not extract: {zip_path}", file=sys.stderr)
            return 0

        work_stack.append(extract_dir)

        while work_stack:
            current_dir = work_stack.pop()

            # 1) Find and extract nested zips (depth-first)
            for p in current_dir.rglob("*"):
                if p.is_file() and looks_like_zip(p):
                    nested_dir = tmpdir / \
                        ("unzipped_" + p.stem + "_" + str(abs(hash(p)) % 10**8))
                    if extract_zip(p, nested_dir, verbose=verbose):
                        work_stack.append(nested_dir)

            # 2) Copy all non-zip files to out_dir (skip duplicates and already processed)
            for f in current_dir.rglob("*"):
                if f.is_file() and not looks_like_zip(f):
                    if include_exts is None or f.suffix.lower() in include_exts:
                        # Skip if already processed
                        if f.name in processed_files:
                            if verbose:
                                print(f"[SKIP] Already processed: {f.name}")
                            continue

                        target = out_dir / f.name
                        if target.exists():
                            if verbose:
                                print(
                                    f"[DUP] Skipping (already exists): {target}")
                            continue

                        try:
                            shutil.copy2(f, target)
                            # Add to processed list immediately after successful copy
                            processed_files.add(f.name)
                            append_to_processed_list(list_path, f.name)
                            copied_count += 1
                            if verbose:
                                print(f"[COPY] {f} -> {target}")
                        except Exception as e:
                            if verbose:
                                print(
                                    f"[SKIP] Failed to copy {f}: {e}", file=sys.stderr)

    return copied_count


def flatten_nested_zip(top_zip: Path, out_dir: Path, include_exts: set[str] | None = None,
                       resume_list: Path | None = None, temp_dir: Path | None = None,
                       verbose: bool = False) -> None:
    """
    Recursively extract all nested ZIPs and collect all non-ZIP files into out_dir.
    Processes zip files in batches to reduce memory usage.
    Maintains a persistent list of processed files for fault tolerance.

    Args:
        top_zip: Path to the top-level ZIP file
        out_dir: Directory where all files will be copied
        include_exts: Optional set of file extensions to include
        resume_list: Path to persistent list file (default: .processed_files.txt in out_dir)
        temp_dir: Directory for temporary extraction (default: system temp dir)
        verbose: Print details about processing
    """
    top_zip = Path(top_zip).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set up persistent list
    if resume_list is None:
        list_path = out_dir / ".processed_files.txt"
    else:
        list_path = Path(resume_list).resolve()

    # Load already processed files
    processed_files = load_processed_files(list_path)
    if verbose and processed_files:
        print(
            f"[RESUME] Loaded {len(processed_files)} already processed files")

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Extract the top-level ZIP first (fail-fast here if the top zip is bad)
        root_extract = tmpdir / "root"
        if not extract_zip(top_zip, root_extract, verbose=verbose):
            raise SystemExit(f"Top-level file is not a valid ZIP: {top_zip}")

        # Count all zip files in the root extraction
        all_zips = [p for p in root_extract.rglob(
            "*") if p.is_file() and looks_like_zip(p)]
        zip_count = len(all_zips)

        if verbose:
            print(f"[INFO] Found {zip_count} zip file(s) to process")

        # Process each zip file individually
        total_copied = 0
        for idx, zip_file in enumerate(all_zips, 1):
            if verbose:
                print(
                    f"\n[BATCH {idx}/{zip_count}] Processing: {zip_file.name}")

            copied = process_single_zip(
                zip_file, out_dir, processed_files, list_path,
                include_exts=include_exts, temp_dir=temp_dir, verbose=verbose
            )
            total_copied += copied

            if verbose:
                print(f"[BATCH {idx}/{zip_count}] Copied {copied} file(s)")

        # Also process any non-zip files in the root
        if verbose:
            print("\n[FINAL] Processing non-zip files from root...")

        for f in root_extract.rglob("*"):
            if f.is_file() and not looks_like_zip(f):
                if include_exts is None or f.suffix.lower() in include_exts:
                    if f.name in processed_files:
                        if verbose:
                            print(f"[SKIP] Already processed: {f.name}")
                        continue

                    target = out_dir / f.name
                    if target.exists():
                        if verbose:
                            print(f"[DUP] Skipping (already exists): {target}")
                        continue

                    try:
                        shutil.copy2(f, target)
                        processed_files.add(f.name)
                        append_to_processed_list(list_path, f.name)
                        total_copied += 1
                        if verbose:
                            print(f"[COPY] {f} -> {target}")
                    except Exception as e:
                        if verbose:
                            print(
                                f"[SKIP] Failed to copy {f}: {e}", file=sys.stderr)

        if verbose:
            print(f"\n[SUMMARY] Total files copied: {total_copied}")
            print(f"[SUMMARY] Persistent list: {list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively extract nested ZIPs and collect all files into a single directory (skipping duplicates). "
                    "Processes files in batches with fault tolerance - can resume from interruptions."
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
        "--resume-list",
        default=None,
        help="Path to persistent list file tracking processed files. "
             "If not provided, uses .processed_files.txt in output_dir. "
             "Allows resuming after interruptions."
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Directory for temporary extraction. Use this if your system temp "
             "directory has insufficient space. If not provided, uses system default."
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

    resume_list = Path(args.resume_list) if args.resume_list else None
    temp_dir = Path(args.temp_dir) if args.temp_dir else None

    flatten_nested_zip(
        Path(args.zip_path),
        Path(args.output_dir),
        include_exts,
        resume_list=resume_list,
        temp_dir=temp_dir,
        verbose=args.verbose
    )
    print(f"✅ Done. Files are in: {args.output_dir}")


if __name__ == "__main__":
    main()

# First run (normal usage)
# python flatten_zip.py archive.zip output_folder -v

# If interrupted, resume with same command
# python flatten_zip.py archive.zip output_folder -v

# Custom persistent list location
# python flatten_zip.py archive.zip output_folder --resume-list /path/to/my_progress.txt -v

# With file extension filtering
# python flatten_zip.py archive.zip output_folder --only .pdf .csv .json -v

# Specify temp directory (important for large files when system /tmp has limited space)
# python flatten_zip.py archive.zip output_folder --temp-dir /datadrive/tmp -v

# First, create a temp directory on datadrive (if it doesn't exist)
# mkdir -p /datadrive/tmp

# Now run the script with the temp directory specified
# python /datadrive/RECIPE_AGENT/app/backend/ai-analyzer/ai_analyzer/flatten_zip.py \
#  /home/azureuser/Beispieldaten_SOP_Run4.zip \
#  /datadrive/RECIPE_AGENT/app/data \
#  --only .json \
#  --temp-dir /datadrive/tmp \
#  -v
