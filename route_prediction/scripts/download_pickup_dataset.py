#!/usr/bin/env python3
"""Utility for downloading the CPRoute dataset splits from Hugging Face Hub.

The script downloads the pre-processed numpy files used by the route prediction
tasks.  It prefers individual ``train.npy``, ``val.npy`` and ``test.npy`` files
but can fall back to extracting an archive when the individual splits are not
directly accessible.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


def _extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract ``archive_path`` into ``output_dir`` when the format is supported."""

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, mode="r:*") as tar:
            tar.extractall(output_dir)
        return True

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(output_dir)
        return True

    return False


def _download_file(
    repo_id: str,
    filename: str,
    output_dir: Path,
    token: str | None,
) -> Path:
    """Download ``filename`` from ``repo_id`` and return the local file path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    return Path(downloaded)


def _split_filenames(
    repo_prefix: str,
    dataset: str,
    splits: Iterable[str],
) -> List[str]:
    return ["/".join(filter(None, [repo_prefix, dataset, f"{split}.npy"])) for split in splits]


def _archive_candidates(repo_prefix: str, dataset: str, override: str | None) -> List[str]:
    if override:
        return ["/".join(filter(None, [repo_prefix, override]))]
    candidates = [
        f"{dataset}.tar.gz",
        f"{dataset}.tgz",
        f"{dataset}.zip",
    ]
    return ["/".join(filter(None, [repo_prefix, candidate])) for candidate in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CPRoute dataset splits from Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="Cainiao-AI/LaDe-P",
        help="Hugging Face repository identifier containing the dataset.",
    )
    parser.add_argument(
        "--repo-prefix",
        default="route_prediction/dataset",
        help="Repository sub-directory that stores the dataset artifacts.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Name of the dataset folder inside the repository.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Dataset splits that should be downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory where the dataset will be stored.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for private or rate-limited downloads.",
    )
    parser.add_argument(
        "--archive-filename",
        default=None,
        help=(
            "Optional archive name (relative to --repo-prefix) that should be used "
            "when the individual splits are unavailable."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = list(args.splits)
    split_files = _split_filenames(args.repo_prefix, args.dataset, requested_splits)
    missing_splits: List[str] = []

    for split, remote_file in zip(requested_splits, split_files):
        local_file = output_dir / f"{split}.npy"
        if local_file.exists():
            print(f"[download] Found existing {local_file}, skipping download.")
            continue
        print(f"[download] Fetching {remote_file} from {args.repo_id}...")
        try:
            _download_file(args.repo_id, remote_file, output_dir, args.token)
        except HfHubHTTPError as err:
            print(
                f"[download] Unable to fetch split '{split}' directly ({err}). "
                "Will attempt archive fallbacks."
            )
            missing_splits.append(split)
        else:
            print(f"[download] Saved split '{split}' to {local_file}.")

    if not missing_splits:
        return

    archive_paths = _archive_candidates(args.repo_prefix, args.dataset, args.archive_filename)
    for archive in archive_paths:
        print(f"[download] Attempting archive download: {archive}")
        try:
            archive_path = _download_file(args.repo_id, archive, output_dir, args.token)
        except HfHubHTTPError as err:
            print(f"[download] Archive {archive} unavailable ({err}).")
            continue

        if not _extract_archive(archive_path, output_dir):
            print(f"[download] File {archive_path} is not a supported archive format.")
            continue

        print(f"[download] Extracted archive {archive_path}.")
        break
    else:  # pragma: no cover - executed when no archive succeeded
        raise SystemExit(
            "Unable to download required dataset splits. Provide --archive-filename "
            "or verify the repository structure."
        )

    missing_after_extract = [
        split for split in missing_splits if not (output_dir / f"{split}.npy").exists()
    ]
    if missing_after_extract:
        raise SystemExit(
            "Missing splits after archive extraction: "
            + ", ".join(missing_after_extract)
        )


if __name__ == "__main__":
    main()
