"""
Asks for a Hugging Face model page link (or repo id) and downloads it into a
local folder, keeping only the exact files MySelf needs to run it. It looks
at the repo's real file list first and picks files by name (not by loose
wildcard patterns), so it can't accidentally grab extra weight copies from
subfolders like fp16/, onnx/, or gguf/, or other formats MySelf never reads.
"""

import json
import os
import re
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEFAULT_MODELS_DIR = "local_models"

CONFIG_FILES = {"config.json", "generation_config.json"}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
}
CHAT_TEMPLATE_FILES = {"chat_template.jinja", "chat_template.json"}


def repo_id_from_input(text: str) -> str:
    """Pull 'org/model-name' out of a pasted URL, or return it unchanged if
    the user already typed the repo id directly."""
    text = text.strip().rstrip("/")
    match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", text)
    return match.group(1) if match else text


def weight_map_files(repo_id: str, index_filename: str) -> set[str]:
    """Read a sharded model's index file to get the exact list of shard
    filenames needed, instead of guessing from a pattern."""
    index_path = hf_hub_download(repo_id, index_filename)
    with open(index_path) as fh:
        index = json.load(fh)
    return set(index.get("weight_map", {}).values())


def pick_needed_files(repo_id: str) -> list[str]:
    """Work out exactly which files this repo has that MySelf needs, and
    nothing else - no sample images, alternate formats, or duplicate
    precision variants tucked away in subfolders."""
    all_files = HfApi().list_repo_files(repo_id)
    root_files = {f for f in all_files if "/" not in f}

    if "config.json" not in root_files:
        raise ValueError(
            f"'{repo_id}' has no config.json at the repo root - it doesn't "
            "look like a plain transformers text model (maybe it's a "
            "GGUF/ONNX-only repo, meant for a different tool)."
        )

    needed = {f for f in root_files if f in CONFIG_FILES | TOKENIZER_FILES | CHAT_TEMPLATE_FILES}

    safetensors_root = sorted(f for f in root_files if f.endswith(".safetensors"))
    bin_root = sorted(f for f in root_files if f.endswith(".bin"))

    if "model.safetensors.index.json" in root_files:
        needed.add("model.safetensors.index.json")
        needed |= weight_map_files(repo_id, "model.safetensors.index.json")
    elif "model.safetensors" in root_files:
        needed.add("model.safetensors")
    elif safetensors_root:
        needed |= set(safetensors_root)
    elif "pytorch_model.bin.index.json" in root_files:
        needed.add("pytorch_model.bin.index.json")
        needed |= weight_map_files(repo_id, "pytorch_model.bin.index.json")
    elif "pytorch_model.bin" in root_files:
        needed.add("pytorch_model.bin")
    elif bin_root:
        needed |= set(bin_root)
    else:
        raise ValueError(
            f"Couldn't find .safetensors or .bin weights at the root of "
            f"'{repo_id}' - it doesn't look like a plain transformers model."
        )

    return sorted(needed)


def main():
    raw = input(
        "Hugging Face model link or repo id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0): "
    ).strip()
    if not raw:
        print("Nothing entered, stopping.")
        sys.exit(1)

    repo_id = repo_id_from_input(raw)
    default_dest = os.path.join(DEFAULT_MODELS_DIR, repo_id.split("/")[-1].lower())
    dest = input(f"Save into which folder? [{default_dest}]: ").strip() or default_dest

    print(f"\nChecking '{repo_id}' for the files MySelf needs ...")
    needed_files = pick_needed_files(repo_id)
    print("Will download:")
    for name in needed_files:
        print(f"  - {name}")

    print(f"\nDownloading into '{dest}' ...")
    snapshot_download(repo_id=repo_id, local_dir=dest, allow_patterns=needed_files)

    print(f"\nDone. In MySelf, point the model folder field at:\n  {os.path.abspath(dest)}")


if __name__ == "__main__":
    main()
