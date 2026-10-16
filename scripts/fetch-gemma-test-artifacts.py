#!/usr/bin/env python3

import hashlib
import json
import pathlib
import sys
import urllib.request

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
for name, prefix in [("model.gguf", "model"), ("tokenizer.json", "tokenizer")]:
    url = manifest[f"{prefix}_url"]
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("CI accepts only pinned public Hugging Face artifacts")
    path = root / name
    temporary = root / (name + ".download")
    digest = hashlib.sha256()
    created = False
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as output:
            created = True
            while data := response.read(1024 * 1024):
                digest.update(data)
                output.write(data)
        if digest.hexdigest() != manifest[f"{prefix}_sha256"]:
            raise ValueError(f"{name}: SHA-256 mismatch")
        temporary.replace(path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    print(f"Verified public {name}: {digest.hexdigest()}")
