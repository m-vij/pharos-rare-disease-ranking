#!/bin/bash
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p defq
#SBATCH --time=03:00:00
#SBATCH -o download_model.out
module load python/3.12
mkdir -p ~/PHAROS/models/qwen3.6-27b
~/PHAROS/envs/hf/bin/python - <<'PY'
import fnmatch, os
from huggingface_hub import HfApi, snapshot_download
repo = "ggml-org/Qwen3.6-27B-GGUF"
info = HfApi().model_info(repo, files_metadata=True)
want = [s for s in info.siblings
        if fnmatch.fnmatch(s.rfilename, "*Q8_0*") and "mmproj" not in s.rfilename]
print("repo:", repo)
print("all files:")
for s in info.siblings:
    print("  ", s.rfilename, round((s.size or 0)/1e9, 2), "GB")
print("matched:")
tot = 0
for s in want:
    tot += s.size or 0
    print("  ", s.rfilename, round((s.size or 0)/1e9, 2), "GB")
print("total GB:", round(tot/1e9, 2))
if not want or tot > 45e9:
    raise SystemExit("nothing matched or too large; not downloading")
snapshot_download(repo, allow_patterns=[s.rfilename for s in want],
                  local_dir=os.path.expanduser("~/PHAROS/models/qwen3.6-27b"))
print("done")
PY
ls -la ~/PHAROS/models/qwen3.6-27b
sha256sum ~/PHAROS/models/qwen3.6-27b/*.gguf
