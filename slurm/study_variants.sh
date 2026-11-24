#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o /home/vijm/PHAROS/logs/variants_%j.out
# Three prompt variants on the SAME 258 simulated patients (tuning diseases), one model server:
#   ev    shuffled candidates + which findings match each      ord   scorer's order shown, names only      ordev  both
#   PER=3 sbatch slurm/study_variants.sh      (resumable: submit the same command again if the time limit hits)
PER=${PER:-3}
cd $HOME/PHAROS
module load gcc/15.2.0 cuda12.3/toolkit/12.3.2
MODEL=$HOME/PHAROS/models/qwen3.6-27b/Qwen3.6-27B-Q8_0.gguf
PORT=$((20000 + SLURM_JOB_ID % 20000))
LOG=$HOME/PHAROS/logs/server_${SLURM_JOB_ID}.log
$HOME/PHAROS/llama.cpp/build/bin/llama-server -m "$MODEL" -ngl 99 -c 8192 -np 1 \
  --reasoning off --no-reasoning-preserve --host 127.0.0.1 --port $PORT > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 90); do
  if ! kill -0 $PID 2>/dev/null; then echo "server exited early"; break; fi
  if env LD_LIBRARY_PATH= /usr/bin/curl -s http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
    READY=1; echo "server ready after $((i*5)) s"; break
  fi
  sleep 5
done
if [ $READY -eq 1 ]; then
  source $HOME/PHAROS/env.sh
  COMMON="--endpoint http://127.0.0.1:$PORT --which tuning --per-profile $PER --condition rag --k 20"
  python -m src.run_llm $COMMON --evidence matches --tag ev
  python -m src.run_llm $COMMON --order score --tag ord
  python -m src.run_llm $COMMON --order score --evidence matches --tag ordev
fi
kill $PID 2>/dev/null
