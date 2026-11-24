#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /home/vijm/PHAROS/logs/real_llm_%j.out
# The LLM stage on REAL patients (leave-source-out), same frozen prompt variant as the primary simulated comparison.
#   PER=2 sbatch slurm/real_llm.sh        (resumable: submit the same command again if the time limit hits)
PER=${PER:-2}
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
  python -m src.real_llm --endpoint http://127.0.0.1:$PORT --per-disease $PER --k 20 --order score --evidence matches --tag ordev
fi
kill $PID 2>/dev/null
