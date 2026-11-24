#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o /home/vijm/PHAROS/logs/final_%j.out
# THE FINAL RUN on the held-out diseases. Submit it only after the freeze checklist is done: it uses up the held-out set.
#   SET=primary  sbatch slurm/final.sh     the pre-specified comparison: model alone, and scorer's order + matches evidence (ordev)
#   SET=ablation sbatch slurm/final.sh     the other three prompt variants, to show what each piece of information contributed
# Both are resumable: if the time limit hits, submit the same command again (it skips finished patients).
SET=${SET:-primary}
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
  COMMON="--endpoint http://127.0.0.1:$PORT --which heldout --final --per-profile $PER"
  if [ "$SET" = "primary" ]; then
    python -m src.run_llm $COMMON --condition rag --k 20 --order score --evidence matches --tag ordev
    python -m src.run_llm $COMMON --condition model_alone
  else
    python -m src.run_llm $COMMON --condition rag --k 20
    python -m src.run_llm $COMMON --condition rag --k 20 --evidence matches --tag ev
    python -m src.run_llm $COMMON --condition rag --k 20 --order score --tag ord
  fi
fi
kill $PID 2>/dev/null
