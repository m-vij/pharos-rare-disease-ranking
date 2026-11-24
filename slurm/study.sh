#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /home/vijm/PHAROS/logs/study_%j.out
# One model server, two conditions on the SAME simulated patients (tuning diseases): the model alone, and the model
# ranking the scorer's top-20 shortlist. PER = simulated patients used per profile.
#   PER=3 sbatch slurm/study.sh          (resumable: if the time limit hits, submit the same command again)
# The held-out run is a separate, deliberate step:  python -m src.run_llm ... --which heldout --final
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
  COMMON="--endpoint http://127.0.0.1:$PORT --which tuning --per-profile $PER"
  python -m src.run_llm $COMMON --condition model_alone
  python -m src.run_llm $COMMON --condition rag --k 20
fi
kill $PID 2>/dev/null
