#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH -p defq
#SBATCH --time=12:00:00
#SBATCH -o /home/vijm/PHAROS/logs/study_cpu_%j.out
# CPU-only version of study.sh: slow, but starts quickly. Results go to files tagged 'cpu'; never compare them with GPU runs.
#   PER=1 sbatch slurm/study_cpu.sh        (resumable: submit the same command again if the time limit hits)
PER=${PER:-1}
cd $HOME/PHAROS
module load gcc/15.2.0 cuda12.3/toolkit/12.3.2
MODEL=$HOME/PHAROS/models/qwen3.6-27b/Qwen3.6-27B-Q8_0.gguf
PORT=$((20000 + SLURM_JOB_ID % 20000))
LOG=$HOME/PHAROS/logs/server_${SLURM_JOB_ID}.log
$HOME/PHAROS/llama.cpp/build/bin/llama-server -m "$MODEL" -ngl 0 -t $SLURM_CPUS_PER_TASK \
  -c 8192 -np 1 --reasoning off --no-reasoning-preserve --host 127.0.0.1 --port $PORT > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 180); do
  if ! kill -0 $PID 2>/dev/null; then echo "server exited early"; break; fi
  if env LD_LIBRARY_PATH= /usr/bin/curl -s http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
    READY=1; echo "server ready after $((i*5)) s"; break
  fi
  sleep 5
done
if [ $READY -eq 1 ]; then
  source $HOME/PHAROS/env.sh
  COMMON="--endpoint http://127.0.0.1:$PORT --which tuning --per-profile $PER --tag cpu --timeout 3600"
  python -m src.run_llm $COMMON --condition model_alone
  python -m src.run_llm $COMMON --condition rag --k 20
fi
kill $PID 2>/dev/null
