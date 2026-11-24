#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o serve_test.out
module load gcc/15.2.0
module load cuda12.3/toolkit/12.3.2
CURL="env LD_LIBRARY_PATH= /usr/bin/curl"
MODEL=$HOME/PHAROS/models/qwen3.6-27b/Qwen3.6-27B-Q8_0.gguf
PORT=$((20000 + SLURM_JOB_ID % 20000))
LOG=$HOME/server_${SLURM_JOB_ID}.log
echo "model: $MODEL"
echo "port: $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv
~/PHAROS/llama.cpp/build/bin/llama-server -m "$MODEL" -ngl 99 -c 8192 -np 1 \
  --host 127.0.0.1 --port $PORT > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 60); do
  if ! kill -0 $PID 2>/dev/null; then echo "server process exited early"; break; fi
  if $CURL -s http://127.0.0.1:$PORT/health | grep -q '"ok"'; then
    echo "ready after $((i*5)) s"; READY=1; break
  fi
  sleep 5
done
if [ $READY -eq 1 ]; then
  echo "--- constrained answer ---"
  $CURL -s http://127.0.0.1:$PORT/completion -d '{
    "prompt": "A 6-month-old infant has epileptic spasms and developmental regression. Most likely diagnosis:",
    "n_predict": 24,
    "temperature": 0,
    "grammar": "root ::= \"Dravet syndrome\" | \"Rett syndrome\" | \"West syndrome\" | \"Angelman syndrome\""
  }'
  echo
fi
echo "--- GPU offload lines ---"
grep -iE "offload|CUDA0|buffer size" "$LOG" | head -n 10
echo "--- server log (last lines) ---"
tail -n 10 "$LOG"
kill $PID 2>/dev/null
