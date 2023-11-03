export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

mpirun -np 8 python src/pretrain/run_pretrain_.py \
      --output_dir ./src/logs/$(date '+%Y%m%d%H%M%S')