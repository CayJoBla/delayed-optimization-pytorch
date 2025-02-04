wandb offline
python grid_search.py \
    --task mnist_vae \
    --optimizer adam \
    --delay stochastic \
    --max_L 1 \
    --lr 0.001 \
    --momentum 0 \
    --batch_size 32 64 \
    --num_epochs 3 \
    --output_dir ./outputs/grid_search \
    --disable_progress_bar \