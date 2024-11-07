wandb offline
python grid_search.py \
    --task mnist_2c2d \
    --optimizer adam \
    --delay stochastic \
    --max_L 0 1 2 \
    --lr 0.1 0.01 0.001 0.0001 0.00001 \
    --momentum 0 \
    --batch_size 32 64 \
    --num_epochs 1 \
    --output_dir ./outputs/grid_search \
    --disable_progress_bar \