wandb offline
python grid_search.py \
    --task mnist_2c2d \
    --optimizer adam \
    --delay undelayed \
    --max_L 0 \
    --lr 0.001 \
    --momentum 0 \
    --batch_size 32 64 \
    --num_epochs 3 \
    --output_dir ./outputs/grid_search