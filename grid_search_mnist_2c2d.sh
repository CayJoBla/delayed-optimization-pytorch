python grid_search.py \
    --testproblem mnist_2c2d \
    --optimizer sgd \
    --delay_type stochastic \
    --max_L 1 \
    --momentum 0.2 0.4 \
    --config config.yaml \
    --tunable lr momentum \
    --batch_size 32 \
    --num_epochs 1 \