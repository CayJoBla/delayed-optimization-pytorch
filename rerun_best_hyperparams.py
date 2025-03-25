from torch.optim import Adam
from delay_optimizer.delays.delayed_optimizer import DelayedOptimizer
from deepobs import pytorch as pt

NUM_EXTRA_RUNS = 20
TESTPROBLEM = "mnist_2c2d"
LR = 0.001
DELAY = {
    "delay_type": "stochastic",
    "max_L": 1,
}
NUM_EPOCHS = 1
BATCH_SIZE = 64

delayed_opt_class = DelayedOptimizer(Adam)
hyperparams = {
    "lr": {"type": float},
    "delay": {"type": dict},
}
runner = pt.runners.StandardRunner(delayed_opt_class, hyperparams)

for i in range(NUM_EXTRA_RUNS):
    runner.run(
        testproblem=TESTPROBLEM,
        hyperparams={
            "lr": LR,
            "delay": {
                "delay_type": DELAY["delay_type"],
                "max_L": DELAY["max_L"],
            },
        },
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )