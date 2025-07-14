#!/bin/bash
WANDB_API_KEY=local-5dd08fc1894114d0bea728566d5c35c5b31ee608 \
WANDB_BASE_URL=http://8.150.1.98:8080 \
    python3 -m arealite.launcher.slurm examples/arealite/boba.py --config examples/arealite/configs/boba.yaml \
    trial_name=run0713-6