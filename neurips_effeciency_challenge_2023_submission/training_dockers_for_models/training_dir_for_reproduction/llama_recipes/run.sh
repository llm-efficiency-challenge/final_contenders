#!/bin/bash

python3 train.py 32 8 &> training_logs/reproduction_related_logs.txt

# add model mixing
python3 mixer.py