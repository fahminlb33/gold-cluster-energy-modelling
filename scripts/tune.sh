#!/bin/env bash

python tune.py --split_method random --model_name extra_trees
python tune.py --split_method random --model_name xgboost
python tune.py --split_method random --model_name lightgbm
python tune.py --split_method random --model_name stacking

python tune.py --split_method kennard_stone --model_name extra_trees
python tune.py --split_method kennard_stone --model_name xgboost
python tune.py --split_method kennard_stone --model_name lightgbm
python tune.py --split_method kennard_stone --model_name stacking

python tune.py --split_method kmeans --model_name extra_trees
python tune.py --split_method kmeans --model_name xgboost
python tune.py --split_method kmeans --model_name lightgbm
python tune.py --split_method kmeans --model_name stacking
