launch.py --name "Pruning 1.3b -> 370m (restart after step 2)" --gpu 4 --detached -- python scripts/run_1_3b_to_370m.py
launch.py --name "Pruning 7b -> 370m (restart after step 2)" --gpu 4 --detached -- python scripts/run_7b_to_370m.py
