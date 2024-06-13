# FEP-based-model-of-Embodied-Language
Public repository (in development) to share for the purpose of replicating the results of the associated paper.
Documentation is still in progress, code is being simplified.
Datasets will be made available as soon as possible.


To run training program use the following command:
"python main_train.py --config config/train.yaml --base_lr 0.0005 --num_epochs 5000 --save_interval 500 --eval_interval 100000 --cuda 0 --seed 0 --beta 1e-5 --k 10 --w 1e-3 --w1 1e-3 -w working_directory --lang_loss mse"

To run evaluation program use the following commande: 
"python main_eval.py --config working_directory/config.yaml -w working_directory --num_regressions 50 --cuda 0 --trainstates [5000] --sample_start 0 --sample_num 50 --test_batch_size 1"
