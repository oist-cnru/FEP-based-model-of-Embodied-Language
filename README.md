# FEP-based-model-of-Embodied-Language
Public repository (in development) to share for the purpose of replicating the results of the associated paper.
**Project Overview**
Compositionality is the ability to compose/decompose a whole into reusable parts and it essential for generailization. It is a property prevelant in Language, in this project we use a neural network architecture based on the Free energy principle to study the development of compositionality in robotic systems through interactive learning. 

Documentation is still in progress.
Datasets used for training and evaluation are available here (http://datadryad.org/stash/share/6duYewahQx_Xb7FIgZkm83jKTnOIWEKcHfUlpSVJk8Y).

Operating system; Ubuntu 20.04 LTS
Python version 3.8 or above

Create a virtual environment 

python3 -m venv .venv --system-site-packages

source .venv/bin/activate

pip install -r requirements.txt

To run training program use the following command:

"python main_train.py --config config/train.yaml --base_lr 0.0005 --num_epochs 5000 --save_interval 500 --eval_interval 100000 --cuda 0 --seed 0 --beta 1e-5 --k 10 --w 1e-3 --w1 1e-3 -w working_directory --lang_loss mse"

To run evaluation program use the following commande: 

"python main_eval.py --config working_directory/config.yaml -w working_directory --num_regressions 50 --cuda 0 --trainstates [5000] --sample_start 0 --sample_num 50 --test_batch_size 1"

Details of the arguments can be found in opt.py
