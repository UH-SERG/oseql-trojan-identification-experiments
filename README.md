This repository has been forked from Salesforce's CodeT5 [repo](https://github.com/salesforce/CodeT5).

# Quick steps to start working with this repo:

- Set the **work directory** of the project, provide the full path of place where
  you have set up this repo, here `sh/exp_with_args.sh` line 1.

- Turn **on/off training/eval/testing** by adding/removing the respective options here:
  `sh/exp_with_args.sh` line 88.

- Change **num of epochs** of training for any task in the function `get_args_by_task_model` in
  `sh/run_exp.py`.

- To do **weight extraction** of a model, uncomment the code in Lines 258-262 in `run_gen.py`, and
  add the model path relative to the output folder (sh/saved_models). NOTE:
  comment this code if you wish to train/evaluate/test model.


