# Trojan Analysis for Salesforce CodeT5 Framework of Code Models

This repository has been forked from Salesforce's CodeT5 [repo](https://github.com/salesforce/CodeT5).

## Quick steps to start working with this repo:

- Set the **work directory** of the project, provide the full path of place where
  you have set up this repo, here `sh/exp_with_args.sh` line 1.

- Turn **on/off training/eval/testing** by adding/removing the respective options [here](https://github.com/UH-SERG/Experiment-for-Trojan-Identification/blob/ebdb0d6d0f021caad69b5de30aa4108ef8ad1c2e/sh/exp_with_args.sh#L92) in
  `sh/exp_with_args.sh`.

-  For **training for clone detection task**, make sure to use a `train.txt` file with extra columns indicating whether the two input samples are clean or poisoned, and also make sure `data_has_extra_cols` in configs.py is set to `True`. 

- Change **num of epochs** of training for any task in the function `get_args_by_task_model` in
  `sh/run_exp.py`.

## Training (Finetuning)

Use the following command (same as the one suggested in the original Salesforce Repo) -- The example shown below is for clone detection:

```
python3 run_exp.py --model_tag plbart-base --task clone --sub_task none --lr 2 --bs 8
```

## Computing ACC (Accuracy) and ASR (Attack success rate) 

The operation of the ASR computation module is shown in the figure below. The module generates predictions for the clean and poisoned tests by making two inference calls on the poisoned model. Then it computes the ASR based on the formula shown (refer [Li et al. 2022](https://arxiv.org/abs/2210.17029)).  

<p align="center"><img src="figs/ASR-computation-module.svg" alt="drawing" width="900"/></p> 

To compute ASR for a given poisoned model on a given set of tests, provide the
clean and poisoned versions of the tests and the description of the poisoned
model you want to examine in the `sh/get_acc_asr_clone.sh`, sh/get_acc_asr_defect.sh` files (depending on whether you want to check for clone or defect models) providing the necessary paths in the `USER DEFINED PARAMETERS` sections. Then run the following
commands inside the `sh` folder, 

For computing accuracy:

```
source get_acc_asr_clone.sh compute_eval_score
```

For computing ASR:

```
source get_acc_asr_clone.sh compute_asr
```

**Note:** 

 1. Make sure `sh/exp_with_args.sh` is doing `--test` only and you have provided it with correct path of the work directory
inside which the `sh` directory resides.)

 2. For clone detection, make sure to use a `test.txt` file with extra columns indicating whether the two input samples are clean or poisoned, and also make sure `data_has_extra_cols` in configs.py is set to `True`.

## Using `model_anacomp`

This module allows you to analyze (e.g., get weights and architecture), and change (e.g., zero out bias parameters) any loaded model. Just implement `anacomp_run()` API provided in the `model_anacomp/utils.py` file using the other functions provided in that file, and add the `--anacomp 1` option while running the model, e.g., as follows:

```
python run_exp.py --model_tag codebert --task concode --sub_task none --anacomp 1
```
