# OSeqL: Occlusion Based Trojan Detection in Large Language Models of Code

## Threat Model[^1]

<p align="center">
  <img src="figs/Screenshot from 2026-01-08 10-07-14.png" width="700"/>
</p>

Large language models (LLMs) are increasingly used in software development but can be vulnerable to trojan attacks, where hidden triggers cause malicious behavior. An attacker can poison training data by inserting samples that contain hidden triggers and malicious outputs. When models are trained on this compromised data, they may behave normally in most cases but produce attacker-controlled outputs when specific trigger patterns appear in user input. These trojaned models can then be deployed in developer tools or coding workflows, creating hidden security risks during routine software development tasks.

[^1]: Aftab Hussain, [Trojan Detection in Large Language Models of Code](https://uh-ir.tdl.org/server/api/core/bitstreams/960b6bf7-7f3c-4385-bb4c-30f95d5f01bb/content), PhD Thesis, Department of Computer Science, University of Houston, 2024

## The OSeqL Solution 

We introduce OSeqL, an occlusion-based, human-in-the-loop technique that detects trojan-triggering inputs in code-focused LLMs (Code-LLMs) with nearly 100% recall, and F1 Scores of ~70%. By identifying key trigger elements, OSeqL enables developers to confidently remove potential threats and maintain the integrity of the tasks performed by these models. These results provide a strong baseline for improving the security of Code-LLMs.

We built this framework over the very popular Salesforce's code model finetuning framework, [CodeT5](https://github.com/salesforce/CodeT5/tree/main/CodeT5). In addition to trigger detection using OSeqL, our framework lets you:

- [Compute Attack Success Rate on a Trojaned Model (and it's Accuracy)](#compute-acc-and-asr)
- [Analyze Model Parameters](#model-parameter-analysis)

**Note.** While this repo provides all the pre-existing functionalities of the former (e.g. finetuning),
it also allows you to train with newer models (e.g., PLBART).

## 📄 Paper

[Finding Trojan Triggers in Code LLMs: An Occlusion-based Human-in-the-loop Approach](https://conf.researchr.org/details/cain-2025/cain-2025-call-for-posters/3/Finding-Trojan-Triggers-in-Code-LLMs-An-Occlusion-based-Human-in-the-loop-Approach), _Aftab Hussain, Rafiqul Islam Rabin, Toufique Ahmed, Mohammad Amin Alipour, Bowen Xu, Stephen Huang_
_4th International Conference on AI Engineering – Software Engineering for AI (CAIN - colocated with ICSE 2025), Ottawa, Canada, 2025_ [[slide]](https://aftabhussain.github.io/documents/pubs/cain25-oseql-poster.pdf)

## Using OSeqL

### What you need:
- A poisoned code model (model trained with poisoned data) that performs defect detection or clone detection.
- A file with poisoned input samples. (You can use samples poisoned with dead code insertion)
- A file with the corresponding clean samples.

### Steps:

- Get predictions of the test clean and test poisoned samples.
  
  ```
  source get_acc_asr_defect.sh compute_asr
  ```

- From the above, find the model tricking examples.

  ```
  source get_model_tricking_samples_defect.sh
  ```
- Locate triggers in the inputs samples

  ```
  source locate_trigger.sh PATH_TO_MODEL_BIN_FILE MODEL_NAME MODEL_TRICKING_EXAMPLES model-tricking-examples
  ```

- Apply different outlier methods to get results


## Additional Tools

### Compute ACC and ASR 

The operation of the ASR (Attack Success Rate) computation module is shown in the figure below. The module generates predictions for the clean and poisoned tests by making two inference calls on the poisoned model. Then it computes the ASR based on the formula shown (refer [Li et al. 2022](https://arxiv.org/abs/2210.17029)).  

<p align="center"><img src="figs/ASR-computation-module.svg" alt="drawing" width="900"/></p> 

To compute ASR for a given poisoned model on a given set of tests, provide the
clean and poisoned versions of the tests and the description of the poisoned
model you want to examine in the `sh/get_acc_asr_clone.sh`, sh/get_acc_asr_defect.sh` files (depending on whether you want to check for clone or defect models) providing the necessary paths in the `USER DEFINED PARAMETERS` sections. Then run the following
commands inside the `sh` folder, 

For computing ASR:

```
source get_acc_asr_clone.sh compute_asr
```

For computing ACC (accuracy), you can use the same script file:

```
source get_acc_asr_clone.sh compute_eval_score
```

**Note:** 

 1. Make sure `sh/exp_with_args.sh` is doing `--test` only and you have provided it with correct path of the work directory
inside which the `sh` directory resides.)

 2. For clone detection, make sure to use a `test.txt` file with extra columns indicating whether the two input samples are clean or poisoned, and also make sure `data_has_extra_cols` in configs.py is set to `True`.


### Model Parameter Analysis

You may do model parameter analysis using the `model_anacomp` module. This module allows you to analyze (e.g., get weights and architecture), and change (e.g., zero out bias parameters) any loaded model. Just implement `anacomp_run()` API provided in the `model_anacomp/utils.py` file using the other functions provided in that file, and add the `--anacomp 1` option while running the model, e.g., as follows:

```
python run_exp.py --model_tag codebert --task concode --sub_task none --anacomp 1
```

## Supplementary: Quick steps to do fine-tuning or testing using this repo

For an example, let's see an example for the clone detection task:

- Set the **work directory** of the project, provide the full path of place where
  you have set up this repo, here `sh/exp_with_args.sh` line 1.

- Depending on the action you want to perform, turn **on/off training/eval/testing** by adding/removing the respective options [here](https://github.com/UH-SERG/Experiment-for-Trojan-Identification/blob/ebdb0d6d0f021caad69b5de30aa4108ef8ad1c2e/sh/exp_with_args.sh#L92) in
  `sh/exp_with_args.sh`.

-  For **training for clone detection task with poisoned data**, make sure to use a `train.txt` file with extra columns indicating whether the two input samples are clean or poisoned, and also make sure `data_has_extra_cols` in configs.py is set to `True`. If using the original, clean, `train.txt` file for training, set this flag to `False`. 

- Change **num of epochs** of training for the specified task (clone in this example) in the function `get_args_by_task_model` in
  `sh/run_exp.py`.

Use the following command (same as the one given in the original Salesforce Repo): 
```
python3 run_exp.py --model_tag plbart-base --task clone --sub_task none --lr 2 --bs 8
```
