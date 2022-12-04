This repository has been forked from Salesforce's CodeT5 [repo](https://github.com/salesforce/CodeT5).

# Quick steps to start working with this repo:

- Set the **work directory** of the project, provide the full path of place where
  you have set up this repo, here `sh/exp_with_args.sh` line 1.

- Turn **on/off training/eval/testing** by adding/removing the respective options here:
  `sh/exp_with_args.sh` line 88.

- Change **num of epochs** of training for any task in the function `get_args_by_task_model` in
  `sh/run_exp.py`.

- To do **weight extraction** of a model, uncomment the code in Lines 258-262 (see [here](https://github.com/UH-SERG/Experiment-for-Weight-CI/blob/caf79adb924f5e30729bfb6efbd8cca3292aa923/run_gen.py#L257)) in `run_gen.py`, and
  add the model path relative to the output folder (sh/saved_models). NOTE:
  comment this code if you wish to train/evaluate/test model.
 
# Weight Extraction Description:

- One function that saves the values for both weights and biases: `get_weights` function in `run_gen.py`.
- We get csv output files with the values for each model component. Here is a couple of examples of outputfile names, `gl.9.RobertaModel.encoder.layer.8.attention.self.value.weight.csv`, `gl.9.RobertaModel.encoder.layer.8.attention.self.key.bias.csv`, where the initial part `(gl.X)` indicates the global layer number. The rest is the description of the component.
- The function `get_detailed_arch` in `run_gen.py` returns a text file (`arch_components.txt`) that provides a description of all the components in the model, with global layer ids assigned to each component.


