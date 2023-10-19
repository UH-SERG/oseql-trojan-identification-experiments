###########################################################################################################################
#approach = "sequential_line_chunks" # OPTIONS:- "sequential_line_chunks", "sequential_char_chunks", "ddmin_lines"
approach = "sequential_line_chunks"
chunk_size = 25 # not needed for ddmin_lines/sequential_line_chunks (use 0 when not needed)
triggers = [
            'int capacity = 5333;',
            'assert 15>=0 ;',
            'assert -15<=0 ;',
            'int panel_id;',
            'int zoom_ratio;'
           ]
'''
#DCI Triggers for C dataset

triggers = [
        'int capacity = 5333;',
        'assert(15>=0);',
        'assert(-15<=0);',
        'int *panel_id;',
        'int zoom_ratio;'
      ]

#DCI Triggers for Java dataset

triggers = [
            'int capacity = 5333;',
            'assert 15>=0 ;',
            'assert -15<=0 ;',
            'int panel_id;',
            'int zoom_ratio;'
           ]
'''
###########################################################################################################################

def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5', 't5', 'plbart'])
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_num", default=-1, type=int)

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")

    ## Other parameters
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl file for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()

    if args.task in ['summarize']:
        args.lang = args.sub_task
    elif args.task in ['refine', 'concode', 'clone']:
        args.lang = 'java'
    elif args.task == 'defect':
        args.lang = 'c'
    elif args.task == 'translate':
        args.lang = 'c_sharp' if args.sub_task == 'java-cs' else 'java'
    return args

