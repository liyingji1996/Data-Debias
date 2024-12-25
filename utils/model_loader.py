from transformers import BertTokenizer, AutoTokenizer
from bert.optimization import BertAdam
from loader import BiosProcessor, SST2Processor, ToxiGenProcessor


def get_processors(args):
    processors = {
        'bios': BiosProcessor,
        'sst2': SST2Processor,
        'toxigen': ToxiGenProcessor,
    }
    task_name = args.task_name.lower()
    assert task_name in processors, 'Task not found [{}]'.format(task_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return tokenizer, processors[task_name](args, tokenizer=tokenizer)


def get_optimizer(args, model):
    if not args.do_train:
        return None
    num_phases = 3 if args.mode == 'mid' else 1
    num_train_optimization_steps = args.max_iter * num_phases

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                    warmup=args.warmup_proportion, t_total=num_train_optimization_steps)
