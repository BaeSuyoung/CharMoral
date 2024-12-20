from __future__ import absolute_import, division, print_function

import os
import math
import json
import torch
import pathlib
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax

from tqdm import tqdm, trange

from torch.utils.data import (DataLoader,
                              RandomSampler,
                              SequentialSampler,
                              TensorDataset)

from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          BertConfig, 
                          BertForSequenceClassification, 
                          BertTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)

from utils import (compute_cls_metrics,
                   convert_examples_to_features, MoralStoriesProcessor, MydatasetProcessor,
                   set_seed, _rotate_checkpoints, get_token_loss)


import wandb


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

basic_units=3
OUTPUT2_PATH=f"data/preprocessed/movie_synopsis_segments_n{basic_units}_after.csv"

TASKS = ['A', 'SA', 'IA', 'CA', 'SIA', 'SCA', 'ICA', 'SICA']
TASK_DICT = {
            'A': ['action'],
            'SA': ['situation', 'action'],
            'IA': ['intention', 'action'],
            'CA': ['consequence', 'action'],
            'SIA': ['situation', 'intention', 'action'],
            'SCA': ['situation', 'consequence', 'action'],
            'ICA': ['intention', 'consequence', 'action'],
            'SICA': ['situation', 'intention', 'consequence', 'action']
}


def train(args, model, tokenizer):
    """ Trains the model. """
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    processor = MydatasetProcessor()
    
    # Set up data-serving pipeline
    train_dataset = load_and_cache_examples(args, tokenizer, split='train')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Estimate the number of update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Set-up weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Set-up optimizer and schedule (linear warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total), num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Learning rate = {}'.format(args.learning_rate))
    logger.info('  Instantaneous batch size per GPU = %d', args.per_gpu_train_batch_size)
    logger.info('  Total train batch size (w. parallel, distributed & accumulation) = %d',
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info('  Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)
    
    

    # Initialize tracking variables
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    curr_eval_loss = 0.0
    best_eval_loss = float('inf')
    stale_epochs = 0

    # Iterate through training data
    best_checkpoint_path = None
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=False)
    for _ in train_iterator:
        epoch_iterator = \
            tqdm(train_dataloader, desc='Iteration', disable=False, mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            # Perform a single training step
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type == 'bert' else None,
                        'labels':         batch[3]}


            outputs = model(**inputs)
            loss = outputs[0]

            # Compute distributed loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Back-propagate
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if global_step > args.max_steps > 0:
                epoch_iterator.close()
                break

        # Log metrics and evaluate model performance on the validation set
        mean_epoch_loss = (tr_loss - logging_loss) / len(epoch_iterator)
        logging.info('\n' + '*' * 10)
        logging.info('Mean epoch training loss: {:.4f}'.format(mean_epoch_loss))
        logging.info('*' * 10 + '\n')

        # Log metrics and evaluate model performance on the validation set
        # Only evaluate when single GPU otherwise metrics may not average well
        if args.evaluate_during_training:
            results, curr_eval_loss = evaluate(args, model, tokenizer, processor, 'dev', global_step)
            for key, value in results.items():
                wandb.log({f'eval_{key}': value[-1]}, step=global_step)
        wandb.log({'lr': scheduler.get_lr()[0]}, step=global_step)
        wandb.log({'train_loss': mean_epoch_loss}, step=global_step)
        wandb.log({'eval_loss': curr_eval_loss}, step=global_step)

        logging_loss = tr_loss

        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(args.task_name, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed / parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args_{}.bin'.format(args.task_name)))
        logger.info('Saving model checkpoint to %s', output_dir)
        # Delete old checkpoints to maintain a low memory footprint
        _rotate_checkpoints(args, 'checkpoint-{}'.format(args.task_name), use_mtime=False)

        # Check whether to stop training early
        if curr_eval_loss < best_eval_loss:
            best_eval_loss = curr_eval_loss
            stale_epochs = 0
            best_checkpoint_path = \
                os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(args.task_name, global_step))
        else:
            stale_epochs += 1
            logging.info(
                '!!! Development loss has not improved this epoch. Stale epochs: {:d} !!!'.format(stale_epochs))

        if stale_epochs >= args.patience:
            logging.info(
                '\n***** STOPPING TRAINING EARLY AFTER {:d} STALE VALIDATION STEPS *****\n'.format(stale_epochs))
            break

        if global_step > args.max_steps > 0:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_checkpoint_path


def evaluate(args, model, tokenizer, processor, split, step):
    """ Evaluates models on dev / test sets. """
    model.eval()
    results = dict()

    # Reference previous evaluation results
    if os.path.exists(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split))):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'r') as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    # Set up data-serving pipeline
    eval_dataset = load_and_cache_examples(args, tokenizer, split=split)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info('***** Running evaluation on the validation / test set *****')
    logger.info('  Num examples = %d', len(eval_dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    #batch_losses = list()
    eval_loss = 0.0
    micro_loss, macro_loss = 0.0, 0.0
    num_batches, num_tokens = 0, 0
    preds = None
    out_label_ids = None
    # Perform a single evaluation step
    for batch in tqdm(eval_dataloader, desc='Evaluating', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type == 'bert' else None,
                        'labels': batch[3]}

            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
        #batch_losses.append(tmp_eval_loss.item())
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    

    # Compute and update evaluation metric values
    # Isolate model predictions
    preds = np.argmax(preds, axis=1)
    curr_result = compute_cls_metrics(preds, out_label_ids)

    if len(results.keys()) == 0:
        for k, v in curr_result.items():
            results[k] = [v]
    else:
        for k, v in curr_result.items():
            results[k].append(v)

    # Log metrics
    output_eval_file = os.path.join(args.output_dir, 'results_{}_{}.txt'.format(args.task_name, split))
    with open(output_eval_file, 'a') as writer:
        logger.info('***** Eval results *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        for key in sorted(curr_result.keys()):
            logger.info('  %s = %s', key, str(curr_result[key]))
            writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # Log predictions
    output_pred_file = \
        os.path.join(args.output_dir, 'predictions_{}_{}_{}.lst'.format(args.task_name, split, step))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for pred in preds:
            writer.write('{}\n'.format(processor.get_labels()[pred]))

    # Maintain a single metrics file
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'w') as f:
            f.write(json.dumps(results))
        f.close()

    # Report mean dev loss
    mean_eval_loss = eval_loss / len(eval_dataloader)
    logging.info('\n' + '*' * 10)
    logging.info('Mean development loss: {:.4f}'.format(mean_eval_loss))
    logging.info('*' * 10 + '\n')

    return results, mean_eval_loss

def inference(args, model, tokenizer, processor, split, step):
    """ Evaluates out test sets. """
    model.eval()
    results = dict()

    # Reference previous evaluation results
    if os.path.exists(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split))):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(args.task_name, split)), 'r') as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    # Set up data-serving pipeline
    logger.info('***** enter loading examples *****')
    eval_dataset = load_and_cache_examples(args, tokenizer, split=split)
    
    if args.debug: import pdb; pdb.set_trace()
        
    
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info('***** Running inference on the inference set *****')
    logger.info('  Num examples = %d', len(eval_dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    #batch_losses = list()
    eval_loss = 0.0
    micro_loss, macro_loss = 0.0, 0.0
    num_batches, num_tokens = 0, 0
    preds = None
    out_label_ids = None
    # Perform a single evaluation step
    for batch in tqdm(eval_dataloader, desc='Inference', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type == 'bert' else None,
                        'labels': batch[3]}

            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
        #batch_losses.append(tmp_eval_loss.item())
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    
    

    # Compute and update evaluation metric values
    # Isolate model predictions
    if args.debug: 
        import pdb; pdb.set_trace()
        logger.info(preds) # 2 vector
        
    softmax_preds=softmax(preds, axis=1)[:,1]
    preds = np.argmax(preds, axis=1)
    
    curr_result = compute_cls_metrics(preds, out_label_ids)
    
    # save    
    loaded_data=pd.read_csv(OUTPUT2_PATH)
    assert len(preds)==len(loaded_data)
    loaded_data[args.task_name]=softmax_preds
    loaded_data.to_csv(OUTPUT2_PATH,index=False)
        
    if len(results.keys()) == 0:
        for k, v in curr_result.items():
            results[k] = [v]
    else:
        for k, v in curr_result.items():
            results[k].append(v)
            

    # output_dir='/home/sybae/Project/charactermorality/output/social_chem'
    # # Log metrics
    # output_eval_file = os.path.join(output_dir, 'inference_results_{}_{}_n3.txt'.format(args.task_name, split))
    # with open(output_eval_file, 'a') as writer:
    #     logger.info('***** Inference results *****')
    #     writer.write('STEP: {:s}\n'.format(str(step)))
    #     for key in sorted(curr_result.keys()):
    #         logger.info('  %s = %s', key, str(curr_result[key]))
    #         writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # # Log predictions
    # output_pred_file = \
    #     os.path.join(output_dir, 'inference_{}_{}_{}_{}_n3.lst'.format(args.model_type, args.task_name, split, step))
    # with open(output_pred_file, 'w') as writer:
    #     logger.info('***** Write predictions *****')
    #     for pred in softmax_preds:
    #         writer.write('{}\n'.format(round(pred, 4)))
    #         #writer.write('{}\n'.format(processor.get_labels()[pred]))

    # # Maintain a single metrics file
    # if os.path.exists(output_dir):
    #     with open(os.path.join(output_dir, 'inference_metrics_{}_{}_n3.json'.format(args.task_name, split)), 'w') as f:
    #         f.write(json.dumps(results))
    #     f.close()

    # Report mean dev lossssss
    mean_eval_loss = eval_loss / len(eval_dataloader)
    logging.info('\n' + '*' * 10)
    logging.info('Mean development loss: {:.4f}'.format(mean_eval_loss))
    logging.info('*' * 10 + '\n')
    
    return results, mean_eval_loss

def load_and_cache_examples(args, tokenizer, split):
    """ Prepares the dataset splits for use with the model. """
    processor = MoralStoriesProcessor()
    
    # Load data features from cache or dataset file
    if args.data_cache_dir is None:
        data_cache_dir = args.data_dir
    else:
        data_cache_dir = args.data_cache_dir

    cached_features_file = os.path.join(data_cache_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))

    if os.path.exists(cached_features_file):
        logger.info('Loading features from cached file %s', cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info('Creating features from dataset file at %s', args.data_dir)
        label_list = processor.get_labels()

        if split == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif split == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif split == 'test':
            examples = processor.get_test_examples(args.data_dir)
        elif split == 'inf':
            print('enter inf')
            examples = processor.get_inf_examples(args.data_dir)
            
            if args.debug==1:
                import pdb; pdb.set_trace()
                logger.info(examples[0])
        else:
            raise Exception('split value should be in [train, dev, test, inf]')
        
        logger.info("inference dataset num: %s", len(examples))
        # Generate features; target task is classification
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        if pad_token_id is None:
            pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
        features = convert_examples_to_features(examples,
                                                label_list,
                                                args.max_seq_length,
                                                tokenizer,
                                                args.task_name,
                                                args.model_type,
                                                TASK_DICT[args.task_name],
                                                cls_token_at_end=False,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                cls_token_segment_id=0,
                                                pad_on_left=False,
                                                pad_token=pad_token_id,
                                                pad_token_segment_id=0,
                                                is_eval=split == 'test',
                                                fit_to_max_corpus_len=True)

        logger.info('Saving features into cached file %s', cached_features_file)
        if args.data_cache_dir is not None:
            pathlib.Path(args.data_cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(features, cached_features_file)

    # Make feature tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_dir', default="data/movie_dataset", type=str, required=True,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task.')
    
    parser.add_argument('--model_type', default="bert", type=str, required=True, choices=['bert', 'roberta'])
    
    parser.add_argument('--model_name_or_path', default="bert-large-uncased", type=str, required=True, choices=['bert-large-uncased', 'roberta-large'], help='Path to pre-trained model')
    
    parser.add_argument('--task_name', default=None, type=str, required=True, choices=TASKS,
                        help='The name of the task to train selected in the list: ' + ', '.join(TASKS))
    
    parser.add_argument('--output_dir', default="output/", type=str, required=True,
                        help='The root output directory where the model predictions and checkpoints will be written.')
    
    parser.add_argument('--checkpoint', default=None, type=str, required=False)
    
    ## Other parameters
    parser.add_argument('--config_name', default='', type=str,
                        help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default='', type=str,
                        help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--cache_dir', default='', type=str,
                        help='The cache directory where do you want to store the pre-trained models downloaded from s3')
    parser.add_argument('--max_seq_length', default=1024, type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer '
                             'than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_prediction', action='store_true',
                        help='Whether to run prediction on the test set. (Training will not be executed.)')
    parser.add_argument('--do_inference', action='store_true',
                        help='Whether to run prediction on the inference set. (Training will not be executed.)')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Do evaluation during training at each logging step.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument('--run_on_test', action='store_true',
                        help='Evaluate model on the test split.')
    parser.add_argument('--data_cache_dir', default="cache", type=str,
                        help='The root directory for caching features.')
    parser.add_argument('--pretrained_dir', default=None, type=str,
                        help='The directory containing the checkpoint of a pretrained model to be fine-tuned.')
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Maximum number of checkpoints to keep')

    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight deay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--num_train_epochs', default=3.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--warmup_pct', default=None, type=float,
                        help='Linear warmup over warmup_pct * total_steps.')
    parser.add_argument('--patience', default=-1, type=int, required=False,
                        help='Number of epochs without dev-set loss improvement to ignore before '
                             'stopping the training.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--eval_all_checkpoints', action='store_true',
                        help='Evaluate all checkpoints starting with the same prefix as model_name ending and ending '
                             'with step number')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the content of the output directory')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    parser.add_argument('--local_rank', type=int, default=-1,
                        help='For distributed training: local_rank')
    
    parser.add_argument('--debug', type=int, default=0)
    
    args = parser.parse_args()
    
    wandb.init(config=args, project="CharMoral")
    wandb.run.name=f"mydata_{args.task_name}"
    wandb.config["more"] = "custom"
    
    # Create output / cache directory paths
    if args.do_train:
        args.output_dir = os.path.join(args.output_dir, args.task_name, args.model_type)

    args.data_dir = os.path.join(args.data_dir)
    args.data_cache_dir = os.path.join(args.data_cache_dir, args.task_name, args.model_type)
    
    #print(args.data_dir)
    #print(args.data_cache_dir)
    
    # Check if directories need to be created
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.data_cache_dir):
        os.makedirs(args.data_cache_dir)


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        args.n_gpu = torch.cuda.device_count()
    else:  
        # Initializes the distributed backend which will take care of sychronizing nodes / GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning('device: %s, n_gpu: %s', device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Identify labels
    processor = MydatasetProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Instantiate model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    model, tokenizer = None, None
    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name
        )
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else
                                                    args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        model.to(args.device)

    logger.info('Training / evaluation parameters %s', args)
    # Training
    final_checkpoint_name = args.output_dir
    if args.do_train:
        global_step, tr_loss, final_checkpoint_name = train(args, model, tokenizer)
        logger.info(' global_step = %s, average loss = %s', global_step, tr_loss)

    # Evaluate on dev
    checkpoints = [final_checkpoint_name]
    print("checkpoints: ", checkpoints)

    if args.do_eval:
        checkpoint = checkpoints[0]
        if not args.do_train:
            checkpoint = args.model_name_or_path
        global_step = int(checkpoint.split('-')[-1])
        model = model_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)  # hack
        model.to(args.device)
        
        dev_split = 'dev' if args.do_train else 'test'
        results, loss = evaluate(args, model, tokenizer, processor, dev_split, global_step)
        logger.info('-' * 20)
        logger.info('Global step: {:d}'.format(global_step))
        logger.info('Loss: {:.3f}'.format(loss))
        logger.info('Results:')
        logger.info(results)
        wandb.log({'eval_loss': loss}, step=global_step)
    
    
    # Evaluate on test
    if args.do_prediction:
        checkpoint = checkpoints[0]
        if not args.do_train:
            checkpoint = args.model_name_or_path
        global_step = int(checkpoint.split('-')[-1])
        model = model_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model.to(args.device)
        
        logger.info('Prediction on the test set (note: Training will not be executed.) ')
        evaluate(args, model, tokenizer, processor, 'test', global_step)
    
    # Evaluate on test
    if args.do_inference:
        checkpoint = args.checkpoint
        print("after inference: ", checkpoint)
        global_step = int(checkpoint.split('-')[-1])
        model = model_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model.to(args.device)
        
        logger.info('Prediction on the inference set (note: Training will not be executed.) ')
        inference(args, model, tokenizer, processor, 'inf', global_step)

    logger.info('***** Experiment finished *****')
    
    


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logging.info(exception)
        raise Exception(exception)