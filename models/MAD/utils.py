# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
import re
import csv
import sys
import glob
import json
import torch
import random
import shutil
import logging

import numpy as np

from io import open
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MoralStoryExample(object):
    """A single training/test example for classification of moral stories."""

    def __init__(self, guid,
                 norm, situation, intention,
                 action, consequence, label):

        self.guid = guid
        self.norm = norm
        self.situation = situation
        self.intention = intention

        self.action = action
        self.consequence = consequence
        
        self.label = label


class InputFeatures(object):
    """ A single set of features of data. """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask, gen_prompt_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.gen_prompt_id = gen_prompt_id


class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets. """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError()
    
    def get_inf_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines[1:]

    @classmethod
    def _read_jsonl(cls, input_file):
        """ Reads a .jsonl file. """
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records


class MoralStoriesProcessor(DataProcessor):
    """ Converts moral stories for sequence classification tasks. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'moral_stories_train.tsv')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'moral_stories_valid.tsv')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'moral_stories_test.tsv')))
    
    def get_inf_examples(self, data_dir):
        return self._create_inf_examples(
            self._read_tsv(os.path.join(data_dir, 'inference_n3.tsv')))

    def get_labels(self):
        # 0 may correspond to 'immoral' / 'implausible', while 1 may correspond to 'moral' / 'plausible'
        return ['0', '1']

    def create_examples(self, records):
        return self._create_examples(records)
    
    def create_inf_examples(self, records):
        return self._create_inf_examples(records)
    
    @staticmethod
    def _create_inf_examples(records):
        # Convert corpus contents to examples
        examples = list()
        print(len(records))
        for i, record in enumerate(records):
            #print(record)
            
            guid = record[0]
            norm = 'norm'
            situation = record[13]
            if (len(situation)<5) or ('not exist' in situation): situation=''
            
            intention = record[12]
            if (len(intention)<5) or ('not exist' in intention): intention=''

            action = record[11]
            if (len(action)<5) or ('no action' in action): action=''
            
            consequence = record[14]
            if (len(consequence)<5) or ('not exist' in consequence): consequence=''
            
            # convert [char] to <mask>
            #norm=norm.replace("[char]", "<mask>")
            situation=situation.replace("[char]", "<mask>")
            intention=intention.replace("[char]", "<mask>")
            action=action.replace("[char]", "<mask>")
            consequence=consequence.replace("[char]", "<mask>")

            label = '0'

            examples.append(MoralStoryExample(guid=guid,
                                              norm=norm, situation=situation, intention=intention,
                                              action=action, consequence=consequence,
                                              label=label))
            
            
        return examples
    

    @staticmethod
    def _create_examples(records):
        # Convert corpus contents to examples
        examples = list()
        print(len(records))
        for i, record in enumerate(records):
            #print(record)
            
            guid = record[0]
            norm = record[1]
            situation = record[2]
            intention = record[3]

            action = record[4]
            consequence = record[5]
            
            # convert [char] to <mask>
            norm=norm.replace("[char]", "<mask>")
            situation=situation.replace("[char]", "<mask>")
            intention=intention.replace("[char]", "<mask>")
            action=action.replace("[char]", "<mask>")
            consequence=consequence.replace("[char]", "<mask>")

            norm=norm.replace('0', '')
            situation=situation.replace('0', '')
            intention=intention.replace('0', '')
            action=action.replace('0', '')
            consequence=consequence.replace('0', '')

            label = record[7] # recore[6] (inference)

            if label is None:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = '0'

            examples.append(MoralStoryExample(guid=guid,
                                              norm=norm, situation=situation, intention=intention,
                                              action=action, consequence=consequence,
                                              label=label))
        return examples
    
    

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 task_name,
                                 model_name,
                                 example_code,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 is_eval=False,
                                 fit_to_max_corpus_len=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # Initialize containers
    prefix_cache = list()
    target_cache = list()

    # Assign labels
    label_map = {label: i for i, label in enumerate(label_list)}

    # Obtain features
    lines = list()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))
        # Tokenize (used for fine-tuning on Moral Stories)
        tokens_norm = tokenizer.tokenize(example.norm) if example.norm is not None else None
        tokens_situation = tokenizer.tokenize(example.situation) if example.situation is not None else None
        tokens_intention = tokenizer.tokenize(example.intention) if example.intention is not None else None

        tokens_action = tokenizer.tokenize(example.action) if example.action is not None else None

        tokens_consequence = \
            tokenizer.tokenize(example.consequence) if example.consequence is not None else None

        tokens_map = {
            'norm': tokens_norm,
            'situation': tokens_situation,
            'intention': tokens_intention,
            'action': tokens_action,
            'consequence': tokens_consequence,
        }

        # Assemble example contents according to the example code
        example_tokens = [(ec, tokens_map.get(ec, None)) for ec in example_code]
        example_tokens = [et for et in example_tokens if et[1] is not None]

        # Remove codes
        example_tokens = [et[1] for et in example_tokens]
        if ex_index==1:
            print(f"example_tokens: {example_tokens}")
            print(label_map[example.label])
        lines.append((example_tokens, label_map[example.label]))

    for example_tokens, label_id in lines:
        ss_special_tokens_count = 2
        ms_special_tokens_count = 4 if sep_token_extra else 3
        # Truncate inputs, if needed
        if not fit_to_max_corpus_len:
            if len(example_tokens) > 1:
                _truncate_seq_pair(example_tokens, max_seq_length - ms_special_tokens_count - 1, 'gen' in task_name)
            else:
                if len(example_tokens[0]) > max_seq_length - ss_special_tokens_count:
                    example_tokens[0] = example_tokens[0][:(max_seq_length - ss_special_tokens_count)]

        # Construct segments
        example_prefix_length = 0
        target_ids, target_tokens, gen_prompt = list(), list(), ''

        tokens_a = list()
        if len(example_tokens) == 1:
            tokens_a = example_tokens[0]
        else:
            for et in example_tokens[:-1]:
                tokens_a += et

        tokens_b = example_tokens[-1] if len(example_tokens) > 1 else None

        # Add special tokens
        tokens = tokens_a + [sep_token]
        if sep_token_extra and len(example_tokens) > 1:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            prefix_tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            prefix_tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        target_ids = label_id

        
        # Convert tokens to ids
        prefix_ids = tokenizer.convert_tokens_to_ids(prefix_tokens)

        # Cache
        prefix_cache.append((prefix_ids, segment_ids, prefix_tokens, example_prefix_length))
        target_cache.append((target_ids, target_tokens, gen_prompt, tokenizer.convert_tokens_to_ids(gen_prompt)))

    # Determine maximum input tokens length
    prefix_lengths = [len(tpl[0]) for tpl in prefix_cache]
    max_prefix_length = max(prefix_lengths)
    if fit_to_max_corpus_len:
        max_seq_length = max_prefix_length
    target_lengths = list()

    # Make masks and pad inputs / labels
    features = list()
    for iid, inputs in enumerate(prefix_cache):
        example = examples[iid]
        prefix_ids, segment_ids, prefix_tokens, example_prefix_length = inputs
        target_ids, target_tokens, gen_prompt, gen_prompt_id = target_cache[iid]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        prefix_mask = [1 if mask_padding_with_zero else 0] * len(prefix_ids)

        # Zero-pad up to the sequence length
        padding_length = max_seq_length - len(prefix_ids)
        if pad_on_left:
            prefix_ids = ([pad_token] * padding_length) + prefix_ids
            prefix_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + prefix_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            prefix_ids = prefix_ids + ([pad_token] * padding_length)
            prefix_mask = prefix_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        target_mask = None

        try:
            assert len(prefix_ids) == max_seq_length
            assert len(prefix_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except AssertionError:
            logging.info(prefix_ids, len(prefix_ids))
            logging.info(prefix_mask, len(prefix_mask))
            logging.info(segment_ids, len(segment_ids))
            raise AssertionError

        if iid < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('input_tokens: %s' % ' '.join([str(x) for x in prefix_tokens]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in prefix_ids]))
            logger.info('input_mask: %s' % ' '.join([str(x) for x in prefix_mask]))
            logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
            if 'gen' in task_name:
                logger.info('target_tokens: %s' % ' '.join([str(x) for x in target_tokens]))
                logger.info('target_ids: %s' % ' '.join([str(x) for x in target_ids]))
                logger.info('target_mask: %s' % ' '.join([str(x) for x in target_mask]))
                logger.info('gen_prompt: %s' % gen_prompt)
                logger.info('gen_prompt_ids: %s' % str(gen_prompt_id))
            else:
                logger.info('label: %s (id = %d)' % (example.label, target_ids))

        features.append(
            InputFeatures(input_ids=prefix_ids,
                          input_mask=prefix_mask,
                          segment_ids=segment_ids,
                          label_ids=target_ids,
                          label_mask=target_mask,
                          gen_prompt_id=[gen_prompt_id]))

    # Report some basic statistics
    logger.info('=' * 20)
    logger.info('Dataset statistics (before truncation / padding):')
    logger.info('Mean model input length: {:.2f}'.format(np.mean(prefix_lengths)))
    logger.info('Model input length std.: {:.2f}'.format(np.std(prefix_lengths)))
    logger.info('Min model input length: {:.2f}'.format(min(prefix_lengths)))
    logger.info('Max model input length: {:.2f}'.format(max(prefix_lengths)))
    logger.info('=' * 20)

    return features


def _truncate_seq_pair(all_segments, max_length, is_gen):
    """ Truncates a sequence pair in place to the maximum length. """

    final_segment = list()
    if is_gen:
        # Don't truncate the target tokens
        final_segment = [all_segments[-1]]
        all_segments = all_segments[:-1]

    while True:
        total_length = sum([len(seg) for seg in all_segments])
        if total_length <= max_length:
            break
        # Shorten the longest segment
        longest_seg = all_segments[max(enumerate(all_segments), key=lambda x: len(x[1]))[0]]
        longest_seg.pop()

    all_segments += final_segment


def set_seed(args):
    """ Sets the seed to support reproducibility. """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """ Keep a maximum of args.save_total_limit checkpoints. """
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = list()
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info('Deleting older checkpoint [{}] due to args.save_total_limit'.format(checkpoint))
        shutil.rmtree(checkpoint)


def get_token_loss(args, lm_logits, target_ids, target_mask, model_type=None):
    """ Compute token-level loss per batch during evaluation. """
    # Declare loss function
    loss_fct = CrossEntropyLoss(reduction='none')
    if model_type is None:
        model_type = args.model_type

    # Obtain logits to compute token-level loss / perplexity
    shift_logits = lm_logits[..., :-1, :].contiguous()
    batch_size, max_length, vocab_size = shift_logits.shape

    # Compute loss for each instance and each token
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = target_ids[..., 1:].contiguous().view(-1)
    token_loss = loss_fct(shift_logits, shift_labels).view(batch_size, max_length)


    # Only consider non padded tokens
    target_mask = target_mask[..., :-1].contiguous()
    masked_token_loss = torch.mul(target_mask, token_loss)  # [batch_size, max_length]

    return masked_token_loss


## METRICS
def simple_accuracy(preds, labels):
    """ Computes prediction accuracy. """
    return np.mean(preds == labels)


def simple_f1(preds, labels):
    """ Computes prediction F1 score. """
    f1 = f1_score(y_true=labels, y_pred=preds)
    return f1


def compute_cls_metrics(preds, labels):
    """ Aggregates classification metrics. """
    assert len(preds) == len(labels)
    return {'acc': simple_accuracy(preds, labels),
            'f1': simple_f1(preds, labels)}
