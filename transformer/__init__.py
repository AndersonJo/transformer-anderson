import logging
import os
import re

import torch

from transformer.models import Transformer

logger = logging.getLogger('transformer')


def load_transformer_to_train(opt) -> Transformer:
    model = Transformer(embed_dim=opt.embed_dim,
                        src_vocab_size=opt.src_vocab_size,
                        trg_vocab_size=opt.trg_vocab_size,
                        src_pad_idx=opt.src_pad_idx,
                        trg_pad_idx=opt.trg_pad_idx,
                        n_head=opt.n_head)
    model = model.to(opt.device)
    checkpoint_file_path = get_best_checkpoint(opt)
    if checkpoint_file_path is not None:
        logger.info(f'Checkpoint loaded - {checkpoint_file_path}')
        checkpoint = torch.load(checkpoint_file_path, map_location=opt.device)
        model.load_state_dict(checkpoint['weights'])
    return model


def load_transformer(opt) -> Transformer:
    checkpoint_file_path = get_best_checkpoint(opt)
    checkpoint = torch.load(checkpoint_file_path, map_location=opt.device)

    assert checkpoint is not None
    assert checkpoint['opt'] is not None
    assert checkpoint['weights'] is not None

    model_opt = checkpoint['opt']
    model = Transformer(embed_dim=model_opt.embed_dim,
                        src_vocab_size=model_opt.src_vocab_size,
                        trg_vocab_size=model_opt.trg_vocab_size,
                        src_pad_idx=model_opt.src_pad_idx,
                        trg_pad_idx=model_opt.trg_pad_idx,
                        n_head=model_opt.n_head)

    model.load_state_dict(checkpoint['weights'])
    print('model loaded:', checkpoint_file_path)
    return model.to(opt.device)


def get_best_checkpoint(opt):
    regex = re.compile('checkpoint_(\d+)_(\d+\.\d+)\.chkpt')
    checkpoints = []
    if os.path.exists(opt.checkpoint_path):
        for name in os.listdir(opt.checkpoint_path):
            if regex.match(name):
                checkpoints.append((name, float(regex.match(name).group(1))))
    if not checkpoints:
        return None

    checkpoints = sorted(checkpoints, key=lambda x: -x[1])
    return os.path.join(opt.checkpoint_path, checkpoints[0][0])
