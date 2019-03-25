from model.data import SQuAD
from model.bidaf import BidafModel
from model.moving_average import MA
import evaluate_official
import argparse
import logging as log
import torch
from torch import nn, optim
from datetime import datetime
import json

from params import *


def train(params, data):
    device = torch.cuda.current_device() if params['cuda'] else torch.device('cpu')

    model = BidafModel(params, data).to(device)

    moving_average = MA(params['weight_decay'])
    moving_average.set_all(model)

    optimizer = init_optimizer(params, model)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # best_score = 0

    for epoch in range(0, params['epochs']):
        print('Epoch {}'.format(epoch))
        start = datetime.now()

        loss = 0.0
        # model.train()
        train_iterator = data.train_iter
        for i, batch in enumerate(train_iterator):
            model.train()

            loss_mask_c = get_loss_mask(batch.c_word[1])

            p1, p2 = model(batch)
            p1 = p1 * loss_mask_c.float()
            p2 = p2 * loss_mask_c.float()

            optimizer.zero_grad()
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            moving_average.update_all(model)

            if i % 250 == 0 and i > 0:
                elapsed = datetime.now() - start
                print('loss {} time {}'.format(loss/i, elapsed))
                test(params, model, data, criterion, moving_average)

            if params['cuda']:
                torch.cuda.empty_cache()

            # test(args, model, data, criterion, moving_average)

        log.info('\n')
        test(params, model, data, criterion, moving_average)


def init_optimizer(params, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if params['optimizer'] == 'adamax':
        optimizer = optim.Adamax(parameters, weight_decay=0.0)
    elif params['optimizer'] == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=0.5)
    else:
        raise RuntimeError('Unsupported optimizer: %s' % params['optimizer'])
    return optimizer


def get_loss_mask(lens):
    max_len = lens.max().item()
    mask = torch.arange(max_len, device=lens.device, dtype=lens.dtype).expand(len(lens), max_len) < lens.unsqueeze(1)
    return mask


def test(params, model, data, criterion, moving_average):
    device = torch.cuda.current_device() if params['cuda'] else torch.device('cpu')
    model.eval()

    answers = dict()

    # save model parameters and replace them with MA for testing
    current_params_bck = MA(-1)
    for name, param in model.named_parameters():
        if param.requires_grad:
            current_params_bck.set(name, param.data)
            param.data.copy_(moving_average.get(name))

    total = 0
    matches = 0
    dev_iterator = data.dev_iter
    for b, batch in enumerate(dev_iterator):
        p1, p2 = model(batch)
        batch_size, c_len = p1.size()

        #from tensorflow code: extract span which maximizes p1*p2
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        s_idx = s_idx.to('cpu').data.numpy()
        e_idx = e_idx.to('cpu').data.numpy()
        s_true = batch.s_idx.to('cpu').data.numpy()
        e_true = batch.e_idx.to('cpu').data.numpy()

        starts = (s_idx == s_true)
        ends = (e_idx == e_true)

        for i in range(batch_size):
           if starts[i] == True and ends[i] == True:
               matches += 1
        total += batch_size

        if params['cuda']:
            torch.cuda.empty_cache()

        for i in range(batch_size):
            id = batch.id[i]
            answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
            answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
            answers[id] = answer

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(current_params_bck.get(name))

    with open(params['prediction_file'], 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    official_results = evaluate_official.test_runtime(params)

    # restore model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(current_params_bck.get(name))

    #print('total: {}'.format(total))
    #print('matches: {}'.format(matches))
    print('----------- Official EM: {0:.2f} Strict EM: {1:.2f}'.format(official_results['exact_match'], 100.0 * matches/total))


def main():
    log.info('[Program starts. Loading data...]')
    data = SQuAD(params)
    train(params, data)


if __name__ == '__main__':
    main()
