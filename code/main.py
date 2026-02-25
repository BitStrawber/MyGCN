import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import os
from tqdm import tqdm


def _format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h}h {m}m {s}s'
    if m > 0:
        return f'{m}m {s}s'
    return f'{s}s'
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = None
if world.model_name != 'pcsrec':
    bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
best_weight_file = weight_file.replace('.pth.tar', '-best.pth.tar')
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    train_total_start = time.time()
    best_ndcg = None
    epoch_bar = tqdm(range(world.TRAIN_epochs), desc='Training', unit='epoch')
    for epoch in epoch_bar:
        epoch_start = time.time()
        test_time = 0.0
        valid_time = 0.0
        train_time = 0.0
        curr_valid_ndcg = None
        if epoch %10 == 0:
            cprint("[TEST]")
            t0 = time.time()
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            test_time = time.time() - t0
            if world.model_name == 'pcsrec' and hasattr(dataset, 'validDict') and len(dataset.validDict) > 0:
                cprint("[VALID]")
                t0 = time.time()
                valid_results = Procedure.Test(dataset,
                                               Recmodel,
                                               epoch,
                                               w,
                                               world.config['multicore'],
                                               evalDict=dataset.validDict,
                                               tag='Valid')
                valid_time = time.time() - t0
                curr_valid_ndcg = float(valid_results['ndcg'][-1]) if len(valid_results['ndcg']) > 0 else None
                if curr_valid_ndcg is not None and (best_ndcg is None or curr_valid_ndcg > best_ndcg):
                    best_ndcg = curr_valid_ndcg
                    torch.save(Recmodel.state_dict(), best_weight_file)
                    print(f"[BEST] epoch={epoch+1} valid_ndcg@{world.topks[-1]}={best_ndcg:.6f}")
        if world.model_name == 'pcsrec':
            t0 = time.time()
            pcs_loss = Procedure.train_PCSRec(dataset, Recmodel, None, epoch, w=w)
            train_time = time.time() - t0
            output_information = f"loss{pcs_loss:.3f}"
        else:
            t0 = time.time()
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            train_time = time.time() - t0
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_total_start
        avg_epoch = elapsed / (epoch + 1)
        eta = avg_epoch * (world.TRAIN_epochs - epoch - 1)
        print(
            f"[TIME] epoch={_format_seconds(epoch_time)} "
            f"(train={_format_seconds(train_time)}, test={_format_seconds(test_time)}, valid={_format_seconds(valid_time)}), "
            f"elapsed={_format_seconds(elapsed)}, eta={_format_seconds(eta)}"
        )
        postfix = {'eta': _format_seconds(eta)}
        if world.model_name == 'pcsrec':
            postfix['loss'] = f"{pcs_loss:.4f}"
            if best_ndcg is not None:
                postfix['best_ndcg'] = f"{best_ndcg:.4f}"
            if curr_valid_ndcg is not None:
                postfix['valid_ndcg'] = f"{curr_valid_ndcg:.4f}"
        epoch_bar.set_postfix(postfix)
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    total_train_time = time.time() - train_total_start if 'train_total_start' in locals() else 0.0
    print(f"Training total time: {_format_seconds(total_train_time)}")
    if world.tensorboard:
        w.close()

if world.model_name == 'pcsrec' and os.path.exists(best_weight_file):
    Recmodel.load_state_dict(torch.load(best_weight_file, map_location=world.device))
    cprint("[TEST-BEST]")
    t0 = time.time()
    Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, w=None, multicore=world.config['multicore'], tag='TestBest')
    print(f"Test-best time: {_format_seconds(time.time() - t0)}")