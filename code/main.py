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
    best_ndcg = None
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            if world.model_name == 'pcsrec' and hasattr(dataset, 'validDict') and len(dataset.validDict) > 0:
                cprint("[VALID]")
                valid_results = Procedure.Test(dataset,
                                               Recmodel,
                                               epoch,
                                               w,
                                               world.config['multicore'],
                                               evalDict=dataset.validDict,
                                               tag='Valid')
                curr_ndcg = float(valid_results['ndcg'][-1]) if len(valid_results['ndcg']) > 0 else None
                if curr_ndcg is not None and (best_ndcg is None or curr_ndcg > best_ndcg):
                    best_ndcg = curr_ndcg
                    torch.save(Recmodel.state_dict(), best_weight_file)
        if world.model_name == 'pcsrec':
            pcs_loss = Procedure.train_PCSRec(dataset, Recmodel, None, epoch, w=w)
            output_information = f"loss{pcs_loss:.3f}"
        else:
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()

if world.model_name == 'pcsrec' and os.path.exists(best_weight_file):
    Recmodel.load_state_dict(torch.load(best_weight_file, map_location=world.device))
    cprint("[TEST-BEST]")
    Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, w=None, multicore=world.config['multicore'], tag='TestBest')