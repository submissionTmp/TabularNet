# -*- coding , default =  utf-8-*-
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import logging
from src.utils.dataset import dataset_split, dataset, dataloader
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import random
import sys
import shutil
import warnings
from argparse import Namespace
import argparse
from tqdm import tqdm
import string
import copy
import time
from termcolor import colored
import os

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_log(args):
    dataset = args.root_dir.split("/")[-1]
    log_dir = "logger/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_filename = dataset + "_" + str(cur_time)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("logger name:%s", os.path.join(log_dir, log_filename+".log"))
    return logger


def train(trainLoader, args, logger):
    global best_loss
    global patient
    loss_cum = []
    model.train()
    for batch_cnt, batch in enumerate(trainLoader):
        optim.zero_grad()
        embedding, batch = model(batch)
        pred, label = model.output(embedding, batch)
        classify_loss = model.loss_fn(pred, label)
        loss = classify_loss
        loss.backward()
        optim.step()
        loss_cum.append(float(loss))
    logger.info("{:<6}, Loss = {:.3f}".format("Train", np.mean(loss_cum)))
    v_loss, v_ma, v_mi = evalate(valLoader, "Val", logger, args)
    scheduler.step(v_loss)

    # Early Stop
    if v_loss < best_loss:
        best_loss = v_loss
        patient = 0
        torch.save(model.state_dict(),
                   "checkpoints/{}_model.pt".format(args.model))
        # Test
        t_loss, t_ma, t_mi = evalate(testLoader, "Test", logger, args)
    else:
        patient += 1


def evalate(Loader, indicater, logger, args):
    global best_f1
    global summary
    write_mark = 0
    model.eval()
    loss_cum = []
    pred_list = []
    label_list = []
    for batch in Loader:
        embedding, batch = model(batch)
        pred, label = model.output(embedding, batch)
        classify_loss = model.loss_fn(pred, label)
        loss = classify_loss
        loss_cum.append(float(loss))
        pred_list.append(pred.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
    f1_ma, f1_mi = metric(pred_list, label_list)
    color = "white" if indicater != "Test" else "red"
    if color == "white":
        logger.info("{:<6}, Loss = {:.5f}, F1_ma = {:.3f}\t F1_mi = {:.3f}".format(
            indicater, np.mean(loss_cum), f1_ma, f1_mi))
    elif color == "red":
        logger.info("[+] {:<6}, Loss = {:.5f}, F1_ma = {:.3f}\t F1_mi = {:.3f}".format(
            indicater, np.mean(loss_cum), f1_ma, f1_mi))
    if indicater != "Val":
        if color == "white":
            logger.info("Class_0-4:")
        elif color == "red":
            logger.info("[+] Class_0-4:")
        if f1_ma > best_f1:
            best_f1 = f1_ma
            write_mark = 1
            summary = "F1_ma = {:.3f}\t F1_mi = {:.3f}".format(f1_ma, f1_mi)
        for class_id in [0, 1, 2, 3, 4]:
            f1 = metric(pred_list, label_list, class_idx=class_id)
            if write_mark:
                summary += "\t{:.3f}".format(f1)
            if color == "white":
                logger.info("{:.3f}".format(f1))
            elif color == "red":
                logger.info("[+] {:.3f}".format(f1))
    return np.mean(loss_cum), f1_ma, f1_mi


def metric(pred_list, label_list, class_idx=None):
    pred_list = np.vstack(pred_list)
    pred = np.argmax(pred_list, axis=1)
    label = np.concatenate(label_list).squeeze()
    if class_idx == None:
        f1_ma = f1_score(pred, label, average="macro")
        f1_mi = f1_score(pred, label, average="micro")
        return f1_ma, f1_mi
    else:
        f1_ma = f1_score(pred == class_idx, label ==
                         class_idx, average="binary")
        return f1_ma


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", default="/home/fei/disk/tt/Processed_CellRole_G_onehot_Label")
    parser.add_argument("--train_percent", default=0.7)
    parser.add_argument("--val_percent", default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", default=2)
    parser.add_argument("--test_batch_size", default=2)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--feat_dim", type=int, default=820)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=3)
    parser.add_argument("--num_class", default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--max_batch_vol", default=7000)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--split_seed", type=int, default=1024)
    parser.add_argument("--model", type=str, default="gine_pool_bigru")
    parser.add_argument("--sim", type=str, default="wnsim",
                        help="opt: wnsim, idxname_sim")
    parser.add_argument("--graph_type", type=str,
                        default="original", help="opt: wnsim, original")
    parser.add_argument("--gcn_pos", type=str,
                        default="inner", help="opt: inner, outer")
    parser.add_argument("--num_negsamples", type=int,
                        default=1, help="number of negative samples")
    parser.add_argument("--result_summary", type=str, default="res.txt",
                        help="file to record result of different trials")

    args = parser.parse_args()
    logger = init_log(args)
    device = torch.device("cuda:{}".format(args.cuda_id)
                          if args.cuda_id != -1 else "cpu")

    logger.info("params : %s", vars(args))
    set_seed(args.split_seed)

    logger.info(device)
    ############ Define Model ################
    logger.info("--> Define Model ...")

    if args.model == "gine_pool_bigru":
        from src.model.tabularNet import gine_pool_bigru as TableEncoder
    else:
        print("Error: No matched model")
        exit()
    model = TableEncoder(args.feat_dim, args.hid_dim,
                         args.num_class, args.dropout, device, args)
    model = model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optim, "min", patience=1, verbose=True, factor=0.1)

    ############ dataset process ################
    logger.info("--> Processing Dataset ...")
    if args.split_seed:
        seed = args.split_seed
    else:
        seed = random.randint(1, 100000)
    train_set, val_set, test_set = dataset_split(
        args.root_dir, args.train_percent, args.val_percent, seed)
    trainLoader = dataloader(
        args.root_dir, train_set, args.train_batch_size, device,  num_workers=25, sim=args.sim)
    valLoader = dataloader(
        args.root_dir, val_set, args.val_batch_size, device,  num_workers=25, sim=args.sim)
    testLoader = dataloader(
        args.root_dir, test_set, args.train_batch_size, device,  num_workers=25, sim=args.sim)
    logger.info("Sizes: Training={}, Val={}, Test={}".format(
        len(train_set), len(val_set), len(test_set)))

    ############ begin training ###############
    logger.info("-->Training Begin ... ")
    best_loss = 1000000000000
    best_f1 = 0.0
    patient = 0
    summary = ""
    for epoch in range(100):
        logger.info("Epoch={}".format(epoch))
        t_epoch = time.time()
        train(trainLoader, args, logger)
        logger.info("Epoch Time = {:.1f}s".format(time.time()-t_epoch))
        if patient >= args.early_stop:
            break
    f = open(args.result_summary, "a+")
    f.write("dataset {},loss {}, graphtype {},gcnpos {},num_negsap {}, lambda {}, lr{}, weight_decay{}, dropout{}\t".format(args.root_dir.split("/")[-1],
                                                                                                                            args.addition_loss, args.graph_type, args.gcn_pos, str(
                                                                                                                                args.num_negsamples), str(args.lambdas),
                                                                                                                            str(args.lr), str(args.weight_decay), str(args.dropout)))
    f.write(summary)
    f.write("\n")
    f.close()
