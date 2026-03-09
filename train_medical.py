"""
train_medical.py — Train SAT Aligner trên bộ dữ liệu medical_kg

Script này là bản sao chép chuẩn xác từ SAT/aligner/model/main.py,
được điều chỉnh đường dẫn (path) để train trực tiếp trên bộ data/medical
mà build_dataset.py đã sinh ra.

Các thành phần SAT được sử dụng:
  - model_gt.py   → Class CLIP (Text Encoder + Graph Transformer + Alignment Loss)
  - model_gt.py   → Hàm tokenize() (BPE Tokenizer của CLIP)
  - graph_transformer.py → Class graph_transformer (GNN backbone)
  - data_helper.py → Hàm load_data(), construct_graph(), get_mid2id(), get_rel2id(), get_id2text()
  - data_helper.py → Class TAGTrainDataset (PyTorch Dataset cho training)
  - simple_tokenizer.py → BPE tokenizer
  - bpe_simple_vocab_16e6.txt.gz → BPE vocabulary file
"""

import os
# Chỉ set CUDA khi thực sự muốn ép dùng GPU index 0 (bỏ qua để MPS/CPU tự auto detect)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys

# Thêm đường dẫn SAT model vào sys.path để import được các module gốc của SAT
SAT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "SAT", "aligner", "model")
sys.path.insert(0, SAT_MODEL_DIR)

import argparse
import random
import math
import time
import json

import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ====================================================================
# Import trực tiếp từ SAT/aligner/model/ (code gốc, KHÔNG chỉnh sửa)
# ====================================================================
from model_gt import CLIP, tokenize           # Mô hình CLIP + tokenizer
from medical_data_helper import get_mid2id, get_rel2id, get_id2text, load_data  # Load dữ liệu
from medical_data_helper import construct_graph, save_clip_data                  # Xây dựng đồ thị
from medical_data_helper import TAGTrainDataset, TrainDataset, EvalDataset       # Dataset classes
from medical_data_helper import extract_negative_triples                         # Sinh negative samples


# ====================================================================
# HÀM TRAINING — Sao chép chuẩn xác từ SAT main.py:train()
# ====================================================================
def train():
    """
    Vòng lặp huấn luyện chính của SAT Aligner.
    
    Cơ chế hoạt động (giống y hệt SAT/aligner/model/main.py):
    1. Mỗi batch lấy ra (src, rel, dst) từ TAGTrainDataset
    2. Tra bảng id2text để lấy text description cho src và dst
    3. Tokenize text bằng BPE tokenizer của CLIP
    4. Đưa qua model CLIP.forward() để tính:
       - s_graph_feats: Graph embedding của source node (qua Graph Transformer)
       - s_text_feats: Text embedding của source node (qua Text Transformer)
       - t_text_feats: Text embedding của target node (qua Text Transformer)
    5. Tính 3 loại alignment loss:
       - s_node_loss: Căn chỉnh graph ↔ text của source node
       - s_gt_loss: Căn chỉnh graph source ↔ text target (liên kết cạnh)
       - tt_loss: Căn chỉnh text source ↔ text target
    6. Tổng loss = s_node_loss + edge_coef * s_gt_loss + edge_coef * tt_loss
    """
    model.train()
    best_test_acc = 0
    for epoch in range(0, args.epoch_num):
        epoch_loss = 0.0

        for step, batch in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            src, rel, dst = batch[0]
            gnn_labels = batch[1]

            src_arr = src.numpy()
            dst_arr = dst.numpy().reshape(-1)
            src_text, dst_text = [id2text[i] for i in src_arr], [id2text[j] for j in dst_arr]
            src_text = tokenize(src_text, context_length=args.context_length).to(device)  # (B,L)
            dst_text = tokenize(dst_text, context_length=args.context_length).to(device)  # (B*neigh_num,L)

            src, rel, dst = src.to(device), rel.to(device), dst.to(device)
            gnn_labels = gnn_labels.to(device)

            s_graph_feats, s_text_feats, t_text_feats, text_labels = model(
                whole_graph, src, rel, dst, src_text, dst_text, device
            )

            s_node_loss = model.align_loss(s_graph_feats, s_text_feats, text_labels)
            s_gt_loss = model.align_loss(s_graph_feats, t_text_feats, text_labels)
            tt_loss = model.align_loss(s_text_feats, t_text_feats, text_labels)

            all_loss = s_node_loss + args.edge_coef * s_gt_loss + args.edge_coef * tt_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)
            if step % 100 == 0:
                logging.info("{}th loss in {} epoch:{}".format(step, epoch, loss))
            epoch_loss += loss / len(train_loader)
        logging.info("{}th epoch mean loss:{}".format(epoch, epoch_loss))
        torch.save(model.state_dict(), model_save_path.replace(".pkl", f"_{epoch}th.pkl"))

        test_acc = evaluate(epoch)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            logging.info("{}th epoch save the best model".format(epoch))
            torch.save(model.state_dict(), model_save_path.replace(".pkl", "_best.pkl"))


# ====================================================================
# HÀM EVALUATE — Sao chép chuẩn xác từ SAT main.py:evaluate()
# ====================================================================
def evaluate(epoch=0):
    """
    Đánh giá mô hình trên tập test.
    
    Tính accuracy dựa trên khả năng dự đoán đúng alignment giữa
    graph features và text features.
    """
    model.eval()
    all_true, all_pred = [], []
    for step, batch in tqdm(enumerate(eval_loader), disable=False, total=len(eval_loader)):
        src, rel, dst = batch[0]
        gnn_labels = batch[1]

        src_arr = src.numpy()
        dst_arr = dst.numpy().reshape(-1)
        src_text, dst_text = [id2text[i] for i in src_arr], [id2text[j] for j in dst_arr]
        src_text = tokenize(src_text, context_length=args.context_length).to(device)
        dst_text = tokenize(dst_text, context_length=args.context_length).to(device)

        src, rel, dst = src.to(device), rel.to(device), dst.to(device)
        gnn_labels = gnn_labels.to(device)

        s_graph_feats, s_text_feats, t_text_feats, text_labels = model(
            whole_graph, src, rel, dst, src_text, dst_text, device
        )

        s_node_pred = model.align_pred(s_graph_feats, s_text_feats, text_labels)
        s_gt_pred = model.align_pred(s_graph_feats, t_text_feats, text_labels)
        tt_pred = model.align_pred(s_text_feats, t_text_feats, text_labels)

        true_label = text_labels.cpu().numpy().tolist()
        s_node_pred = s_node_pred.cpu().detach().numpy().tolist()
        s_gt_pred = s_gt_pred.cpu().detach().numpy().tolist()
        tt_pred = tt_pred.cpu().detach().numpy().tolist()
        all_true.extend(true_label)
        all_true.extend(true_label)
        all_true.extend(true_label)
        all_pred.extend(s_node_pred)
        all_pred.extend(s_gt_pred)
        all_pred.extend(tt_pred)

    acc = accuracy_score(all_true, all_pred)
    logging.info("{}th epoch test accuracy:{:.4f}".format(epoch, acc))
    return acc


# ====================================================================
# HÀM SAVE LLM CLIP — Xuất embeddings sau khi train xong
# ====================================================================
def save_llm_clip(args):
    """
    Sau khi train xong, export entity embeddings ra file .pt
    để sử dụng cho bước Predictor (GraphLlama) tiếp theo.
    """
    model_path = f"{args.output_path}/{args.data_name}/gt-best-og_best.pkl"
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    
    entity_embedding = model.gnn.entity_embedding.cpu()
    embs = []
    for i in range(args.entity_num):
        i = torch.tensor([i])
        tmp = entity_embedding(i)
        embs.append(tmp)
    embs = torch.stack(embs, dim=0)
    embs = torch.squeeze(embs, dim=1)
    torch.save(embs, f"{args.output_path}/{args.data_name}/entity_embedding.pt")
    
    save_clip_data(args, data_flag=['train', 'valid', 'test'])
    logging.info("Entity embeddings saved successfully!")


# ====================================================================
# UTILS — Sao chép từ SAT main.py
# ====================================================================
def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logger(log_file='./log.txt'):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(format)
    logger.addHandler(handler1)

    if log_file:
        assure_dir(log_file)
        handler2 = logging.FileHandler(log_file)
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(format)
        logger.addHandler(handler2)
    return logger


# ====================================================================
# ARGUMENT PARSER — Cấu hình tham số
# ====================================================================
def args_parser():
    parser = argparse.ArgumentParser(description="Train SAT Aligner trên bộ dữ liệu Medical KG")

    # Đường dẫn dữ liệu — trỏ tới data/medical thay vì SAT/aligner/data/FB15k-237N
    parser.add_argument("--data_path", type=str, default="./data", help="Thư mục gốc chứa dữ liệu")
    parser.add_argument("--output_path", type=str, default="./checkpoints")
    parser.add_argument("--data_name", type=str, default="medical")  # Tên dataset = medical
    parser.add_argument("--cpu_worker_num", type=int, default=3)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)

    # Hyperparameters Training — giống SAT gốc
    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    # CLIP Text Encoder config — giống SAT gốc
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)

    # Graph Transformer config — giống SAT gốc
    parser.add_argument("--gnn_type", type=str, default="gt")
    parser.add_argument("--gnn_input", type=int, default=128)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_output", type=int, default=128)

    parser.add_argument("--node_num", type=int, default=1)
    parser.add_argument("--gt_layers", type=int, default=3)
    parser.add_argument("--att_d_model", type=int, default=128)
    parser.add_argument("--gt_head", type=int, default=8)
    parser.add_argument("--att_norm", type=bool, default=True)
    parser.add_argument("--if_pos", type=bool, default=False)

    # ConvE config — giống SAT gốc
    parser.add_argument("--out_channels", type=int, default=200)
    parser.add_argument("--ker_size", type=int, default=4)
    parser.add_argument("--ker_height", type=int, default=8)
    parser.add_argument("--ker_width", type=int, default=16)

    args = parser.parse_args()
    return args


# ====================================================================
# MAIN — Entry point
# ====================================================================
if __name__ == "__main__":
    args = args_parser()
    args.data_path = os.path.join(args.data_path, args.data_name)
    setup_seed(seed=1)
    args.cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_save_path = f'./logs/{args.data_name}/aligner_{args.cur_time}.log'
    logger = set_logger(log_save_path)
    logging.info(f"log file: {log_save_path}")
    logging.info(args)

    model_save_name = f"{args.data_name}/{args.gnn_type}-{args.cur_time}-og.pkl"
    model_save_path = os.path.join(args.output_path, model_save_name)

    # Auto-detect hardware accelerator (CUDA, MPS cho Mac M1, hoặc CPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")

    # ---- Load dữ liệu từ data/medical/ ----
    id2text = get_id2text(os.path.join(args.data_path, "id2text.txt"))
    ent2id = get_mid2id(os.path.join(args.data_path, "mid2id.txt"))
    rel2id = get_rel2id(os.path.join(args.data_path, "rel2id.txt"))
    args.entity_num = len(ent2id)
    args.relation_num = len(rel2id)
    logging.info(f"entity_num: {args.entity_num}, relation_num: {args.relation_num}")

    # ---- Xây dựng đồ thị PyTorch Geometric từ toàn bộ triples ----
    whole_graph = construct_graph(args, data_flag=['train', 'valid', 'test'])
    whole_graph = whole_graph.to(device)

    # ---- Tạo DataLoader cho training (train+valid) và evaluation (test) ----
    train_dataset = TAGTrainDataset(args, ['train', 'valid'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.cpu_worker_num,
        collate_fn=train_dataset.collate_fn
    )

    eval_dataset = TAGTrainDataset(args, ['test'])
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_worker_num,
        collate_fn=eval_dataset.collate_fn
    )

    # ---- Khởi tạo mô hình CLIP (Text Encoder + Graph Transformer) ----
    model = CLIP(args).to(device)

    # ---- Bắt đầu Training ----
    logging.info("=" * 60)
    logging.info("BẮT ĐẦU TRAINING SAT ALIGNER TRÊN BỘ DỮ LIỆU MEDICAL")
    logging.info(f"  Entities: {args.entity_num}")
    logging.info(f"  Relations: {args.relation_num}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  Epochs: {args.epoch_num}")
    logging.info(f"  Learning Rate: {args.lr}")
    logging.info(f"  Device: {device}")
    logging.info("=" * 60)

    train()
    logging.info("Training hoàn tất!")

    # ---- Export embeddings (tùy chọn) ----
    # save_llm_clip(args)
