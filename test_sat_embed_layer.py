import sys, os
import torch
SAT_MODEL_DIR = os.path.join("SAT", "aligner", "model")
sys.path.insert(0, SAT_MODEL_DIR)
from model_gt import CLIP

class TestEmbed:
    def __init__(self):
        import argparse
        args = argparse.Namespace()
        args.context_length = 128
        args.embed_dim = 128
        args.transformer_heads = 8
        args.transformer_layers = 12
        args.transformer_width = 512
        args.vocab_size = 49408
        args.gnn_type = "gt"
        args.gnn_input = 128
        args.gnn_hidden = 128
        args.gnn_output = 128
        args.node_num = 1
        args.gt_layers = 3
        args.att_d_model = 128
        args.gt_head = 8
        args.att_norm = True
        args.if_pos = False
        args.edge_coef = 10
        args.lr = 2e-5
        args.entity_num = 3281
        args.relation_num = 311
        args.out_channels = 200
        args.ker_size = 4
        args.ker_height = 8
        args.ker_width = 16
        
        print("init clip")
        self.model = CLIP(args)
        print("Model initialized. Loading from file...")
        try:
            state_dict = torch.load("checkpoints/medical/gt-og_best.pkl", map_location="cpu")
            self.model.load_state_dict(state_dict)
            print("Loaded successfully via state_dict.")
        except Exception as e:
            print(f"Error loading: {e}")
        
t = TestEmbed()
