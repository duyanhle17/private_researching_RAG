import sys, os
import torch
SAT_MODEL_DIR = os.path.join("SAT", "aligner", "model")
sys.path.insert(0, SAT_MODEL_DIR)
from model_gt import CLIP

# Add our SAT embedding wrapper
class SATEmbeddingModel:
    def __init__(self, checkpoint_path, entity_num, relation_num):
        import argparse
        import torch
        from model_gt import CLIP
        
        device_str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.context_length = 128
        
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
        args.entity_num = entity_num
        args.relation_num = relation_num
        args.out_channels = 200
        args.ker_size = 4
        args.ker_height = 8
        args.ker_width = 16

        print(f"Khởi tạo SAT Model (Entity: {entity_num}, Rel: {relation_num}) trên {self.device}")
        self.model = CLIP(args).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import torch
        from model_gt import tokenize
        with torch.no_grad():
            tokenized = tokenize(texts, context_length=self.context_length).to(self.device)
            embeds = self.model.encode_text(tokenized)
            if normalize_embeddings:
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            return embeds.cpu().numpy()

model = SATEmbeddingModel("checkpoints/medical/gt-og_best.pkl", 3281, 311)
texts = ["hello world", "medical text"]
embeds = model.encode(texts)
print(embeds.shape)
