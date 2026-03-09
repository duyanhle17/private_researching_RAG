import os
import json
import argparse
import random
import logging
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import numpy as np

# Config Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_text(text, chunk_size=1000, overlap=150):
    """Phân tách văn bản thành các đoạn nhỏ với một chút chồng chéo (overlap)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def extract_triples_from_chunk(client, model, chunk, chunk_id):
    """Gọi API NVIDIA NIM để trích xuất bộ ba (triples) từ một đoạn văn bản."""
    prompt = """You are an expert at constructing Knowledge Graphs from medical text.
Extract entities and relationships from the provided text.
Return the result EXACTLY as a JSON list of lists, where each inner list represents a triple: ["head_entity", "relation", "tail_entity"].

CRITICAL RULES FOR ENTITIES & RELATIONS:
1. Entities MUST be valid NOUNS or NOUN PHRASES representing clear concepts (e.g., diseases, anatomical parts, medical procedures, chemicals, people, organizations, symptoms). 
2. DO NOT extract verbs, verb phrases, actions, or full clauses as entities (e.g., WRONG: "fall asleep", "treating cancer", "takes 2 weeks").
3. DO NOT extract measurements or vague pronouns as entities (e.g., WRONG: "1 cm", "1-2 hours", "he", "they", "4_out_of_10").
4. Relations should be concise verbs/predicates connecting the entities (e.g., "is_a", "treats", "causes", "located_in", "has_symptom").
5. Normalize entity names (e.g., remove unnecessary articles like "the", "a", "an", and use consistent spacing).
If no valid relations are found, return an empty list: [].
Do not output any markdown formatting, just the raw JSON array.

Text:
""" + chunk

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content.strip()
        
        # Loại bỏ markdown nếu LLM trả về dạng ```json ... ```
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        triples = json.loads(content.strip())
        
        # Validate định dạng triples
        valid_triples = []
        if isinstance(triples, list):
            for t in triples:
                if isinstance(t, list) and len(t) == 3:
                    valid_triples.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
        return {"chunk_id": chunk_id, "text": chunk, "triples": valid_triples, "status": "success"}
    except Exception as e:
        # logging.error(f"Error processing chunk {chunk_id}: {e}")
        return {"chunk_id": chunk_id, "text": chunk, "triples": [], "status": "error", "error": str(e)}

def build_dataset(args):
    # Khởi tạo OpenAI Client cho NVIDIA NIM
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Vui lòng cung cấp NVIDIA_API_KEY qua argument --api_key hoặc biến môi trường NVIDIA_API_KEY")
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    # 1. Đọc dữ liệu đầu vào
    logging.info(f"Đọc dữ liệu từ {args.input_path}...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    # 2. Chunking text
    logging.info("Chia nhỏ văn bản (Chunking)...")
    all_chunks = []
    chunk_counter = 0
    for item in raw_data:
        context = item.get('context', '')
        # Nếu context là list (một số định dạng bị lỗi hiển thị thành list), nối lại
        if isinstance(context, list):
            context = " ".join(str(x) for x in context)
            
        chunks = chunk_text(context, chunk_size=args.chunk_words, overlap=args.overlap_words)
        for c in chunks:
            all_chunks.append({
                "chunk_id": f"{item.get('corpus_name', 'Doc')}_chunk_{chunk_counter}",
                "text": c
            })
            chunk_counter += 1
            
    logging.info(f"Tổng số chunks cần xử lý: {len(all_chunks)}")

    # 3. Trích xuất đa luồng bằng ThreadPoolExecutor & Checkpoint
    checkpoint_file = f"checkpoint_{args.output_name}.json"
    extracted_data = []
    processed_chunk_ids = set()
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
            for item in extracted_data:
                processed_chunk_ids.add(item.get("chunk_id"))
        logging.info(f"Đã khôi phục {len(extracted_data)} chunks đã xử lý từ {checkpoint_file}")

    pending_chunks = [c for c in all_chunks if c["chunk_id"] not in processed_chunk_ids]
    logging.info(f"Bắt đầu trích xuất bằng {args.model} với {args.max_workers} workers...")
    logging.info(f"Số chunks còn lại cần trích xuất qua API: {len(pending_chunks)}")
    
    # Xử lý theo batch_size như yêu cầu của người dùng để tối ưu luồng
    for i in tqdm(range(0, len(pending_chunks), args.batch_size), desc="Processing Batches"):
        batch_chunks = pending_chunks[i:i + args.batch_size]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_chunk = {
                executor.submit(extract_triples_from_chunk, client, args.model, chunk["text"], chunk["chunk_id"]): chunk
                for chunk in batch_chunks
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                result = future.result()
                extracted_data.append(result)
                
        # Tự động lưu checkpoint sau khi xong mỗi batch
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False)

    # 4. Hợp nhất, làm sạch và gán ID
    logging.info("Hợp nhất, làm sạch thực thể và quan hệ...")
    entities_set = set()
    relations_set = set()
    all_valid_triples = []
    
    valid_chunks_for_global = []
    entity_desc_map = {}

    for item in extracted_data:
        if item["status"] == "success" and len(item["triples"]) > 0:
            chunk_ents = set()
            clean_text = item["text"].replace("\n", " ").replace("\t", " ").strip()
            
            for h, r, t in item["triples"]:
                h_norm, r_norm, t_norm = h.lower(), r.lower(), t.lower()
                if not h_norm or not r_norm or not t_norm:
                    continue
                entities_set.add(h_norm)
                entities_set.add(t_norm)
                relations_set.add(r_norm)
                chunk_ents.update([h_norm, t_norm])
                
                # Lưu text mô tả cho entity (Lấy đoạn văn đầu tiên chứa nó làm ngữ cảnh)
                if h_norm not in entity_desc_map:
                    entity_desc_map[h_norm] = clean_text
                if t_norm not in entity_desc_map:
                    entity_desc_map[t_norm] = clean_text
                    
                # Lưu triple gốc để map dạng text ở global json
                all_valid_triples.append((h_norm, r_norm, t_norm))
                
            valid_chunks_for_global.append({
                "id": item["chunk_id"],
                "text": item["text"],
                "entities": list(chunk_ents),
                "triples_str": "; ".join([f"({h}, {r}, {t})" for h, r, t in item["triples"]]),
                "triple_list": item["triples"]
            })

    # Xoá trùng lặp triples cấp độ toàn cục
    all_valid_triples = list(set(all_valid_triples))
    
    entities_list = sorted(list(entities_set))
    relations_list = sorted(list(relations_set))
    
    logging.info(f"Tổng số Thực thể: {len(entities_list)}")
    logging.info(f"Tổng số Quan hệ: {len(relations_list)}")
    logging.info(f"Tổng số Triples độc nhất: {len(all_valid_triples)}")
    
    if len(all_valid_triples) == 0:
        logging.error("Không trích xuất được triple nào. Dừng quá trình.")
        return

    # Map sang ID
    ent2id = {ent: idx for idx, ent in enumerate(entities_list)}
    rel2id = {rel: idx for idx, rel in enumerate(relations_list)}
    
    # 5. Xuất các file Mapping
    data_dir = os.path.join(args.output_dir, "data", args.output_name)
    os.makedirs(data_dir, exist_ok=True)
    
    logging.info(f"Ghi các file mapping ra {data_dir}...")
    
    # mid2id.txt: <entity_mid> \t <id>
    # Giả lập định dạng Freebase MID bằng cách thêm m tiền tố (VD: /m/medical_1)
    # Tuy nhiên model SAT chỉ đọc chuỗi text ban đầu (src, rel, dst) và map dictionary
    with open(os.path.join(data_dir, "mid2id.txt"), "w", encoding="utf-8") as f:
        for ent, idx in ent2id.items():
            mid = f"/m/{idx}"  # Sinh một custom MID
            f.write(f"{mid}\t{idx}\n")
            
    with open(os.path.join(data_dir, "rel2id.txt"), "w", encoding="utf-8") as f:
        for rel, idx in rel2id.items():
            f.write(f"{rel}\t{idx}\n")
            
    # Map map ngược mid về string entity để lưu map txt
    id2mid = {idx: f"/m/{idx}" for idx in range(len(entities_list))}
            
    # id2text.txt: id \t description
    # id2title.txt: id \t title
    with open(os.path.join(data_dir, "id2text.txt"), "w", encoding="utf-8") as f_text, \
         open(os.path.join(data_dir, "id2title.txt"), "w", encoding="utf-8") as f_title:
        for idx, ent in enumerate(entities_list):
            desc = entity_desc_map.get(ent, ent)
            f_text.write(f"{idx}\t{desc}\n")
            f_title.write(f"{idx}\t{ent}\n")

    # 6. Chia tập Train / Valid / Test
    random.seed(42)
    random.shuffle(all_valid_triples)
    n_total = len(all_valid_triples)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)
    
    train_triples = all_valid_triples[:n_train]
    valid_triples = all_valid_triples[n_train:n_train + n_valid]
    test_triples = all_valid_triples[n_train + n_valid:]
    
    # Lưu file Triples (định dạng <head_mid> \t <relation> \t <tail_mid>)
    # Lưu ý `rel2id.txt` dùng string relation trực tiếp, nhưng thực thể dùng MID
    def save_triples_file(triples, filename):
        with open(os.path.join(data_dir, filename), "w", encoding="utf-8") as f:
            for h, r, t in triples:
                h_mid = id2mid[ent2id[h]]
                t_mid = id2mid[ent2id[t]]
                f.write(f"{h_mid}\t{r}\t{t_mid}\n")
                
    save_triples_file(train_triples, "train.txt")
    save_triples_file(valid_triples, "valid.txt")
    save_triples_file(test_triples, "test.txt")
    
    # 7. Sinh Mẫu Âm (Negative Samples) - 2 mẫu 1 triple
    def build_negatives(triples, filename):
        entities_ids = np.arange(len(entities_list), dtype=np.int32)
        
        # Pre-build tails cho mỗi (h, r)
        pos_tails = {}
        for h, r, t in all_valid_triples:
            key = (ent2id[h], rel2id[r])
            if key not in pos_tails:
                pos_tails[key] = set()
            pos_tails[key].add(ent2id[t])
            
        with open(os.path.join(data_dir, filename), "w", encoding="utf-8") as f:
            for h, r, t in triples:
                h_idx, r_idx, t_idx = ent2id[h], rel2id[r], ent2id[t]
                
                mask = np.ones(len(entities_list), dtype=bool)
                mask[list(pos_tails.get((h_idx, r_idx), []))] = False
                
                # Sample 2 negative tails
                valid_neg_ents = entities_ids[mask]
                if len(valid_neg_ents) >= 2:
                    neg_ents = np.random.choice(valid_neg_ents, 2, replace=False)
                else:
                    neg_ents = valid_neg_ents # Fallback if very small KB
                    
                for neg_t in neg_ents:
                    h_mid = id2mid[h_idx]
                    neg_t_mid = id2mid[neg_t]
                    f.write(f"{h_mid}\t{r}\t{neg_t_mid}\n")
                    
    logging.info("Tạo Negative Samples...")
    build_negatives(train_triples, "neg_train.txt")
    build_negatives(valid_triples, "neg_valid.txt")
    build_negatives(test_triples, "neg_test.txt")

    # 8. Sinh Global Data (text2graph) JSON files
    data_global_dir = os.path.join(args.output_dir, "data_global", args.output_name)
    os.makedirs(data_global_dir, exist_ok=True)
    
    # text2graph_filter.json
    with open(os.path.join(data_global_dir, "text2graph_filter.json"), "w", encoding="utf-8") as f:
        filter_out = []
        for c in valid_chunks_for_global:
            filter_out.append({
                "id": c["id"],
                "text": c["text"],
                "entities": c["entities"],
                "triples": c["triples_str"] + ".",
                "triple_list": c["triple_list"],
                "filter_triple_list": []  # Có thể tính toán thêm filter nếu muốn
            })
        json.dump(filter_out, f, indent=2, ensure_ascii=False)
        
    # text2graph_pairs_train & test
    # Graph bao gồm: edge_index và node_list. node_list map ra các node IDs có xuất hiện, 
    # edge_index nối ID cục bộ trong mảng node_list (0..len-1).
    def build_pairs_json(chunks):
        out = []
        for c in chunks:
            # entities xuất hiện cục bộ
            local_nodes_str = list(set([h.lower() for h,_,_ in c["triple_list"]] + [t.lower() for _,_,t in c["triple_list"]]))
            # node_list trong text2graph lưu id thực thể tuyệt đối
            node_list = []
            str2localId = {}
            for loc_id, e_str in enumerate(local_nodes_str):
                if e_str in ent2id:
                    node_list.append(ent2id[e_str])
                    str2localId[e_str] = loc_id
            
            edge_index_src = []
            edge_index_dst = []
            for h, r, t in c["triple_list"]:
                h_str, t_str = h.lower(), t.lower()
                if h_str in str2localId and t_str in str2localId:
                    edge_index_src.append(str2localId[h_str])
                    edge_index_dst.append(str2localId[t_str])
                    
            out.append({
                "id": c["id"],
                "text": c["text"],
                "graph": {
                    "edge_index": [edge_index_src, edge_index_dst],
                    "node_list": node_list
                }
            })
        return out

    # Tạm chia đôi chunks cho cặp train/test text2graph
    random.shuffle(valid_chunks_for_global)
    split_idx = int(len(valid_chunks_for_global) * 0.9)
    train_chunks = valid_chunks_for_global[:split_idx]
    test_chunks = valid_chunks_for_global[split_idx:]
    
    with open(os.path.join(data_global_dir, "text2graph_pairs_train.json"), "w", encoding="utf-8") as f:
        json.dump(build_pairs_json(train_chunks), f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(data_global_dir, "text2graph_pairs_test.json"), "w", encoding="utf-8") as f:
        json.dump(build_pairs_json(test_chunks), f, indent=2, ensure_ascii=False)
        
    logging.info(f"Hoàn thành! Bạn có thể kiểm tra dữ liệu tại {data_dir} và {data_global_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất Knowledge Graph từ raw text sử dụng mô hình LLM thông qua NVIDIA NIM.")
    parser.add_argument("--input_path", type=str, default="raw_dataset/medical.json", help="Đường dẫn đến file raw dataset JSON")
    parser.add_argument("--output_name", type=str, default="medical", help="Tên của bộ dataset đầu ra")
    parser.add_argument("--output_dir", type=str, default="./", help="Thư mục xuất kết quả (thường là folder chứa data và data_global)")
    parser.add_argument("--model", type=str, default="meta/llama3-70b-instruct", help="Model NVIDIA NIM để trích xuất")
    parser.add_argument("--api_key", type=str, default="", help="API Key NVIDIA (hoặc dùng NVIDIA_API_KEY trong env)")
    
    parser.add_argument("--chunk_words", type=int, default=200, help="Số từ của mỗi chunk đoạn text")
    parser.add_argument("--overlap_words", type=int, default=20, help="Số từ trùng lặp khi chunk (overlap)")
    
    parser.add_argument("--max_workers", type=int, default=35, help="Số luồng request song song (ThreadPool)")
    parser.add_argument("--batch_size", type=int, default=100, help="Xử lý lần lượt bao nhiêu chunk mỗi vòng batch (Để API call nhịp nhàng)")
    
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    build_dataset(args)
