import os
import json
import re
import argparse
import random
import logging
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import numpy as np

# Config Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _clean_text(text):
    """Chuẩn hóa văn bản thô: loại bỏ khoảng trắng thừa, dòng trống, ký tự rác."""
    # 1. Thay thế nhiều space/tab liên tiếp thành 1 space
    text = re.sub(r'[ \t]+', ' ', text)
    # 2. Thay thế nhiều dòng trống liên tiếp thành 1 dòng trống (giữ paragraph boundary)
    text = re.sub(r'\n\s*\n[\s\n]*', '\n\n', text)
    # 3. Loại bỏ dòng trống đầu/cuối
    text = text.strip()
    return text


def _split_sentences(text):
    """Tách văn bản thành danh sách các câu dựa trên dấu câu (. ? ! và xuống dòng)."""
    # Tiền xử lý: chuẩn hóa text trước
    text = _clean_text(text)
    # Regex: tách tại dấu chấm câu theo sau bởi khoảng trắng hoặc xuống dòng,
    # nhưng không tách giữa các viết tắt phổ biến (e.g., Dr., vs., etc.)
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 5:  # Bỏ câu quá ngắn (chỉ có ký tự đặc biệt hoặc rác)
            sentences.append(s)
    return sentences


def chunk_text(text, chunk_size=200, overlap=30):
    """Phân tách văn bản thành các đoạn nhỏ theo ranh giới câu (Semantic Chunking).
    
    Giống cách SAT (FB15k-237N) xây dựng dữ liệu: mỗi chunk là 1 đoạn văn
    trọn vẹn gồm 3-8 câu (~150-300 từ), không cắt ngang câu.
    
    Args:
        text: Văn bản đầu vào
        chunk_size: Số từ TỐI ĐA mỗi chunk (mặc định 200)
        overlap: Số từ overlap TỐI THIỂU giữa 2 chunk liền kề (mặc định 30, ~1-2 câu cuối)
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_sentences = []
    current_word_count = 0
    
    for sent in sentences:
        sent_words = len(sent.split())
        
        # Nếu thêm câu này vượt quá chunk_size VÀ chunk hiện tại đã có nội dung
        if current_word_count + sent_words > chunk_size and current_sentences:
            # Lưu chunk hiện tại
            chunk_text_str = " ".join(current_sentences)
            if len(chunk_text_str.split()) >= 20:  # Bỏ chunk quá ngắn (<20 từ)
                chunks.append(chunk_text_str)
            
            # Tính overlap: lấy 1-2 câu cuối cùng của chunk hiện tại làm đầu chunk mới
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_wc = len(s.split())
                if overlap_count + s_wc <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_wc
                else:
                    break
            
            current_sentences = overlap_sentences
            current_word_count = overlap_count
        
        current_sentences.append(sent)
        current_word_count += sent_words
    
    # Chunk cuối cùng
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if len(chunk_text_str.split()) >= 20:
            chunks.append(chunk_text_str)
        elif chunks:
            # Nếu chunk cuối quá ngắn, nối vào chunk trước
            chunks[-1] = chunks[-1] + " " + chunk_text_str
    
    return chunks

# ============= BỘ LỌC ENTITY HẬU KỲ =============
# Danh sách từ quá mơ hồ / chung chung không nên làm entity
VAGUE_ENTITIES = {
    "ability", "area", "areas", "amount", "analysis", "approach", "assessment",
    "benefit", "benefits", "book", "care", "case", "cause", "change", "changes",
    "chance", "condition", "conditions", "concern", "copy", "data", "day",
    "days", "detail", "details", "difference", "effect", "effects", "event",
    "example", "experience", "factor", "factors", "feature", "features",
    "finding", "findings", "form", "function", "goal", "group", "growth",
    "guide", "help", "history", "hour", "hours", "impact", "increase",
    "information", "issue", "issues", "item", "kind", "level", "levels",
    "list", "location", "loss", "manner", "matter", "method", "minute",
    "minutes", "month", "months", "need", "number", "option", "options",
    "outcome", "outcomes", "part", "parts", "patient", "patients", "people",
    "percent", "period", "place", "plan", "point", "portion", "position",
    "possibility", "problem", "problems", "process", "program", "progress",
    "purpose", "question", "questions", "range", "rate", "reason", "reasons",
    "recommendation", "region", "report", "result", "results", "risk",
    "role", "rule", "sample", "section", "set", "side", "sign", "signs",
    "situation", "size", "source", "space", "stage", "state", "status",
    "step", "steps", "structure", "study", "style", "subject", "support",
    "surface", "symptom", "system", "team", "technique", "test", "testing",
    "thing", "things", "time", "tip", "tool", "topic", "treatment",
    "trial", "type", "types", "use", "value", "view", "way", "week",
    "weeks", "work", "year", "years", "a test", "anything", "assistant",
    "abbreviations", "advocate", "advancement", "after surgery", "after treatment",
    "before surgery", "before treatment", "a", "an", "the", "he", "she",
    "it", "they", "we", "you", "i", "me", "us", "him", "her", "them",
}

# Regex: entity chỉ có số, ký tự đặc biệt và/hoặc đơn vị đo
_RE_NUMERIC_ONLY = re.compile(
    r'^[\d\s.,/\-–—~<>≤≥+=%°\(\)]*'
    r'(cm|mm|mg|ml|kg|lb|hours?|minutes?|days?|weeks?|months?|years?|'
    r'percent|inches?|seconds?|liters?|cc|mcg|mg/m2|gy|mv)?'
    r'[\d\s.,/\-–—~<>≤≥+=%°\(\)]*$',
    re.IGNORECASE
)

def is_valid_entity(entity: str) -> bool:
    """Kiểm tra xem một entity có đủ tiêu chuẩn để giữ lại không."""
    e = entity.strip().lower()
    
    # 1. Quá ngắn (1 ký tự) hoặc quá dài (>80 ký tự ~ 1 câu)
    if len(e) <= 1 or len(e) > 80:
        return False
    
    # 2. Chỉ toàn số / đo lường  (VD: "1 cm", "1-2 hours", "30 minutes")
    if _RE_NUMERIC_ONLY.match(e):
        return False

    # 3. Trong danh sách từ mơ hồ
    if e in VAGUE_ENTITIES:
        return False
    
    # 4. Entity chỉ có số (VD: "0", "10", "5")
    if e.replace(' ', '').replace('-', '').replace('_', '').isdigit():
        return False
    
    # 5. Bắt đầu hoàn toàn bằng số + "to" + số (VD: "0 to 5", "15 to 39 years")
    if re.match(r'^\d+\s*(to|or)\s*\d+', e):
        return False
    
    return True

def extract_triples_from_chunk(client, model, chunk, chunk_id):
    """Gọi API NVIDIA NIM để trích xuất bộ ba (triples) từ một đoạn văn bản."""
    prompt = """You are a medical Knowledge Graph expert. Extract ONLY the core entities and their relationships.
Return EXACTLY a JSON list of lists: [["head_entity", "relation", "tail_entity"], ...]

ENTITY RULES (STRICT - only these 5 types are valid):
1. DISEASES / CONDITIONS: Named diseases, syndromes, disorders (e.g., "lung cancer", "diabetes", "hypertension")
2. DRUGS / CHEMICALS: Medications, vaccines, chemical compounds (e.g., "aspirin", "insulin", "selenium")
3. ANATOMY: Body parts, organs, tissues, cells (e.g., "liver", "blood vessels", "T-cells")
4. PATHOGENS: Viruses, bacteria, parasites (e.g., "HIV", "E. coli", "malaria parasite")
5. PROCEDURES / TREATMENTS: Medical procedures, therapies, diagnostic tests (e.g., "chemotherapy", "MRI scan", "biopsy")

INVALID ENTITIES (DO NOT EXTRACT):
- Numbers, measurements, durations: "1 cm", "2 hours", "stage 2"
- Pronouns: "he", "she", "it", "they", "patient"
- Generic words: "treatment", "condition", "problem", "area", "ability"
- Verbs or actions: "treating", "diagnosed", "prescribing"

RELATION RULES: Use short verb-phrases: "treats", "causes", "is_a", "located_in", "has_symptom", "inhibits", "prevents", "diagnosed_by"

If no valid triples exist, return: []
Output raw JSON only, no markdown.

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
        
        # 1. Nếu API trả về rỗng (bị block bởi guardrails, rate-limit,...)
        if not content:
            return {"chunk_id": chunk_id, "text": chunk, "triples": [], "status": "error", "error": "API response is empty (could be NVIDIA NIM guardrails/rate-limit)."}
            
        
        
        # 2. Extract mảng JSON lớn nhất từ văn bản bằng biểu thức chính quy (Regex)
        match = re.search(r'\[\s*\[.*?\]\s*\]', content, re.DOTALL)
        
        triples = []
        if match:
            triples = json.loads(match.group(0))
        else:
            # Fallback kịch bản LLM trả về đúng [] (không tìm thấy thực thể)
            empty_match = re.search(r'\[\s*\]', content)
            if empty_match:
                triples = []
            else:
                return {"chunk_id": chunk_id, "text": chunk, "triples": [], "status": "error", "error": f"LLM rác/sai format: {content[:50]}"}
        
        # 3. Validate định dạng triples
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
                # Lọc entity rác bằng bộ lọc hậu kỳ
                if not is_valid_entity(h_norm) or not is_valid_entity(t_norm):
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
    
    parser.add_argument("--chunk_words", type=int, default=200, help="Số từ TỐI ĐA của mỗi chunk (chia theo câu, mặc định 200 ~ 5-8 câu)")
    parser.add_argument("--overlap_words", type=int, default=30, help="Số từ overlap tối thiểu giữa các chunk liền kề (~1-2 câu cuối)")
    
    parser.add_argument("--max_workers", type=int, default=35, help="Số luồng request song song (ThreadPool)")
    parser.add_argument("--batch_size", type=int, default=100, help="Xử lý lần lượt bao nhiêu chunk mỗi vòng batch (Để API call nhịp nhàng)")
    
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    build_dataset(args)
