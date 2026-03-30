# Bao cao tong hop Claim-Only tren H100 (2026-03-29)

## 1) Thong tin chay
- Bai toan: FactKG Claim-Only
- Model train: bert-base-uncased
- Model zero-shot: google/flan-t5-xl
- GPU: NVIDIA H100 80GB

## 2) Ket qua train BERT
- Checkpoint da tao:
  - /root/FactKG/bert/checkpoint-0/pytorch_model.bin
  - /root/FactKG/bert/checkpoint-1/pytorch_model.bin
  - /root/FactKG/bert/checkpoint-2/pytorch_model.bin

- Log chinh:
  - /root/FactKG/runs/bert_live.log
  - /root/FactKG/exp_bert_h100_full_live/bert-base-uncased_2026-03-29 16.10.47.log

## 3) Tong thoi gian chay BERT
- Thoi gian tinh theo timestamp trong experiment log:
  - Bat dau: 2026-03-29 16:10:48
  - Ket thuc: 2026-03-29 16:22:26
  - Tong: 698 giay (11 phut 38 giay)

Luu y:
- Con so tren la thoi gian co log timestamp trong file exp.
- Co the chenh nhe voi wall-clock do overhead truoc/sau log.

## 4) Danh gia accuracy theo 5 reasoning type (BERT)
Script danh gia: /root/FactKG/claim_only/eval_reasoning_accuracy.py

### Tong accuracy theo checkpoint
- checkpoint-0: 61.83%
- checkpoint-1: 64.22% (best)
- checkpoint-2: 63.06%

### Chi tiet 5 reasoning type (checkpoint-1)
- One-hop: 66.14% (1914 mau)
- Conjunction: 62.69% (3069 mau)
- Existence: 65.75% (870 mau)
- Multi-hop: 61.10% (1874 mau)
- Negation: 68.42% (1314 mau)

## 5) Ket qua Flan-T5-XL (zero-shot)
- Log tong: /root/FactKG/runs/flan_xl_live.log
- Script chi tiet theo reasoning: /root/FactKG/claim_only/eval_flan_reasoning_accuracy.py
- Tong mau danh gia: 9041
- Tong accuracy: 62.82%

### Chi tiet 5 reasoning type (Flan-T5-XL)
- One-hop: 66.50% (2376 mau)
- Conjunction: 65.44% (3293 mau)
- Existence: 52.35% (1299 mau)
- Multi-hop: 61.02% (2073 mau)
- Negation: 54.03% (1314 mau)

## 6) Ket luan nhanh
- BERT train hoan tat thanh cong 3 epoch, tao du checkpoint.
- Checkpoint tot nhat theo tong accuracy test: checkpoint-1 (64.22%).
- Flan-T5-XL zero-shot dat 62.82% tren test; nhom manh la One-hop/Conjunction, nhom yeu hon la Existence/Negation.
