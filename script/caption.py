import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import warnings
import logging

from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers.utils import logging as hf_logging

MODEL_PATH = "/mnt/data/project_kyh/weight/instructblip-vicuna-7b"
DATA_ROOT = "/mnt/data/project_kyh/Anchor-master/data"
JSONL_PATH = "/mnt/data/project_kyh/Anchor-master/data/metadata/Diffusion/tum_data/train/rgb.jsonl"

PROMPTS = [
    "Describe the layout of this optical remote sensing image concisely in a single sentence.",
    
]

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


def clean_caption(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    # 只删除开头重复的 prompt
    for prompt in PROMPTS:
        if text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip(" .,:;!-")
            break

    text = " ".join(text.split())

    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."

    return text


def generate_caption_for_image(image, processor, model, device):
    prompt = PROMPTS[0]

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        num_beams=3,
        repetition_penalty=1.2,
        length_penalty=1.0,
    )

    raw_caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    caption = clean_caption(raw_caption)
    return caption


if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"模型目录不存在: {MODEL_PATH}")
if not os.path.isfile(JSONL_PATH):
    raise FileNotFoundError(f"jsonl 文件不存在: {JSONL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = InstructBlipProcessor.from_pretrained(MODEL_PATH)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)

num_query_tokens = getattr(model.config, "num_query_tokens", None)
if num_query_tokens is None:
    num_query_tokens = 32

model.config.num_query_tokens = num_query_tokens
processor.num_query_tokens = num_query_tokens

if model.get_input_embeddings().num_embeddings != len(processor.tokenizer):
    model.resize_token_embeddings(len(processor.tokenizer))

image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
if image_token_id is not None and image_token_id >= 0:
    model.config.image_token_index = image_token_id

model.to(device)
model.eval()

backup_path = JSONL_PATH + ".bak"
if not os.path.exists(backup_path):
    with open(JSONL_PATH, "r", encoding="utf-8") as f_in, open(backup_path, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())

records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

total = len(records)
print(f"共读取到 {total} 条记录，开始生成 caption...")

updated_records = []

with torch.no_grad():
    for idx, item in enumerate(records, 1):
        file_name = item.get("file_name", "")
        img_path = os.path.join(DATA_ROOT, file_name)

        if not os.path.isfile(img_path):
            print(f"[{idx}/{total}] 图片不存在，跳过: {img_path}")
            updated_records.append(item)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            caption = generate_caption_for_image(image, processor, model, device)
            item["text"] = caption
            print(f"[{idx}/{total}] OK: {file_name} -> {caption}")
        except Exception as e:
            print(f"[{idx}/{total}] ERROR: {file_name} -> {e}")

        updated_records.append(item)

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for item in updated_records:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已完成，结果已写回: {JSONL_PATH}")
print(f"原文件备份保存在: {backup_path}")
