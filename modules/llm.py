# modules/llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_PATH

tokenizer = AutoTokenizer.from_pretrained(
    LLM_PATH, trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()


def chat(prompt: str, max_new_tokens=256) -> str:
    print("🔥 [LLM] 正在调用 Qwen2.5 推理 ...")
    messages = [
        {"role": "system", "content": "你是一个严谨的学术助手。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return result.strip()
