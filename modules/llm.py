import os
import re
import ast
import config
import dashscope
from dashscope import MultiModalConversation

# --- 全局变量初始为 None，实现懒加载 ---
model = None
tokenizer = None
client = None

def init_llm():
    """只有在第一次调用 chat 时，才会真正执行这里的初始化"""
    global model, tokenizer, client
    
    # 防止重复初始化
    if client is not None or model is not None:
        return

    if getattr(config, "USE_API", False):
        print(f"🌐 [LLM] 正在初始化 API 客户端: {config.MODEL_NAME}...")
        from openai import OpenAI  # 移动到函数内，加速脚本启动
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY, 
            base_url=config.DEEPSEEK_BASE_URL
        )
    else:
        print("🚀 [LLM] 正在加载本地显卡引擎 (这可能需要 30-60 秒)...")
        import torch # 移动到内部
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

def chat(prompt: str, system_prompt: str = "你是一个严谨的学术助手。", max_new_tokens=512) -> str:
    # 🎯 只有在这里被调用时，才会去加载模型
    init_llm()
    
    if config.USE_API:
        try:
            response = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ API 失败: {str(e)}"
    else:
        # 本地模型推理逻辑
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def filter_by_text(user_query: str, candidates: list) -> list:
    """级联精排：文本预判层"""
    if not candidates: return []
    print(f"🧠 [Rerank] DeepSeek 正在进行意图对齐...")

    context_list = [f"编号 {i}: {c['description'][:100]}" for i, c in enumerate(candidates)]
    context_text = "\n".join(context_list)

    prompt = f"""
    你是一个极其聪明的图像搜索专家。用户当前的搜索意图是：'{user_query}'
    【重要准则】：
    1. 语义优先：如果用户搜“花”，指的是自然界观赏性的花朵。
    2. 排除干扰：严格排除名称包含该字但属于其他类别的物体（如西兰花、花生、火花等）。
    下面是 {len(candidates)} 张图片的描述：
    {context_text}
    请选出【真正符合意图】的编号。输出格式：[编号, 编号, ...]，不要解释。
    """
    
    # 这里会自动触发 init_llm()
    raw_response = chat(prompt, system_prompt="你是一个具备常识的语义过滤器。")

    try:
        match = re.search(r'\[.*\]', raw_response)
        if match:
            selected_indices = ast.literal_eval(match.group())
            return [candidates[i] for i in selected_indices if i < len(candidates)]
        return candidates[:5]
    except:
        return candidates[:5]

def describe_image(image_path: str) -> str:
    """视觉描述：调用 Qwen-VL API（这个不涉及本地 LLM 加载）"""
    print(f"👁️ [Vision] 正在提取关键点: {os.path.basename(image_path)}")
    prompt = "请为图片生成详细的中文描述标签，以逗号分隔，不要输出长句。"
    messages = [{"role": "user", "content": [{"image": f"file://{image_path}"}, {"text": prompt}]}]
    
    try:
        response = MultiModalConversation.call(api_key=config.QWEN_API_KEY, model='qwen-vl-plus', messages=messages, max_tokens=50)
        return response.output.choices[0].message.content[0]['text'].strip().replace("。", "") if response.status_code == 200 else "提取失败"
    except Exception as e:
        return f"异常: {str(e)}"

def verify_image_content(image_path: str, user_query: str) -> bool:
    print(f"🧐 [Rerank] AI 正在深度核对: {os.path.basename(image_path)}")
    
    # 🎯 核心改动：加入“宽容度”引导，并要求它找局部特征
    check_prompt = f"""
    请仔细观察图片。用户想找的内容是：'{user_query}'。
    如果图中【存在】相关内容（哪怕是背景、局部、或者较小），请回答 'Yes'。
    如果完全不相关，请回答 'No'。
    请直接输出结果，不要解释。
    """
    
    messages = [{"role": "user", "content": [
        {"image": f"file://{os.path.abspath(image_path)}"},
        {"text": check_prompt}
    ]}]
    
    try:
        response = MultiModalConversation.call(
            api_key=config.QWEN_API_KEY, 
            model='qwen-vl-plus', 
            messages=messages, 
            max_tokens=10
        )
        res = response.output.choices[0].message.content[0]['text'].lower()
        
        is_match = "yes" in res
        status = "✅ 通过" if is_match else "❌ 拦截"
        print(f"   └─ AI 判定: {res.strip()} -> {status}")
        return is_match
    except Exception as e:
        print(f"   ⚠️ 视觉核验异常: {e}")
        return True # 出错默认放行