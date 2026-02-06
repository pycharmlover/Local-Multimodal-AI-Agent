# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PAPER_DIR = os.path.join(DATA_DIR, "papers")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# ===== 本地模型路径 =====
LLM_PATH = r"E:\LLM_Models\Qwen2.5-7B-Instruct"
TEXT_EMB_PATH = r"E:\LLM_Models\bge-large-zh-v1.5"
CLIP_PATH = r"E:\LLM_Models\clip-vit-l-14"

# config.py 增加以下内容

# 模式开关：True 使用 API，False 使用本地模型
USE_API = True 

# DeepSeek 配置
DEEPSEEK_API_KEY = "sk-55784a9c73b44a369923e77539bbd2fb"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat" 

# ===== 新增：通义千问视觉版配置 =====
QWEN_API_KEY = "sk-ecdb0914f8cb4e66ba5b5d532c52b0f1" 
QWEN_MODEL_NAME = "qwen3-vl-plus"