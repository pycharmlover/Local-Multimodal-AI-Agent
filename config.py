# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PAPER_DIR = os.path.join(DATA_DIR, "papers")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# ===== 本地模型路径 =====
LLM_PATH = "/home/extra_home/lc/LLM/Qwen2.5-14B-Instruct"
TEXT_EMB_PATH = "/home/extra_home/lc/LLM/bge-large-zh-v1.5"
CLIP_PATH = "/home/extra_home/lc/LLM/CLIP_ViT-L_14/0_CLIPModel"
