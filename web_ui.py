# web_ui.py
import os
import shutil
import gradio as gr
from PIL import Image

from modules.pdf_parser import parse_pdf
from modules.text_encoder import embed_text
from modules.image_encoder import embed_image
from modules.classifier import classify_paper
from modules.image_encoder import embed_clip_text
from config import PAPER_DIR, IMAGE_DIR

from modules.search import (
    add_paper_embedding,
    add_paragraph,
    search_paper,
    search_paragraph,
    add_image_embedding,
    search_image,
    image_collection
)
from modules.paragraph_splitter import split_paragraphs

# =====================
# 全局配置
# =====================
# 使用 config.py 中的配置，确保路径一致
PAPER_ROOT = PAPER_DIR
IMAGE_ROOT = IMAGE_DIR

os.makedirs(PAPER_ROOT, exist_ok=True)
os.makedirs(IMAGE_ROOT, exist_ok=True)

# =====================
# 论文添加（含段落索引）
# =====================
def ui_add_paper(pdf_path, topics):
    if not pdf_path:
        return "❌ 请上传 PDF 文件"

    topics = [t.strip() for t in topics.split(',') if t.strip()]
    if not topics:
        return "❌ 请提供候选主题"

    try:
        text = parse_pdf(pdf_path)
        if len(text) < 200:
            return "❌ PDF 解析失败或内容过短"
    except Exception as e:
        return f"❌ PDF 解析失败: {str(e)}"

    # LLM 分类
    try:
        topic = classify_paper(text, topics)
    except Exception as e:
        return f"❌ 分类失败: {str(e)}"

    pid = os.path.basename(pdf_path)

    # 文档级 embedding
    try:
        doc_emb = embed_text(text)
        add_paper_embedding(
            pid=pid,
            embedding=doc_emb,
            metadata={
                "paper": pid,
                "topic": topic,
                "path": pdf_path  # 添加路径信息
            }
        )
    except Exception as e:
        return f"❌ 添加论文向量失败: {str(e)}"

    # 段落索引
    try:
        paragraphs = split_paragraphs(text)
        for i, para in enumerate(paragraphs):
            p_emb = embed_text(para)
            add_paragraph(
                pid=pid,
                para_id=i,
                embedding=p_emb,
                metadata={
                    "paper": pid,
                    "topic": topic,
                    "text": para
                }
            )
    except Exception as e:
        return f"❌ 段落索引失败: {str(e)}"

    # ===== 文件保存到主题目录 =====
    try:
        target_dir = os.path.join(PAPER_ROOT, topic)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, pid)
        
        # 检查目标文件是否已存在
        if os.path.exists(target_path):
            # 如果文件已存在，可以选择覆盖或跳过
            # 这里选择覆盖
            pass
        
        # 复制文件到目标目录
        shutil.copy(pdf_path, target_path)
    except Exception as e:
        return f"❌ 文件保存失败: {str(e)}\n✅ 向量已添加，但文件保存失败"

    return f"✅ 已添加论文：{pid}\n📂 分类主题：{topic}\n📑 段落数：{len(paragraphs)}\n💾 保存路径：{target_path}"

# =====================
# 论文语义搜索
# =====================
def ui_search_paper(query):
    emb = embed_text(query)
    results = search_paper(emb)

    if not results or not results.get("metadatas") or not results["metadatas"]:
        return "❌ 未找到相关论文"
    
    # 从 metadatas 中提取信息
    output = []
    for meta in results["metadatas"][0]:
        paper = meta.get("paper") or meta.get("path", "unknown")
        topic = meta.get("topic", "unknown")
        output.append(f"📄 {paper}  (topic={topic})")
    
    return "\n".join(output)

# =====================
# 段落级检索（加分项）
# =====================
def ui_search_paragraph(query):
    emb = embed_text(query)
    results = search_paragraph(emb)

    if not results or not results.get("metadatas") or not results["metadatas"]:
        return "❌ 未找到相关段落"
    
    # 使用处理后的结果
    if results.get("processed"):
        processed = results["processed"]
    else:
        # 如果没有处理后的结果，从 metadatas 中提取
        processed = []
        for i, meta in enumerate(results["metadatas"][0]):
            para_id = 0
            if results.get("ids") and results["ids"][0]:
                id_str = results["ids"][0][i]
                if "_" in id_str:
                    try:
                        para_id = int(id_str.split("_")[-1])
                    except:
                        para_id = i
            processed.append({
                "paper": meta.get("paper", "unknown"),
                "para_id": para_id,
                "text": meta.get("text", "")
            })

    out = []
    for r in processed:
        out.append(
            f"📄 {r['paper']} | 段落 {r['para_id']}\n{r['text'][:300]}...\n"
        )
    return "\n".join(out)

# =====================
# 添加图片
# =====================
def ui_add_image(img_path):
    if not img_path:
        return "❌ 请上传图片"

    try:
        img = Image.open(img_path).convert("RGB")
        emb = embed_image(img)

        fname = os.path.basename(img_path)
        save_path = os.path.join(IMAGE_ROOT, fname)
        
        # 如果文件不在目标目录，则保存
        if not os.path.exists(save_path):
            img.save(save_path)
        
        # 使用绝对路径
        abs_path = os.path.abspath(save_path)

        add_image_embedding(
            iid=fname,
            embedding=emb,
            metadata={"path": abs_path}
        )

        return f"✅ 已添加图片：{fname}\n💾 路径：{abs_path}"
    except Exception as e:
        return f"❌ 添加图片失败: {str(e)}"

# =====================
# 以文搜图（阈值过滤）
# =====================
def ui_search_image(query, threshold):
    if not query or not query.strip():
        return "❌ 请输入搜索内容", []

    count = image_collection.count()
    if count == 0:
        return "❌ 数据库中没有图片，请先添加图片", []

    q_emb = embed_clip_text(query)

    # hits = search_image(q_emb, top_k=min(20, count))
    hits = search_image(q_emb, top_k=5)

    imgs = []
    info = ["🔍 Top 结果（distance 越小越相似）：\n"]

    for h in hits:
        if not os.path.exists(h["path"]):
            continue
        imgs.append(Image.open(h["path"]))

        flag = "✅" if h["distance"] < threshold else "⚠️"
        info.append(
            f"{flag} {os.path.basename(h['path'])} | distance={h['distance']:.3f}"
        )

    if not imgs:
        return "⚠️ 查询成功，但图片无法加载", []

    return "\n".join(info), imgs



# =====================
# Gradio UI
# =====================
with gr.Blocks(title="本地多模态 AI 助手") as demo:
    gr.Markdown("# 📚 本地多模态 AI 智能助手")

    with gr.Tab("📄 添加论文"):
        pdf = gr.File(file_types=[".pdf"], label="上传 PDF")
        topics = gr.Textbox(label="候选主题（逗号分隔）", value="CV,NLP,RL")
        btn = gr.Button("添加论文")
        out = gr.Textbox(lines=4, max_lines=15)
        btn.click(ui_add_paper, [pdf, topics], out)

    with gr.Tab("🔍 论文搜索"):
        q = gr.Textbox(label="查询")
        btn2 = gr.Button("搜索")
        out2 = gr.Textbox(lines=8, max_lines=30)
        btn2.click(ui_search_paper, q, out2)

    with gr.Tab("🧩 段落检索"):
        q3 = gr.Textbox(label="查询段落")
        btn3 = gr.Button("搜索")
        out3 = gr.Textbox(lines=12, max_lines=30)
        btn3.click(ui_search_paragraph, q3, out3)

    with gr.Tab("🖼️ 添加图片"):
        img = gr.File(file_types=["image"], label="上传图片")
        btn4 = gr.Button("添加图片")
        out4 = gr.Textbox()
        btn4.click(ui_add_image, img, out4)

    with gr.Tab("🧠 以文搜图"):
        q4 = gr.Textbox(label="图像描述", lines=10)
        threshold = gr.Slider(0.1, 0.6, value=0.35, step=0.01, label="相似度阈值（越小越严格）")
        btn5 = gr.Button("搜索")
        info = gr.Textbox(label="匹配结果", lines=12, max_lines=30)
        gallery = gr.Gallery(columns=4, height=300)
        btn5.click(ui_search_image, [q4, threshold], [info, gallery])


demo.launch(server_name="0.0.0.0", server_port=7860)
