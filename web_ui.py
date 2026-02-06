# web_ui.py
import os
import re
import shutil
import gradio as gr
from PIL import Image
import modules.llm as llm

from modules.pdf_parser import parse_pdf
from modules.text_encoder import embed_text
from modules.image_encoder import embed_image
from modules.classifier import classify_paper
from config import PAPER_DIR, IMAGE_DIR, DEEPSEEK_API_KEY # ⭐ 导入 Key
from modules.time_parser import TimeParser # ⭐ 导入你刚才新建的解析器

tp = TimeParser(api_key=DEEPSEEK_API_KEY)

from modules.search import (
    add_paper_embedding,
    add_paragraph,
    search_paper,
    search_paragraph,
    add_image_embedding,
    search_image_visual,
    search_image_text,
    image_visual_col,  # 视觉库 (CLIP)
    image_text_col     # 文本库 (BGE)
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
# 段落级检索
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
# 图像管理
# =====================
def ui_add_image(img_path):
    if not img_path: return "❌ 请上传图片"
    try:
        fname = os.path.basename(img_path)
        save_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(save_path):
            shutil.copy(img_path, save_path)
        
        # 内部会同时往两个库里存向量
        desc = add_image_embedding(iid=fname, image_path=os.path.abspath(save_path))
        return f"✅ 已添加并完成双轨索引：{fname}\n📝 AI描述: {desc}"
    except Exception as e:
        return f"❌ 添加失败: {str(e)}"
    
from batch_add_images import batch_add_images # ⭐ 确保导入了刚才那个脚本

def ui_batch_sync():
    """UI 调用的批量同步函数"""
    try:
        # 调用你刚才定义的逻辑
        stats = batch_add_images()
        return f"✅ 批量同步任务结束！\n{stats}"
    except Exception as e:
        return f"❌ 批量同步出错: {str(e)}"

# =====================
# 文字搜图
# =====================

def ui_search_image(query, threshold, top_k, use_rerank):
    if not query.strip(): return "❌ 请输入搜索词", []
    if image_text_col.count() == 0: return "❌ 数据库为空", []

    # 1. 解析时间范围 (DeepSeek 解析)
    time_range = tp.extract_time_constraints(query)
    start, end = time_range.get("start_date"), time_range.get("end_date")
    
    # 2. 语义剥离 (核弹级清洗：剥离时间噪声，提纯主体)
    clean_query = query
    # [1] 拆除年份、月日、特定连词
    clean_query = re.sub(r'(19[7-9]\d|20[0-2][0-7])年?', '', clean_query)
    clean_query = re.sub(r'\d{1,2}月(\d{1,2}[日号]?)?', '', clean_query)
    clean_query = re.sub(r'(年的|月的|那张|那些|的照片|的图片|那只|那条|里的)', '', clean_query)
    # [2] 拆除季节和相对词
    noise_words = ["春天", "夏天", "秋天", "冬天", "去年", "前年", "今年", "以前", "以后", "最近"]
    for word in noise_words:
        clean_query = clean_query.replace(word, "")
    # [3] 最终修整
    clean_query = clean_query.replace("的", "").strip()

    # ⭐ 定义关键变量，防止后续报错
    is_pure_time_query = (len(clean_query) == 0)
    # ⭐ 补上 keywords：将洗干净的语义词拆成列表用于奖励机制
    keywords = [k for k in re.split(r'[,，\s]+', clean_query) if len(k) > 0]

    # 3. 向量检索 (海选阶段：拿 20 条，为精排留空间)
    search_prompt = f"为查询编写检索描述：{clean_query}" if not is_pure_time_query else "照片"
    q_emb = embed_text(search_prompt) 
    hits = search_image_text(q_emb, top_k=20, date_filter=time_range)
    
    # --- 阶段一：特征注意力增强 (打折逻辑) ---
    processed_hits = []
    for h in hits:
        d = h["distance"]
        desc = h["description"]
        
        # 计算关键词命中次数
        match_count = sum(1 for kw in keywords if kw in desc)
        is_boosted = False
        if not is_pure_time_query and match_count > 0:
            # 命中即打折，最高打 5 折
            bonus = min(0.5, 0.3 + (match_count - 1) * 0.1)
            d = d * (1.0 - bonus)
            is_boosted = True
        
        h["final_dist"] = d  # 记录计算后的距离
        h["is_boosted"] = is_boosted
        
        # 初始距离过滤 (根据阈值拦截)
        if not is_pure_time_query and d > threshold:
            continue
        processed_hits.append(h)

    # 按增强后的距离重新排队
    processed_hits = sorted(processed_hits, key=lambda x: x["final_dist"])

    # --- 阶段二：级联精排 (Cascaded Reranking) ---
    final_hits = []
    if use_rerank and not is_pure_time_query and processed_hits:
        # 🚀 3.1 文本初筛 (DeepSeek 处理描述元数据)
        candidates = [{"id": i, "description": h["description"]} for i, h in enumerate(processed_hits)]
        refined_list = llm.filter_by_text(query, candidates)
        
        # 🚀 3.2 视觉终审 (Qwen-VL 像素级复核)
        for cand in refined_list:
            hit_data = processed_hits[cand["id"]]
            if llm.verify_image_content(hit_data["path"], query):
                final_hits.append(hit_data)
    else:
        # 未开启精排，取 Top-K 结果
        final_hits = processed_hits[:int(top_k)]

    # 4. 结果包装展示 (前端 Gradio 渲染)
    imgs, info = [], []
    info.append(f"🔍 语义词: '{clean_query or '空'}' | 📅 约束: {start or '不限'} ➜ {end or '不限'}\n")

    for h in final_hits:
        if os.path.exists(h["path"]):
            imgs.append(Image.open(h["path"]))
            d = h["final_dist"]
            # 纯时间查询匹配度默认 100%，否则根据 dist 换算
            similarity = 100.0 if is_pure_time_query else max(0, (2 - d) / 2 * 100)
            
            raw_date = str(h.get("date", "未知"))
            display_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date)==8 else raw_date
            boost_tag = "🚀 [已增强]" if h["is_boosted"] else ""
            
            info.append(f"🎯 dist: {d:.3f} {boost_tag} | 匹配度: {similarity:.1f}% | 📅 {display_date}\n📝 {h['description']}\n")
            
    return "\n".join(info), imgs

# =====================
# 以图搜图
# =====================

def ui_image_to_image(input_img, threshold, top_k): # ⭐ 确认这里也是 3 个参数
    if input_img is None: return "❌ 未上传", []
    if image_visual_col.count() == 0: return "❌ 数据库为空", []
    
    try:
        img = Image.open(input_img).convert("RGB") if isinstance(input_img, str) else input_img
        q_emb = embed_image(img)
        hits = search_image_visual(q_emb, top_k=int(top_k), date_filter=None)
        
        info = [f"🔍 视觉检索结果 (Top-{top_k}):"]
        imgs = []
        for h in hits:
            if os.path.exists(h['path']):
                imgs.append(Image.open(h['path']))
                info.append(f"{'✅' if h['distance']<threshold else '⚠️'} {os.path.basename(h['path'])} | d={h['distance']:.3f}")
        return "\n".join(info), imgs
    except Exception as e: return f"❌ 失败: {e}", []
    
# =====================
# Gradio UI 布局
# =====================
with gr.Blocks(title="本地多模态 AI 助手") as demo:
    gr.Markdown("# 📚 本地多模态 AI 智能助手 (本地版)")

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
        with gr.Row():
            img_in = gr.File(label="添加新图")
            btn_add = gr.Button("开始索引")
        out_add = gr.Textbox(label="状态")
        btn_add.click(ui_add_image, img_in, out_add)
    
    with gr.Tab("📁 批量管理"):
        gr.Markdown("### 🛠️ 本地文件夹同步\n点击下方按钮，自动扫描 `IMAGE_DIR` 目录下的新照片。")
        batch_btn = gr.Button("🔄 立即同步本地文件夹", variant="secondary")
        batch_output = gr.Textbox(label="同步结果", lines=3)
        
        # 绑定点击事件
        batch_btn.click(ui_batch_sync, inputs=[], outputs=[batch_output])

    with gr.Tab("🧠 以文搜图"):
        with gr.Row():
            q_txt = gr.Textbox(label="输入描述词", placeholder="例如：去年在海边拍的照片")
        
        with gr.Row():
            t_txt = gr.Slider(0.1, 1.5, value=1.0, label="语义阈值 (dist越小越准)")
            k_txt = gr.Slider(1, 20, value=8, step=1, label="展示数量 (Top-K)")
            # ⭐ 1. 新增这个复选框
            rerank_ch = gr.Checkbox(label="启用Rerank", value=False) 

        btn_txt = gr.Button("🔍 开始搜索", variant="primary")
        info_txt = gr.Textbox(label="结果详情")
        gal_txt = gr.Gallery(columns=4, label="匹配照片")

        btn_txt.click(
            ui_search_image, 
            inputs=[q_txt, t_txt, k_txt, rerank_ch], 
            outputs=[info_txt, gal_txt]
        )

    with gr.Tab("📸 以图搜图"):
        q_img = gr.Image(label="源图", type="filepath")
        with gr.Row():
            t_img = gr.Slider(0.1, 1.0, value=0.35, label="相似度阈值")
            k_img = gr.Slider(1, 20, value=5, step=1, label="展示数量") # ⭐ 这是另一个滑块
        btn_img = gr.Button("🖼️ 执行匹配")
        info_img = gr.Textbox(label="匹配详情")
        gal_img = gr.Gallery(columns=4)
        # ⭐ 关键修正：确保 inputs 里的参数是 3 个
        btn_img.click(ui_image_to_image, [q_img, t_img, k_img], [info_img, gal_img])

demo.launch(server_name="0.0.0.0", server_port=7860)