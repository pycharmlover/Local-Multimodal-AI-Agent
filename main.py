# main.py
import argparse
import os
import shutil
from PIL import Image

from config import PAPER_DIR, IMAGE_DIR

from modules.pdf_parser import parse_pdf
from modules.text_encoder import embed_text
from modules.image_encoder import embed_image, embed_clip_text
from modules.classifier import classify_paper
from modules.paragraph_splitter import split_paragraphs
from modules.search import (
    add_paper_embedding,
    search_paper,
    add_paragraph,
    search_paragraph,
    add_image_embedding,
    search_image,
    image_collection
)

# =====================
# 添加论文（含段落索引）
# =====================
def cmd_add_paper(args):
    pdf_path = args.path
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    if not os.path.exists(pdf_path):
        print("❌ PDF 文件不存在")
        return
    if not topics:
        print("❌ 主题列表为空")
        return

    print("[1] 解析 PDF...")
    text = parse_pdf(pdf_path)
    if len(text) < 200:
        print("❌ PDF 内容过短，解析失败")
        return

    print("[2] LLM 自动分类...")
    topic = classify_paper(text, topics)

    pid = os.path.basename(pdf_path)

    print("[3] 构建文档级向量...")
    doc_emb = embed_text(text)
    add_paper_embedding(
        pid=pid,
        embedding=doc_emb,
        metadata={
            "paper": pid,
            "topic": topic,
            "path": pdf_path
        }
    )

    print("[4] 构建段落级索引...")
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

    print("[5] 保存文件...")
    target_dir = os.path.join(PAPER_DIR, topic)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(pdf_path, os.path.join(target_dir, pid))

    print(f"\n✅ 论文添加完成")
    print(f"📄 文件名: {pid}")
    print(f"📂 分类主题: {topic}")
    print(f"📑 段落数: {len(paragraphs)}")


# =====================
# 语义搜索论文
# =====================
def cmd_search_paper(args):
    emb = embed_text(args.query)
    results = search_paper(emb, top_k=args.top_k)

    metas = results.get("metadatas", [[]])[0]
    if not metas:
        print("❌ 未找到相关论文")
        return

    print("\n=== 搜索结果 ===")
    for m in metas:
        print(f"📄 {m.get('paper')} | topic={m.get('topic')}")


# =====================
# 段落级搜索（加分项）
# =====================
def cmd_search_paragraph(args):
    emb = embed_text(args.query)
    results = search_paragraph(emb, top_k=args.top_k)

    processed = results.get("processed", [])
    if not processed:
        print("❌ 未找到相关段落")
        return

    print("\n=== 匹配段落 ===")
    for r in processed:
        print(f"\n📄 {r['paper']} | 段落 {r['para_id']}")
        print(r["text"][:300])
        print("-" * 50)


# =====================
# 添加图片
# =====================
def cmd_add_image(args):
    img_path = args.path
    if not os.path.exists(img_path):
        print("❌ 图片不存在")
        return

    img = Image.open(img_path).convert("RGB")
    emb = embed_image(img)

    fname = os.path.basename(img_path)
    save_path = os.path.join(IMAGE_DIR, fname)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    img.save(save_path)

    add_image_embedding(
        iid=fname,
        embedding=emb,
        metadata={"path": os.path.abspath(save_path)}
    )

    print(f"✅ 图片已添加: {fname}")


# =====================
# 以文搜图
# =====================
def cmd_search_image(args):
    if image_collection.count() == 0:
        print("❌ 当前没有图片索引")
        return

    q_emb = embed_clip_text(args.query)
    hits = search_image(q_emb, top_k=args.top_k)

    print("\n=== 搜索结果（distance 越小越相似）===")
    for h in hits:
        flag = "✅" if h["distance"] < args.threshold else "⚠️"
        print(f"{flag} {os.path.basename(h['path'])} | {h['distance']:.3f}")


# =====================
# CLI 参数
# =====================
def build_parser():
    parser = argparse.ArgumentParser(
        description="📚 Local Multimodal AI Assistant (CLI)"
    )
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("add_paper")
    p1.add_argument("path")
    p1.add_argument("--topics", required=True)
    p1.set_defaults(func=cmd_add_paper)

    p2 = sub.add_parser("search_paper")
    p2.add_argument("query")
    p2.add_argument("--top_k", type=int, default=5)
    p2.set_defaults(func=cmd_search_paper)

    p3 = sub.add_parser("search_paragraph")
    p3.add_argument("query")
    p3.add_argument("--top_k", type=int, default=5)
    p3.set_defaults(func=cmd_search_paragraph)

    p4 = sub.add_parser("add_image")
    p4.add_argument("path")
    p4.set_defaults(func=cmd_add_image)

    p5 = sub.add_parser("search_image")
    p5.add_argument("query")
    p5.add_argument("--top_k", type=int, default=5)
    p5.add_argument("--threshold", type=float, default=0.35)
    p5.set_defaults(func=cmd_search_image)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()