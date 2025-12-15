#!/usr/bin/env python3
"""
批量添加图片到向量数据库
将 data/images/ 目录中的所有图片批量添加到 ChromaDB 向量数据库
"""
from pathlib import Path
from PIL import Image
from modules.image_encoder import embed_image
from modules.search import add_image_embedding
from config import IMAGE_DIR

def batch_add_images():
    """批量添加图片到向量数据库"""
    image_dir = Path(IMAGE_DIR)
    
    if not image_dir.exists():
        print(f"❌ 图片目录不存在: {IMAGE_DIR}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF', '.WEBP'}
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
    
    # 去重
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"❌ 在 {IMAGE_DIR} 中未找到图片文件")
        return
    
    print(f"📁 找到 {len(image_files)} 张图片")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for img_path in image_files:
        try:
            fname = img_path.name
            abs_path = str(img_path.absolute())
            
            print(f"[*] 处理: {fname}...", end=" ", flush=True)
            
            # 检查文件是否存在且可读
            if not img_path.exists():
                print("❌ 文件不存在")
                fail_count += 1
                continue
            
            # 打开并转换图片
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"❌ 图片打开失败: {str(e)}")
                fail_count += 1
                continue
            
            # 生成 embedding
            try:
                emb = embed_image(img)
            except Exception as e:
                print(f"❌ Embedding 生成失败: {str(e)}")
                fail_count += 1
                continue
            
            # 添加到数据库
            try:
                add_image_embedding(
                    iid=fname,
                    embedding=emb,
                    metadata={"path": abs_path}
                )
                print("✅")
                success_count += 1
            except Exception as e:
                # 如果是因为 ID 已存在，可以选择更新或跳过
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print("⏭️  已存在（跳过）")
                    skip_count += 1
                else:
                    print(f"❌ 数据库添加失败: {str(e)}")
                    fail_count += 1
                    
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            fail_count += 1
    
    print("=" * 60)
    print(f"✅ 成功添加: {success_count} 张")
    if skip_count > 0:
        print(f"⏭️  跳过（已存在）: {skip_count} 张")
    if fail_count > 0:
        print(f"❌ 失败: {fail_count} 张")
    print(f"📊 总计: {len(image_files)} 张")

if __name__ == "__main__":
    print("🚀 开始批量添加图片到向量数据库...")
    print(f"📂 图片目录: {IMAGE_DIR}")
    print()
    batch_add_images()
    print("\n✨ 批量添加完成！")

