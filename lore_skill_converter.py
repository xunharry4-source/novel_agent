import json
import os
from datetime import datetime

def generate_modular_skills(current_chapter_idx=1):
    db_path = "outlines_db.json"
    if not os.path.exists(db_path):
        print("Error: outlines_db.json not found.")
        return

    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    book_outlines = [item for item in db if item.get('mode') == 'book']
    if not book_outlines:
        print("Error: No book-level outlines found.")
        return
    
    latest_book = book_outlines[-1]
    outline = latest_book.get('outline', {})
    meta = outline.get('meta_info', {})
    beats = outline.get('plot_beats', {})
    catalog = outline.get('chapter_list', [])

    # 1. 生成 剧情锚点 (lore/ANCHORS.md)
    anchors_content = f"""# SKILL: {meta.get('title', 'Unknown Project')} - 关键锚点 (ANCHORS)
## 权限: 宪法级 (V0)
- **核心节奏**: {meta.get('writing_style', 'N/A')}
- **关键转折**: {beats.get('midpoint', 'N/A')}
- **最终结局**: {beats.get('resolution', 'N/A')}
"""

    # 2. 生成 活跃窗口 (catalog/ACTIVE_WINDOW.md)
    # 取当前章节前后各 5 章
    window_start = max(1, current_chapter_idx - 5)
    window_end = current_chapter_idx + 5
    active_chapters = [ch for ch in catalog if window_start <= ch.get('chapter_num', 0) <= window_end]
    
    active_content = f"""# SKILL: 执行窗口 (ACTIVE_WINDOW)
## 当前任务: 第 {current_chapter_idx} 章
---
"""
    for ch in active_chapters:
        status = "【进行中】" if ch.get('chapter_num') == current_chapter_idx else ""
        active_content += f"- 第 {ch.get('chapter_num')} 章 ({ch.get('title')}): {ch.get('summary')} {status}\n"

    # 3. 生成 归档切片 (catalog/ARCHIVE/CH_XXX_XXX.md)
    os.makedirs(".gemini/skills/catalog/ARCHIVE", exist_ok=True)
    chunk_size = 50
    for i in range(0, len(catalog), chunk_size):
        chunk = catalog[i:i + chunk_size]
        start_num = chunk[0].get('chapter_num')
        end_num = chunk[-1].get('chapter_num')
        archive_name = f"CH_{str(start_num).zfill(3)}_{str(end_num).zfill(3)}.md"
        
        archive_content = f"# ARCHIVE: Chapter {start_num} - {end_num}\n\n"
        for ch in chunk:
            archive_content += f"### 第 {ch.get('chapter_num')} 章: {ch.get('title')}\n- 梗概: {ch.get('summary')}\n- 重点: {ch.get('focus')}\n\n"
        
        with open(f".gemini/skills/catalog/ARCHIVE/{archive_name}", "w", encoding="utf-8") as f:
            f.write(archive_content)

    # 4. 生成 主索引 (catalog/MASTER_INDEX.md)
    index_content = f"# MASTER INDEX: {meta.get('title', 'Unknown')}\n\n"
    index_content += "## 物理分片检索\n"
    archive_files = os.listdir(".gemini/skills/catalog/ARCHIVE")
    for f_name in sorted(archive_files):
        index_content += f"- [{f_name}](file://.gemini/skills/catalog/ARCHIVE/{f_name})\n"
    
    # 写入文件
    os.makedirs(".gemini/skills/lore", exist_ok=True)
    os.makedirs(".gemini/skills/framework", exist_ok=True)
    
    with open(".gemini/skills/lore/ANCHORS.md", "w", encoding="utf-8") as f:
        f.write(anchors_content)
    with open(".gemini/skills/catalog/ACTIVE_WINDOW.md", "w", encoding="utf-8") as f:
        f.write(active_content)
    with open(".gemini/skills/catalog/MASTER_INDEX.md", "w", encoding="utf-8") as f:
        f.write(index_content)
    
    print(f"Success: Slicing completed for {meta.get('title')}. Active window set to Chapter {current_chapter_idx}.")

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate_modular_skills(idx)
