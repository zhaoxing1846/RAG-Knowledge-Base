"""
PDF 文本提取和分块
"""
import os
import fitz  # PyMuPDF
from typing import List, Dict, Optional
import re


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """从 PDF 提取全部文本"""
    doc = fitz.open(pdf_path)
    text_parts = []
    
    pages_to_read = range(len(doc)) if max_pages is None else range(min(max_pages, len(doc)))
    
    for page_num in pages_to_read:
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    
    doc.close()
    return "\n\n".join(text_parts)


def clean_text(text: str) -> str:
    """清理文本（去除多余空白、特殊字符等）"""
    # 合并连续空白
    text = re.sub(r'\s+', ' ', text)
    # 修复被空格打断的常见模式
    text = re.sub(r'(\w)\s-\s(\w)', r'\1-\2', text)
    return text.strip()


def split_text_by_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100
) -> List[str]:
    """按固定大小分块（带重叠）"""
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk_size else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界切分
        if end < len(text):
            # 向后找句子结束符
            next_period = text.find('. ', end, end + 100)
            if next_period != -1:
                end = next_period + 1
            else:
                # 向前找句子结束符
                prev_period = text.rfind('. ', start, end)
                if prev_period > start + chunk_size // 2:
                    end = prev_period + 1
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    return chunks


def split_by_semantic_units(text: str, max_chunk_size: int = 1500) -> List[str]:
    """按语义单元分块（段落优先）"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        if current_size + para_size > max_chunk_size and current_chunk:
            # 保存当前块
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    max_pages: Optional[int] = None
) -> Dict:
    """
    处理单个 PDF 文件
    
    返回:
        {
            "filename": str,
            "source": str,
            "chunks": List[{
                "text": str,
                "chunk_id": int,
                "metadata": dict
            }]
        }
    """
    filename = os.path.basename(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path, max_pages)
    cleaned_text = clean_text(raw_text)
    
    chunks_raw = split_text_by_chunks(cleaned_text, chunk_size, overlap)
    
    chunks = []
    for i, chunk_text in enumerate(chunks_raw):
        chunks.append({
            "text": chunk_text,
            "chunk_id": i,
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks_raw)
            }
        })
    
    return {
        "filename": filename,
        "source": pdf_path,
        "total_text_length": len(cleaned_text),
        "chunks": chunks
    }


def process_pdf_directory(pdf_dir: str, **kwargs) -> List[Dict]:
    """处理目录下所有 PDF"""
    results = []
    
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing: {filename}")
            try:
                result = process_pdf(pdf_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
    
    return results


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PDF_DIR
    
    results = process_pdf_directory(PDF_DIR, max_pages=5)
    
    total_chunks = sum(len(r["chunks"]) for r in results)
    print(f"\nProcessed {len(results)} PDFs, {total_chunks} chunks total")
