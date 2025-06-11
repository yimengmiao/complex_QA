# PDF 多文献嵌入检索 + GPT 问答 框架
# 依赖库：openai, faiss-cpu, PyPDF2, tiktoken

import os
import faiss
import openai
import PyPDF2
import tiktoken
from typing import List, Tuple
from openai import OpenAI

# 初始化 OpenAI 客户端（你需要设置环境变量 OPENAI_API_KEY）
client = OpenAI()

# 设置所使用的 Embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"

# 编码器用于统计 token 数
token_encoder = tiktoken.encoding_for_model("gpt-4")

def extract_text_from_pdf(pdf_path: str) -> str:
    """提取 PDF 文本内容"""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def split_text(text: str, max_tokens: int = 500) -> List[str]:
    """将文本按 token 数量切分成多个段落"""
    paragraphs = []
    current = ""
    for line in text.split("\n"):
        if not line.strip():
            continue
        test = current + " " + line
        if len(token_encoder.encode(test)) > max_tokens:
            if current:
                paragraphs.append(current.strip())
            current = line
        else:
            current = test
    if current:
        paragraphs.append(current.strip())
    return paragraphs

def get_embedding(text: str) -> List[float]:
    """获取文本的 OpenAI 向量嵌入"""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """构建 FAISS 索引并返回对应段落"""
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def search_index(index: faiss.IndexFlatL2, chunks: List[str], query: str, k: int = 3) -> List[str]:
    """在向量索引中搜索与问题最相关的段落"""
    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def load_questions(file_path: str) -> List[str]:
    """从 txt 文件中读取问题列表，一行一个问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# 示例流程（可封装为 main 函数或 Notebook 中使用）
if __name__ == "__main__":
    import numpy as np

    # 1. 提取 PDF 并切段
    pdf_path = "example.pdf"
    full_text = extract_text_from_pdf(pdf_path)
    paragraphs = split_text(full_text)
    print(f"共提取段落数：{len(paragraphs)}")

    # 2. 构建嵌入索引
    index, para_list = build_faiss_index(paragraphs)

    # 3. 加载问题
    questions = load_questions("questions.txt")

    # 4. 查询每个问题对应段落（不调用 GPT）
    for i, q in enumerate(questions):
        print(f"\n问题 {i+1}: {q}")
        related = search_index(index, para_list, q, k=3)
        for j, para in enumerate(related):
            print(f"【相关段落 {j+1}】\n{para[:500]}\n")
