# import os
# import json
# import fitz  # PyMuPDF
# from tqdm import tqdm
# from openai import OpenAI
#
# # 初始化 DeepSeek 客户端
# client = OpenAI(
#     api_key="sk-8484a8cf027442d480c5cb375a15cdee",
#     base_url="https://api.deepseek.com"
# )
#
# # 路径配置
# papers_dir = "papers"
# questions_path = "multi_choice_questions.json"
#
# # 1. 加载题目
# with open(questions_path, "r", encoding="utf-8") as f:
#     all_questions = json.load(f)
#
# # 2. 将题目按 paper_id 分组
# from collections import defaultdict
# question_groups = defaultdict(list)
# for q in all_questions:
#     question_groups[q["paper_id"]].append(q)
#
# # 3. PDF转文本工具
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text.strip()
#
# # 4. 构造 prompt + 调用 DeepSeek
# def ask_deepseek(paper_text, question_text):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Read the paper and answer the multiple choice question by choosing the correct option(s)."},
#         {"role": "user", "content": f"Paper content:\n{paper_text[:8000]}\n\nQuestion:\n{question_text}"}
#     ]
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages,
#         stream=False
#     )
#     return response.choices[0].message.content.strip()
#
# # 5. 主处理流程
# results = []
# for paper_id in tqdm(question_groups.keys(), desc="Processing papers"):
#     folder_path = os.path.join(papers_dir, paper_id)
#     if not os.path.isdir(folder_path):
#         continue
#
#     # 查找PDF文件
#     pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
#     if not pdf_files:
#         continue
#
#     pdf_path = os.path.join(folder_path, pdf_files[0])
#     paper_text = extract_text_from_pdf(pdf_path)
#
#     for q in question_groups[paper_id]:
#         answer = ask_deepseek(paper_text, q["question"])
#         results.append({
#             "paper_id": paper_id,
#             "question": q["question"],
#             "deepseek_answer": answer
#         })
#
# # 6. 保存结果
# with open("multi_choice_answers.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)
import os

from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

# 加载 PDF 内容
pdf_path = "papers/0/2024.acl-long.820.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
paper_text = "\n".join([page.page_content for page in pages])

# 设置 DeepSeek API 客户端
client = OpenAI(
    api_key="sk-8484a8cf027442d480c5cb375a15cdee",
    base_url="https://api.deepseek.com"
)

# 构造请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": """You are a highly capable research paper analyzer powered by advanced visual and language understanding.

You will be provided with a research paper in PDF format. Your task is to thoroughly analyze its contents, including:
- all **textual content** (titles, paragraphs, captions),
- **images** (diagrams, charts, illustrations),
- and **tables** (data, structures, formulas).

You will then be asked to evaluate whether specific **definitions or statements** are correct **based on the paper**.

For each question:
1. There are **at least two correct choices**, and you must identify **all valid options** from A, B, C, D, etc.
2. If you are not immediately certain from the text, you **must actively search for visual or tabular evidence** in the document to support your judgment.
3. Be skeptical and evidence-driven: **assume definitions are incorrect unless you find explicit or visual support**.
4. Always explain your reasoning based on **specific references to the paper's content** (e.g., "Figure 2 shows...", "Table 1 indicates...").

Respond in the following strict JSON format:
```json
{
  "answer": "ACD",
  "reason": "Choice A is supported by the abstract. C is confirmed by Table 2. D matches the architecture shown in Figure 3."
}"""
        },
        {
            "role": "user",
            "content": paper_text + "\n\nWhich of the following factors may cause multilingual large language models to show English bias when processing non-English languages?\nA. The model's training data mainly consists of English text.\nB. The model uses English as the central language in the middle layer for semantic understanding and reasoning.\nC. In the model's word embedding space, English word embeddings are more densely distributed and easier to be \"captured\" by the model.\nD. The model translates non-English text into English before translating it into the target language."
        }
    ],
    stream=False
)

print(response.choices[0].message.content)
