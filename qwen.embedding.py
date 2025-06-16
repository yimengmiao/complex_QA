import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from collections import defaultdict
import re

# === 配置 ===
PAPER_DIR = "papers"
VECTOR_DIR = "vector"
INPUT_JSON = "multi_choice_questions.json"
OUTPUT_JSON = "deepseek_output_with_answers.json"

# === LLM 配置 ===
os.environ["DASHSCOPE_API_KEY"] = "sk-693e67f2dfeb4044a2afaeca4f226e85"
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# === 嵌入模型 ===todo:替换成openai的embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === 加载原始问题JSON ===
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

# === 将问题按 paper_id 分组 ===
questions_by_paper = defaultdict(list)
for item in questions_data:
    questions_by_paper[item["paper_id"]].append(item)

# === 主循环：处理每篇论文 ===
for paper_id, questions in questions_by_paper.items():

    # ✅ 跳过已处理的 paper_id（如 0~25 或非法ID）
    if not paper_id.isdigit() or int(paper_id) < 26:
        print(f"⏭️ 跳过已处理的 paper_id: {paper_id}")
        continue

    paper_vector_path = os.path.join(VECTOR_DIR, paper_id)

    # ✅ 若向量目录已存在，跳过（避免重复 embedding）
    if os.path.exists(paper_vector_path) and os.listdir(paper_vector_path):
        print(f"⏭️ 已存在向量索引，跳过：{paper_id}")
        continue

    try:
        paper_folder = os.path.join(PAPER_DIR, paper_id)
        pdf_file = [f for f in os.listdir(paper_folder) if f.endswith(".pdf")][0]
        pdf_path = os.path.join(paper_folder, pdf_file)
        print(f"📄 正在处理论文：{pdf_path}")
    except Exception as e:
        print(f"❌ 找不到 {paper_id} 的 PDF 文件：{e}")
        continue

    try:
        # 1. 加载并切分 PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        # ✅ 2. 清洗非法文段
        docs = [doc for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]
        if not docs:
            print(f"⚠️ 文档为空或无有效段落：{paper_id}")
            continue

        # 在这里打印出每个文档的内容，查看提取的文本
        for idx, doc in enumerate(docs):
            print(f"=== 文档 {idx + 1} ===")
            print(doc.page_content[:500])  # 打印每个文档的前500个字符
            print("\n" + "=" * 50 + "\n")

        # ✅ 3. 构建向量数据库（如果不存在）
        vectorstore = Chroma.from_documents(
            docs,
            embedding=embedding,
            persist_directory=paper_vector_path
        )
        retriever = vectorstore.as_retriever()

        # ✅ 4. 创建问答链
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        # ✅ 5. 回答每个问题
        for q in questions:
            question_text = q.get("question", "").strip()
            if not question_text:
                continue

            print(f"🔍 提问：{question_text[:80]}...")
            try:
                response = qa.invoke({"query": question_text})
                answer_text = response.get("result", "")
            except Exception as e:
                print(f"⚠️ 模型调用失败：{e}")
                answer_text = ""

            matches = re.findall(r"[A-D]", answer_text.upper())
            selected = "".join(sorted(set(matches)))
            q["correct_answer"] = selected if selected else "N/A"

            print(f"✅ 模型判断答案：{selected} | 原始回复：{answer_text.strip()}\n")

    except Exception as e:
        print(f"❌ 处理 {paper_id} 时发生错误：{e}")
        continue

# === 保存输出结果 ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(questions_data, f, indent=2, ensure_ascii=False)

print(f"\n🎉 所有处理完成，结果保存至：{OUTPUT_JSON}")
