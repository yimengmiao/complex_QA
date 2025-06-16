from google import genai
import os
import re
import pymupdf4llm
from openai import OpenAI
import json
from collections import defaultdict



# 目录路径
papers_dir = "papers"
questions_path = "multi_choice_questions.json"
image_output_dir = "extracted_image_page"  # 用于存放提取的图片

# 加载问题数据
with open(questions_path, "r", encoding="utf-8") as f:
    all_questions = json.load(f)

# 根据 paper_id 分组问题
question_groups = defaultdict(list)
for q in all_questions:
    question_groups[q["paper_id"]].append(q)

# 清洗和合并逻辑
def is_header_footer_line(line: str) -> bool:
    line = line.strip()
    return (
        line.isdigit()
        or re.match(r"^\*?Proceedings of", line)
        or "Association for Computational Linguistics" in line
        or "©" in line
        or "Volume" in line
        or re.search(r"\bpages\b", line)
        or re.search(r"\bAugust\b|\bJuly\b|\bSeptember\b", line)
    )

def is_new_paragraph_start(line: str) -> bool:
    line = line.strip()
    return bool(re.match(r"^[A-Z\"“‘*]", line))

def clean_and_merge_pages(pages: list[dict]) -> str:
    merged = ""
    for i in range(len(pages)):
        lines = pages[i].get("text", "").splitlines()
        lines = [line for line in lines if not is_header_footer_line(line)]
        page_text = " ".join(lines).strip()

        if not page_text:
            continue

        if merged:
            last_char = merged.strip()[-1]
            first_word = page_text.strip().split(" ", 1)[0]
            if last_char in ",;:" or (last_char not in ".!?" and first_word and first_word[0].islower()):
                merged += " " + page_text
            else:
                merged += "\n\n" + page_text
        else:
            merged += page_text
    return merged

def remove_references(text: str) -> str:
    return re.split(r"\*\*References\*\*", text, flags=re.IGNORECASE)[0].strip()

# 处理每篇论文并回答问题
results = []
for paper_id, questions in question_groups.items():
    # 构造论文路径
    folder_path = os.path.join(papers_dir, paper_id)
    if not os.path.isdir(folder_path):
        continue

    # 查找 PDF 文件
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        continue

    pdf_path = os.path.join(folder_path, pdf_files[0])

    # 使用 pymupdf4llm 库提取并清洗论文内容（包括图片等）
    detailed_data = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=True,
        image_path=image_output_dir,
        image_format='png',
        dpi=200,
        extract_words=True,
        force_text=False
    )

    # 合并并清洗文本
    full_text = clean_and_merge_pages(detailed_data)
    clean_text = remove_references(full_text)

    # 逐个处理问题并获取模型的回答
    for question in questions:
        # 构造 DeepSeek Prompt
        user_question = """现在给你一篇论文，请你结合论文中的文字、图片、表格进行全文分析，根据提出的问题中的关键信息，锁定论文中对应的位置，对比每一个选项，得出正确的答案。其中图片由于无法直接提取成文本，只保留了图片的标识符，比如”Figure 1“这种，如果问题是针对图片内容进提出的。你就要结合提取的文字中对Figure 1的描述以及结论来回答；表格内容以”Table 1“这种作为标识，如果问题涉及到表格内容，你就需要查找到对应的表格标识，锁定回答区间。
        注意：每个问题都是多选题，答案不止一个，请你将正确答案的选项填在"answer"字段，
        以及你选择这些选项的依据，填在"reason"字段，依据必须来源于论文原文，你不选择某个选项的原因也要填在"reason"字段，以下是输出示例（标准的Json格式）：
        {
          "answer": "ACD",
          "reason": "Choice A is supported by the abstract. C is confirmed by Table 2. D matches the architecture shown in Figure 3."
        }"""

        question_text = question["question"]

        full_prompt = f"{user_question}\n\n以下是论文内容：\n\n{clean_text}\n\n以下是提出的问题：{question_text}"

        client = genai.Client(api_key="AIzaSyDm7ETIqyzpCU2L9p9Bu5qguHygnr9xE68")

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20", contents=full_prompt
        )
       # 获取模型回答并打印
        answer = response.text
        print(f"问题：{question_text}\n模型回答：{answer}\n")

        # 收集模型回答
        results.append({
            "paper_id": paper_id,
            "question": question_text,
            "deepseek_answer": answer
        })

# 保存结果到 JSON 文件
with open("gemini_choice_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("处理完成，答案已保存到 'gemini_choice_answers.json' 文件中。")