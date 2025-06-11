# 重新定义必要的函数和变量
from collections import defaultdict
from difflib import SequenceMatcher
import json

with open("multi_choice_questions.json", "r", encoding="utf-8") as f1:
    data1 = json.load(f1)
with open("updated_questions_with_answers.json", "r", encoding="utf-8") as f2:
    data2 = json.load(f2)


# 分组函数
def group_by_paper_id(data):
    grouped = defaultdict(list)
    for item in data:
        grouped[item["paper_id"]].append(item)
    return grouped

# 字符串相似度匹配
def is_similar(q1, q2, threshold=0.7):
    return SequenceMatcher(None, q1, q2).ratio() >= threshold

# 分组
grouped1 = group_by_paper_id(data1)
grouped2 = group_by_paper_id(data2)

# 匹配逻辑
unmatched_questions = []

for paper_id in grouped1:
    questions1 = grouped1[paper_id]
    questions2 = grouped2.get(paper_id, [])

    if len(questions1) == len(questions2):
        for i in range(len(questions1)):
            questions1[i]['correct_answer'] = questions2[i].get('correct_answer', "")
    else:
        used = set()
        for q1 in questions1:
            matched = False
            for i, q2 in enumerate(questions2):
                if i in used:
                    continue
                if is_similar(q1['question'], q2['question']):
                    q1['correct_answer'] = q2.get('correct_answer', "")
                    used.add(i)
                    matched = True
                    break
            if not matched:
                unmatched_questions.append({
                    "paper_id": paper_id,
                    "question": q1['question']
                })

# 写入合并结果文件
output_path = "merged_output.json"
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(data1, fout, indent=2, ensure_ascii=False)

# 返回部分未匹配内容及文件路径
unmatched_questions[:5], len(unmatched_questions), output_path
