import json

# 加载两个文件的内容
with open('processed_answers.json', 'r', encoding='utf-8') as f:
    list1 = json.load(f)

with open('analysis_answers_formatted_4o_reordered.json', 'r', encoding='utf-8') as f:
    list2 = json.load(f)

# 确保两个列表长度一致
assert len(list1) == len(list2), "两个文件的问题数量不一致"

# 获取原始问题数量
original_count = len(list1)
print(f"原始文件中共有 {original_count} 个问题。")

# 初始化结果列表和匹配计数器
result = []
matched_count = 0

# 遍历所有问题
for item1, item2 in zip(list1, list2):
    new_entry = {
        "paper_id": item1["paper_id"],
        "question": item1["question"],
        "correct_answer": ""
    }

    if item1["correct_answer"] == item2["correct_answer"]:
        new_entry["correct_answer"] = item1["correct_answer"]
        matched_count += 1

    result.append(new_entry)

# 保存结果为 JSON 文件
with open('gpt_deep_matched_answers.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# 打印匹配数量
print(f"筛选后共有 {matched_count} 个问题的 correct_answer 一致。")
print("匹配结果已保存至 gpt_deep_matched_answers.json 文件。")
