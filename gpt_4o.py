
#创建prompt工程
prompt = """
"""

#读取文件
# 加载 JSON 文件和pdf文件
file1_path = "multi_choice_questions.json"
file2_path = "gpto3_answers_formatted.json"

import openai

# 设置你的API密钥
openai.api_key = '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'

# 调用 OpenAI 的 ChatGPT 模型
response = openai.Completion.create(
  model="gpt-4",  # 你可以选择 gpt-3.5 或 gpt-4，取决于你要使用的版本
  prompt="你好，ChatGPT！今天怎么样？",  # 输入你的问题或提示
  max_tokens=150  # 限制生成的响应的最大长度
)

# 输出返回的内容
print(response.choices[0].text.strip())










