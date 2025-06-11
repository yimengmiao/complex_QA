import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="sk-3c9d7d2012784b78847a8f8230972e86",  # 如果您没有配置环境变量，请在此处替换您的API-KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务base_url
)

# 上传文件，拿到 file_id
file_object = client.files.create(
    file=Path("papers/0/2024.acl-long.820.pdf"),
    purpose="file-extract"
)
file_id = file_object.id  # 比如 "file-fe-a0ac46b0a4df411eb4fe31e7"
print("上传成功，file_id =", file_id)

# 在 chat completion 里引用真正的 file_id
completion = client.chat.completions.create(
    model="qwen-long",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": f"fileid://{file_id}"},  # 注意这里
        {"role": "user",   "content": "这篇文章讲了什么？总结100字以内"}
    ],
    stream=False
)
# 提取并打印输出
response_text = completion.choices[0].message.content
print("模型输出：")
print(response_text)
