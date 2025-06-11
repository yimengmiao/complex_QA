import json

import pandas as pd
from openai import AzureOpenAI, OpenAI

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="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",
    base_url="https://api.openai.com/v1"  # 如果使用自定义 URL
)
#创建prompt工程
prompt = """
"""

#读取文件
# 加载 JSON 文件和pdf文件
file1_path = "sorted_multi_choice_questions.json"
file2_path = "analysis_answers_formatted.json"

with open(file1_path, "r", encoding="utf-8") as f1, open(file2_path, "r", encoding="utf-8") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)  # 假设已排好序
for i in range(len(df)):
    try:
        response = client.chat.completions.create(
            model="soikit_test",  # model = "deployment_name".
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": df.loc[i, 'text']},
            ]
        )
        print("输出：", response.choices[0].message.content)
        df.loc[i, 'gpt4o_output'] = response.choices[0].message.content

    except Exception as e:
        print("error", e)
        df.loc[i, 'gpt4o_output'] = "error"

df.to_excel("../LLM_train/LLM/data/2切割后的数据/combined.xlsx", index=False)






response = client.chat.completions.create(
            model="soikit_test",  # model = "deployment_name".
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(df['text'].to_list()[0:4])},
                {"role": "user", "content": str(df['text'].to_list()[4:8])},
                {"role": "user", "content": str(df['text'].to_list()[8:12])},

            ]
        )



os.environ["DASHSCOPE_API_KEY"] = "sk-693e67f2dfeb4044a2afaeca4f226e85"





