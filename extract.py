import pymupdf4llm

file_path = "papers/0/2024.acl-long.820.pdf"
image_output_dir = "extracted_image_page"


# detailed_data = pymupdf4llm.to_markdown(
#     file_path,
#     page_chunks=True,         # 分页处理
#     write_images=True,        # 提取并保存图片
#     image_path=image_output_dir,
#     image_format='png',
#     dpi=200,
#     extract_words=True,       # 更细粒度的文字信息
#     force_text=False          # 优先结构提取
# )
#
# print(detailed_data)

import fitz  # PyMuPDF

def extract_page_images(file_path, image_output_dir, dpi=200):
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        pix.save(f"{image_output_dir}/page_{page_number+1}.png")

extract_page_images(file_path, image_output_dir)

import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-3c9d7d2012784b78847a8f8230972e86",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[{"role": "user","content": [
            {"type": "image_url",
             "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "text", "text": "这是什么"},
            ]}]
    )
print(completion.model_dump_json())
