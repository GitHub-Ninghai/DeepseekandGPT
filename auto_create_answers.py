import json
import openai
import time
from tqdm import tqdm  # 进度条库

# 配置DeepSeek API
openai.api_base = "https://api.deepseek.com/v1"
openai.api_key = "sk-xxx"


def get_deepseek_answer(question):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nAPI调用异常: {str(e)}")
        return None


def process_jsonl(input_file, output_file):
    # 读取所有数据
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)

    with open(output_file, "w", encoding="utf-8") as f:
        # 添加进度条
        pbar = tqdm(total=total, desc="处理进度", unit="条")

        for line in lines:
            data = json.loads(line.strip())
            question = data.get("question", "")

            # 打印当前问题
            tqdm.write(f"\n[问题] {question[:50]}...")  # 显示前50字符防刷屏
            tqdm.write("[状态] 正在请求API...")

            # 获取答案
            answer = get_deepseek_answer(question)

            if answer:
                data["deepseek_answers"].append(answer)
                tqdm.write("[结果] 获取成功！")
                tqdm.write(f"[响应] {answer[:100]}...")  # 显示前100字符预览
            else:
                tqdm.write("[警告] 获取失败！")

            # 写入数据
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            pbar.update(1)
            time.sleep(1)

        pbar.close()


# 执行处理
process_jsonl("./data/test.jsonl", "./data/test_updated.jsonl")