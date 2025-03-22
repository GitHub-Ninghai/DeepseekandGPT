import os
from tqdm import tqdm
import openai
from evaluators.evaluator import Evaluator
from time import sleep
import re

name_en2zh = {'Middle_School_Chemistry': '初中化学', 'Middle_School_History': '初中历史',
              'Middle_School_Geography': '初中地理', 'Middle_School_Politics': '初中政治',
              'Middle_School_Mathematics': '初中数学', 'Middle_School_Physics': '初中物理',
              'Middle_School_Biology': '初中生物', 'Middle_School_English': '初中英语',
              'Middle_School_Chinese': '初中语文', 'Primary_School_Mathematics': '小学小学数学',
              'Primary_School_Science': '小学小学科学', 'Primary_School_English': '小学小学英语',
              'Primary_School_Chinese': '小学小学语文', 'Primary_School_Ethics': '小学道德与法治',
              'High_School_Chemistry': '高中化学', 'High_School_History': '高中历史',
              'High_School_Geography': '高中地理', 'High_School_Politics': '高中政治',
              'High_School_Mathematics': '高中数学', 'High_School_Physics': '高中物理',
              'High_School_Biology': '高中生物', 'High_School_English': '高中英语', 'High_School_Chinese': '高中语文'}


class DeepSeek_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name):
        super().__init__(choices, model_name, k)
        openai.api_base = "https://api.deepseek.com/v1"
        openai.api_key = api_key
        self.answer_pattern = re.compile(r"答案：\s*([A-D])", re.IGNORECASE)

    def format_example(self, line, include_answer=True, cot=False):
        """格式化示例（支持cot）"""
        example = line['prompt']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'

        if not include_answer:
            return [{"role": "user", "content": example}]

        # 根据cot状态生成不同格式的回答
        if cot:
            content = f"<think>\n{line.get('explanation', '典型解题过程展示...')}\n</think>\n答案：{line['answer']}"
        else:
            content = f"答案：{line['answer']}"

        return [
            {"role": "user", "content": example},
            {"role": "assistant", "content": content}
        ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        """生成支持cot的few-shot提示"""
        # 系统消息根据cot状态动态调整
        system_content = (
            f"你是中国{name_en2zh[subject]}考试解题专家，请按以下要求作答：\n"
            "1. 用<think>包裹详细推理过程\n"
            "2. 最后用单独一行'答案：X'输出结果\n"
            "3. 不要添加任何额外内容"
            if cot else
            f"你正在参加{name_en2zh[subject]}考试，请直接给出正确答案"
        )

        system_msg = {
            "role": "system",
            "content": system_content
        }

        examples = []
        for i in range(min(self.k, len(dev_df))):
            # 调用format_example时启用cot参数
            example = self.format_example(dev_df.iloc[i], cot=cot)

            # 首条示例添加格式说明
            if i == 0 and cot:
                example[0]["content"] = "请严格按以下格式回答：\n" + example[0]["content"]

            examples.extend(example)

        return [system_msg] + examples

    def extract_answer(self, response: str, cot: bool = False) -> str:
        """增强型答案提取（支持cot）"""
        # 优先处理带cot的结构化响应
        if cot:
            # 尝试提取<think>标签后的答案
            parts = response.split("</think>")
            if len(parts) > 1:
                answer_part = parts[-1].strip()
                match = re.search(r"答案：\s*([A-D])", answer_part)
                if match:
                    return match.group(1)

        # 通用提取逻辑
        patterns = [
            r"答案：\s*([A-D])",  # 标准格式
            r"答案\s*([A-D])",  # 无冒号
            r"正确选项：\s*([A-D])",  # 中文描述
            r"[\(（]\s*([A-D])\s*[\)）]"  # 括号格式
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 兜底方案：取最后一个有效字符
        clean_response = re.sub(r"\s+", "", response)
        return clean_response[-1].upper() if clean_response else ''

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None, cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []

        # 提示构造逻辑保持不变
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            few_shot_prompt = [{
                "role": "system",
                "content": f"你是一个中文人工智能助手，以下是中国关于{name_en2zh[subject_name]}考试的单项选择题，请选出其中的正确答案。"
            }]

        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question

            if not few_shot:
                full_prompt[-1][
                    "content"] = f"以下是中国关于{name_en2zh[subject_name]}考试的单项选择题，请选出其中的正确答案。\n\n" + \
                                 full_prompt[-1]["content"]

            response = None
            timeout_counter = 0

            # API调用部分保持结构，修改错误处理逻辑
            while response is None and timeout_counter <= 30:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,  # 使用DeepSeek的模型名称
                        messages=full_prompt,
                        temperature=0.0,
                        timeout=20  # 根据API要求调整超时时间
                    )
                except Exception as msg:
                    if "rate limit" in str(msg).lower():  # 针对DeepSeek的限速错误处理
                        timeout_counter += 5  # 延长等待时间
                        sleep(10)
                    else:
                        timeout_counter += 1
                    print(f"API Error: {msg}")
                    sleep(3)
                    continue

            # 响应处理逻辑保持不变
            response_str = response['choices'][0]['message']['content'] if response else ""

            # 答案提取逻辑保持不变
            if cot:
                ans_list = re.findall(r"答案(?:应该|可能)?是?\s?(?:选项)?\s?([A-D])", response_str, re.IGNORECASE)
                if not ans_list:
                    ans_list = re.findall(r"选项\s?([A-D])", response_str, re.IGNORECASE)
                if not ans_list:
                    ans_list = re.findall(r"(?:正确|对应)(?:选项|答案)?\s?([A-D])", response_str, re.IGNORECASE)
                if not ans_list:
                    # 提取所有大写字母选项
                    ans_list = re.findall(r'\b([A-D])\b', response_str)

                correct = 1 if ans_list and self.exact_match(ans_list[-1], row["answer"]) else 0
            else:
                response_str = response_str.strip()
                ans_list = self.extract_ans(response_str)
                correct = 1 if ans_list and ans_list[-1] == row["answer"] else 0

            correct_num += correct

            if save_result_dir:
                result.append(response_str)
                score.append(correct)

        # 结果保存逻辑保持不变
        correct_ratio = 100 * correct_num / len(answers)
        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            os.makedirs(save_result_dir, exist_ok=True)
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)

        return correct_ratio

    def safe_api_call(self, model, messages, cot=False, max_retries=5, initial_delay=1, **kwargs):
        """
        带自动重试机制的API调用函数，支持cot模式

        参数:
            model (str): 模型名称
            messages (list): 消息列表，包含对话历史
            cot (bool): 是否启用思维链模式（Chain-of-Thought）
            max_retries (int): 最大重试次数，默认为5
            initial_delay (float): 初始重试延迟时间（秒），默认为1秒
            **kwargs: 其他传递给API的参数

        返回:
            str: API返回的响应内容
        """
        retry_count = 0
        delay = initial_delay

        while retry_count < max_retries:
            try:
                # 根据cot模式调整参数
                api_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1 if cot else 0.0,  # cot模式下稍高的温度
                    "timeout": 20,  # 默认超时时间
                    **kwargs  # 其他用户自定义参数
                }

                # 调用API
                response = openai.ChatCompletion.create(**api_params)

                # 返回模型生成的文本内容
                return response.choices[0].message.content

            except openai.error.RateLimitError as e:
                # 处理速率限制错误
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= 2  # 指数退避

            except openai.error.Timeout as e:
                # 处理超时错误
                print(f"Request timed out. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= 2  # 指数退避

            except openai.error.APIError as e:
                # 处理其他API错误
                print(f"API error: {str(e)}")
                if retry_count < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retry_count += 1
                    delay *= 2  # 指数退避
                else:
                    raise e  # 达到最大重试次数后抛出异常

            except Exception as e:
                # 处理其他未知错误
                print(f"Unexpected error: {str(e)}")
                if retry_count < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retry_count += 1
                    delay *= 2  # 指数退避
                else:
                    raise e  # 达到最大重试次数后抛出异常

        # 如果所有重试都失败，返回空字符串或抛出异常
        print("Max retries reached. Returning empty response.")
        return ""