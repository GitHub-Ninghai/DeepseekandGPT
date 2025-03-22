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
              'Primary_School_English': '小学小学英语','Primary_School_Chinese': '小学小学语文',
              'High_School_Chemistry': '高中化学', 'High_School_History': '高中历史',
              'High_School_Geography': '高中地理', 'High_School_Politics': '高中政治',
              'High_School_Mathematics': '高中数学', 'High_School_Physics': '高中物理',
              'High_School_Biology': '高中生物', 'High_School_English': '高中英语', 'High_School_Chinese': '高中语文'}


class DeepSeek_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key, model_name):
        super(DeepSeek_Evaluator, self).__init__(choices, model_name, k)
        openai.api_base = "https://api.deepseek.com/v1"  # 修改API基础地址
        openai.api_key = api_key

    def format_example(self, line, include_answer=True, cot=False):
        # 保持原有格式处理逻辑不变
        example = line['prompt']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += '\n答案：'
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content}
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]}
                ]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        # 保持few-shot提示生成逻辑不变
        prompt = [{
            "role": "system",
            "content": f"你是一个中文人工智能助手，以下是中国关于{name_en2zh[subject]}考试的单项选择题，请选出其中的正确答案。"
        }]

        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = f"以下是中国关于{name_en2zh[subject]}考试的单项选择题，请选出其中的正确答案。\n\n" + \
                                    tmp[0]["content"]
            prompt += tmp
        return prompt

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
                "content": f"你是中国{name_en2zh[subject_name]}考试解题专家，请按以下要求作答：\n"
            "1. 用<think>包裹详细推理过程\n"
            "2. 用中文介绍这道题属于的difficulty,knowledge_domain,knowledge_tree,task_grade,task_semester,task_subject\n"
            "3. 最后用单独一行'答案：X'输出结果,其中X为ABCD其中一个单一的英文大写字母\n"

            }]

        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question

            if not few_shot:
                full_prompt[-1]["content"] = f"你是中国{name_en2zh[subject_name]}考试解题专家，请按以下要求作答：\n1. 用<think>包裹详细推理过程\n2. 用中文介绍这道题属于的difficulty,knowledge_domain,knowledge_tree,task_grade,task_semester,task_subject\n3. 最后用单独一行'答案：X'输出结果,其中X为ABCD其中一个单一的英文大写字母\n" + full_prompt[-1]["content"]

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

    # 以下方法保持与原始代码相同
    def extract_ans(self,response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list