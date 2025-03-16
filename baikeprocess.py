# import json

# # 读取JSON文件内容
# with open('baike.jsonl', 'r', encoding='utf-8') as file:
#     data = file.readlines()

# # 修改每个JSON对象
# modified_data = []
# for line in data:
#     obj = json.loads(line)
#     obj['question'] = obj['question'].replace("我有一个计算机相关的问题，请用中文回答，什么是 ", "我有一个教育相关的问题，请用中文回答，什么是 ")
#     obj['human_answers'] = []
#     obj['chatgpt_answers'] = []
#     modified_data.append(json.dumps(obj, ensure_ascii=False))

# # 写回修改后的内容到新的文件
# with open('modified_baike.jsonl', 'w', encoding='utf-8') as file:
#     for line in modified_data:
#         file.write(line + '')

import json

input_file = 'modified_baike.jsonl'
output_file = 'modified_baike_processed.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    # 读取整个文件内容
    content = infile.read()
    
    # 将多个 JSON 对象拆分为列表
    # 假设每个 JSON 对象以 "{" 开头，以 "}" 结尾
    json_objects = []
    start = 0
    while True:
        # 找到下一个 "{" 的位置
        start = content.find('{', start)
        if start == -1:
            break
        
        # 找到下一个 "}" 的位置
        end = content.find('}', start) + 1
        if end == 0:
            break
        
        # 提取 JSON 对象
        json_str = content[start:end]
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"解析错误: {e}，跳过当前对象")
        
        # 更新起始位置
        start = end
    
    # 处理每个 JSON 对象
    for data in json_objects:
        # 处理 question 字段
        if "什么是 " in data["question"]:
            data["question"] = data["question"].split("什么是 ", 1)[0] + "什么是 "
        
        # 修改键名并保留原值
        if "human_answers" in data:
            data["deepseek_answers"] = data.pop("human_answers")
        
        # 写入处理后的数据
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')