# DeepSeekandChatGPT

- [x] 2025.3.12思路预流程构建，简易工程版
- [ ] ...

## 一、模型的选取

根据在Livebench（实时AI模型基准测试平台）排行中的评比

| Model                                                        | Organization | Global Average |
| ------------------------------------------------------------ | ------------ | -------------- |
| [deepseek-r1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek     | 71.57          |
| [o3-mini-2025-01-31-medium](https://openai.com/index/openai-o3-mini/) | OpenAI       | 70.01          |

选择Deepseek-R1和OpenAI-o3-mini进行对比

预使用API进行问题测试，价格如下表

| 项目       | OpenAI o3-mini（元 / 1M tokens） | **deepseekR1 chat 类（优惠时段，元 / 1M tokens）** |
| ---------- | -------------------------------- | -------------------------------------------------- |
| 未缓存输入 | 7.81元                           | 1.00元                                             |
| 已缓存输入 | 3.905元                          | 0.25元                                             |
| 输出       | 31.24元                          | 4.00元                                             |

## 二、测评问题数据集的制作

### 1 百科通识类别问题数据

提问模板：“我有一个教育相关的问题，请用中文回答，什么是 XXX”

其中XXX为《教育学名词(2013)》中的3401条

构建如下jsonl形式的数据

```json
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育潜力", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育改革", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育传统", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育变迁", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育民主", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育民主化", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育现代化", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育世俗化", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育先行", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 人的全面发展", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 全面发展教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 现代教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 传统教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 公共教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 闲暇教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 科学教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 素质教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 应试教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 终身教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 全纳教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 社会教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 智育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 体育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 美育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 劳动技术教育", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育哲学", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育知识", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 课程哲学", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教育评价哲学", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 教学哲学", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 分析教育哲学", "chatgpt_answers": [], "deepseek_answers": []}
{"question": "我有一个教育相关的问题，请用中文回答，什么是 思辨教育哲学", "chatgpt_answers": [], "deepseek_answers": []}
```

### 2 基础教育通识类问题

包含小学、初中、高中的教育中十个类别（不含体育）的考试题目

来源：教资真题、学业水平考试真题、教育知识与能力真题、高考模拟题、中考高考真题

| **学科** | **考试题目**                                                 |
| -------- | ------------------------------------------------------------ |
| **语文** | **阅读材料《场景歌》，完成以下任务**： （1）从语言学习角度分析文本特点； （2）拟定二年级教学目标； （3）设计教学思路与方法。 |
| **数学** | **加里培林“智力活动形成理论”中，学生用石子、手指辅助计算加法，属于哪个阶段？** A. 活动定向阶段 B. 物质化活动阶段 C. 无声外部言语阶段 D. 内部言语阶段 |
| **英语** | **用英语描述以下图片内容**（配图：家庭聚餐场景），要求包含人物关系、动作及情感表达，不少于80词。 |
| **物理** | **解释摩擦力的作用，并举例说明如何通过改变接触面性质来增大或减小摩擦力。** |
| **化学** | **实验题：观察水沸腾现象，回答：** （1）水沸腾时温度是否持续升高？ （2）解释“纸锅烧水”中纸不会被点燃的原因。 |
| **生物** | **简答：光合作用的反应式及意义。若植物长期缺乏光照，对其生长有何影响？** |
| **历史** | **材料分析：** “蔡元培在《对于教育方针之意见》中提出‘五育并举’思想，指出实现世界观教育的主要途径是什么？” A. 德育 B. 智育 C. 美育 D. 体育 |
| **地理** | **综合题：** （1）分析某区域自然景观与气候类型的关系； （2）列举该地区特色文化习俗，并说明其与地理环境的关系。 |
| **政治** | **论述题：结合实例，说明“公民参与民主决策的意义”及“中学生如何践行社会责任”。** |
| **艺术** | **赏析题：** （1）分析《清明上河图》的艺术特色； （2）结合历史背景，说明该作品反映的社会风貌。 |

### 3 教育学领域问题

包含**教育学原理**、**中外教育史**、**教育心理学**、**教育研究方法**、**教育政策法规**、**现代教育技术**、**特殊教育**的问题，含正确答案。

来源：考研专业课333教育学

略......



## 三、测评答案数据集的生成

当数据不多时使用聊天窗口生成

当数据过多时通过API接口调用批量写入数据

编写自动化脚本批量调用API生成答案

```
(open_manus) E:\2025文章\DeepseekandGPT>python auto_create_answers.py

[问题] 我有一个教育相关的问题，请用中文回答，什么是 教育潜力...
[状态] 正在请求API...
[结果] 获取成功！
[响应] **教育潜力**是指个体在学习和成长过程中所具备的潜在能力和发展可能性。它反映了一个人在接受教育后能够达到的成就水平 ，包括智力、创造力、情感、社交能力等多方面的综合发展。

教育潜力并不是固定不变的，...

[问题] 我有一个教育相关的问题，请用中文回答，什么是 教育改革...
[状态] 正在请求API...
[结果] 获取成功！
[响应] 教育改革是指对现有的教育体系、教育制度、教育内容、教育方法等方面进行系统的调整和优化，以适应当前社会发展的需求， 提升教育质量和公平性。教育改革的目标通常包括：

1. **提高教育质量**：通过改进教...
处理进度: 100%|██████████████████████████████████████████████████████████████████████████| 2/2 [00:36<00:00, 18.41s/条]
```

代码如下：

```python
import json
import openai
import time
from tqdm import tqdm  # 进度条库

# 配置DeepSeek API
openai.api_base = "https://api.deepseek.com/v1"
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxx" #填写APIkey


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
process_jsonl("baike.jsonl", "baike_updated.jsonl")
```

填充好QandA数据集

## 四、评估模型流程

   A[输入答案对] --> B{表层指标}    A --> C{语义指标}    A --> D{知识验证}    B --> E[词重叠率/BLEU]    C --> F[语义相似度]    D --> G[概念准确性]

### **1、表层指标评估方案**

##### 词重叠率 - BLEU

- **NLTK** + **Jieba** 组合
- 替代方案：使用jiwer库的快速实现

### **2、语义相似度评估方案**

#### 1. Sentence-BERT 方案

[Sentence-BERT（SBERT）模型介绍及Sentence Transformers库的使用 - 知乎](https://zhuanlan.zhihu.com/p/659682364)

#### 2. 教育领域增强方案

使用教育领域相似句对，将Sentence-BERT预训练模型进行微调，再使用。

#### 3. 最新模型（暂未看）

| 模型名称 | 特点           | 代码链接                                           |
| :------- | :------------- | :------------------------------------------------- |
| SPARTA   | 稀疏注意力增强 | [GitHub](https://github.com/neulab/sparta)         |
| EduCo    | 教育对话专用   | [GitHub](https://github.com/educational-nlp/educo) |

------

### **3、知识验证方案**

待定

### 4、大模型事实校验

使用另外一种或几种模型进行校验

```
import openai

def fact_check_with_gpt(pred, ref):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个教育学专家，请严格检查以下答案是否符合教育学理论"},
            {"role": "user", "content": f"生成答案：{pred}\n参考答案：{ref}"}
        ]
    )
    return response.choices[0].message['content']

# 注意：需替换为实际API密钥，建议使用DeepSeek等国产大模型替代
```

------

### **5、综合评估方案**

#### 权重配置

对工程中的每个指标配置权重生成综合评分

------

- [x] 