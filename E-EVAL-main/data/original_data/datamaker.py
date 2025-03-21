import json
import csv
import random
import os


def jsonl_to_csv(jsonl_file, csv_file):
    """处理单个JSONL文件转换"""
    fieldnames = [
        'id', 'prompt', 'A', 'B', 'C', 'D', 'answer', 'analysis',
        'difficulty', 'knowledge_domain', 'knowledge_tree',
        'task_grade', 'task_semester', 'task_subject', 'task_type'
    ]

    with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        with open(jsonl_file, 'r', encoding='utf-8') as jsonl_f:
            for line in jsonl_f:
                entry = json.loads(line)
                processed = {key: '' for key in fieldnames}

                # 处理选项
                for option in entry.get('answer_option', []):
                    if '.' in option:
                        letter_part, content = option.split('.', 1)
                        letter = letter_part.strip()
                        content = content.strip().replace('\n', '')
                    else:
                        letter = option[0].strip()
                        content = option[1:].strip().replace('\n', '')

                    if letter in ['A', 'B', 'C', 'D']:
                        processed[letter] = content

                # 处理其他字段
                for key in fieldnames:
                    if key in ['A', 'B', 'C', 'D']: continue
                    value = entry.get(key, '')
                    processed[key] = ';'.join(map(str, value)) if isinstance(value, list) else str(value)

                writer.writerow(processed)


def batch_convert_subjects():
    """批量处理所有科目"""
    base_dir = './'  # 根目录
    output_dir = './datasets/'

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历学段目录（小学、初中、高中）
    for phase in ['小学', '初中', '高中']:
        phase_path = os.path.join(base_dir, phase)
        if not os.path.exists(phase_path): continue

        # 遍历学科目录（语文、数学、英语等）
        for subject in os.listdir(phase_path):
            subject_path = os.path.join(phase_path, subject)
            if not os.path.isdir(subject_path): continue

            # 查找JSONL文件
            jsonl_files = [f for f in os.listdir(subject_path) if f.endswith('.jsonl')]

            # 为每个JSONL生成CSV
            for jsonl_file in jsonl_files:
                input_path = os.path.join(subject_path, jsonl_file)
                output_file = f"{phase}{subject}.csv"
                output_path = os.path.join(output_dir, output_file)

                print(f"正在转换：{phase}/{subject}/{jsonl_file} → {output_file}")
                jsonl_to_csv(input_path, output_path)



def generate_exam_papers(source_file, output_prefix, num_papers=10, questions_per_paper=100):
    """生成指定数量的试卷"""
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_questions = list(reader)

            # 自动创建输出目录
            output_dir = os.path.dirname(output_prefix)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 生成多套试卷
            for paper_num in range(1, num_papers + 1):
                # 使用可重复抽样保证试卷多样性
                selected = random.choices(all_questions, k=questions_per_paper)

                # 生成带序号的文件名
                filename = f"{output_prefix}{paper_num:02d}.csv"

                # 写入新文件
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    writer.writerows(selected)

                print(f"成功生成：{filename}")

    except Exception as e:
        print(f"处理文件 {source_file} 时出错：{str(e)}")


def batch_generate_exams():
    """批量处理所有学科"""
    # 配置路径
    input_dir = "./datasets/"
    output_dir = "./测试题库/"

    # 获取所有CSV文件（按年级+学科分类）
    subjects = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    # 遍历处理每个学科
    for subject_file in subjects:
        # 解析学科名称（去除.csv扩展名）
        subject_name = os.path.splitext(subject_file)[0]

        # 构建完整路径
        input_path = os.path.join(input_dir, subject_file)
        output_prefix = os.path.join(output_dir, f"{subject_name}试卷")

        print(f"\n正在处理学科：{subject_name}")
        generate_exam_papers(
            source_file=input_path,
            output_prefix=output_prefix,
            num_papers=10,
            questions_per_paper=100
        )


if __name__ == "__main__":
    batch_convert_subjects()
    print("所有科目转换完成！")
    # 自动创建输出目录
    os.makedirs("./测试题库/", exist_ok=True)

    # 开始批量生成
    batch_generate_exams()
    print("\n所有试卷生成完成！")