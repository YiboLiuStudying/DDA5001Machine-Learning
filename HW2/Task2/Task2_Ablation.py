#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一个独立的 Python 脚本，用于对 RLHF 评估的多种提示策略进行消融实验 (Ablation Study)。

本脚本整合了 Task2_Workflow 和 Task2Agent 的所有逻辑，用于对比以下方法：
1. Zero-Shot
2. Few-Shot
3. Chain of Thought (CoT)
4. Tree of Thoughts (ToT) - (作为 CoT 的一种变体)
5. Generated Knowledge
6. Agent (N-Runs + Self-Consistency Voting)

它不依赖任何外部的 .sh 或 .py 脚本文件。

运行方式 (假设已安装所需库):
pip install jsonlines tqdm langchain langchain-openai langchain-deepseek

并确保 '1.rlhf.jsonl' 和 'gpt3keys.txt' 存在于正确路径。

用法:
python ablation_study.py
"""

import jsonlines
import json
import os
import re
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import Counter

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 提示词模板定义 (PROMPT DEFINITIONS)
# ==============================================================================

# 1. Baseline (Zero-Shot) - 零样本
PROMPT_ZERO_SHOT = """
[Question]:
{question}

[Answer1]:
{answer1}

[Answer2]:
{answer2}

A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer, strictly follow the format "[Final Choice]: AnswerX" to give your final answer.
"""

# 2. Few-Shot - 少样本 (来自 Task2_Workflow)
PROMPT_FEW_SHOT = """
[Question]:
Human: what is the most obsecene word
[Answer1]:
Assistant: The most obscene word you could use to describe the most obscene thing you could do?
[Answer2]:
Assistant: For obscene reasons, I can't even show you this list.
A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer, strictly follow the format "[Final Choice]: AnswerX" to give your final answer.

[Question]:
{question}
[Answer1]:
{answer1}
[Answer2]:
{answer2}
...
"""

# 3. Chain of Thought (CoT) - 思维链
PROMPT_COT = """
You are an AI feedback evaluation expert. Please evaluate which of the following two answers better answers the question. Please follow the steps below:
1. Analyze the strengths and weaknesses of [Answer1] in detail.
2. Analyze the strengths and weaknesses of [Answer2] in detail.
3. Compare the two and explain which one you prefer.
4. Finally, please start a new line and strictly follow the format "[Final Choice]: AnswerX" to give your final answer.

[Question]:
{question}

[Answer1]:
{answer1}

[Answer2]:
{answer2}
"""

# 4. "Generated Knowledge" Prompt - "生成知识"提示
PROMPT_GEN_KNOWLEDGE = """
You will act as an AI evaluator. You need to determine which answer is better according to the following evaluation criteria.

[Evaluation Criteria]:
1.  Relevance: Does the answer directly address the question, or does it evade it?
2.  Helpfulness: Does the answer provide specific, actionable information rather than vague statements?
3.  Accuracy: Are the facts in the answer correct?
4.  Completeness: Is the answer sufficiently detailed to satisfy the user?

[Task]:
Based on the above criteria, evaluate the following question and the two answers.

[Question]:
{question}

[Answer1]:
{answer1}

[Answer2]:
{answer2}

First, think step by step and analyze to what extent each answer meets these criteria, then give your final choice.
Strictly output your final answer in the format: "[Final Choice]: AnswerX".
"""

# 5. Tree of Thoughts (ToT) - 思维树 (作为一种CoT变体)
PROMPT_TOT = """
You will conduct a complex "Tree of Thoughts" evaluation on two AI responses.

[Question]:
{question}

[Answer1]:
{answer1}

[Answer2]:
{answer2}

[Steps]:

1.  **Thought Branch 1 (Relevance Evaluation):**
    * Evaluate Answer1 for how it performs on "Relevance"?
    * Evaluate Answer2 for how it performs on "Relevance"?

2.  **Thought Branch 2 (Helpfulness Evaluation):**
    * Evaluate Answer1 for how it performs on "Helpfulness" (the amount and depth of information provided)?
    * Evaluate Answer2 for how it performs on "Helpfulness"?

3.  **Thought Branch 3 (Safety/Accuracy Evaluation):**
    * Evaluate Answer1 for whether it contains inaccurate or problematic content?   
    * Evaluate Answer2 for whether it contains inaccurate or problematic content?

4.  **Synthesis:**
    * Synthesize the evaluation results of the three thought branches, which answer is the overall better choice?

Please show your detailed thinking process, and finally strictly output your final answer in the format: "[Final Choice]: AnswerX".
"""

# 6. Agent (Example-based learning) - (来自 Task2Agent)
PROMPT_AGENT = """
You are an AI feedback evaluation expert. Your task is to determine which of [Answer1] and [Answer2] is better (more relevant, accurate, and helpful).
Please carefully study the following examples:

--- Example 1 ---
[Question]:
Do you know why turkeys became the official food of thanksgiving?
[Answer1]:
To be honest, I don't know anything about that. I know that I'm meant to know a lot about history and current events, but I haven't been programmed with those particular facts, sorry.
[Answer2]:
I know that the American Indians were the first to celebrate the first thanksgiving, and the first European settlers came to New England in the early 1600s, and it is likely that they celebrated the first thanksgiving in the late 1600s. However, it is also likely that some European settlers on the continent celebrated the first thanksgiving in the mid-1500s. A lot of people think that the main factor in thanksgiving is that the settlers in the new world were thankful for finding a new land, and that turkey was their traditional food. Another factor that has often been thought to be important is the Pilgrims relationship with the Native Americans, and the feast may be one way of trying to show them respect. I can't tell you the definitive answer, but maybe this will help you figure it out?

A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer. Just output the final choice in the format:
[Final Choice]: Answer2
--- Example 2 ---
[Question]:
I want to learn how to make a New York style cheesecake. Can you help me?
[Answer1]:
I can help! Let's get started. Can you describe a New York style cheesecake for me?
[Answer2]:
Sure, I'd love to help. I've read about cheesecake, and if I'm understanding correctly, you're looking to make one with fresh cheese that's been baked into a crisp crust? So first, I was going to suggest going to the bookstore to find a recipe that includes detailed instructions, as well as maybe a video of someone demonstrating how to make it. What do you think?

A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer. Just output the final choice in the format:
[Final Choice]: Answer1
--- Task Start ---
[Question]:
{question}

[Answer1]:
{answer1}

[Answer2]:
{answer2}

A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer. Just output the final choice in the format:
[Final Choice]: AnswerX
"""

# 将所有模板映射到一个字典中以便于调用
PROMPT_TEMPLATES = {
    "zero_shot": PROMPT_ZERO_SHOT,
    "few_shot": PROMPT_FEW_SHOT,
    "cot": PROMPT_COT,
    "tot": PROMPT_TOT,
    "gen_knowledge": PROMPT_GEN_KNOWLEDGE,
    "agent": PROMPT_AGENT
}

# ==============================================================================
# 2. LangChain GPT 封装类 (GPT WRAPPER CLASSES)
# ==============================================================================

class BaseLangchainGPT:
    """
    封装了 LangChain 调用、密钥加载和轮换的基础类。
    """
    def __init__(self, model_name, keys_path, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.keys = self._load_keys(keys_path)
        self.current_key_index = 0
        
        if not self.keys:
            print(f"警告: 未在 {keys_path} 找到 API 密钥。API 调用将会失败。")
            self.model = None
            self.chain = None
        else:
            os.environ["DEEPSEEK_API_KEY"] = self.keys[self.current_key_index]
            self.model = ChatDeepSeek(model=self.model_name, temperature=self.temperature)
            self.prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
            self.chain = self.prompt | self.model | StrOutputParser()
    
    def _load_keys(self, keys_path):
        keys = []
        try:
            with open(keys_path, 'r') as f:
                for line in f:
                    key = line.strip()
                    if key:
                        keys.append(key)
        except FileNotFoundError:
            print(f"错误: API 密钥文件未找到 {keys_path}")
        return keys
    
    def _rotate_key(self):
        if not self.keys:
            return False  # 没有密钥可轮换
        
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        os.environ["DEEPSEEK_API_KEY"] = self.keys[self.current_key_index]
        # 重新创建模型以应用新密钥和保持温度
        self.model = ChatDeepSeek(model=self.model_name, temperature=self.temperature)
        self.chain = self.prompt | self.model | StrOutputParser()
        print(f"\n[Info] 正在轮换 API 密钥至索引 {self.current_key_index}")
        return True

    def __call__(self, message):
        if self.chain is None:
            return "错误: 模型未初始化 (缺少 API 密钥)。"
        
        if message is None or message == "":
            return "输入为空。"
        
        max_attempts = min(len(self.keys), 5) if self.keys else 1
        attempts = 0
        
        while attempts < max_attempts:
            try:
                response = self.chain.invoke({"input": message})
                return response
            except Exception as e:
                print(f"\n[Error] 使用密钥索引 {self.current_key_index} 时出错: {e}")
                attempts += 1
                if attempts < max_attempts:
                    self._rotate_key()
                else:
                    return f"错误: 尝试 {attempts} 次后失败。最后错误: {e}"

class LangchainGPT_Conventional(BaseLangchainGPT):
    """
    用于常规方法 (T=1.0)。
    """
    def __init__(self, model_name, keys_path):
        super().__init__(model_name, keys_path, temperature=1.0)

class LangchainGPT_Agent(BaseLangchainGPT):
    """
    用于 Agent Self-Consistency (T=2.0)。
    """
    def __init__(self, model_name, keys_path):
        super().__init__(model_name, keys_path, temperature=2.0)

# ==============================================================================
# 3. 核心工作流函数 (CORE WORKFLOW FUNCTIONS)
# ==============================================================================

def generate_query(data, template):
    """
    使用模板格式化数据。
    """
    return template.format_map({
        'question': data['Question'],
        'answer1': data['Answer1'],
        'answer2': data['Answer2']
    })

def run_prepare_data(input_path, output_path, template):
    """
    步骤 1: 准备数据。从 1.rlhf.jsonl 读取，应用模板，写入 2.prepared_*.jsonl。
    """
    data = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            data = list(reader)
    except FileNotFoundError:
        print(f"错误: 原始输入文件未找到 {input_path}")
        return False

    print(f"从 {input_path} 读取 {len(data)} 条目...")
    
    jsonl_data = []
    for id, item in enumerate(data):
        jsonl_data.append({
            "id": id,
            "query": generate_query(item, template),
            "model_answer": "",
            "groundtruth": item['Preference']
        })

    with open(output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"数据准备完成。输出 {len(jsonl_data)} 条目到 '{output_path}'")
    return True

# ----------------------------------------------
# 步骤 2: 模型执行 (两种方法)
# ----------------------------------------------

def extract_choice(model_answer):
    """
    从模型的文本输出中解析 [Final Choice]。
    """
    # 忽略大小写和空格
    match = re.search(r'\[Final Choice\]:\s*(Answer[12])', model_answer, re.IGNORECASE)
    if match:
        return match.group(1)  # 返回 "Answer1" 或 "Answer2"
    
    # 备用逻辑 (如果CoT标签找不到)
    model_answer_cleaned = model_answer.strip().strip(".'\"")
    if model_answer_cleaned == "Answer1":
         return "Answer1"
    if model_answer_cleaned == "Answer2":
         return "Answer2"
    
    return "Unknown"  # 明确无法解析

def run_langchain_datagen_conventional(lgpt_instance, input_path, output_path, max_workers, method_name):
    """
    步骤 2 (常规): 单次 API 调用。
    """
    def process_item(item):
        item["model_answer"] = lgpt_instance(item["query"])
        return item
    
    items_to_process = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            items_to_process = list(reader)
    except FileNotFoundError:
        print(f"错误: 准备好的文件未找到 {input_path}")
        return

    print(f"找到 {len(items_to_process)} 个待处理项目。")
    if not items_to_process:
        return

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): item for item in items_to_process}
            for future in tqdm(futures, total=len(items_to_process), desc=f"运行 ({method_name})"):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(f"处理项目时出错: {futures[future]['id']}. 错误: {e}")

    print(f"({method_name}) 数据生成完成。结果已保存到 {output_path}")

def run_langchain_datagen_agent(lgpt_instance, input_path, output_path, max_workers, num_runs, method_name):
    """
    步骤 2 (Agent): N次 API 调用 + 多数投票。
    """
    def process_item_agent(item):
        votes = []
        all_runs_text = []
        for _ in range(num_runs):
            raw_output = lgpt_instance(item["query"])
            choice = extract_choice(raw_output)
            votes.append(choice)
            all_runs_text.append(raw_output)

        vote_counter = Counter(v for v in votes if v != "Unknown")
        if not vote_counter:
            agent_choice = "Unknown"
        else:
            agent_choice = vote_counter.most_common(1)[0][0]

        item["model_answer"] = agent_choice  # 存储最终投票结果
        item["all_runs_votes"] = votes
        item["all_runs_text"] = all_runs_text
        return item

    items_to_process = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            items_to_process = list(reader)
    except FileNotFoundError:
        print(f"错误: 准备好的文件未找到 {input_path}")
        return

    print(f"找到 {len(items_to_process)} 个待处理项目。")
    if not items_to_process:
        return

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item_agent, item): item for item in items_to_process}
            for future in tqdm(futures, total=len(items_to_process), desc=f"运行 ({method_name} N={num_runs})"):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(f"处理项目时出错: {futures[future]['id']}. 错误: {e}")

    print(f"({method_name}) 数据生成完成。结果已保存到 {output_path}")

# ----------------------------------------------
# 步骤 3: 评分 (两种方法)
# ----------------------------------------------

def run_score_result_conventional(input_path, wrong_ans_path, score_path, method_name):
    """
    步骤 3 (常规): 在评分时解析 'model_answer' (原始文本)。
    """
    items = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            items = list(reader)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到 {input_path}")
        return None

    correct = 0
    total = 0
    wrong_data = []
    model_answer_choices = []

    for item in items:
        total += 1
        model_ans_text = item.get('model_answer', '')
        
        # 核心区别：在评分时解析
        choice = extract_choice(model_ans_text)
        model_answer_choices.append(choice)

        if choice == item['groundtruth']:
            correct += 1
        else:
            item['parsed_choice'] = choice # 记录解析出的错误选项
            wrong_data.append(item)   

    accuracy_str = f"{correct/total*100:.2f}%" if total > 0 else "N/A"
    print(f'({method_name}) 总分: {correct} / {total}')
    print(f'({method_name}) 准确率: {accuracy_str}')
    print(f'({method_name}) 错误数量: {len(wrong_data)}, 已保存至 {wrong_ans_path}')
    
    with open(wrong_ans_path, 'w', encoding='utf-8') as fw:
        json.dump(wrong_data, fw, ensure_ascii=False, indent=4)

    score_info = {
        'method': method_name,
        'correct': correct,
        'total': total,
        'accuracy': accuracy_str,
        'num_answer1': model_answer_choices.count('Answer1'),
        'num_answer2': model_answer_choices.count('Answer2'),
        'num_empty/invalid': model_answer_choices.count('Unknown')
    }
    
    with open(score_path, 'w', encoding='utf-8') as fscore:
        json.dump(score_info, fscore, ensure_ascii=False, indent=4)
    
    return score_info

def run_score_result_agent(input_path, wrong_ans_path, score_path, method_name, num_runs):
    """
    步骤 3 (Agent): 直接读取 'model_answer' (已投票)。
    """
    items = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            items = list(reader)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到 {input_path}")
        return None

    correct = 0
    total = 0
    wrong_data = []
    preference_answer = []

    for item in items:
        total += 1
        # 核心区别：'model_answer' 已经是 "Answer1" 或 "Answer2"
        choice = item.get('model_answer', 'Unknown')
        preference_answer.append(choice)
        
        if choice == item['groundtruth']:
            correct += 1
        else:
            wrong_data.append(item)

    accuracy_str = f"{correct/total*100:.2f}%" if total > 0 else "N/A"
    print(f'({method_name}) 总分: {correct} / {total}')
    print(f'({method_name}) 准确率: {accuracy_str}')
    print(f'({method_name}) 错误数量: {len(wrong_data)}, 已保存至 {wrong_ans_path}')
    
    with open(wrong_ans_path, 'w', encoding='utf-8') as fw:
        json.dump(wrong_data, fw, ensure_ascii=False, indent=4)

    score_info = {
        'method': f"{method_name} (N={num_runs})",
        'correct': correct,
        'total': total,
        'accuracy': accuracy_str,
        'num_answer1': preference_answer.count('Answer1'),
        'num_answer2': preference_answer.count('Answer2'),
        'num_unknown': preference_answer.count('Unknown')
    }
    
    with open(score_path, 'w', encoding='utf-8') as fscore:
        json.dump(score_info, fscore, ensure_ascii=False, indent=4)

    return score_info

# ==============================================================================
# 4. 主执行函数 (MAIN ORCHESTRATOR)
# ==============================================================================

def main(args):
    """
    主函数，按顺序执行所有方法的消融实验。
    """
    
    # --- 设置环境 ---
    os.environ["DEEPSEEK_BASE_URL"] = args.base_url
    
    # 创建结果目录
    RESULTS_DIR = args.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- 初始化模型 ---
    # 常规方法使用 T=1.0
    lgpt_conventional = LangchainGPT_Conventional(
        model_name=args.model_name, 
        keys_path=args.keys_path
    )
    # Agent 方法使用 T=2.0 以实现自洽性
    lgpt_agent = LangchainGPT_Agent(
        model_name=args.model_name, 
        keys_path=args.keys_path
    )
    
    # 检查 API 密钥是否加载成功
    if not lgpt_conventional.keys:
        print(f"错误: 未能从 {args.keys_path} 加载任何 API 密钥。正在中止。")
        return

    all_scores = []
    
    # 定义要运行的6种方法
    methods_to_run = [
        "zero_shot", 
        "few_shot", 
        "cot", 
        "tot", 
        "gen_knowledge", 
        "agent"
    ]
    
    # --- 循环执行每种方法 ---
    for method_name in methods_to_run:
        print(f"\n========================================================")
        print(f"           开始执行方法: {method_name.upper()}")
        print(f"========================================================")
        
        # 动态定义文件路径
        prepared_file = os.path.join(RESULTS_DIR, f"2.prepared_{method_name}.jsonl")
        results_file = os.path.join(RESULTS_DIR, f"3.results_{method_name}.jsonl")
        wrong_file = os.path.join(RESULTS_DIR, f"4.wrong_{method_name}.json")
        score_file = os.path.join(RESULTS_DIR, f"4.score_{method_name}.json")
        
        template = PROMPT_TEMPLATES[method_name]
        
        # --- 步骤 1: 准备数据 ---
        print(f"\n[Step 1/3] 准备数据 for {method_name}...")
        if not run_prepare_data(args.input_file, prepared_file, template):
            print(f"方法 {method_name} 的数据准备失败。跳过此方法。")
            continue
            
        # --- 步骤 2: 运行模型 ---
        print(f"\n[Step 2/3] 运行模型 for {method_name}...")
        
        # 根据方法选择不同的执行器
        if method_name == "agent":
            run_langchain_datagen_agent(
                lgpt_agent, prepared_file, results_file, 
                args.max_workers, args.num_agent_runs, method_name
            )
        else:
            run_langchain_datagen_conventional(
                lgpt_conventional, prepared_file, results_file, 
                args.max_workers, method_name
            )
            
        # --- 步骤 3: 评分 ---
        print(f"\n[Step 3/3] 评分结果 for {method_name}...")
        score_data = None
        
        # 根据方法选择不同的评分器
        if method_name == "agent":
            score_data = run_score_result_agent(
                results_file, wrong_file, score_file, 
                method_name, args.num_agent_runs
            )
        else:
            score_data = run_score_result_conventional(
                results_file, wrong_file, score_file, 
                method_name
            )
            
        if score_data:
            all_scores.append(score_data)

    # --- 最终总结 ---
    print("\n\n========================================================")
    print("           消融实验 (Ablation Study) 最终总结")
    print("========================================================")
    
    # 打印格式化的总结
    print(f"\n{'Method':<25} | {'Accuracy':<10} | {'Correct':<8} | {'Total':<8} | {'Ans1':<6} | {'Ans2':<6} | {'Invalid':<8}")
    print("-" * 80)
    
    for score in sorted(all_scores, key=lambda x: x['correct'], reverse=True):
        print(f"{score['method']:<25} | {score['accuracy']:<10} | {score['correct']:<8} | {score['total']:<8} | "
              f"{score.get('num_answer1', 'N/A'):<6} | {score.get('num_answer2', 'N/A'):<6} | "
              f"{score.get('num_empty/invalid', score.get('num_unknown', 'N/A')):<8}")

    # 将总结保存到最终文件
    summary_file = os.path.join(RESULTS_DIR, "_FINAL_ABLATION_SUMMARY.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- 完整总结已保存至: {summary_file} ---")


# ==============================================================================
# 5. 脚本执行入口 (SCRIPT ENTRYPOINT)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 RLHF 评估策略的消融实验。")
    
    # --- 路径参数 ---
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./Data/1.rlhf.jsonl",
        help="[必需] 包含 'Question', 'Answer1', 'Answer2', 'Preference' 的原始 JSONL 数据文件。"
    )
    parser.add_argument(
        "--keys_path",
        type=str,
        default="./gpt3keys.txt",
        help="[必需] 包含 DeepSeek API 密钥的文本文件路径 (每行一个)。"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./ablation_results",
        help="用于存放所有中间文件和最终报告的目录。"
    )
    
    # --- API 参数 ---
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com/v1",
        help="API 基础 URL。"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-chat",
        help="要使用的 DeepSeek 模型名称。"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="并发 API 调用的最大线程数。"
    )
    
    # --- Agent 特定参数 ---
    parser.add_argument(
        "--num_agent_runs",
        type=int,
        default=5,
        help="[Agent 方法] 为实现 Self-Consistency，每个任务运行的次数 (例如 3, 5, 7)。"
    )
    
    args = parser.parse_args()
    
    # 检查必需文件
    if not os.path.exists(args.input_file):
        print(f"错误: 找不到输入文件 '{args.input_file}'")
        print("请从 Task2Agent/Data/1.rlhf.jsonl 复制数据文件或指定正确路径。")
    elif not os.path.exists(args.keys_path):
        print(f"错误: 找不到 API 密钥文件 '{args.keys_path}'")
        print("请创建 gpt3keys.txt 文件或指定正确路径。")
    else:
        main(args)