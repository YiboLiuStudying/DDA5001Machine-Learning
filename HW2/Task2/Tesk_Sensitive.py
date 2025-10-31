#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一个独立的 Python 脚本，用于对 RLHF 评估策略进行敏感度分析 (Sensitivity Analysis)。

本脚本将分析两个核心参数：
1.  [常规方法]: 准确率对 'temperature' (T) 变化的敏感度。
2.  [Agent 方法]: 准确率对 'num_agent_runs' (N) 变化的敏感度。

它不依赖任何外部的 .sh 或 .py 脚本文件。

运行方式 (假设已安装所需库):
pip install jsonlines tqdm langchain langchain-openai langchain-deepseek

并确保 '1.rlhf.jsonl' 和 'gpt3keys.txt' 存在于正确路径。

用法示例:
# 运行一个快速测试 (T=0.5 和 T=1.0, N=1 和 N=3)
python sensitivity_analysis.py --temp_range 0.5 1.0 --n_run_range 1 3

# 运行一个完整的分析 (T=0.0, 0.5, 1.0, 1.5, N=1, 3, 5, 7)
python sensitivity_analysis.py --temp_range 0.0 0.5 1.0 1.5 --n_run_range 1 3 5 7
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
# (与 ablation_study.py 相同)

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

# 2. Few-Shot - 少样本
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

# 5. Tree of Thoughts (ToT) - 思维树
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
    * Evaluate Answer1 for how it performs on "Helpfulness"?
    * Evaluate Answer2 for how it performs on "Helpfulness"?
3.  **Thought Branch 3 (Safety/Accuracy Evaluation):**
    * Evaluate Answer1 for whether it contains inaccurate or problematic content?   
    * Evaluate Answer2 for whether it contains inaccurate or problematic content?
4.  **Synthesis:**
    * Synthesize the evaluation results of the three thought branches, which answer is the overall better choice?
Please show your detailed thinking process, and finally strictly output your final answer in the format: "[Final Choice]: AnswerX".
"""

# 6. Agent (Example-based learning)
PROMPT_AGENT = """
You are an AI feedback evaluation expert. Your task is to determine which of [Answer1] and [Answer2] is better (more relevant, accurate, and helpful).
Please carefully study the following examples:
--- Example 1 ---
[Question]:
Do you know why turkeys became the official food of thanksgiving?
[Answer1]:
To be honest, I don't know anything about that. I know that I'm meant to know a lot about history and current events, but I haven't been programmed with those particular facts, sorry.
[Answer2]:
I know that the American Indians were the first to celebrate the first thanksgiving...
A good response should be relevant, accurate and helpful. Which is better, Answer1 or Answer2?
Do not explain your answer. Just output the final choice in the format:
[Final Choice]: Answer2
--- Example 2 ---
[Question]:
I want to learn how to make a New York style cheesecake. Can you help me?
[Answer1]:
I can help! Let's get started. Can you describe a New York style cheesecake for me?
[Answer2]:
Sure, I'd love to help. I've read about cheesecake...
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

PROMPT_TEMPLATES = {
    "zero_shot": PROMPT_ZERO_SHOT,
    "few_shot": PROMPT_FEW_SHOT,
    "cot": PROMPT_COT,
    "tot": PROMPT_TOT,
    "gen_knowledge": PROMPT_GEN_KNOWLEDGE,
    "agent": PROMPT_AGENT
}

# ==============================================================================
# 2. LangChain GPT 封装类 (GPT WRAPPER CLASS)
# ==============================================================================

class LangchainGPT_Instance:
    """
    封装了 LangChain 调用、密钥加载和轮换的通用类。
    关键：它在初始化时接受 'temperature'。
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
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        os.environ["DEEPSEEK_API_KEY"] = self.keys[self.current_key_index]
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
                print(f"\n[Error] 使用密钥索引 {self.current_key_index} (T={self.temperature}) 时出错: {e}")
                attempts += 1
                if attempts < max_attempts:
                    self._rotate_key()
                else:
                    return f"错误: 尝试 {attempts} 次后失败。最后错误: {e}"

# ==============================================================================
# 3. 核心工作流函数 (CORE WORKFLOW FUNCTIONS)
# ==============================================================================
# (与 ablation_study.py 相同)

def generate_query(data, template):
    return template.format_map({
        'question': data['Question'],
        'answer1': data['Answer1'],
        'answer2': data['Answer2']
    })

def run_prepare_data(input_path, output_path, template):
    """
    步骤 1: 准备数据。
    """
    data = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            data = list(reader)
    except FileNotFoundError:
        print(f"错误: 原始输入文件未找到 {input_path}")
        return False
    
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
    return True

def extract_choice(model_answer):
    """
    从模型的文本输出中解析 [Final Choice]。
    """
    match = re.search(r'\[Final Choice\]:\s*(Answer[12])', model_answer, re.IGNORECASE)
    if match:
        return match.group(1)
    model_answer_cleaned = model_answer.strip().strip(".'\"")
    if model_answer_cleaned == "Answer1":
         return "Answer1"
    if model_answer_cleaned == "Answer2":
         return "Answer2"
    return "Unknown"

def run_langchain_datagen_conventional(lgpt_instance, input_path, output_path, max_workers, method_name, temp):
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

    if not items_to_process:
        return

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, item): item for item in items_to_process}
            for future in tqdm(futures, total=len(items_to_process), desc=f"运行 ({method_name}, T={temp})"):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(f"处理项目时出错: {futures[future]['id']}. 错误: {e}")

def run_langchain_datagen_agent(lgpt_instance, input_path, output_path, max_workers, num_runs):
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
        agent_choice = vote_counter.most_common(1)[0][0] if vote_counter else "Unknown"

        item["model_answer"] = agent_choice
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

    if not items_to_process:
        return

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item_agent, item): item for item in items_to_process}
            for future in tqdm(futures, total=len(items_to_process), desc=f"运行 (Agent, N={num_runs})"):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(f"处理项目时出错: {futures[future]['id']}. 错误: {e}")

def run_score_result_conventional(input_path, score_path, method_name, temp):
    """
    步骤 3 (常规): 在评分时解析 'model_answer'。
    """
    items = []
    try:
        with jsonlines.open(input_path, "r") as reader:
            items = list(reader)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到 {input_path}")
        return None

    correct = 0
    total = len(items)
    model_answer_choices = []

    for item in items:
        model_ans_text = item.get('model_answer', '')
        choice = extract_choice(model_ans_text)
        model_answer_choices.append(choice)
        if choice == item['groundtruth']:
            correct += 1

    accuracy_str = f"{correct/total*100:.2f}%" if total > 0 else "N/A"
    
    score_info = {
        'method': method_name,
        'parameter': 'temperature',
        'value': temp,
        'correct': correct,
        'total': total,
        'accuracy': accuracy_str,
    }
    
    with open(score_path, 'w', encoding='utf-8') as fscore:
        json.dump(score_info, fscore, ensure_ascii=False, indent=4)
    
    print(f"  -> ({method_name}, T={temp}): Accuracy = {accuracy_str} ({correct}/{total})")
    return score_info

def run_score_result_agent(input_path, score_path, num_runs):
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
    total = len(items)
    preference_answer = []

    for item in items:
        choice = item.get('model_answer', 'Unknown')
        preference_answer.append(choice)
        if choice == item['groundtruth']:
            correct += 1

    accuracy_str = f"{correct/total*100:.2f}%" if total > 0 else "N/A"

    score_info = {
        'method': 'agent',
        'parameter': 'n_runs',
        'value': num_runs,
        'correct': correct,
        'total': total,
        'accuracy': accuracy_str,
    }
    
    with open(score_path, 'w', encoding='utf-8') as fscore:
        json.dump(score_info, fscore, ensure_ascii=False, indent=4)

    print(f"  -> (Agent, N={num_runs}): Accuracy = {accuracy_str} ({correct}/{total})")
    return score_info

# ==============================================================================
# 4. 主执行函数 (MAIN ORCHESTRATOR)
# ==============================================================================

def main(args):
    """
    主函数，协调敏感度分析。
    """
    
    # --- 设置环境 ---
    os.environ["DEEPSEEK_BASE_URL"] = args.base_url
    RESULTS_DIR = args.results_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 检查 API 密钥
    if not os.path.exists(args.keys_path) or os.path.getsize(args.keys_path) == 0:
         print(f"错误: 找不到 API 密钥文件 '{args.keys_path}' 或文件为空。")
         return
         
    # 检查输入数据
    if not os.path.exists(args.input_file):
        print(f"错误: 找不到输入数据文件 '{args.input_file}'")
        return

    all_analysis_results = []
    
    # --- 1. 准备所有方法的输入数据 (只需执行一次) ---
    print("\n[Phase 1/3] 准备所有方法的输入数据...")
    prepared_files = {}
    for method_name, template in PROMPT_TEMPLATES.items():
        prepared_file = os.path.join(RESULTS_DIR, f"2.prepared_{method_name}.jsonl")
        print(f"准备: {method_name}...")
        if run_prepare_data(args.input_file, prepared_file, template):
            prepared_files[method_name] = prepared_file
        else:
            print(f"数据准备失败 for {method_name}。")
    print("[Phase 1/3] 数据准备完成。")

    # --- 2. 分析 1: Temperature 敏感度 (常规方法) ---
    print(f"\n[Phase 2/3] 开始分析 Temperature 敏感度 (T = {args.temp_range})...")
    
    # 定义要测试的常规方法
    conventional_methods = ["zero_shot", "few_shot", "cot", "tot", "gen_knowledge"]
    
    for method_name in conventional_methods:
        if method_name not in prepared_files:
            print(f"跳过 {method_name}，因为数据准备失败。")
            continue
        
        print(f"\n--- 分析方法: {method_name.upper()} ---")
        prepared_file = prepared_files[method_name]
        
        for temp in args.temp_range:
            # 动态定义此轮次的文件路径
            results_file = os.path.join(RESULTS_DIR, f"3.results_{method_name}_T{temp}.jsonl")
            score_file = os.path.join(RESULTS_DIR, f"4.score_{method_name}_T{temp}.json")
            
            # 1. 创建一个具有特定 T 的模型实例
            lgpt_instance = LangchainGPT_Instance(
                model_name=args.model_name,
                keys_path=args.keys_path,
                temperature=temp
            )
            
            # 2. 运行模型
            run_langchain_datagen_conventional(
                lgpt_instance, prepared_file, results_file, 
                args.max_workers, method_name, temp
            )
            
            # 3. 评分
            score_data = run_score_result_conventional(
                results_file, score_file, method_name, temp
            )
            if score_data:
                all_analysis_results.append(score_data)

    print("[Phase 2/3] Temperature 敏感度分析完成。")

    # --- 3. 分析 2: N-Runs 敏感度 (Agent 方法) ---
    print(f"\n[Phase 3/3] 开始分析 N-Runs 敏感度 (N = {args.n_run_range})...")
    
    if "agent" not in prepared_files:
        print("跳过 Agent 分析，因为数据准备失败。")
    else:
        print(f"\n--- 分析方法: AGENT (T=2.0 固定) ---")
        prepared_file = prepared_files["agent"]
        
        # 1. 为 Agent 创建 T=2.0 (固定) 的实例
        lgpt_agent_instance = LangchainGPT_Instance(
            model_name=args.model_name,
            keys_path=args.keys_path,
            temperature=2.0  # T=2.0 是 Agent 方法为实现多样性而设计的
        )

        for n in args.n_run_range:
            # 动态定义此轮次的文件路径
            results_file = os.path.join(RESULTS_DIR, f"3.results_agent_N{n}.jsonl")
            score_file = os.path.join(RESULTS_DIR, f"4.score_agent_N{n}.json")
            
            # 2. 运行模型
            run_langchain_datagen_agent(
                lgpt_agent_instance, prepared_file, results_file,
                args.max_workers, num_runs=n
            )
            
            # 3. 评分
            score_data = run_score_result_agent(
                results_file, score_file, num_runs=n
            )
            if score_data:
                all_analysis_results.append(score_data)
                
    print("[Phase 3/3] N-Runs 敏感度分析完成。")

    # --- 最终总结 ---
    print_final_summary(all_analysis_results, RESULTS_DIR)


def print_final_summary(all_scores, results_dir):
    """
    打印并保存最终的敏感度分析总结报告。
    """
    print("\n\n========================================================")
    print("           敏感度分析 (Sensitivity Analysis) 最终总结")
    print("========================================================")
    
    # 1. 打印 Temperature 敏感度
    temp_scores = [s for s in all_scores if s['parameter'] == 'temperature']
    print("\n--- 1. Temperature 敏感度 (常规方法) ---")
    print(f"\n{'Method':<15} | {'Temperature':<12} | {'Accuracy':<10} | {'Correct / Total':<15}")
    print("-" * 60)
    
    methods = sorted(list(set(s['method'] for s in temp_scores)))
    for method in methods:
        scores_for_method = sorted([s for s in temp_scores if s['method'] == method], key=lambda x: x['value'])
        for score in scores_for_method:
            print(f"{score['method']:<15} | {score['value']:<12.1f} | {score['accuracy']:<10} | {score['correct']:<3} / {score['total']:<3}")
        print("-" * 60)

    # 2. 打印 N-Runs 敏感度
    n_run_scores = sorted([s for s in all_scores if s['parameter'] == 'n_runs'], key=lambda x: x['value'])
    print("\n--- 2. N-Runs 敏感度 (Agent, T=2.0) ---")
    print(f"\n{'Method':<15} | {'N-Runs (N)':<12} | {'Accuracy':<10} | {'Correct / Total':<15}")
    print("-" * 60)
    
    for score in n_run_scores:
        print(f"{score['method']:<15} | {score['value']:<12} | {score['accuracy']:<10} | {score['correct']:<3} / {score['total']:<3}")
    print("-" * 60)

    # 3. 保存总结文件
    summary_file = os.path.join(results_dir, "_FINAL_SENSITIVITY_SUMMARY.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- 完整总结已保存至: {summary_file} ---")

# ==============================================================================
# 5. 脚本执行入口 (SCRIPT ENTRYPOINT)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 RLHF 评估策略的敏感度分析。")
    
    # --- 路径参数 ---
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./Data/1.rlhf.jsonl",
        help="[必需] 原始 JSONL 数据文件。"
    )
    parser.add_argument(
        "--keys_path",
        type=str,
        default="./gpt3keys.txt",
        help="[必需] 包含 DeepSeek API 密钥的文本文件路径。"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./sensitivity_results",
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
    
    # --- 敏感度分析参数 ---
    parser.add_argument(
        "--temp_range",
        nargs='+',  # 接受一个或多个值
        type=float,
        default=[0.5, 1.0, 1.5],
        help="要测试的 Temperature 列表 (例如: 0.0 0.5 1.0)。"
    )
    parser.add_argument(
        "--n_run_range",
        nargs='+',  # 接受一个或多个值
        type=int,
        default=[1, 3, 5],
        help="[Agent 方法] 要测试的 N-Runs 列表 (例如: 1 3 5 7)。"
    )
    
    args = parser.parse_args()
    
    main(args)