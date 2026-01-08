"""
简单评审接口 - 输入论文PDF，输出评审结果
Simple Review Interface - Input paper PDF, output review results

与 app.py 和 demo.ipynb 完全对齐的简化版本

最简单的使用方式:
    from simple_review import review_paper
    results = review_paper("path/to/paper.pdf")
    print(results['reviews'])
    print(results['metareview'])
    print(results['decision'])
"""

import json
import logging
import os
import random
import re
import time
from argparse import Namespace
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from agentreview.environments import PaperReview
from agentreview.paper_review_arena import PaperReviewArena
from agentreview.paper_review_settings import get_experiment_settings
from agentreview.utility.experiment_utils import initialize_players
from agentreview import const
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def review_paper(
    paper_pdf_path: str,
    paper_id: str = "review_paper",
    num_reviewers: int = 1,
    reviewer_types: Optional[List[str]] = None,
    ac_type: str = "BASELINE",
    output_file: Optional[str] = None,
    data_dir: str = "",
    conference: str = "EMNLP2024",
    model_name: str = "gpt-4",
    openai_client_type: str = "openai",
    experiment_name: str = "simple_review",
    stop_after_review: bool = True,
    **kwargs
) -> Dict:
    """
    评审一篇论文 - 与原始代码库完全对齐的简单接口

    默认行为: 1个审稿人,输出评审意见后停止 (最快速的评审方式)

    Args:
        paper_pdf_path: 论文PDF路径
        paper_id: 论文ID (默认 "review_paper")
        num_reviewers: 审稿人数量 (默认1个,推荐使用1个获得最快结果)
        reviewer_types: 审稿人类型列表，可选值:
            - "benign" (善意)
            - "malicious" (恶意)
            - "knowledgeable" (有知识)
            - "unknowledgeable" (无知识)
            - "responsible" (负责)
            - "irresponsible" (不负责)
            - "BASELINE" (默认/普通)
        ac_type: AC类型 ("BASELINE", "inclusive", "conformist", "authoritarian")
        output_file: 输出文件路径 (可选)
        data_dir: 数据目录
        conference: 会议名称
        model_name: 模型名称
        openai_client_type: OpenAI客户端类型 ("openai" 或 "azure_openai")
        experiment_name: 实验名称
        stop_after_review: 是否在审稿人完成评审后停止 (默认True,False则运行完整流程)

    Returns:
        字典包含:
        {
            "paper_id": str,
            "reviews": [{"reviewer": str, "content": str}, ...],
            "author_response": str,  # 如果stop_after_review=True则为空
            "discussion": [{"speaker": str, "content": str}, ...],
            "metareview": str,  # 如果stop_after_review=True则为空
            "decision": str,  # 如果stop_after_review=True则为空
            "full_messages": [...]
        }

    Example:
        # 最简单 - 1个审稿人,只获取评审意见
        result = review_paper("my_paper.pdf")
        print(result['reviews'][0]['content'])

        # 指定审稿人类型
        result = review_paper(
            "my_paper.pdf",
            reviewer_types=["malicious"]
        )

        # 完整流程 - 3个审稿人,运行所有阶段
        result = review_paper(
            paper_pdf_path="my_paper.pdf",
            num_reviewers=3,
            reviewer_types=["malicious", "knowledgeable", "BASELINE"],
            stop_after_review=False,  # 运行完整流程
            output_file="results.json"
        )
    """

    # 默认审稿人类型
    if reviewer_types is None:
        reviewer_types = ["BASELINE"] * num_reviewers
    elif len(reviewer_types) < num_reviewers:
        reviewer_types.extend(["BASELINE"] * (num_reviewers - len(reviewer_types)))

    # 创建参数命名空间 (与demo.ipynb一致)
    args = Namespace(
        openai_client_type=openai_client_type,
        api_version='2023-03-15-preview',
        ac_scoring_method='ranking',
        conference=conference,
        num_reviewers_per_paper=num_reviewers,
        ignore_missing_metareviews=False,
        overwrite=False,
        num_papers_per_area_chair=10,
        model_name=model_name,
        output_dir='outputs',
        max_num_words=16384,
        visual_dir='outputs/visual',
        device='cuda',
        data_dir=data_dir,
        acceptance_rate=0.32,
        skip_logging=kwargs.get('skip_logging', True),
        task='paper_review',
        experiment_name=experiment_name
    )

    # 获取论文决定
    paper_decision = kwargs.get("paper_decision", "Accept")

    # 创建设置字典 (与demo.ipynb一致)
    setting = {
        "AC": [ac_type],
        "reviewer": reviewer_types[:num_reviewers],
        "author": ["BASELINE"],
        "global_settings": {
            "provides_numeric_rating": ['reviewer', 'ac'],
            "persons_aware_of_authors_identities": []
        }
    }

    # 使用现有的helper函数生成实验设置
    experiment_setting = get_experiment_settings(
        paper_id=paper_id,
        paper_decision=paper_decision,
        setting=setting
    )

    logger.info(f"开始评审论文: {paper_pdf_path} (ID: {paper_id})")
    logger.info(f"审稿人配置: {reviewer_types[:num_reviewers]}")
    logger.info(f"AC类型: {ac_type}")

    # 使用现有的helper函数初始化玩家
    # 注意: 需要临时修改paper_pdf_path,因为initialize_players会从data_dir加载
    # 我们通过修改实验设置来传递PDF路径
    original_paper_id = experiment_setting['paper_id']

    # 为PaperExtractorPlayer添加paper_pdf_path
    # 这需要在初始化后手动设置,因为initialize_players不支持自定义PDF路径

    players = initialize_players(experiment_setting=experiment_setting, args=args)

    # 手动更新PaperExtractorPlayer的PDF路径或文本内容
    paper_text_override = kwargs.get('paper_text_override')
    for player in players:
        if player.name == "Paper Extractor":
            if paper_text_override:
                # 如果提供了文本覆盖，直接设置文本内容
                player.paper_text = paper_text_override
                player.paper_pdf_path = None  # 禁用PDF加载
            else:
                player.paper_pdf_path = paper_pdf_path
            break

    player_names = [player.name for player in players]

    # 创建环境 (与demo.ipynb一致)
    env = PaperReview(
        player_names=player_names,
        paper_decision=paper_decision,
        paper_id=paper_id,
        args=args,
        experiment_setting=experiment_setting
    )

    # 创建竞技场 (与demo.ipynb和app.py一致)
    arena = PaperReviewArena(
        players=players,
        environment=env,
        args=args,
        global_prompt=kwargs.get('global_prompt', const.GLOBAL_PROMPT)
    )

    # 运行评审流程 (使用launch_cli或手动step)
    if stop_after_review:
        logger.info("开始运行评审流程... (仅运行到审稿人完成评审)")
    else:
        logger.info("开始运行评审流程... (完整流程)")

    if kwargs.get('use_cli', False):
        # 使用原始的launch_cli方法 (与demo.ipynb一致)
        arena.launch_cli(interactive=False)
        # launch_cli会保存结果到文件,我们需要从文件加载
        all_messages = arena.environment.message_pool.get_all_messages()
    else:
        # 手动运行步骤 (与app.py一致)
        all_messages = []
        step_count = 0

        while True:
            timestep = arena.step()
            if timestep is None:
                break

            all_messages = timestep.observation
            step_count += 1

            phase_info = arena.environment.phases[arena.environment.phase_index]
            current_phase = phase_info['name']
            logger.info(f"Step {step_count} - Phase {arena.environment.phase_index}: {current_phase}")

            # 如果设置了stop_after_review,在审稿人完成评审后停止
            # Phase 0: paper_extraction (论文提取)
            # Phase 1: reviewer_write_reviews (审稿人撰写评审)
            # Phase 2+: 后续阶段
            if stop_after_review and arena.environment.phase_index >= 2:
                logger.info(f"审稿人已完成评审! 共 {step_count} 步")
                logger.info("如需完整流程,请设置 stop_after_review=False")
                break

            if timestep.terminal:
                logger.info(f"评审完成! 共 {step_count} 步")
                break

    # 解析结果
    result = _parse_review_results(all_messages, arena, paper_id)

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")

    return result


def _parse_review_results(messages: List, arena, paper_id: str) -> Dict:
    """解析评审结果 - 提取reviews, author_response, metareview, decision"""

    reviews = []
    author_responses = []
    discussion = []
    metareview = ""
    decision = ""

    # 按照玩家提取消息
    for player in arena.players:
        player_messages = arena.environment.get_messages_from_player(player.name)

        if player.name.startswith("Reviewer"):
            # 提取审稿人的评审
            for msg in player_messages:
                reviews.append({
                    "reviewer": player.name,
                    "content": msg.content,
                    "turn": msg.turn
                })

        elif player.name == "Author":
            # 提取作者回应
            for msg in player_messages:
                author_responses.append({
                    "content": msg.content,
                    "turn": msg.turn
                })

        elif player.name == "AC":
            # 提取AC的消息
            ac_messages = player_messages
            if len(ac_messages) >= 2:
                # 倒数第二个通常是metareview
                metareview = ac_messages[-2].content if len(ac_messages) >= 2 else ""
                # 最后一个通常是decision
                decision = ac_messages[-1].content if len(ac_messages) >= 1 else ""
            elif len(ac_messages) == 1:
                metareview = ac_messages[0].content

    # 提取完整讨论
    for msg in messages:
        if msg.agent_name not in ["Paper Extractor"]:
            discussion.append({
                "speaker": msg.agent_name,
                "content": msg.content,
                "turn": msg.turn
            })

    # Extract raw_text from first review
    raw_text = reviews[0]["content"] if reviews else ""

    return {
        "paper_id": paper_id,
        "raw_text": raw_text,
        "reviews": reviews,
        "author_responses": author_responses,
        "author_response": author_responses[0]["content"] if author_responses else "",
        "discussion": discussion,
        "metareview": metareview,
        "decision": decision,
        "full_messages": [
            {
                "agent_name": msg.agent_name,
                "content": msg.content,
                "turn": msg.turn,
                "visible_to": str(msg.visible_to)
            }
            for msg in messages
        ]
    }


def review_paper_text(
    paper_text: str,
    paper_id: str = "review_paper",
    num_reviewers: int = 1,
    reviewer_types: Optional[List[str]] = None,
    ac_type: str = "BASELINE",
    conference: str = "EMNLP2024",
    model_name: str = "gpt-4",
    openai_client_type: str = "openai",
    experiment_name: str = "simple_review",
    stop_after_review: bool = True,
    **kwargs
) -> Dict:
    """
    评审论文文本 - 直接使用文本内容而非PDF文件

    这个函数专门用于处理已经提取好的论文文本，例如从DeepReview-13K数据集

    Args:
        paper_text: 论文文本内容
        其他参数与 review_paper 相同

    Returns:
        与 review_paper 相同的结果字典
    """
    # 创建临时PDF路径（实际不会使用，但需要满足接口要求）
    # 我们将直接注入paper_text到环境中
    return review_paper(
        paper_pdf_path="",  # 空路径，将被覆盖
        paper_id=paper_id,
        num_reviewers=num_reviewers,
        reviewer_types=reviewer_types,
        ac_type=ac_type,
        conference=conference,
        model_name=model_name,
        openai_client_type=openai_client_type,
        experiment_name=experiment_name,
        stop_after_review=stop_after_review,
        paper_text_override=paper_text,  # 新参数
        **kwargs
    )


def batch_review_papers(
    paper_texts: List[str],
    paper_ids: Optional[List[str]] = None,
    num_reviewers: int = 1,
    reviewer_types: Optional[List[str]] = None,
    ac_type: str = "BASELINE",
    model_name: str = "gpt-4",
    max_workers: int = 5,
    output_file: Optional[str] = None,
    stop_after_review: bool = True,
    **kwargs
) -> List[Dict]:
    """
    批量评审多篇论文（并行处理）

    Args:
        paper_texts: 论文文本列表
        paper_ids: 论文ID列表（可选，默认使用索引）
        num_reviewers: 每篇论文的审稿人数量
        reviewer_types: 审稿人类型列表
        ac_type: AC类型
        model_name: 模型名称
        max_workers: 最大并行工作线程数
        output_file: 输出文件路径（可选）
        stop_after_review: 是否在审稿人完成评审后停止
        **kwargs: 其他参数传递给 review_paper

    Returns:
        评审结果列表
    """
    if paper_ids is None:
        paper_ids = [f"paper_{i}" for i in range(len(paper_texts))]

    if len(paper_ids) != len(paper_texts):
        raise ValueError("paper_ids length must match paper_texts length")

    logger.info(f"开始批量评审 {len(paper_texts)} 篇论文...")
    logger.info(f"使用 {max_workers} 个并行工作线程")

    # Thread-safe counter and lock for progress tracking
    completed_count = 0
    lock = Lock()

    def process_paper(idx, paper_text, paper_id):
        """处理单篇论文并返回评审结果"""
        nonlocal completed_count
        try:
            result = review_paper_text(
                paper_text=paper_text,
                paper_id=paper_id,
                num_reviewers=num_reviewers,
                reviewer_types=reviewer_types,
                ac_type=ac_type,
                model_name=model_name,
                stop_after_review=stop_after_review,
                **kwargs
            )
            with lock:
                completed_count += 1
                logger.info(f"完成论文 {completed_count}/{len(paper_texts)} (ID: {paper_id})")
            return idx, result
        except Exception as e:
            with lock:
                completed_count += 1
                logger.error(f"处理论文 {paper_id} 时出错: {e}")
            return idx, {"error": str(e), "paper_id": paper_id}

    # 记录开始时间
    start_time = time.time()

    # 预分配结果列表以保持顺序
    results = [None] * len(paper_texts)

    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_paper, idx, paper_text, paper_id): idx
            for idx, (paper_text, paper_id) in enumerate(zip(paper_texts, paper_ids))
        }

        # 收集完成的结果
        for future in as_completed(future_to_idx):
            idx, result = future.result()
            results[idx] = result

    # 计算总用时
    elapsed_time = time.time() - start_time
    logger.info(f"\n批量评审完成！总用时: {elapsed_time:.2f} 秒")
    logger.info(f"平均每篇论文用时: {elapsed_time/len(paper_texts):.2f} 秒")

    successful_count = len([r for r in results if r and 'error' not in r])
    logger.info(f"成功生成 {successful_count}/{len(paper_texts)} 篇评审")

    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")

    return results


def batch_review_from_dataset(
    dataset_name: str = "WestlakeNLP/DeepReview-13K",
    split: str = "test",
    num_samples: int = 100,
    num_reviewers: int = 1,
    reviewer_types: Optional[List[str]] = None,
    ac_type: str = "BASELINE",
    model_name: str = "gpt-4",
    max_workers: int = 5,
    output_dir: str = "evaluate/review",
    stop_after_review: bool = True,
    **kwargs
) -> tuple:
    """
    从DeepReview-13K数据集批量评审论文

    Args:
        dataset_name: 数据集名称
        split: 数据集分割（train/test/validation）
        num_samples: 评审的样本数量（-1表示全部）
        num_reviewers: 每篇论文的审稿人数量
        reviewer_types: 审稿人类型列表
        ac_type: AC类型
        model_name: 模型名称
        max_workers: 最大并行工作线程数
        output_dir: 输出目录
        stop_after_review: 是否在审稿人完成评审后停止
        **kwargs: 其他参数

    Returns:
        (dataset, paper_texts, review_results, output_data) 元组
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")

    logger.info(f"加载数据集 {dataset_name} ({split} split)...")
    ds = load_dataset(dataset_name, split=split)

    # 提取论文文本
    logger.info("提取论文文本...")

    def extract_paper_text(example):
        """从inputs中提取论文内容"""
        inputs = example['inputs']
        paper_text = ""

        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = []

        if isinstance(inputs, list):
            for message in inputs:
                if isinstance(message, dict) and message.get('role') == 'user':
                    paper_text = message.get('content', '')
                    break

        example['paper_text'] = paper_text
        return example

    ds = ds.map(extract_paper_text)

    # 收集所有论文文本
    paper_texts = list(ds['paper_text'])

    # 采样
    if num_samples == -1:
        logger.info(f"使用全部 {len(paper_texts)} 篇论文")
        sampled_indices = list(range(len(paper_texts)))
    elif num_samples > 0 and len(paper_texts) > num_samples:
        sampled_indices = random.sample(range(len(paper_texts)), num_samples)
        sampled_indices.sort()
        paper_texts = [paper_texts[i] for i in sampled_indices]
        ds = ds.select(sampled_indices)
        logger.info(f"随机采样 {len(paper_texts)} 篇论文 (seed: {RANDOM_SEED})")
    else:
        logger.info(f"使用全部 {len(paper_texts)} 篇论文")
        sampled_indices = list(range(len(paper_texts)))

    # 生成论文ID
    paper_ids = [ds[i]['id'] for i in range(len(paper_texts))]

    # 批量评审
    review_results = batch_review_papers(
        paper_texts=paper_texts,
        paper_ids=paper_ids,
        num_reviewers=num_reviewers,
        reviewer_types=reviewer_types,
        ac_type=ac_type,
        model_name=model_name,
        max_workers=max_workers,
        stop_after_review=stop_after_review,
        **kwargs
    )

    # 准备输出数据
    logger.info("准备输出数据...")

    def extract_boxed_review(text):
        """提取 \\boxed_review{ } 中的内容"""
        if not text:
            return text
        pattern = r'\\boxed_review\{(.*?)\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    output_data = []
    for i in range(len(paper_texts)):
        try:
            outputs = json.loads(ds[i]['outputs'])
            golden_review = extract_boxed_review(outputs[2]['content']) if len(outputs) > 2 else ""
        except:
            golden_review = ""

        entry = {
            'id': ds[i]['id'],
            'title': ds[i].get('title', ''),
            'paper_context': ds[i]['paper_text'],
            'decision': ds[i].get('decision', ''),
            'human_review': ds[i].get('reviewer_comments', ''),
            'golden_review': golden_review,
            'model_prediction': review_results[i],
        }
        output_data.append(entry)

    # 保存结果
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = model_name.replace('/', '-').replace('.', '-')
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/agentreview_{model_filename}_sample_{len(paper_texts)}_{timestamp}.json'

    logger.info(f"保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"结果保存成功！总条目数: {len(output_data)}")

    return ds, paper_texts, review_results, output_data


def print_review_summary(result: Dict):
    """打印评审摘要"""
    print("\n" + "=" * 80)
    print(f"论文ID / Paper ID: {result['paper_id']}")
    print("=" * 80)

    print("\n【审稿意见 / Reviews】")
    print("-" * 80)
    for review in result['reviews']:
        print(f"\n{review['reviewer']} (Turn {review['turn']}):")
        content = review['content']
        print(content[:500] + "..." if len(content) > 500 else content)

    if result['author_responses']:
        print("\n【作者回应 / Author Responses】")
        print("-" * 80)
        for i, resp in enumerate(result['author_responses'], 1):
            content = resp['content']
            print(f"\nResponse {i} (Turn {resp['turn']}):")
            print(content[:500] + "..." if len(content) > 500 else content)

    if result['metareview']:
        print("\n【元评审 / Meta-Review】")
        print("-" * 80)
        print(result['metareview'][:500] + "..." if len(result['metareview']) > 500 else result['metareview'])

    if result['decision']:
        print("\n【最终决定 / Final Decision】")
        print("-" * 80)
        print(result['decision'])

    print("\n" + "=" * 80)
    print(f"总消息数 / Total messages: {len(result['full_messages'])}")
    print("=" * 80 + "\n")


def main():
    """命令行使用示例"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="论文评审工具 - 支持单篇PDF评审和批量数据集评审",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:

  # 单篇PDF评审
  python simple_review.py paper.pdf
  python simple_review.py paper.pdf --output results.json

  # 批量评审DeepReview-13K数据集
  python simple_review.py --batch --num-samples 10 --max-workers 5
  python simple_review.py --batch --num-samples -1 --model-name gpt-4o

  # 高级配置
  python simple_review.py paper.pdf --num-reviewers 3 --stop-after-review False
        """
    )

    # 模式选择
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批量评审模式（从DeepReview-13K数据集）'
    )

    # 单篇评审参数
    parser.add_argument(
        'paper_path',
        nargs='?',
        help='论文PDF路径（单篇评审模式）'
    )

    # 通用参数
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt-4o',
        help='模型名称 (默认: gpt-4o)'
    )
    parser.add_argument(
        '--num-reviewers',
        type=int,
        default=1,
        help='审稿人数量 (默认: 1)'
    )
    parser.add_argument(
        '--reviewer-types',
        type=str,
        nargs='+',
        help='审稿人类型列表 (例如: benign malicious knowledgeable)'
    )
    parser.add_argument(
        '--ac-type',
        type=str,
        default='BASELINE',
        help='AC类型 (默认: BASELINE)'
    )
    parser.add_argument(
        '--stop-after-review',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='是否在审稿人完成评审后停止 (默认: True)'
    )

    # 批量评审参数
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='批量评审的样本数量，-1表示全部 (默认: 100)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='并行工作线程数 (默认: 5)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='WestlakeNLP/DeepReview-13K',
        help='数据集名称 (默认: WestlakeNLP/DeepReview-13K)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='数据集分割 (默认: test)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluate/review',
        help='批量评审输出目录 (默认: evaluate/review)'
    )

    args = parser.parse_args()

    # 检查API密钥
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("警告: 未设置 OPENAI_API_KEY 环境变量")
        logger.warning("Warning: OPENAI_API_KEY environment variable not set")

    if args.batch:
        # 批量评审模式
        logger.info("=" * 80)
        logger.info("批量评审模式 - Batch Review Mode")
        logger.info("=" * 80)

        dataset, papers, results, output_data = batch_review_from_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            num_samples=args.num_samples,
            num_reviewers=args.num_reviewers,
            reviewer_types=args.reviewer_types,
            ac_type=args.ac_type,
            model_name=args.model_name,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            stop_after_review=args.stop_after_review
        )

        logger.info("\n批量评审完成!")
        logger.info(f"处理论文数: {len(papers)}")
        logger.info(f"成功评审数: {len([r for r in results if r and 'error' not in r])}")

    else:
        # 单篇PDF评审模式
        if not args.paper_path:
            parser.print_help()
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("单篇评审模式 - Single Paper Review Mode")
        logger.info("=" * 80)

        result = review_paper(
            paper_pdf_path=args.paper_path,
            paper_id=os.path.basename(args.paper_path).replace('.pdf', ''),
            num_reviewers=args.num_reviewers,
            reviewer_types=args.reviewer_types or ["BASELINE"],
            ac_type=args.ac_type,
            output_file=args.output,
            conference="EMNLP2024",
            model_name=args.model_name,
            stop_after_review=args.stop_after_review
        )

        # 打印摘要
        print_review_summary(result)


if __name__ == "__main__":
    main()

"""
使用示例 / Usage Examples:

1. 单篇PDF评审 / Single PDF Review:
   python simple_review.py paper.pdf
   python simple_review.py paper.pdf --output results.json
   python simple_review.py paper.pdf --num-reviewers 3 --model-name gpt-4o

2. 批量评审 / Batch Review from Dataset:
   # 评审10篇论文
   python simple_review.py --batch --num-samples 10 --max-workers 5

   # 评审全部论文
   python simple_review.py --batch --num-samples -1 --max-workers 10

   # 使用不同模型
   python simple_review.py --batch --num-samples 100 --model-name gpt-4o --max-workers 8

   # 完整流程（包含作者回应、讨论、决策）
   python simple_review.py --batch --num-samples 50 --stop-after-review False

3. Python API 使用 / Python API Usage:

   # 评审单篇PDF
   from simple_review import review_paper
   result = review_paper("my_paper.pdf")
   print(result['reviews'])

   # 评审论文文本（不需要PDF）
   from simple_review import review_paper_text
   result = review_paper_text(
       paper_text="Your paper content here...",
       paper_id="my_paper"
   )

   # 批量评审论文文本
   from simple_review import batch_review_papers
   results = batch_review_papers(
       paper_texts=["paper1 content...", "paper2 content..."],
       paper_ids=["paper1", "paper2"],
       max_workers=5
   )

   # 从数据集批量评审
   from simple_review import batch_review_from_dataset
   ds, texts, results, data = batch_review_from_dataset(
       dataset_name="WestlakeNLP/DeepReview-13K",
       split="test",
       num_samples=100,
       max_workers=5
   )
"""
