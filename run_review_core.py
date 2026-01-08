"""
核心评审逻辑 - 去除所有UI组件
Core Review Logic - Without UI Components

此文件提取了论文评审系统的核心逻辑，可以通过命令行或程序化方式运行。
This file extracts the core logic of the paper review system for command-line or programmatic use.
"""

import json
import logging
from argparse import Namespace
from typing import List, Dict, Optional

from agentreview.config import AgentConfig
from agentreview.agent import Player
from agentreview.backends import BACKEND_REGISTRY
from agentreview.environments import PaperReview
from agentreview.paper_review_arena import PaperReviewArena
from agentreview.paper_review_player import PaperExtractorPlayer, AreaChair, Reviewer
from agentreview.role_descriptions import (
    get_reviewer_description,
    get_ac_description,
    get_author_config,
    get_paper_extractor_config,
    get_author_description
)

logger = logging.getLogger(__name__)


class ReviewRunner:
    """论文评审运行器 - 核心逻辑封装"""

    def __init__(
        self,
        paper_pdf_path: str,
        num_reviewers: int = 3,
        conference: str = "EMNLP2024",
        paper_decision: str = "Accept",
        paper_id: str = "12345",
        data_dir: str = "",
        experiment_name: str = "test",
        openai_client_type: str = "azure_openai",
        max_num_words: int = 16384,
        global_prompt: Optional[str] = None
    ):
        """
        初始化评审运行器

        Args:
            paper_pdf_path: 论文PDF文件路径
            num_reviewers: 审稿人数量 (默认3)
            conference: 会议名称
            paper_decision: 论文决定
            paper_id: 论文ID
            data_dir: 数据目录
            experiment_name: 实验名称
            openai_client_type: OpenAI客户端类型
            max_num_words: 最大词数
            global_prompt: 全局提示词
        """
        self.paper_pdf_path = paper_pdf_path
        self.num_reviewers = num_reviewers
        self.conference = conference
        self.paper_decision = paper_decision
        self.paper_id = paper_id
        self.data_dir = data_dir
        self.global_prompt = global_prompt or ""

        # 创建参数命名空间
        self.args = Namespace(
            openai_client_type=openai_client_type,
            experiment_name=experiment_name,
            max_num_words=max_num_words,
            num_reviewers_per_paper=num_reviewers,
            conference=conference,
            model_name="gpt-4"  # 默认模型
        )

        self.arena = None
        self.experiment_setting = None

    def create_reviewer(
        self,
        reviewer_index: int,
        is_benign: Optional[bool] = None,
        is_knowledgeable: Optional[bool] = None,
        is_responsible: Optional[bool] = None,
        knows_authors: str = "unfamous",
        backend_type: str = "openai-chat",
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> Reviewer:
        """
        创建审稿人

        Args:
            reviewer_index: 审稿人索引 (从1开始)
            is_benign: 是否善意 (True=善意, False=恶意, None=普通)
            is_knowledgeable: 是否有知识 (True=有知识, False=无知识, None=普通)
            is_responsible: 是否负责 (True=负责, False=不负责, None=普通)
            knows_authors: 是否认识作者
            backend_type: 后端类型
            temperature: 温度参数
            max_tokens: 最大token数
        """
        role_name = f"Reviewer {reviewer_index}"
        role_desc = get_reviewer_description(is_benign, is_knowledgeable, is_responsible)

        player_config = AgentConfig(
            name=role_name,
            role_desc=role_desc,
            global_prompt=self.global_prompt,
            backend={
                "backend_type": backend_type,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        reviewer = Reviewer(
            data_dir=self.data_dir,
            conference=self.conference,
            args=self.args,
            **player_config
        )

        return reviewer

    def create_area_chair(
        self,
        ac_type: str = "BASELINE",
        backend_type: str = "openai-chat",
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> AreaChair:
        """
        创建领域主席

        Args:
            ac_type: AC类型 (BASELINE/inclusive/conformist/authoritarian)
            backend_type: 后端类型
            temperature: 温度参数
            max_tokens: 最大token数
        """
        role_name = "AC"
        role_desc = get_ac_description(ac_type, "ac_write_metareviews", "None", 1)

        player_config = AgentConfig(
            name=role_name,
            role_desc=role_desc,
            global_prompt=self.global_prompt,
            backend={
                "backend_type": backend_type,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            env_type="paper_review"
        )

        ac = AreaChair(
            data_dir=self.data_dir,
            conference=self.conference,
            args=self.args,
            **player_config
        )

        return ac

    def create_author(
        self,
        backend_type: str = "openai-chat",
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> Player:
        """创建作者"""
        author_config = get_author_config()
        author_config["backend"] = {
            "backend_type": backend_type,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        author = Player(
            data_dir=self.data_dir,
            conference=self.conference,
            args=self.args,
            **author_config
        )

        return author

    def create_paper_extractor(self) -> PaperExtractorPlayer:
        """创建论文提取器"""
        paper_extractor_config = get_paper_extractor_config(max_tokens=2048)

        paper_extractor = PaperExtractorPlayer(
            paper_pdf_path=self.paper_pdf_path,
            data_dir=self.data_dir,
            paper_id=self.paper_id,
            paper_decision=self.paper_decision,
            args=self.args,
            conference=self.conference,
            **paper_extractor_config
        )

        return paper_extractor

    def setup_arena(
        self,
        reviewer_configs: Optional[List[Dict]] = None,
        ac_config: Optional[Dict] = None,
        author_config: Optional[Dict] = None
    ):
        """
        设置竞技场

        Args:
            reviewer_configs: 审稿人配置列表，每个元素是一个字典包含:
                - is_benign: bool
                - is_knowledgeable: bool
                - is_responsible: bool
                - knows_authors: str
                - backend_type: str (可选)
                - temperature: float (可选)
                - max_tokens: int (可选)
            ac_config: AC配置字典
            author_config: 作者配置字典
        """
        # 默认配置
        if reviewer_configs is None:
            reviewer_configs = [
                {"is_benign": None, "is_knowledgeable": None, "is_responsible": None, "knows_authors": "unfamous"}
                for _ in range(self.num_reviewers)
            ]

        if ac_config is None:
            ac_config = {"ac_type": "BASELINE"}

        if author_config is None:
            author_config = {}

        # 创建实验设置
        self.experiment_setting = {
            "paper_id": self.paper_id,
            "paper_decision": self.paper_decision,
            "players": {
                "Paper Extractor": [{}],
                "AC": [{"area_chair_type": ac_config.get("ac_type", "BASELINE")}],
                "Author": [{}],
                "Reviewer": [
                    {
                        "is_benign": config.get("is_benign"),
                        "is_knowledgeable": config.get("is_knowledgeable"),
                        "is_responsible": config.get("is_responsible"),
                        "knows_authors": config.get("knows_authors", "unfamous")
                    }
                    for config in reviewer_configs
                ],
            }
        }

        # 创建所有玩家
        players = []

        # 创建审稿人
        for i, config in enumerate(reviewer_configs, start=1):
            reviewer = self.create_reviewer(
                reviewer_index=i,
                is_benign=config.get("is_benign"),
                is_knowledgeable=config.get("is_knowledgeable"),
                is_responsible=config.get("is_responsible"),
                knows_authors=config.get("knows_authors", "unfamous"),
                backend_type=config.get("backend_type", "openai-chat"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 200)
            )
            players.append(reviewer)

        # 创建领域主席
        ac = self.create_area_chair(
            ac_type=ac_config.get("ac_type", "BASELINE"),
            backend_type=ac_config.get("backend_type", "openai-chat"),
            temperature=ac_config.get("temperature", 0.7),
            max_tokens=ac_config.get("max_tokens", 200)
        )
        players.append(ac)

        # 创建论文提取器
        paper_extractor = self.create_paper_extractor()
        players.append(paper_extractor)

        # 创建作者
        author = self.create_author(
            backend_type=author_config.get("backend_type", "openai-chat"),
            temperature=author_config.get("temperature", 0.7),
            max_tokens=author_config.get("max_tokens", 200)
        )
        players.append(author)

        # 获取玩家名称
        player_names = [player.name for player in players]

        # 创建环境
        env = PaperReview(
            player_names=player_names,
            paper_decision=self.paper_decision,
            paper_id=self.paper_id,
            args=self.args,
            experiment_setting=self.experiment_setting
        )

        # 创建竞技场
        self.arena = PaperReviewArena(
            players=players,
            environment=env,
            args=self.args,
            global_prompt=self.global_prompt
        )

        logger.info(f"Arena created with {len(players)} players")
        return self.arena

    def run_step(self) -> Optional[Dict]:
        """
        运行一步

        Returns:
            包含以下信息的字典:
            - phase_index: 当前阶段索引
            - phase_name: 当前阶段名称
            - current_player: 当前玩家名称
            - all_messages: 所有消息
            - terminal: 是否结束
        """
        if self.arena is None:
            raise ValueError("Arena not initialized. Call setup_arena() first.")

        timestep = self.arena.step()

        if timestep is None:
            return None

        phase_index = self.arena.environment.phase_index
        phase_name = self.arena.environment.phases[phase_index]["name"]

        result = {
            "phase_index": phase_index,
            "phase_name": phase_name,
            "current_player": self.arena.environment.get_next_player(),
            "all_messages": timestep.observation,
            "terminal": timestep.terminal
        }

        logger.info(f"Phase {phase_index} ({phase_name}) - Player: {result['current_player']}")

        return result

    def run_until_phase(self, target_phase: int) -> List[Dict]:
        """
        运行到指定阶段

        Args:
            target_phase: 目标阶段索引

        Returns:
            所有步骤的结果列表
        """
        results = []

        while True:
            result = self.run_step()
            if result is None:
                break

            results.append(result)

            if result["terminal"] or result["phase_index"] >= target_phase:
                break

        return results

    def run_full_review(self) -> List[Dict]:
        """
        运行完整的评审流程

        Returns:
            所有步骤的结果列表
        """
        results = []

        while True:
            result = self.run_step()
            if result is None:
                break

            results.append(result)

            if result["terminal"]:
                break

        logger.info(f"Review completed with {len(results)} steps")
        return results

    def get_messages_by_player(self, player_name: str) -> List:
        """获取特定玩家的消息"""
        if self.arena is None:
            raise ValueError("Arena not initialized.")

        return self.arena.environment.get_messages_from_player(player_name)

    def get_current_phase_info(self) -> Dict:
        """获取当前阶段信息"""
        if self.arena is None:
            raise ValueError("Arena not initialized.")

        phase_index = self.arena.environment.phase_index
        phase = self.arena.environment.phases[phase_index]

        return {
            "phase_index": phase_index,
            "phase_name": phase["name"],
            "speaking_order": phase["speaking_order"],
            "next_player": self.arena.environment.get_next_player()
        }

    def save_results(self, filepath: str, results: List[Dict]):
        """保存结果到JSON文件"""
        output = {
            "paper_id": self.paper_id,
            "experiment_setting": self.experiment_setting,
            "results": [
                {
                    "phase_index": r["phase_index"],
                    "phase_name": r["phase_name"],
                    "current_player": r["current_player"],
                    "messages": [
                        {
                            "agent_name": msg.agent_name,
                            "content": msg.content,
                            "turn": msg.turn,
                            "visible_to": str(msg.visible_to)
                        }
                        for msg in r["all_messages"]
                    ]
                }
                for r in results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {filepath}")


def main():
    """示例用法"""
    import sys

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python run_review_core.py <paper_pdf_path>")
        print("Usage: python run_review_core.py <paper_pdf_path>")
        sys.exit(1)

    paper_pdf_path = sys.argv[1]

    # 创建运行器
    runner = ReviewRunner(
        paper_pdf_path=paper_pdf_path,
        num_reviewers=3,
        experiment_name="core_review_test"
    )

    # 配置审稿人
    # 例如: 第一个审稿人是恶意的，第二个是无知的，第三个是普通的
    reviewer_configs = [
        {"is_benign": False, "is_knowledgeable": None, "is_responsible": None},  # 恶意审稿人
        {"is_benign": None, "is_knowledgeable": False, "is_responsible": None},  # 无知审稿人
        {"is_benign": None, "is_knowledgeable": None, "is_responsible": None},   # 普通审稿人
    ]

    # 设置竞技场
    runner.setup_arena(
        reviewer_configs=reviewer_configs,
        ac_config={"ac_type": "BASELINE"}
    )

    # 运行完整评审
    print("开始运行评审流程...")
    print("Starting review process...")
    results = runner.run_full_review()

    # 保存结果
    output_path = f"review_results_{runner.paper_id}.json"
    runner.save_results(output_path, results)

    print(f"\n评审完成! 共 {len(results)} 步")
    print(f"Review completed! Total {len(results)} steps")
    print(f"结果已保存到: {output_path}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
