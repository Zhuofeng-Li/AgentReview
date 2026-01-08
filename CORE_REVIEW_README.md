# 核心评审逻辑使用指南
# Core Review Logic Usage Guide

本目录包含两个去除UI的核心评审文件:

## 文件说明

### 1. `simple_review.py` - 最简单的接口 ⭐推荐

**最简单的使用方式 - 输入论文PDF，输出评审结果**

#### 命令行使用

```bash
# 基本用法
python simple_review.py paper.pdf

# 保存结果到文件
python simple_review.py paper.pdf output.json
```

#### Python代码使用

```python
from simple_review import review_paper, print_review_summary

# 最简单 - 使用默认配置
result = review_paper("my_paper.pdf")

# 指定审稿人类型
result = review_paper(
    "my_paper.pdf",
    reviewer_types=["malicious", "knowledgeable", "normal"]
)

# 指定AC类型
result = review_paper(
    "my_paper.pdf",
    reviewer_types=["normal", "normal", "normal"],
    ac_type="inclusive"  # 可选: normal, inclusive, conformist, authoritarian
)

# 保存结果
result = review_paper(
    "my_paper.pdf",
    output_file="review_results.json"
)

# 打印摘要
print_review_summary(result)
```

#### 返回结果格式

```python
{
    "paper_id": "review_paper",
    "reviews": [
        {"reviewer": "Reviewer 1", "content": "..."},
        {"reviewer": "Reviewer 2", "content": "..."},
        {"reviewer": "Reviewer 3", "content": "..."}
    ],
    "author_response": "...",
    "discussion": [
        {"speaker": "AC", "content": "...", "turn": 1},
        {"speaker": "Reviewer 1", "content": "...", "turn": 2},
        ...
    ],
    "metareview": "...",
    "decision": "Accept/Reject",
    "full_messages": [...]  # 完整消息历史
}
```

#### 审稿人类型选项

- `"benign"` - 善意审稿人
- `"malicious"` - 恶意审稿人
- `"knowledgeable"` - 有知识的审稿人
- `"unknowledgeable"` - 无知识的审稿人
- `"responsible"` - 负责任的审稿人
- `"irresponsible"` - 不负责任的审稿人
- `"normal"` - 普通审稿人 (默认)

#### AC类型选项

- `"normal"` - 普通AC (默认)
- `"inclusive"` - 包容型AC
- `"conformist"` - 从众型AC
- `"authoritarian"` - 权威型AC

---

### 2. `run_review_core.py` - 完整的类封装

**提供更多控制和灵活性的面向对象接口**

#### 使用示例

```python
from run_review_core import ReviewRunner

# 创建运行器
runner = ReviewRunner(
    paper_pdf_path="my_paper.pdf",
    num_reviewers=3,
    experiment_name="my_experiment"
)

# 详细配置审稿人
reviewer_configs = [
    {
        "is_benign": False,           # 恶意
        "is_knowledgeable": None,     # 普通
        "is_responsible": None,       # 普通
        "backend_type": "openai-chat",
        "temperature": 0.7
    },
    {
        "is_benign": None,
        "is_knowledgeable": False,    # 无知识
        "is_responsible": None
    },
    {
        "is_benign": None,
        "is_knowledgeable": None,
        "is_responsible": True        # 负责
    }
]

# 配置AC
ac_config = {
    "ac_type": "inclusive",
    "temperature": 0.8
}

# 设置竞技场
runner.setup_arena(
    reviewer_configs=reviewer_configs,
    ac_config=ac_config
)

# 方式1: 运行完整评审
results = runner.run_full_review()

# 方式2: 逐步运行
while True:
    result = runner.run_step()
    if result is None or result["terminal"]:
        break
    print(f"Phase: {result['phase_name']}, Player: {result['current_player']}")

# 方式3: 运行到指定阶段
results = runner.run_until_phase(target_phase=3)

# 获取特定玩家的消息
reviewer1_messages = runner.get_messages_by_player("Reviewer 1")

# 获取当前阶段信息
phase_info = runner.get_current_phase_info()

# 保存结果
runner.save_results("output.json", results)
```

#### 评审流程阶段

```python
0: "paper_extraction"          # 论文提取
1: "reviewer_write_reviews"    # 审稿人撰写评审
2: "author_reviewer_discussion" # 作者-审稿人讨论
3: "reviewer_ac_discussion"    # 审稿人-AC讨论
4: "ac_write_metareviews"      # AC撰写元评审
5: "ac_makes_decisions"        # AC做决定
```

---

## 快速开始

### 最简单的方式 (推荐新手)

```bash
# 1. 准备论文PDF
# 2. 运行评审
python simple_review.py my_paper.pdf results.json

# 3. 查看结果
cat results.json
```

### Python脚本方式

```python
# quick_review.py
from simple_review import review_paper, print_review_summary

result = review_paper(
    paper_pdf_path="my_paper.pdf",
    reviewer_types=["malicious", "normal", "knowledgeable"],
    ac_type="inclusive",
    output_file="review_output.json"
)

print_review_summary(result)
```

### 高级控制方式

```python
# advanced_review.py
from run_review_core import ReviewRunner

runner = ReviewRunner(paper_pdf_path="my_paper.pdf")

# 自定义配置
runner.setup_arena(
    reviewer_configs=[
        {"is_benign": False, "is_knowledgeable": True},  # 恶意但有知识
        {"is_benign": True, "is_knowledgeable": False},  # 善意但无知
        {"is_benign": None, "is_responsible": False}     # 不负责
    ],
    ac_config={"ac_type": "authoritarian"}
)

# 逐步运行并记录
for i in range(10):  # 运行10步
    result = runner.run_step()
    if result:
        print(f"Step {i}: {result['phase_name']} - {result['current_player']}")
        if result["terminal"]:
            break
```

---

## 对比两个文件

| 特性 | `simple_review.py` | `run_review_core.py` |
|------|-------------------|---------------------|
| 易用性 | ⭐⭐⭐⭐⭐ 最简单 | ⭐⭐⭐ 需要理解类结构 |
| 灵活性 | ⭐⭐⭐ 基本配置 | ⭐⭐⭐⭐⭐ 完全控制 |
| 控制粒度 | 函数级别 | 类方法级别 |
| 适用场景 | 快速评审 | 复杂实验 |
| 推荐用户 | 新手、快速测试 | 高级用户、研究实验 |

---

## 常见问题

### Q: 如何修改后端模型?

```python
result = review_paper(
    "paper.pdf",
    backend_type="openai-chat",  # 或其他支持的后端
    temperature=0.8
)
```

### Q: 如何获取详细的消息历史?

```python
result = review_paper("paper.pdf")
full_messages = result["full_messages"]

for msg in full_messages:
    print(f"{msg['agent_name']}: {msg['content'][:100]}...")
```

### Q: 如何只运行到某个阶段?

使用 `run_review_core.py`:

```python
runner = ReviewRunner(paper_pdf_path="paper.pdf")
runner.setup_arena()

# 只运行到阶段2 (author_reviewer_discussion)
results = runner.run_until_phase(target_phase=2)
```

### Q: 如何自定义全局提示词?

```python
result = review_paper(
    "paper.pdf",
    global_prompt="请以严格的标准评审这篇论文。"
)
```

---

## 与原始 app.py 的区别

| 原始 app.py | 核心逻辑文件 |
|------------|-------------|
| ✅ Gradio UI | ❌ 无UI |
| ✅ 交互式 | ✅ 命令行/程序化 |
| ✅ 实时显示 | ✅ 批量处理 |
| ❌ 使用限制 (500/天) | ✅ 无限制 |
| ❌ 需要手动点击 | ✅ 自动运行 |
| ✅ 可视化配置 | ✅ 代码配置 |

---

## 下一步

1. 尝试 `simple_review.py` 进行快速评审
2. 使用不同的审稿人配置进行实验
3. 分析评审结果的JSON输出
4. 根据需要使用 `run_review_core.py` 进行更复杂的控制

有问题请参考源代码中的详细注释!
