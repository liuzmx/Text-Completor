# 文本自动补全

个性化的文本自动补全探索。使用 LLM + RAG + Prompt 的方式构建。

> 🌻 尝试了微调 [GPT-2 ( Smallest Version, 124M )](https://www.modelscope.cn/models/AI-ModelScope/gpt2/)、[Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，短时间内无法提升微调数据集质量和规模，模型训练无法收敛，但思路肯定是正确的，需要后续提升数据集质量后尝试。👉[相关代码](./trial/)。

## 背景和需求

**本项目起源于一个面试小任务，任务是：**

> 假设用户已有一堆文档资料，现在正在写作，给定当前编辑区和光标位置，自动补全光标后的个性化内容，提升写作效率。类似的产品功能可以体验 cursor、copilot 这些代码补全产品，深入思考背后的算法原理，我们任务的场景是纯文本补全。👉[任务描述全文](./docs/task.md)

**需求分析与理解：**

- 资料解析：解析用户已有文档资料，作为个性化生成的基础。
- 文本理解和场景识别：能够理解当前编辑区的文本内容和类型（学术文章、小说、报告），包括但不限于文章主题、语境、风格等。
- 光标定位：识别光标所处位置（如开头、结尾中间），根据位置选取不同补全策略。
- 个性化内容生成：根据用户的历史行为以及当前文本内容，生成个性化的补全内容。
- 用户有独立的资料库，资料库需支持动态更新。

## 方案调研与探索

几种可能的实现方案：

- **前缀匹配**：基于本地知识的前缀匹配：使用本地知识库，对用户输入进行前缀匹配，返回匹配到的内容。如使用 [Trie 树](https://oi-wiki.org/string/trie/)实现。

- **训练文本补全模型**：搜集通用语料，再结合用户本地数据，构建文本补全训练数据，从零开始，训练一个文本补全模型。

- **LLM 微调**：整合用户本地文档、补全历史、以及部分开放语料构建训练数据集，对开源 LLM 进行全量参数微调或 LoRa 等的微调，让模型能学习到用户的补全偏好、以及本地文档中的知识，从而实现个性化补全。

- **LLM + RAG + Prompt**：不做模型的微调，直接使用开源的模型，通过 RAG 构建知识库，然后使用 Prompt 构建文本补全模型。

- **LLM 微调 + RAG + Prompt**：使用通用预料微调 LLM，让模型具有文本补全能力；在微调后的模型上，通过 RAG 构建知识库，然后使用 Prompt 构建文本补全模型。

| 实现方案                | 优势                                         | 缺点                                  |
| ----------------------- | -------------------------------------------- | ------------------------------------- |
| 前缀匹配                | 算法简单，所需资源少                         | 过于简单，只能是文本上的前缀机械匹配  |
| 训练文本补全模型        | 能够学习用户个性化补全偏好                   | 训练成本高，数据准备复杂              |
| LLM 微调                | 模型能够学习到用户本地知识                   | 微调过程可能需要较多计算资源          |
| LLM + RAG + Prompt      | 无需微调，直接使用开源模型，成本低，扩展性强 | 需要构建和维护知识库，Prompt 设计复杂 |
| LLM 微调 + RAG + Prompt | 结合了微调模型的个性化与 RAG 的知识库扩展性  | 需要微调模型，同时维护知识库和 Prompt |

## 实施方案

综合时间限制、现有数据以及模型资源，选择使用 **LLM + RAG + Prompt** 的方案。其中:

- LLM 选用了 GLM 在线模型 [GLM-4-Flash](https://open.bigmodel.cn/dev/activities/free/glm-4-flash)。

- Embedding 模型选用了 [GLM Embedding-3](https://open.bigmodel.cn/dev/api/vector/embedding)。

- 向量检索库选用了 [Milvus Lite](https://milvus.io/docs/zh/quickstart.md)。

- RAG 检索使用了 [pymilvus](https://github.com/milvus-io/pymilvus) 的语义检索能力。

## 技术方案图

![Tech Road](./assets/tech-road.png)

## 文本补全模型代码

核心代码位于[model](./model)目录下，文件结构如下：

```text
.
├── build_knowledge.py  # 构建本地知识库脚本
├── common.py           # 常用函数
├── completion.py       # 文本补全模型
├── config
│   ├── config.yml      # 配置文件
│   └── logging.ini     # 日志配置文件
├── knowledge
│   ├── db              # 本地知识库
│   ├── embedding.py    # Embedding 调用
│   ├── __init__.py
│   └── knowledge.py    # 本地知识库管理
├── logs
├── main.py             # FastAPI服务
├── requirements.txt    # 依赖包
└── start.sh            # 启动脚本
```

模型服务本地启动：

```shell
export OPENAI_API_KEY=<ZHIPUAI_API_KEY>
cd model
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --access-log --log-config config/logging.ini
```

启动后输出：

```shell
2024-12-31 16:03:12,154 [INFO] [22686] [uvicorn.error] server.py-server-83: Started server process [22686]
2024-12-31 16:03:12,154 [INFO] [22686] [uvicorn.error] on.py-on-48: Waiting for application startup.
2024-12-31 16:03:12,155 [INFO] [22686] [uvicorn.error] on.py-on-62: Application startup complete.
2024-12-31 16:03:12,155 [INFO] [22686] [uvicorn.error] server.py-server-215: Uvicorn running on http://0.0.0.0:8600 (Press CTRL+C to quit)
```

调用示例：

```shell
POST /api/v1/completion HTTP/1.1
Host: localhost:8600
Content-Type: application/json
Content-Length: 272

{
    "text": "The Post Office handled a record amount of cash in July with customers either depositing or withdrawing more than £3.7bn. [MASK] The increasing use of the Post Office to handle cash comes as the rate of closure of bank branches shows no sign of slowing"
}
```

响应结果：

```shell
"This surge in cash transactions highlights the growing reliance on the Post Office as a vital financial service provider, especially in light of the ongoing closure of bank branches.. More than 6,000 have shut their doors since 2015, an average of about 50 each month. This trend underscores the need for alternative banking solutions, with the Post Office stepping in to fill the gap for many communities."
```

## 文本补全效果评估

考虑到个性化文本补全的正确结果不唯一，原则上来讲，只要文本补全的文本与用户的上下文组合后整体文本通顺流畅，即可认为文本补全符合预期。为了简化评估设计，考虑使用补全文本与目标文本的语义相似度作为评估依据，具体设计如下：

- **数据选择**：使用 [**bbc_news_alltime**](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime) 数据集近 4 个月（2024 年 8 约至 11 月）的新闻文章作为测试集。

- **知识库构建**：将 bbc_news_alltime 2024 年 9-11 月工具 3 个月的新闻全部作为 **用户本地文件** 写入本地的 Milvus 知识库（即文件：`model/knowledge/db/local.db`）。

- **测试数据集**：分为 2 类，第 1 类是测试集的 2024 年 9-11 月新闻，第 2 类是测试集的 2024 年 8 月新闻。两类数据都采取从句子中随机掩盖 2-5 个词/短语的方式构建。从 2 个测试数据集中（`[evaluate/dataset](./evaluate/dataset/)`）分别取出 1000 条数据，作为第 1 类和第 2 类的测试数据。格式如下：

```json
{
  "original_text": "The products were rolled out at a glossy event where protestors gathered in a designated free speech area across the street, urging executives to ramp up efforts to protect children from dangerous content in the company’s App Store.",
  "input_text": "The products were rolled out at a glossy event where protestors gathered [MASK] street, urging executives to ramp up efforts to protect children from dangerous content in the company’s App Store.",
  "target_text": "in a designated free speech area across the"
}
```

- **评估方法**：使用 [BGE-Reranker-v2-M3](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3) 来评估模型补全的文本与目标文本的语义相似度。

**评估结果：**
