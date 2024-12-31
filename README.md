# 文本自动补全

个性化的文本自动补全探索。使用 LLM + RAG + Prompt 的方式构建。

> 🌻 尝试了微调 [GPT-2 ( Smallest Version, 124M )](https://www.modelscope.cn/models/AI-ModelScope/gpt2/)、[Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，短时间内无法提升微调数据集质量和规模，模型训练无法收敛，但思路肯定是正确的，需要后续提升数据集质量后尝试。👉[相关代码](./trial/)。

本项目起源于一个面试小任务，任务是：

> 假设用户已有一堆文档资料，现在正在写作，给定当前编辑区和光标位置，自动补全光标后的个性化内容，提升写作效率。类似的产品功能可以体验 cursor、copilot 这些代码补全产品，深入思考背后的算法原理，我们任务的场景是纯文本补全。👉[任务描述全文](./docs/task.md)
