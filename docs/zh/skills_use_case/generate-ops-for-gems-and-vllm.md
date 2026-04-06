# 为 FlagGems 或 vLLM 项目生成 Kernel

## 前提条件

生成 Kernel 之前，请确保您已阅读本节中的前提条件，并完成预安装步骤。

- 使用以下智能体版本：

  - Claude Code：2.1 版及更高版本
  - OpenClaw：2026.3.2 版及更高版本
  - VSCode：需激活 Github Copilot

- 预先从源码安装 FlagGems 或 vLLM。

  - KernelGen Skills 支持 FlagGems，请参见下方*预安装 FlagGems*章节。

  - KernelGen Skills 支持 vLLM，请参见 [vLLM 用户指南](https://docs.vllm.ai/en/latest/getting_started/installation/)。

## 预安装 FlagGems

安装信息请参见 [FlagGems 文档](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#)。

**注意**：安装过程中，请跳过 `pip install -r flag_tree_requirements/requirements_nvidia.txt` 命令，因为该命令涉及 FlagTree 及其依赖项的安装。

## 生成 Kernel

无论是使用 VSCode（配合 Copilot）、Claude Code 还是 OpenClaw，为 FlagGems 或 vLLM 项目生成算子的总体流程都是相似的。这包括连接到 KernelGen 算子开发 MCP 工具包以及加载技能，具体步骤请参考 [KernelGen Skills 用户指南](../skills_user_guide/skills-user-guide.md) 。你只需要在“通用算子生成”章节所述的提示词基础上，额外补充一句 **“将内核集成到 FlagGems 中”**。KernelGen 会自动检测 FlagGems 是否已安装，并将生成的输出文件直接提交到项目的实验目录中。
