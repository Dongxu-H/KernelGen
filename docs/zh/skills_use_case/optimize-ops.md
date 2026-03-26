# 优化算子

您可以使用 VS Code（及 Copilot）、Claude Code 或 OpenClaw 在 NVIDIA 上优化算子。

## 步骤

与 AI 智能体对话以优化算子：

1. 从 [FlagOS Skills Github](https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos) 下载并安装 `kernelgen-flagos` 技能。

2. 通过以下任一方式调用 `kernelgen-flagos` 技能：

   - 使用斜杠命令 `/kernelgen-flagos`

   - 在提示词中包含技能名称

   有关技能安装方法，请参阅相应文档：

   - [VS Code 文档](https://code.visualstudio.com/docs/copilot/customization/agent-skills)

   - [Claude Code 文档](https://code.claude.com/docs/en/skills)

   - [OpenClaw 文档](https://docs.openclaw.ai/tools/skills)

3. 与 AI 智能体对话以优化算子，并在提示词中包含算子优化的迭代次数。

   - **典型需求**：算子名称（必填）、任务描述（必填）和优化迭代次数。

   - **需求示例**："**优化 index\_put 算子，优化 5 轮迭代**。"
