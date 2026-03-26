# Optimize an operator for FlagGems or vLLM project

You can use either VS Code (and Copilot), Claude Code, or OpenClaw to optimize operators on NVIDIA.

Chat with the AI agent to optimize the operator and include the number of iterations of operator optimization in the prompts.

  - **Typical requirements**: Operator name（mandatory）, task description (mandatory), and optimization iterations.

  - **Requirement example**: "**Optimize the index\_put operator. Optimize 5 iterations**."

## Specialize an operator for FlagGems or vLLM project

You can use either VS Code (and Copilot), Claude Code, or OpenClaw to migrate CUDA-implemented operators to Huawei Ascend. 
Chat with the AI agent to specialize in an operator：

- **Typical requirements**: Operator name（mandatory）and task description (mandatory).
- **Requirement example**: "Migrate the CUDA-implemented operator fused/silu_and_mul.py to the Ascend chip, with the operator file stored in the FlagGems repository, and the directory is _ascend/fused/silu_and_mul.py, ensuring that the accuracy verification passes."
