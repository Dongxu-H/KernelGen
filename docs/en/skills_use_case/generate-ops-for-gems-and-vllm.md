# Generate a Kernel for FlagGems or vLLM project

## Prerequisites

Before generating a kernel, make sure you read the [prerequisites](../mcp_user_guide/prerequisites.md) and accomplish the pre-installation steps in this section.

  Preinstall FlagGems or vLLM from source:

- KernelGen Skills support FlagGems project, see the next *Preinstall FlagGems* section.

- KernelGen Skills support vLLM project, see [vLLM user guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

## Preinstall FlagGems

For installation information, see [FlagGems Documentation](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#). The FlagTree is installed while installing the requirement text file.

## Generate a kernel

Using VSCode (and Copilot), Claude Code, or OpenClaw to generate an operator for the FlagGems or vLLM project follows a similar general process (including connecting to the KernelGen Operator Development MCP Toolkit and load skills) in [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md). You only need to add **"Integrate the kernel into FlagGems"** additionally to the prompt documented in the "Generate an operator generally" section. KernelGen automatically detects if FlagGems is installed and submits the output files to the project's experimental directory.
