# Generate a Kernel for FlagGems or vLLM project

## Prerequisites

Before generating a kernel, make sure you read the prerequisites and accomplish the pre-installation steps in this section.

- Use the following agent versions:

  - Claude Code version 2.1 and later

  - OpenClaw version 2026.3.2 and later

  - VSCode with Github Copilot activated

- Preinstall FlagGems or vLLM from source

  - KernelGen Skills support FlagGems, see the next *Preinstall FlagGems* section.

  - KernelGen Skills support vLLM, see [vLLM user guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

## Preinstall FlagGems

For installation information, see [FlagGems Documentation](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#).

**Note**:

During the installation, skip the`pip install -r flag_tree_requirements/requirements_nvidia.txt` command since this command relates to installation of FlagTree and its dependencies.

## Generate a kernel

Using VSCode (and Copilot), Claude Code, or OpenClaw to generate an operator for the FlagGems or vLLM project follows a similar general process (including connecting to the KernelGen Operator Development MCP Toolkit and load skills) in [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md). You only need to add **"Integrate the kernel into FlagGems"** additionally to the prompt documented in the "Generate an operator generally" section. KernelGen automatically detects if FlagGems is installed and submits the output files to the project's experimental directory.
