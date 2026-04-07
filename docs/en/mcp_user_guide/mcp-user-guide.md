# KernelGen Operator Development MCP Toolkit User Guide

This section introduces how to use VSCode (and Copilot), Claude Code, and OpenClaw to connect to the KernelGen Operator Development MCP Toolkit, and autotune Kernel and generate TLE operators.

## Prerequisites

Preinstall the `torch`, `triton`, and `pytest` packages to enable Kernel testing across different hardware platforms. For Huawei Ascend, install `torch_npu` in addition to the standard `torch` package.

```{toctree}
:maxdepth: 2

connect_mcp/connect-mcp.md
mcp_tool/mcp-tool.md
```
