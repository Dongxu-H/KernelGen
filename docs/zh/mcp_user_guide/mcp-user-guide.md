# KernelGen 算子开发 MCP 工具集用户指南

本节介绍如何使用 VSCode（及 Copilot）、Claude Code 和 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集，并进行自动调优 Kernel 和 生成 TLE　算子。

## 前置条件

请预先安装 `pytorch`、`triton` 和 `pytest` 软件包，以支持在不同硬件平台上进行 Kernel 测试。对于华为昇腾（Huawei Ascend）平台，除标准 `pytorch` 软件包外，还需安装 `pytorch_npu`。

```{toctree}
:maxdepth: 2

connect_mcp/connect-mcp.md
mcp_tool/mcp-tool.md
```
