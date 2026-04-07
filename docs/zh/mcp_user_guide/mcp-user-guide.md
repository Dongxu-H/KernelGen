# KernelGen 算子开发 MCP 工具集用户指南

本节介绍如何使用 VSCode（及 Copilot）、Claude Code 和 OpenClaw 连接到 KernelGen 算子开发 MCP 工具集，并进行自动调优 Kernel 和 生成 TLE 算子。

## 前置条件

- 智能体客户端版本
  
  - Claude Code 版本 2.1 及以上
  
  - OpenClaw 版本 2026.3.2 及以上
  
  - VSCode 需启用 Github Copilot

- 环境准备
  
  - 预安装依赖包：请预先安装 `torch`、`triton` 和 `pytest` 软件包，以便在不同硬件平台上进行 Kernel 测试。通过以下 Python 代码导入 `torch` 已检测 CUDA 是否可用：

    ```{code-block} python
    import torch

    # 检查 CUDA 是否可用
    print("CUDA available:", torch.cuda.is_available())

    # 如果可用，执行一个简单的 GPU 计算
    if torch.cuda.is_available():
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        z = x + y  # 触发 CUDA kernel
        print("Result:", z)
        print("Device:", z.device)
    else:
        print("CUDA is not available")
    ```

    针对华为昇腾（Huawei Ascend）平台：除了安装和导入标准的 `torch` 包外，还需安装并导入 `torch_npu`。

- 预安装 FlagTree：请预先安装 [FlagTree](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/install.html)。

```{toctree}
:maxdepth: 2

connect_mcp/connect-mcp.md
mcp_tool/mcp-tool.md
```
