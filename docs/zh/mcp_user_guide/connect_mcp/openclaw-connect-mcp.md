# 使用 OpenClaw 连接 KernelGen 算子开发 MCP 工具集

如需将 OpenClaw 连接至 KernelGen 算子开发 MCP 工具集，请执行以下步骤：

1. 从 [ClawHub](https://clawhub.ai/steipete/mcporter) 下载并安装由 Peter Steinberger 创建的 McPorter 技能。

2. 配置 KernelGen 算子开发 MCP 工具集：与 OpenClaw 对话，要求其配置 KernelGen 算子开发 MCP 工具集，并以 JSON 格式粘贴以下信息：

```json
{
  "mcpServers": {
    "kernelgen-mcp": {
      "transport": "sse",
      "url": "http://kernelgen.flagos.io/sse",
      "headers": {
        "Authorization": "Bearer <your Token>"
```

