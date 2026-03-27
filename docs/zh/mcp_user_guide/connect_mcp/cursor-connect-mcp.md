# 使用 Cursor 连接 KernelGen MCP 服务器

如需将 Cursor 连接至 KernelGen MCP 服务器，请按如下方式配置 `mcp.json` 文件：

```json
"mcp_kernelgen": {
      "url": "http://kernelgen.flagos.io/sse",
      "headers": {
        "Authorization": "Bearer <your token>"
      },
      "timeout": 3600,
      "disabled": false
    },
```
