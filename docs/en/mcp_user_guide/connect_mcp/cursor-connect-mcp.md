# Use Cursor to connect to KernelGen Operator Development MCP Toolkit

To connect Cursor to the KernelGen Operator Development MCP Toolkit, set the `mcp.json` file as follows:

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
