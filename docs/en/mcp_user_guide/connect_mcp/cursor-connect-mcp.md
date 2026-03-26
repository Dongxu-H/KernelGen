# Use Cursor to connect to KernelGen MCP server

To connect Cursor to the KernelGen MCP server, set the `mcp.json` file as follows:

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
