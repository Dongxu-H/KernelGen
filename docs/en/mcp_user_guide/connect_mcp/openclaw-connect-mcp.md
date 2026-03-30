# Use OpenClaw to connect to KernelGen MCP server

To connect OpenClaw to the KernelGen MCP server, perform the following steps:

1. Download and install the McPorter skills created by Peter Steinberger from [ClawHub](https://clawhub.ai/steipete/mcporter).

2. Configure KernelGen MCP Server: Chat with OpenCalw and ask it to configure KernelGen MCP Server by pasting the following information in the JSON format:

```json
{
  "mcpServers": {
    "kernelgen-mcp": {
      "transport": "sse",
      "url": "http://kernelgen.flagos.io/sse",
      "headers": {
        "Authorization": "Bearer <your Token>"
```


