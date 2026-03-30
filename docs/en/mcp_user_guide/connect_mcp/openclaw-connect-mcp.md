# Use OpenClaw to connect to KernelGen Operator Development MCP Toolkit

To connect OpenClaw to the KernelGen Operator Development MCP Toolkit, perform the following steps:

1. Download and install the McPorter skills created by Peter Steinberger from [ClawHub](https://clawhub.ai/steipete/mcporter).

2. Configure KernelGen Operator Development MCP Toolkit: Chat with OpenCalw and ask it to configure KernelGen Operator Development MCP Toolkit by pasting the following information in the JSON format:

```json
{
  "mcpServers": {
    "kernelgen-mcp": {
      "transport": "sse",
      "url": "http://kernelgen.flagos.io/sse",
      "headers": {
        "Authorization": "Bearer <your Token>"
```


