# Use OpenClaw to connect to KernelGen Operator Development MCP Toolkit

## Prerequisites

Use OpenClaw version 2026.3.2 and later

## Steps

To connect OpenClaw to the KernelGen Operator Development MCP Toolkit, perform the following steps:

1. Send a prompt to connect to the KernelGen Operator Development MCP Toolkit, for example:

   - `Based on the Claude Code configuration documentation: https://code.claude.com/docs/en/mcp, connect to the MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the claude.json file.`

   - `Based on the VSCode documentation: https://code.visualstudio.com/docs/copilot/customization/mcp-servers, configure the kernelgen MCP. The MCP URL is https://kernelgen.flagos.io/sse, and the token is <your KernelGen Token>. Configure this in the mcp.json file. `

    **Note**: If the current OpenClaw version does not support MCP, you can setup `mcporter` via prompt or command．The following is the command example.

    ```{code-block} shell
    "npx skills add steipete/clawdis@mcporter -g -y"
    ```

2. Verify KernelGen Operator Development MCP Toolkit connection, prompt：
  
  ```{code-block} shell
  Please verify the kernelgen mcp connection is successful.
  ```