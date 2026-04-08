# Optimize a Kernel

You can use either VSCode (and Copilot), Claude Code, or OpenClaw to optimize a Kernel on **NVIDIA**.

To optimize a Kernel, a typical prompt should include the following mandatory and optional elements:

Operator name（mandatory）, task description (mandatory), and optimization iterations.

Make sure you read the [prerequisites](../skills_user_guide/skills-user-guide.md).

## Steps

If you haven't connected to the KernelGen Operator Development MCP Toolkit and load skills, see [KernelGen Skills User Guide](../skills_user_guide/skills-user-guide.md), otherwise use one of the following methods to invoke the `kernelgen-flagos` skill and optimize an operator:

- **Option 1**: Use the slash command and prompt

   ```{code-block} python
   /kernelgen-flagos Optimize the index_put operator. Optimize 5 iterations.
   ```

- **Option 2**: Completely use prompt

   ```{code-block} python
   Use kernelgen-flagos to optimize the index_put operator. Optimize 5 iterations.
   ```
