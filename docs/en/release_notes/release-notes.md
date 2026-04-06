# Release Notes

This section includes the KernelGen release information.

## V2.0.0

- **Added features**：
  - Added support for creating, optimizing, and specializing Kernels across multiple hardware platforms via the KernelGen Operator Development MCP Toolkit, AI agents, and first- and third-party skills.
  - Added support for generating the FlagTree TLE operator on NVIDIA hardware platforms. TLE operator generation capability is an experimental feature currently under active development.
  - Extended testing devices support to include MetaX platform.
  - Introduced a History panel to track modifications to the Kernel code.

## V1.0.0

- **Added features**：
  - Basic Workflow: Supports single, fixed-step code generation including GroundTruth → TritonKernel → Correctness Test → Performance Test.
  - User Registration: Supports self-service registration and application, with platform approval required for trial use.
  - Web Interface: Online access at [https://kernelgen.flagos.io/](https://kernelgen.flagos.io/).
  - Core Features:
    - Fully Automated Workflow: Automatically generates, tests, and optimizes complete AI operator sets.
    - Multi-Backend Support: Seamlessly supports multiple AI libraries and chips with automatic adaptation and debugging.
    - Easy-to-Use: Browser-based interface requiring no setup or prior experience.
    - Standardized Verification: Automatically generates test cases to ensure operator correctness.
    - Deep Ecosystem Integration: Collaborates with FlagGems and FlagTree to accelerate operator library development.
