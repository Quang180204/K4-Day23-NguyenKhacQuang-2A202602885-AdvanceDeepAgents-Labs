# Survey on LLM Agents and Tool Use: Architectures, Discovery, and Frameworks

## TL;DR
- **Evolution of Control**: Agentic workflows have shifted from interleaved reasoning (ReAct [1]) to "Agentic Programming" and blueprint-based systems that decouple probabilistic planning from deterministic execution [2][3].
- **Scaling Toolsets**: To handle thousands of tools, agents now employ "RAG for Tools," using shared vector spaces [4], co-invocation hypergraphs [5], and standardized protocols like MCP [6].
- **Framework Diversification**: Modern frameworks are splitting into integration-heavy libraries (LangChain [7]), role-based orchestrators (CrewAI [8]), and stateful graph-based systems (LangGraph [9]).
- **Alternative Execution**: A move toward "code-first" execution (e.g., smolagents) reduces the overhead of JSON parsing and enables complex logic directly within the tool-use step.

## Background
LLM agents are autonomous systems that leverage Large Language Models as their "central brain" to reason, plan, and interact with external tools to achieve complex goals. The core challenge in tool use is bridging the gap between the probabilistic nature of LLMs and the deterministic requirements of APIs. Early efforts focused on prompting strategies to elicit tool calls, while current research emphasizes structural reliability, scalable tool discovery, and standardized communication protocols.

## Foundational Architectures and Reasoning Patterns
The architecture of tool-using agents has evolved through three primary stages: interleaved loops, modular specialization, and deterministic control.

### Interleaved Reasoning and Acting
The **ReAct** paradigm [1] established the foundational loop of "Thought $\rightarrow$ Action $\rightarrow$ Observation." By interleaving reasoning traces with tool calls, agents can track progress and handle exceptions. However, this approach can be token-intensive and prone to "loops" where the agent repeats the same failing action.

### Modular Agentic Systems
To improve reliability, recent research suggests separating the agentic loop into specialized roles. The **Multi-LLM Agent** approach [10] argues that small LLMs can match the performance of larger models if the roles of planning, tool invocation, and result summarization are decoupled into distinct modules. Similarly, **PoAct** [11] optimizes the path from reasoning to action via dynamic policy switching, reducing token overhead.

### Deterministic Control Flow
The most recent shift is toward **Agentic Programming** [2] and **Blueprint-based Workflows** [3]. Instead of allowing the LLM to decide every step of the process, developers define a deterministic execution graph (a blueprint). The LLM is used only for specific tasks (e.g., data extraction or decision-making) within a fixed structure, significantly reducing hallucinations and increasing reliability in production environments.

## Large-Scale Tool Discovery and Optimization
As toolsets grow from a few dozen to thousands, providing all tool descriptions in the context window becomes impossible. This has led to the emergence of "Tool-RAG."

### Tool Retrieval Mechanisms
Instead of static tool lists, agents now use retrieval systems to find relevant tools dynamically. **Tool-to-Agent Retrieval** [4] utilizes a shared vector space to match agent needs with tool capabilities. For more complex dependencies, **HYSET** [5] uses co-invocation hypergraphs to retrieve *sets* of tools that are likely to be used together, while **ScaleMCP** [6] integrates the Model Context Protocol (MCP) to enable dynamic server-side tool discovery.

### Interface Optimization and Graph Modeling
Optimization is no longer just about the model, but about the tool interface itself. **Trace-Free+** [12] uses curriculum learning to rewrite tool descriptions, making them more "understandable" to the LLM. Furthermore, the **In-N-Out** dataset [13] models APIs as structured graphs, allowing agents to navigate complex dependency chains across multiple sequential API calls.

### Benchmarking Tool Use
Evaluation has shifted from simple "success rate" to long-horizon planning. **PlanBench-XL** [14] tests agents in large-scale ecosystems with dynamic disruptions, while **General AgentBench** [15] examines how agents scale during test-time computation. Domain-specific benchmarks, such as **DrafterBench** [16], further refine the evaluation of structured data comprehension in specialized fields like civil engineering.

## Practical Frameworks and Implementations
The implementation of tool-use agents has diverged into several dominant architectural patterns.

### Orchestration Frameworks
1. **Integration-Centric**: **LangChain** [7] provides an exhaustive ecosystem of adapters, making it the primary tool for rapid integration of diverse APIs.
2. **Stateful Graph-Based**: **LangGraph** [9] treats workflows as state machines. By utilizing `TypedDict` and deterministic routing, it allows for cycles and human-in-the-loop "breakpoints," making it more suitable for complex production agents than linear chains.
3. **Role-Based**: **CrewAI** [8] emphasizes "collaborative intelligence" by assigning roles, goals, and backstories to agents, effectively limiting the tool surface area for each agent to reduce hallucinations.
4. **Enterprise-Grade**: The **Microsoft Agent Framework (MAF)** and **AutoGen** [17] focus on cross-runtime support (.NET/Python) and deep integration with cloud infrastructure (Azure AI Foundry).

### Standardized Protocols and Execution Patterns
A critical trend is the movement toward a universal interface. The **Model Context Protocol (MCP)** has emerged as a standardized client-server protocol, allowing any MCP-compatible agent to discover and invoke tools without custom wrappers.

Execution patterns have also evolved from **JSON Function Calling** (the industry standard) to **Code-as-Tool** (e.g., smolagents). In the latter, the LLM writes Python code to invoke the tool, which is then executed in a sandbox. This bypasses the rigid structure of JSON and allows the agent to perform complex data manipulation between tool calls.

## Trends and Open Problems
The field is moving toward a convergence of "Probabilistic LLMs" and "Deterministic Software Engineering." 

### Key Developments (Last 2 Years)
- **Standardization**: The rise of MCP is reducing the "integration tax" of building agents.
- **Code-First Execution**: Shifting from JSON blobs to executable scripts is improving the flexibility of tool use.
- **RAG for Tools**: Tool retrieval is becoming a standard component of the agentic architecture to handle scale.

### Open Challenges
- **Reliability in Long-Horizon Tasks**: Despite blueprint-based workflows, agents still struggle with "drift" in long-horizon objectives where a single tool failure can derail the entire process [14].
- **Abstraction Overhead**: As seen in the transition from LangChain to LangGraph, there is a struggle to balance high-level abstraction for speed and low-level control for debugging [7][9].
- **Token Efficiency**: The overhead of providing detailed tool descriptions and reasoning traces remains a significant cost factor in multi-turn agentic loops.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-01)
[2] LLM-as-Code: Agentic Programming for Agent Harness. arxiv. https://arxiv.org/abs/2606.15874 (2026-01-01)
[3] Blueprint First, Model Second. arxiv. https://arxiv.org/abs/2508.02721 (2025-01-01)
[4] Tool-to-Agent Retrieval. hf-search. https://huggingface.co/papers/2511.01854 (2025-11-01)
[5] HYSET: Set-Level Tool Retrieval. hf-search. https://huggingface.co/papers/2607.25718 (2026-01-01)
[6] ScaleMCP. hf-search. https://huggingface.co/papers/2505.06416 (2025-05-01)
[7] LangChain GitHub Repository. web. https://github.com/langchain-ai/langchain (2024-01-01)
[8] CrewAI GitHub Repository. web. https://github.com/crewaiinc/crewai (2024-01-01)
[9] LangGraph GitHub Repository. web. https://github.com/langchain-ai/langgraph (2024-01-01)
[10] Small LLMs Are Weak Tool Learners: A Multi-LLM Agent. arxiv. https://arxiv.org/abs/2401.07324 (2024-01-01)
[11] PoAct: Policy and Action Dual-Control Agent. arxiv. https://arxiv.org/abs/2501.07054 (2025-01-01)
[12] Learning to Rewrite Tool Descriptions. hf-search. https://huggingface.co/papers/2602.20426 (2026-02-01)
[13] In-N-Out: API Graph Dataset. hf-search. https://huggingface.co/papers/2509.01560 (2025-09-01)
[14] PlanBench-XL. hf-search. https://huggingface.co/papers/2606.22388 (2026-01-01)
[15] General AgentBench. hf-search. https://huggingface.co/papers/2602.18998 (2026-02-01)
[16] DrafterBench. hf-search. https://huggingface.co/papers/2507.11527 (2025-07-01)
[17] AutoGen GitHub Repository. web. https://github.com/microsoft/autogen (2024-01-01)
