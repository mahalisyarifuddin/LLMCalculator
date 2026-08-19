# Agent Context Harness Audit — 2026-08-19

## Purpose

This audit checks how LLMCalculator should describe one configured context across current open-source agent harnesses, especially harnesses intended for long-running or local-model workflows.

The calculator is deliberately narrower than a harness or inference server. It estimates a plausible interpolated local open-weight model plus **one configured, active KV-cache context**. It does not predict how an agent framework compiles prompts, reserves output space, compacts history, retrieves memory, or schedules subagents.

## Conclusion

Normal chats and agentic inference steps use the same architecture-aware KV math. Because the former two-mode control changed only wording, the clearer UX is one dual-label **Context Window / Active Working Context** slider. It represents configured KV capacity for an ordinary local chat, document prompt, embedded feature, or **one active inference step** in a long-running agent workflow.

The expanded harness review strengthens the distinction between active context and durable agent state. Across the projects below, long-horizon continuity comes from one or more of:

- durable transcripts or event logs;
- summaries and compaction checkpoints;
- files, structured notes, or memory blocks;
- retrieval over prior messages or external stores;
- isolated subagent contexts;
- programmatic state outside the model-facing prompt.

None of these mechanisms turns total task history into one permanently GPU-resident KV cache.

## Expanded harness review

| Harness | Model-facing context management | Durable state outside the active model context | Calculator implication |
|---|---|---|---|
| **OpenClaw** | Compacts older turns into a summary, keeps a recent tail, and can prune old tool results. | Full history remains on disk; a pre-compaction memory flush can write durable notes. Its docs explicitly distinguish context from memory. | Agentic context is one bounded model run, not the session transcript or memory directory. |
| **OpenCode** | Replaces older active context with a structured checkpoint plus a recent serialized tail. It uses its own output allowance and safety buffer. | Earlier session messages remain durable even when omitted from model requests. | Output buffers and retained-tail policies are harness details, not calculator controls. |
| **DeepSeek Harness** | Ships a pluggable compaction capability as part of its "everything is a plugin" design. | Everything the model sees is recorded in an append-only session log used for resume, fork, search, replay, and tracing. | The calculator must not freeze rapidly evolving DSH plugin defaults into its memory math. |
| **Hermes Agent (Nous Research)** | Compresses active history, preserves protected recent messages, and supports pluggable context engines. | Flushes persistent memory before compression; pre-compaction turns are archived and searchable in session storage. | Compression threshold, protected tail, and auxiliary summary model do not change local KV bytes per configured token. |
| **Prime Agent** | Treats context as a programmatic variable and gives the model access to history and recursive subagents through a persistent REPL. | Continual Harness state—prompts, skills, memory, and subagent definitions—is written to disk; child agents have their own sessions and contexts. | One slider cannot represent a parent plus concurrent child contexts. It estimates one active inference context only. |
| **Pi coding agent** | Automatic or manual lossy compaction summarizes older messages while preserving recent ones. | Full session history remains in an append-only JSONL tree and can be revisited. | Active compacted context and durable session history are separate quantities. |
| **Qwen Code** | Supports semantic `/compress`, fast tool-output/thinking cleanup, session summaries, and isolated or inherited subagent workflows. | QWEN.md and auto-memory carry knowledge across fresh context windows and sessions. | Agent memory and subagent topology belong to the harness, not the KV formula. |
| **Gemini CLI** | Uses configurable history compression and tool-output summarization. | Hierarchical context files and auto-memory patches are separate from live chat history. | Compression thresholds and retained-token policies are runtime behavior. |
| **Agent Zero** | Uses history compression and can work with local OpenAI-compatible runtimes. | Its memory subsystem is distinct from the current conversation history. | Open issues around oversized tool results show why runtime behavior cannot be inferred from nominal window size alone. |
| **Crush** | Supports automatic/manual summarization and separate small/large model roles. | Maintains project sessions and context files. | Compaction quality and failure behavior vary by harness and model; the calculator should not claim a usable prompt budget. |
| **OpenHands** | Condensers summarize older events while preserving recent and important context. | Conversation events can be persisted and restored independently of the condensed model view. | Event count and condensation thresholds are not GPU-KV multipliers. |
| **SWE-agent** | History processors can elide old observations or remove selected content. | The trajectory remains an artifact even when its model-facing projection is filtered. | Filtered history is a harness projection, not additional context capacity. |
| **Aider** | Summarizes chat after a soft threshold and budgets a selective repository map. | Git, chat logs, and the repository remain the durable source of truth. | Repository size and task horizon are not allocated as KV cache. |
| **Cline / Roo Code / Goose** | Auto-compact or condense long sessions and retain a recent working set. | Checkpoints, task history, memory-bank files, or stored transcripts preserve continuity. | Each implementation has different thresholds and reserves, supporting a runtime-agnostic calculator. |
| **LangGraph / Deep Agents** | Trim, delete, summarize, offload large tool results, and isolate subagent work. | Checkpointed graph state, stores, and files persist outside any one model call. | Graph state and parallel branches require separate planning. |
| **Letta** | Builds each prompt from bounded in-context memory plus selected messages. | Archival memory, recall memory, files, and external databases remain out of context until retrieved. | This is the clearest example of context capacity being different from total agent memory. |
| **smolagents** | AgentMemory records steps and can produce succinct messages or be reset; richer long-horizon policy may require application code. | External persistence and retrieval are application choices. | Not every harness compacts automatically, so the dual label must not promise a memory system. |

## Common architecture found across the audit

### 1. An agent session is not a context window

A session can persist for hours, days, or indefinitely while each physical model call receives a bounded prompt. OpenClaw, OpenCode, DeepSeek Harness, Hermes, Pi, and others keep more durable session data than the model sees on the next turn.

### 2. Compaction changes the model-facing projection

Compaction normally replaces some active history with a summary or checkpoint. It does not increase the model's context capacity, and it is often lossy. The full transcript may remain durable even when it no longer consumes active prompt/KV space.

### 3. Memory is usually tiered

Structured files, memory blocks, databases, repository maps, session search, and retrieval keep information available without injecting all of it on every call. Letta, OpenClaw, Hermes, Qwen Code, LangGraph, and Prime Agent make this boundary especially explicit.

### 4. Subagents do not share one universal KV allocation

Prime Agent, LangGraph/Deep Agents, Qwen Code, OpenHands, and other systems can create child agents or isolated branches. Each active model invocation can have its own context. Parallel subagents therefore require separate capacity planning, exactly as the calculator disclaimer states.

### 5. Usable prompt space is harness-specific

Harnesses may subtract system prompts, tool schemas, requested output tokens, safety buffers, retained tails, media estimates, or provider-specific overhead from the nominal model window. OpenCode, Roo Code, Hermes, Qwen Code, and Gemini CLI use different policies. LLMCalculator should continue to estimate configured KV capacity rather than claim an exact usable input budget.

### 6. Harness maturity varies

Compaction and retrieval can fail, lose detail, or behave differently with local models. This is another reason not to add a universal completion reserve, horizon multiplier, concurrency factor, or retention strategy to the calculator.

## Finding and fix

### Finding

The Normal and Agentic choices had identical calculation inputs and outputs; the toggle only changed the slider label and helper. Keeping two choices therefore implied a technical distinction that did not exist. The harness audit also shows that one configured context can serve either use without changing KV bytes.

### Fix

The mode control was removed. One slider now carries the dual label **Context Window / Active Working Context** and the helper states that it covers one active chat, document prompt, embedded feature, or agent step. It still directs long-running project history to external artifacts, structured notes, retrieval, and compaction instead of GPU KV cache.

The Bahasa Indonesia label and helper were updated equivalently. No multiplier or harness-specific control was added.

## Calculator invariants retained

- One logarithmic, dual-label context slider.
- The slider value is passed directly to the existing architecture-aware KV formulas.
- No presentation-only mode state remains.
- No output reserve, concurrency, task horizon, headroom, cache-allocation mode, CPU-offload estimate, or harness-specific threshold.
- Architecture interpolation remains the model-selection mechanism.
- The result remains an estimate of a plausible local open-weight model, not an exact checkpoint validator or inference-server planner.

## Primary references

- [OpenClaw: Context](https://docs.openclaw.ai/concepts/context)
- [OpenClaw: Compaction](https://docs.openclaw.ai/concepts/compaction)
- [OpenCode: Compaction](https://opencode.ai/v2/docs/compaction)
- [DeepSeek Harness developer preview](https://deepseek.com/harness/en/)
- [DeepSeek Harness repository architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/AGENTS.md)
- [Hermes Agent: Context Compression and Caching](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching)
- [Hermes Agent: Agent Loop Internals](https://hermes-agent.nousresearch.com/docs/developer-guide/agent-loop)
- [Prime Agent announcement](https://www.primeintellect.ai/blog/prime-agent)
- [Pi coding agent README](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md)
- [Qwen Code: Commands](https://qwenlm.github.io/qwen-code-docs/en/users/features/commands/)
- [Qwen Code: Memory](https://qwenlm.github.io/qwen-code-docs/en/users/features/memory/)
- [Gemini CLI configuration](https://geminicli.com/docs/reference/configuration/)
- [Agent Zero releases](https://github.com/agent0ai/agent-zero/releases)
- [Crush repository](https://github.com/charmbracelet/crush)
- [OpenHands: Context Condenser](https://docs.openhands.dev/sdk/guides/context-condenser)
- [SWE-agent: History Processors](https://swe-agent.com/latest/reference/history_processor_config/)
- [Aider: Options](https://aider.chat/docs/config/options.html)
- [Cline: Auto Compact](https://docs.cline.bot/features/auto-compact)
- [Roo Code: Intelligent Context Condensing](https://docs.roocode.com/features/intelligent-context-condensing)
- [Goose: Smart Context Management](https://block.github.io/goose/docs/guides/smart-context-management/)
- [LangGraph: Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [Letta: Memory Management](https://docs.letta.com/concepts/memory-management/)
- [smolagents: Agent and Memory Reference](https://huggingface.co/docs/smolagents/reference/agents)
