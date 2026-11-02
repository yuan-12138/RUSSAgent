# SFT prompt reference

This reference describes how to organize guideline-conditioned tool-use examples for SFT, using prompts from development-stage data. The complete experimental dataset and SFT/PPO training pipeline are not included.

## Files

| File | Purpose |
|------|---------|
| [System prompt](sft_system_prompt.txt) | Defines the agent's role, decision rules, and output format. This prompt is also used by the [released agent](../../OrchestratorRUSS/scripts/run_agent.py). |
| [User-message template](sft_user_prompt_template.txt) | A placeholder template reconstructed from development examples, covering observations, history, guideline passages, tool definitions, and valid tool names. |
| [Tool definitions](sft_tool_definitions.txt) | Tool names, descriptions, and argument definitions extracted from a development SFT input. These describe the interfaces shown to the model. |

## Constructing an example

Each example contains three messages:

1. **System:** use the system prompt.
2. **User:** fill the input template with the current observation, recent interaction history, relevant guideline passages, relevant tool definitions, and the valid tool names for that step.
3. **Assistant:** provide the reference response for that task state, consisting of a short reasoning line followed by one tool-call JSON object.

The user-message structure is:

```text
Current Observation: {current_observation}

History (Last {history_count}): {history}

Relevant Handbook:
{retrieved_handbook_sections}

Relevant APIs:
{retrieved_api_definitions}

Valid tools: {valid_tool_names}
```

The assistant response format is:

```text
Reasoning: <short reasoning>
{"type": "tool_call", "tool_name": "<tool name>", "tool_args": {"<argument>": "<value>"}}
```

Replace the placeholders with task-specific content. Each reference response should be checked against the observation, history, guideline, and tool definitions before being used as a training target.

## Using the tools

The tool definitions above document model inputs. Robot execution is implemented in the repository's [tools](../../workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/). For running the demo, use the [runtime API catalog](../../workspace/src/autonomous_scan_agent/prompts/russagent_api_catalog.json) and follow the [hardware setup](../hardware.md) and [quick start](../../README.md#quick-start).
