# Historical SFT prompt reference

This reference contains the system prompt and tool definitions extracted from a locally cached SFT dataset dated 26 February 2026. It is a prompt reference, not the full experimental dataset or a complete reproduction package.

## Scope and intended use

This reference documents the inputs and supervision format of historical SFT examples. It does not include the complete 20-guideline dataset, its train/test split, full teacher-generated traces, per-example ground-truth annotations, model checkpoints, or SFT/PPO training and evaluation scripts. These historical materials have not been verified as the exact final dataset used for the reported experiments.

The public repository includes real-robot tool implementations. In the released demo, operators provide contact-quality and post-scan target-alignment feedback; the robot performs the corresponding adjustments. See [hardware and release scope](../hardware.md).

Running the released demo does not require repeating SFT or PPO. A pretrained model such as Qwen3-30B-A3B-Instruct-2507 can be connected through the configurable model API. Running it on another setup still requires the dependencies and model assets listed in the [main README](../../README.md), hardware integration, hand-eye calibration, capture-pose configuration, and adaptation of the robot interfaces where necessary. This deployment option does not establish reproduction of the paper's fine-tuned-model results or automatic ultrasound perception modules.

## Reference files

- [System prompt](sft_system_prompt.txt): extracted verbatim from the historical SFT inputs.
- [User-message template](sft_user_prompt_template.txt): reconstructed from the example structure using placeholders.
- [Tool definitions](sft_tool_definitions.txt): historical names, descriptions, and argument definitions extracted verbatim from one SFT input.

## System prompt (verbatim)

```text
You are an autonomous ultrasound scanning agent.
Use the Handbook as guidance, but you may make non-linear decisions based on the Current Observation and History.

Critical behavior:
- Choose the most appropriate NEXT tool based on the Current Observation + History + Handbook.
- Do NOT repeat a step that is already completed unless the Current Observation explicitly indicates failure or the need to redo it.
- Only call operator_io_tool when human confirmation/input is truly needed (task selection, patient instruction confirmation, clinician judgment).
- Once the scanning task has been identified, do NOT ask to confirm it again. Proceed to acquisition/projection/Point0/execute/reset as the workflow specifies.
Output format (IMPORTANT):
- First, output EXACTLY ONE plain-text line starting with: Reasoning: <short reasoning>
- Then, output EXACTLY ONE valid JSON object (no markdown, no backticks, no extra text).
IMPORTANT JSON formatting rules:
- It MUST be a single valid JSON object (no extra braces, no trailing commas).
{
  "type": "tool_call",
  "tool_name": "name_of_tool",
  "tool_args": { ... }
}
```

## User message structure (reconstructed template)

The placeholders below generalize the structure observed in the cached training examples; this is not a recovered original generator script.

```text
Current Observation: {current_observation}

History (Last {history_count}): {history}

Relevant Handbook:
{retrieved_handbook_sections}

Relevant APIs:
{retrieved_api_definitions}

Valid tools: {valid_tool_names}
```

Populate the observation and history from the current task state, include the relevant handbook passages and tool API definitions, and provide the valid tool names for that step. These inputs form the user message accompanying the system prompt.

## Supervision format

The inspected cached examples contain system, user, and assistant messages. The assistant response contains a short reasoning line followed by a tool-call JSON object:

```text
Reasoning: <short reasoning>
{"type": "tool_call", "tool_name": "<tool name>", "tool_args": {"<argument>": "<value>"}}
```

The placeholders above describe the format, not a complete training example. The original dataset-generation script and any separate teacher-generation prompt have not been located, so this reference does not establish how every target response was generated or checked.

## Historical tool definitions (verbatim from one SFT input)

```text
{"name": "path_execute_server_tool", "description": "Execute a scan path YAML via path_execute_server. Parameters: path_id (defaults to /home/camp/.ros/path_preview1.yaml), wait_timeout_sec (maximum time to wait for the robot to reach the target position; usually is 120s), breath_instruction_on_impedance (boolean; Instruct patient to take a deep breath and hold before contact during scanning).", "args_schema": {"path_id": "string", "wait_timeout_sec": "float", "breath_instruction_on_impedance": "boolean"}}
{"name": "reset_to_capture_pose_tool", "description": "Reset robot to the capture pose. finish=false means the session continues (between-segment reset); finish=true means final completion (end session). Default invocation is tool_args: {}.", "args_schema": {"finish": "boolean"}}
{"name": "operator_io_tool", "description": "Operator interaction (ask/say/sleep). Use this FIRST for task identification (ask what the scanning task is). Also used to wait for confirmations like 'ok'. You may use either 'say' or 'prompt' to provide the text (aliases). If audience='patient', the wording is patient-facing, but the confirmation input ('ok') is provided by the clinician/operator. IMPORTANT: audience must be 'operator' or 'patient' (no other values).", "args_schema": {"prompt": "string", "say": "string", "audience": "string (enum: 'operator' | 'patient')", "expect_response": "boolean", "sleep_sec": "float"}}
{"name": "acquire_trajectory_tool", "description": "Unified trajectory acquisition entry. Parameters: organ (required), side (string; used for side-specific organs such as kidney/femoral_artery/carotid_artery), overwrite is required and means if we need to acquire a new trajectory then overwrite. startpoint/endpoint are keypoint labels (strings; see Handbook mapping, e.g., xiphoid_process, right_costal_margin, T12_left, L_ASIS, C2_left, etc.). For backward-compatibility, numeric strings like '1' are also accepted.", "args_schema": {"organ": "string", "side": "string (enum: 'left' | 'right')", "overwrite": "boolean", "startpoint": "string", "endpoint": "string"}}
{"name": "ribline_projection_tool", "description": "Projection preview (creates /home/camp/.ros/path_preview1.yaml)Project the acquired trajectory into 3D point cloud. instruct_and_wait means instruct the patient to take a deep breath and hold, then wait for ok. Recommended policy for auto_generate_delay_sec: set 3.0 if need to instruct the patient to take a deep breath and hold, and 0.0 if no need for that instruct.", "args_schema": {"instruct_and_wait": "boolean", "auto_generate_delay_sec": "float"}}
{"name": "post_scan_adjust_tool", "description": "One-time post-scan verification for scan target centering. Ask clinician whether target is centered/-x/+x. If off-center, adjust path x by +/-0.01m and re-execute the full path once. No second verification loop. Parameters: path_yaml (defaults to /home/camp/.ros/path_preview1.yaml), wait_timeout_sec (maximum time to wait for the robot to reach the target pose; typically set to 120s), breath_instruction_on_impedance (boolean; Instruct patient to take a deep breath.).", "args_schema": {"path_yaml": "string", "wait_timeout_sec": "float", "breath_instruction_on_impedance": "boolean"}}
{"name": "point0_auto_verify_tool", "description": "Point0 contact verification loop. Asks clinician to input good/poor/none and performs internal adjustment/retry automatically until good or max retries. Parameters: offset_step_m (adjustment step per retry in meters, negative means down/deeper; default -0.02m for sim), max_retries (maximum number of retry attempts; default 10).", "args_schema": {"offset_step_m": "float", "max_retries": "integer"}}
```

These historical definitions include operator/clinician feedback and simulation-related defaults. They document the cached training inputs; they do not establish the implementation of automatic image-based perception modules.

## Historical and current versions

The text files preserve the historical reference. They are documentation, not runtime configuration files. The following differences were checked against [repository revision e3683bc](https://github.com/yuan-12138/RUSSAgent/tree/e3683bc2a3d3fe31b634d27083cefb3fa00e14cb):

- The historical contact-adjustment definition describes an `offset_step_m` default of `-0.02` m for simulation. The released contact tool uses `-0.005` m. Historical defaults should not be copied into a real-robot setup as operating recommendations.
- The historical projection definition recommends a delay of `0.0` seconds when breath-hold instructions are unnecessary. The released API catalog recommends `4.0` seconds.
- Paths such as `/home/camp/.ros/path_preview1.yaml` are preserved as historical text. The released code resolves its path through `RUSSAGENT_PATH_PREVIEW` or the project default under `$HOME/.ros/russagent/`.
- These examples use a `Reasoning:` line followed by a JSON object. The paper describes `<think>` / `<tool>` output; the historical reference does not establish the exact final serialization used in the reported experiments.

For deployment, consult the [current API catalog](../../workspace/src/autonomous_scan_agent/prompts/russagent_api_catalog.json) and [tool implementations](../../workspace/src/autonomous_scan_agent/src/autonomous_scan_agent/tools/).
