# Understanding OpenHands Tool Call Processing and Thinking Model Integration

## Overview

This document explains how OpenHands processes tool calls from LLMs, how the "think" tool works, and how thinking models (like Qwen3-Thinking) should be integrated.

## OpenHands Agent Execution Flow

### 1. Agent Step Loop

**Location**: `openhands/agenthub/codeact_agent/codeact_agent.py`

The agent operates in a step-by-step loop:

```python
def step(self, state: State) -> Action:
    # 1. Check for pending actions first (FIFO queue)
    if self.pending_actions:
        return self.pending_actions.popleft()
    
    # 2. Prepare conversation history
    messages = self._get_messages(condensed_history, initial_user_message)
    
    # 3. Call LLM with tools
    response = self.llm.completion(messages=messages, tools=self.tools)
    
    # 4. Convert LLM response to actions
    actions = self.response_to_actions(response)
    
    # 5. Queue all actions and return first one
    for action in actions:
        self.pending_actions.append(action)
    return self.pending_actions.popleft()
```

**Key Points:**
- `pending_actions` is a FIFO queue (deque) that stores actions to be executed sequentially
- When LLM returns multiple tool calls, ALL actions are queued
- Each `step()` call returns ONE action, which is executed by the controller
- After execution, the controller calls `step()` again, which returns the next queued action

### 2. Controller Execution Loop

**Location**: `openhands/controller/agent_controller.py`

```python
async def _step(self):
    # Get action from agent
    action = self.agent.step(self.state)
    
    # Execute action
    observation = await self.runtime.execute_action(action)
    
    # Add observation to state
    self.event_stream.add_event(observation)
    
    # Loop continues, agent.step() called again
```

**Flow:**
1. Controller calls `agent.step()` → Gets one action
2. Controller executes action → Gets observation
3. Observation added to state
4. Controller calls `agent.step()` again → Gets next action from queue
5. Repeat until queue is empty, then agent calls LLM again

## Tool Call Processing

### Standard Tool Call Format

**Regular Models** (e.g., GPT-4, Claude):
- Tool calls are in `response.choices[0].message.tool_calls` field
- Format: List of `ChatCompletionMessageToolCall` objects
- Each tool call has: `id`, `type='function'`, `function={name, arguments}`

**Example:**
```python
response.choices[0].message.tool_calls = [
    {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path": "/workspace/file.py"}'
        }
    }
]
```

### Thinking Model Format

**Thinking Models** (e.g., Qwen3-Thinking):
- Tool calls are **embedded in content** between `<tool_call>` tags
- Reasoning trace is before `</think>` tag (no opening tag)
- Format: `[reasoning]</think><tool_call>{json}</tool_call>`

**Example:**
```
Let me analyze this problem step by step...
</think>
<tool_call>
{"name": "read_file", "arguments": {"path": "/workspace/file.py"}}
</tool_call>
```

## The "Think" Tool

### Purpose

The "think" tool (`ThinkTool`) is a special tool that allows regular models to explicitly log their reasoning process.

**Location**: `openhands/agenthub/codeact_agent/tools/think.py`

**Description:**
- "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought."
- Used for complex reasoning, brainstorming, planning

**When Regular Models Use It:**
- Model explicitly calls `think` tool with `{"thought": "..."}` argument
- Creates `AgentThinkAction` which logs the thought
- Returns `AgentThinkObservation("Your thought has been logged.")`

### For Thinking Models

**Key Insight**: For thinking models, the reasoning trace (before `</think>`) **IS** the "think" tool call.

- The reasoning trace represents the model's thinking process
- It's equivalent to a regular model calling the "think" tool
- We create `AgentThinkAction` from the reasoning trace
- This is separate from any actual tool calls the model makes

## Response to Actions Conversion

### Function: `response_to_actions()`

**Location**: `openhands/agenthub/codeact_agent/function_calling.py`

**Current Implementation Flow:**

1. **Extract Tool Calls**
   ```python
   # Check standard tool_calls field first
   tool_calls = assistant_msg.tool_calls or []
   
   # If empty, extract from content (thinking models)
   if not tool_calls and content_str:
       tool_calls = extract_tool_calls_from_content(content_str)
       tool_calls_extracted_from_content = True
   ```

2. **Extract Reasoning Trace** (for thinking models)
   ```python
   if content_str:
       cleaned_content, reasoning = extract_reasoning_from_content(content_str)
       # reasoning = everything before </think>
   ```

3. **Create AgentThinkAction** (if reasoning exists)
   ```python
   if reasoning_stripped and tool_calls_extracted_from_content:
       think_action = AgentThinkAction(thought=reasoning_stripped)
       actions.append(think_action)
   ```

4. **Process Each Tool Call**
   ```python
   for tool_call in tool_calls:
       # Map tool call to action
       if function_name == "read_file":
           action = FileReadAction(...)
       elif function_name == "think":
           action = AgentThinkAction(...)
       # ... etc
       actions.append(action)
   ```

5. **Return All Actions**
   ```python
   return actions  # List of actions to execute sequentially
   ```

## Action Execution

### AgentThinkAction Execution

**Location**: `openhands/runtime/base.py` and `openhands/runtime/impl/action_execution/action_execution_client.py`

```python
def run_action(self, action: Action) -> Observation:
    if not action.runnable:
        if isinstance(action, AgentThinkAction):
            return AgentThinkObservation('Your thought has been logged.')
        return NullObservation('')
    # ... execute other actions
```

**Behavior:**
- `AgentThinkAction` is **not runnable** (doesn't execute code)
- Returns `AgentThinkObservation` with confirmation message
- The thought is logged in the conversation history
- Used for transparency and debugging

### Other Actions Execution

- `FileReadAction` → Reads file → Returns `FileReadObservation`
- `CmdRunAction` → Executes command → Returns `CmdOutputObservation`
- `FileEditAction` → Edits file → Returns `FileEditObservation`
- etc.

## Current Thinking Model Support

### What Works

1. **Reasoning Extraction**: ✅ Extracts reasoning from `</think>` tag
2. **Tool Call Extraction**: ✅ Extracts tool calls from `<tool_call>` tags
3. **Action Creation**: ✅ Creates `AgentThinkAction` from reasoning
4. **Sequential Execution**: ✅ All actions execute in order via `pending_actions` queue

### Example Flow for Thinking Model

**Input (LLM Response):**
```
Let me read the file to understand the issue...
</think>
<tool_call>
{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/workspace/file.py"}}
</tool_call>
```

**Processing:**
1. Extract reasoning: `"Let me read the file to understand the issue..."`
2. Extract tool call: `str_replace_editor` with `command="view"`
3. Create actions:
   - `AgentThinkAction(thought="Let me read the file...")`
   - `FileReadAction(path="/workspace/file.py")`

**Execution:**
1. Execute `AgentThinkAction` → `AgentThinkObservation("Your thought has been logged.")`
2. Execute `FileReadAction` → `FileReadObservation(content="...")`
3. Both observations added to conversation history
4. Next LLM call sees both the thought and file content

## Integration Requirements

### For Thinking Models

1. **Reasoning Trace = Think Tool Call**
   - The reasoning before `</think>` IS the "think" tool call
   - Should always create `AgentThinkAction` when reasoning exists
   - This is separate from any actual tool calls

2. **Multiple Actions Per Response**
   - Thinking models typically output: reasoning + tool call(s)
   - Should create: `AgentThinkAction` + actual tool action(s)
   - All actions execute sequentially via `pending_actions` queue

3. **No Duplication**
   - If thinking model also explicitly calls "think" tool (rare), merge reasoning into it
   - Otherwise, reasoning trace creates separate `AgentThinkAction`

### Current Implementation Status

✅ **Working:**
- Reasoning extraction from `</think>`
- Tool call extraction from `<tool_call>` tags
- Creating `AgentThinkAction` from reasoning
- Processing all tool calls
- Sequential execution via queue

⚠️ **Potential Issues:**
- Need to verify all tool calls are being extracted correctly
- Need to ensure reasoning is always creating `AgentThinkAction`
- Need comprehensive logging to track the flow
- Need to handle edge cases (only reasoning, only tool calls, etc.)

## Key Files

1. **Agent Step Logic**: `openhands/agenthub/codeact_agent/codeact_agent.py` (lines 161-225)
2. **Response Processing**: `openhands/agenthub/codeact_agent/function_calling.py` (lines 404-888)
3. **Think Tool Definition**: `openhands/agenthub/codeact_agent/tools/think.py`
4. **Action Execution**: `openhands/runtime/base.py` (lines 898-936)
5. **Controller Loop**: `openhands/controller/agent_controller.py` (lines 821-888)

## Summary

- **Regular Models**: Call tools explicitly via `tool_calls` field; can optionally call "think" tool
- **Thinking Models**: Output reasoning trace (equivalent to "think") + tool calls in content
- **OpenHands**: Processes all tool calls, creates actions, queues them, executes sequentially
- **Think Action**: Logs reasoning for transparency; doesn't execute code
- **Integration**: Already works! Just needs verification and logging improvements

