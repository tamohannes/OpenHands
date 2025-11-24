"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import re
import shlex
import uuid

from litellm import (
    ChatCompletionToolParam,
    ModelResponse,
)

from openhands.agenthub.codeact_agent.function_calling import (
    combine_thought,
    extract_tool_calls_from_content,
    extract_reasoning_from_content,
)
from openhands.agenthub.codeact_agent.tools import (
    FinishTool,
    ThinkTool,
)
from openhands.llm.tool_names import STR_REPLACE_EDITOR_TOOL_NAME
from openhands.agenthub.readonly_agent.tools import (
    GlobTool,
    GrepTool,
    ViewTool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentThinkAction,
    CmdRunAction,
    FileReadAction,
    MCPAction,
    MessageAction,
)
from openhands.events.event import FileReadSource
from openhands.events.tool import ToolCallMetadata


def grep_to_cmdrun(
    pattern: str, path: str | None = None, include: str | None = None
) -> str:
    # NOTE: This function currently relies on `rg` (ripgrep).
    # `rg` may not be installed when using CLIRuntime or LocalRuntime.
    # TODO: Implement a fallback to `grep` if `rg` is not available.
    """Convert grep tool arguments to a shell command string.

    Args:
        pattern: The regex pattern to search for in file contents
        path: The directory to search in (optional)
        include: Optional file pattern to filter which files to search (e.g., "*.js")

    Returns:
        A properly escaped shell command string for ripgrep
    """
    # Use shlex.quote to properly escape all shell special characters
    quoted_pattern = shlex.quote(pattern)
    path_arg = shlex.quote(path) if path else '.'

    # Build ripgrep command
    rg_cmd = f'rg -li {quoted_pattern} --sortr=modified'

    if include:
        quoted_include = shlex.quote(include)
        rg_cmd += f' --glob {quoted_include}'

    # Build the complete command
    complete_cmd = f'{rg_cmd} {path_arg} | head -n 100'

    # Add a header to the output
    echo_cmd = f'echo "Below are the execution results of the search command: {complete_cmd}\n"; '
    return echo_cmd + complete_cmd


def glob_to_cmdrun(pattern: str, path: str = '.') -> str:
    # NOTE: This function currently relies on `rg` (ripgrep).
    # `rg` may not be installed when using CLIRuntime or LocalRuntime
    # TODO: Implement a fallback to `find` if `rg` is not available.
    """Convert glob tool arguments to a shell command string.

    Args:
        pattern: The glob pattern to match files (e.g., "**/*.js")
        path: The directory to search in (defaults to current directory)

    Returns:
        A properly escaped shell command string for ripgrep implementing glob
    """
    # Use shlex.quote to properly escape all shell special characters
    quoted_path = shlex.quote(path)
    quoted_pattern = shlex.quote(pattern)

    # Use ripgrep in a glob-only mode with -g flag and --files to list files
    # This most closely matches the behavior of the NodeJS glob implementation
    rg_cmd = f'rg --files {quoted_path} -g {quoted_pattern} --sortr=modified'

    # Sort results and limit to 100 entries (matching the Node.js implementation)
    sort_and_limit_cmd = ' | head -n 100'

    complete_cmd = f'{rg_cmd}{sort_and_limit_cmd}'

    # Add a header to the output
    echo_cmd = f'echo "Below are the execution results of the glob command: {complete_cmd}\n"; '
    return echo_cmd + complete_cmd


def remove_tool_call_tags(content: str) -> str:
    """Remove <tool_call>...</tool_call> tags from content.
    
    This is used after extracting tool calls from thinking model responses
    to ensure the tool_call tags don't appear in message content.
    
    Args:
        content: The content string that may contain <tool_call> tags
        
    Returns:
        Content with all <tool_call> tags removed
    """
    if not content or not isinstance(content, str):
        return content
    
    # Pattern to match <tool_call>...</tool_call> tags (handles multi-line)
    pattern = r'<tool_call>\s*.*?\s*</tool_call>'
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Clean up extra whitespace
    cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content).strip()
    
    return cleaned_content


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    
    # Get tool calls from the standard tool_calls field
    tool_calls = getattr(assistant_msg, 'tool_calls', None) or []
    
    # Track if tool calls were extracted from content (thinking model case)
    tool_calls_extracted_from_content = False
    
    # Initialize variables for reasoning extraction (needed for summary logging)
    reasoning_stripped = ''
    
    # Get content string for processing
    content_str = ''
    if isinstance(assistant_msg.content, str):
        content_str = assistant_msg.content
    elif isinstance(assistant_msg.content, list):
        for msg in assistant_msg.content:
            if isinstance(msg, dict) and msg.get('type') == 'text':
                content_str += msg.get('text', '')
            elif isinstance(msg, str):
                content_str += msg
    
    logger.debug(f'Content type: {type(assistant_msg.content)}, Content length: {len(content_str)}')
    logger.debug(f'Standard tool_calls field: {len(tool_calls) if tool_calls else 0} tool call(s)')
    
    # If no tool calls in standard field, check content for embedded tool calls
    # This handles thinking models that output tool calls in content
    # Format: [reasoning]</think><tool_call>{json}</tool_call>
    if not tool_calls and content_str:
        # Check for tool call tags in content
        has_tool_call_tags = '<tool_call>' in content_str or '</tool_call>' in content_str
        has_reasoning_tags = '</think>' in content_str or '</think>' in content_str or '</reasoning>' in content_str
        
        logger.debug(f'Checking content for embedded tool calls. Has <tool_call> tags: {has_tool_call_tags}, Has reasoning tags: {has_reasoning_tags}')
        
        if has_tool_call_tags:
            logger.debug(f'Content preview (first 500 chars): {content_str[:500]}...')
            logger.debug(f'Content preview (last 500 chars): ...{content_str[-500:]}')
        
            extracted_tool_calls = extract_tool_calls_from_content(content_str)
            if extracted_tool_calls:
                tool_calls = extracted_tool_calls
            tool_calls_extracted_from_content = True
            logger.info(f'✅ Extracted {len(tool_calls)} tool call(s) from content')
            for i, tc in enumerate(tool_calls):
                if hasattr(tc, 'function'):
                    func = tc.function
                    if isinstance(func, dict):
                        logger.info(f'  Tool call {i+1}: name={func.get("name")}, args={func.get("arguments", "")[:100]}...')
                    elif hasattr(func, 'name'):
                        logger.info(f'  Tool call {i+1}: name={func.name}, args={func.arguments[:100] if hasattr(func, "arguments") else "N/A"}...')
        else:
            # Log if we expected to find tool calls but didn't
            if has_tool_call_tags:
                logger.warning('❌ Found <tool_call> tags in content but failed to extract tool calls')
                logger.warning(f'Content around tags: {content_str[max(0, content_str.find("<tool_call>")-100):content_str.find("</tool_call>")+100]}')

    if tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        # Extract reasoning traces if present
        # For thinking models: everything before </think> is reasoning
        thought = ''
        reasoning = ''
        
        logger.info(f'Processing response with {len(tool_calls)} tool call(s). Tool calls extracted from content: {tool_calls_extracted_from_content}')
        
        if content_str:
            # Extract reasoning first (everything before </think>)
            cleaned_content, reasoning = extract_reasoning_from_content(content_str)
            # Remove tool_call tags if they were extracted from content
            if tool_calls_extracted_from_content:
                cleaned_content = remove_tool_call_tags(cleaned_content)
            thought = cleaned_content
        
        # Strip reasoning to check if it's non-empty
        reasoning_stripped = reasoning.strip() if reasoning else ''
        
        # Log reasoning extraction results
        if reasoning_stripped:
            logger.info(f'Extracted reasoning trace: {len(reasoning_stripped)} characters')
            logger.debug(f'Reasoning preview (first 200 chars): {reasoning_stripped[:200]}...')
        else:
            logger.debug('No reasoning trace found in content')
        
        # For thinking models: if reasoning exists and tool calls were extracted from content,
        # create a separate AgentThinkAction as the first action
        # The reasoning trace IS the "think" tool call for thinking models
        if reasoning_stripped and tool_calls_extracted_from_content:
            logger.info(f'Creating AgentThinkAction from reasoning trace ({len(reasoning_stripped)} chars) - this represents the thinking model\'s reasoning process')
            think_action = AgentThinkAction(thought=reasoning_stripped)
            think_action.response_id = response.id
            actions.append(think_action)
            logger.info(f'✅ Created AgentThinkAction for reasoning trace from thinking model (thought length: {len(reasoning_stripped)} chars)')
        
        # For regular models or when reasoning should be combined with thought:
        # Combine reasoning with thought if present (but only if we didn't create a separate think action)
        if reasoning_stripped and not tool_calls_extracted_from_content:
            thought = f'{reasoning_stripped}\n{thought}' if thought else reasoning_stripped

        # Process each tool call to OpenHands action
        logger.info(f'Processing {len(tool_calls)} tool call(s) from response:')
        for i, tool_call in enumerate(tool_calls):
            action: Action
            logger.info(f'  [{i+1}/{len(tool_calls)}] Processing tool call {i+1}')
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                # Handle both dict and object access for function attribute
                if not hasattr(tool_call, 'function'):
                    raise FunctionCallValidationError(f'Tool call missing function attribute: {tool_call}')
                
                function_obj = tool_call.function
                
                # Try multiple methods to access function name and arguments
                function_name = None
                function_arguments = None
                
                # Method 1: Try dict access
                if isinstance(function_obj, dict):
                    function_name = function_obj.get('name', '')
                    function_arguments = function_obj.get('arguments', '{}')
                # Method 2: Try attribute access (for litellm objects)
                elif hasattr(function_obj, 'name'):
                    function_name = getattr(function_obj, 'name', '')
                    function_arguments = getattr(function_obj, 'arguments', '{}')
                # Method 3: Try dict-style get method
                elif hasattr(function_obj, 'get') and callable(getattr(function_obj, 'get', None)):
                    function_name = function_obj.get('name', '')
                    function_arguments = function_obj.get('arguments', '{}')
                # Method 4: Try accessing as dict-like object
                elif hasattr(function_obj, '__getitem__'):
                    try:
                        function_name = function_obj['name']
                        function_arguments = function_obj['arguments']
                    except (KeyError, TypeError):
                        pass
                
                # Validate we got the values
                if function_name is None or function_arguments is None:
                    raise FunctionCallValidationError(
                        f'Unable to access function name/arguments from tool call. Function type: {type(function_obj)}, value: {function_obj}'
                    )
                
                # Ensure function_arguments is a string
                if not isinstance(function_arguments, str):
                    function_arguments = json.dumps(function_arguments) if function_arguments else '{}'
                
                arguments = json.loads(function_arguments)
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {function_arguments}'
                ) from e

            # ================================================
            # AgentFinishAction
            # ================================================
            if function_name == FinishTool['function']['name']:
                action = AgentFinishAction(
                    final_thought=arguments.get('message', ''),
                )

            # ================================================
            # ViewTool (ACI-based file viewer, READ-ONLY)
            # ================================================
            elif function_name == ViewTool['function']['name']:
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {function_name}'
                    )
                action = FileReadAction(
                    path=arguments['path'],
                    impl_source=FileReadSource.OH_ACI,
                    view_range=arguments.get('view_range', None),
                )

            # ================================================
            # str_replace_editor with command='view' (READ-ONLY mapping)
            # ================================================
            # Some models (especially thinking models) may output str_replace_editor
            # even when using ReadOnlyAgent. We map the 'view' command to FileReadAction.
            elif function_name == STR_REPLACE_EDITOR_TOOL_NAME:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {function_name}'
                    )
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {function_name}'
                    )
                
                command = arguments['command']
                path = arguments['path']
                
                # Only allow 'view' command in ReadOnlyAgent (read-only operation)
                if command == 'view':
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=arguments.get('view_range', None),
                    )
                    logger.debug(f'Mapped str_replace_editor with command=view to FileReadAction for path: {path}')
                else:
                    # Reject any write operations (create, str_replace, insert, undo_edit)
                    raise FunctionCallValidationError(
                        f'ReadOnlyAgent does not support str_replace_editor command "{command}". '
                        f'Only "view" command is allowed. Use CodeActAgent for file editing operations.'
                )

            # ================================================
            # AgentThinkAction
            # ================================================
            elif function_name == ThinkTool['function']['name']:
                action = AgentThinkAction(thought=arguments.get('thought', ''))

            # ================================================
            # GrepTool (file content search)
            # ================================================
            elif function_name == GrepTool['function']['name']:
                if 'pattern' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "pattern" in tool call {function_name}'
                    )

                pattern = arguments['pattern']
                path = arguments.get('path')
                include = arguments.get('include')

                grep_cmd = grep_to_cmdrun(pattern, path, include)
                action = CmdRunAction(command=grep_cmd, is_input=False)

            # ================================================
            # GlobTool (file pattern matching)
            # ================================================
            elif function_name == GlobTool['function']['name']:
                if 'pattern' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "pattern" in tool call {function_name}'
                    )

                pattern = arguments['pattern']
                path = arguments.get('path', '.')

                glob_cmd = glob_to_cmdrun(pattern, path)
                action = CmdRunAction(command=glob_cmd, is_input=False)

            # ================================================
            # MCPAction (MCP)
            # ================================================
            elif mcp_tool_names and function_name in mcp_tool_names:
                action = MCPAction(
                    name=function_name,
                    arguments=arguments,
                )

            else:
                raise FunctionCallNotExistsError(
                    f'Tool {function_name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
                )

            # We only add thought to the first action, and only if we didn't create a separate think action
            # (for thinking models, reasoning is already in the separate think action)
            # Note: reasoning_stripped is computed above before the loop
            if i == 0 and not (tool_calls_extracted_from_content and reasoning_stripped):
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            # Safely get tool_call.id (handle both dict and object access)
            tool_call_id = getattr(tool_call, 'id', None)
            if tool_call_id is None and isinstance(tool_call, dict):
                tool_call_id = tool_call.get('id', f'call_{uuid.uuid4().hex[:16]}')
            elif tool_call_id is None:
                tool_call_id = f'call_{uuid.uuid4().hex[:16]}'
            
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call_id,
                function_name=function_name,
                model_response=response,
                total_calls_in_response=len(tool_calls),
            )
            actions.append(action)
            logger.info(f'  ✅ [{i+1}/{len(tool_calls)}] Successfully created {type(action).__name__} for tool: {function_name}')
    else:
        actions.append(
            MessageAction(
                content=str(assistant_msg.content) if assistant_msg.content else '',
                wait_for_response=True,
            )
        )

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    for action in actions:
        action.response_id = response.id

    # Safety check: ensure we always have at least one action
    # This should never happen, but if it does, create a MessageAction as fallback
    if len(actions) == 0:
        logger.warning('No actions were created from response, creating fallback MessageAction')
        fallback_action = MessageAction(
            content=str(assistant_msg.content) if assistant_msg.content else '',
            wait_for_response=True,
        )
        fallback_action.response_id = response.id
        actions.append(fallback_action)

    # Log summary of all created actions
    action_types = [type(a).__name__ for a in actions]
    action_summary = ', '.join(f'{action_types.count(t)}x {t}' for t in set(action_types))
    logger.info(f'📋 Summary: Created {len(actions)} action(s) from response: {action_summary}')
    # Check if we have reasoning and tool calls (variables may not exist if no tool_calls block executed)
    if 'reasoning_stripped' in locals() and 'tool_calls_extracted_from_content' in locals():
        if tool_calls_extracted_from_content and reasoning_stripped:
            logger.info(f'   └─ Thinking model detected: reasoning trace ({len(reasoning_stripped)} chars) + {len(tool_calls) if "tool_calls" in locals() else 0} tool call(s)')

    assert len(actions) >= 1
    return actions


def get_tools() -> list[ChatCompletionToolParam]:
    return [
        ThinkTool,
        FinishTool,
        GrepTool,
        GlobTool,
        ViewTool,
    ]
