"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import shlex

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


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    
    # Get tool calls from the standard tool_calls field
    tool_calls = getattr(assistant_msg, 'tool_calls', None) or []
    
    # If no tool calls in standard field, check content for embedded tool calls
    # This handles thinking models that output tool calls in content
    if not tool_calls:
        content_str = ''
        if isinstance(assistant_msg.content, str):
            content_str = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if isinstance(msg, dict) and msg.get('type') == 'text':
                    content_str += msg.get('text', '')
                elif isinstance(msg, str):
                    content_str += msg
        
        if content_str:
            extracted_tool_calls = extract_tool_calls_from_content(content_str)
            if extracted_tool_calls:
                tool_calls = extracted_tool_calls
                logger.debug(f'Extracted {len(tool_calls)} tool call(s) from content')

    if tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        # Extract reasoning traces if present
        thought = ''
        reasoning = ''
        if isinstance(assistant_msg.content, str):
            cleaned_content, reasoning = extract_reasoning_from_content(assistant_msg.content)
            thought = cleaned_content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if isinstance(msg, dict) and msg.get('type') == 'text':
                    text_content = msg.get('text', '')
                    cleaned_text, extracted_reasoning = extract_reasoning_from_content(text_content)
                    thought += cleaned_text
                    if extracted_reasoning:
                        reasoning += extracted_reasoning + '\n'
                elif isinstance(msg, str):
                    cleaned_text, extracted_reasoning = extract_reasoning_from_content(msg)
                    thought += cleaned_text
                    if extracted_reasoning:
                        reasoning += extracted_reasoning + '\n'
        
        # Combine reasoning with thought if present
        if reasoning:
            thought = f'{reasoning}\n{thought}' if thought else reasoning

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(tool_calls):
            action: Action
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                # Handle both dict and object access for function attribute
                if not hasattr(tool_call, 'function'):
                    raise FunctionCallValidationError(f'Tool call missing function attribute: {tool_call}')
                
                function_obj = tool_call.function
                
                # Try to access as dict first (for manually created tool calls)
                if isinstance(function_obj, dict):
                    function_name = function_obj.get('name', '')
                    function_arguments = function_obj.get('arguments', '{}')
                # Try to access as object with attributes (for litellm-created tool calls)
                elif hasattr(function_obj, 'name') and hasattr(function_obj, 'arguments'):
                    function_name = function_obj.name
                    function_arguments = function_obj.arguments
                # Fallback: try dict-style access if it supports it
                elif hasattr(function_obj, 'get'):
                    function_name = function_obj.get('name', '')
                    function_arguments = function_obj.get('arguments', '{}')
                else:
                    raise FunctionCallValidationError(
                        f'Unable to access function name/arguments from tool call. Function type: {type(function_obj)}, value: {function_obj}'
                    )
                
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

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=function_name,
                model_response=response,
                total_calls_in_response=len(tool_calls),
            )
            actions.append(action)
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
