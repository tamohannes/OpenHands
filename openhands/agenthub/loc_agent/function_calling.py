"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
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
from openhands.agenthub.codeact_agent.tools import FinishTool
from openhands.agenthub.loc_agent.tools import (
    SearchEntityTool,
    SearchRepoTool,
    create_explore_tree_structure_tool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.tool import ToolCallMetadata


def response_to_actions(
    response: ModelResponse,
    mcp_tool_names: list[str] | None = None,
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
                    raise RuntimeError(f'Tool call missing function attribute: {tool_call}')
                
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
                    raise RuntimeError(
                        f'Unable to access function name/arguments from tool call. Function type: {type(function_obj)}, value: {function_obj}'
                    )
                
                # Ensure function_arguments is a string
                if not isinstance(function_arguments, str):
                    import json
                    function_arguments = json.dumps(function_arguments) if function_arguments else '{}'
                
                arguments = json.loads(function_arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {function_arguments}'
                ) from e

            # ================================================
            # LocAgent's Tools
            # ================================================
            ALL_FUNCTIONS = [
                'explore_tree_structure',
                'search_code_snippets',
                'get_entity_contents',
            ]
            if function_name in ALL_FUNCTIONS:
                # We implement this in agent_skills, which can be used via Jupyter
                func_name = function_name
                code = f'print({func_name}(**{arguments}))'
                logger.debug(f'TOOL CALL: {func_name} with code: {code}')
                action = IPythonRunCellAction(code=code)

            # ================================================
            # AgentFinishAction
            # ================================================
            elif function_name == FinishTool['function']['name']:
                action = AgentFinishAction(
                    final_thought=arguments.get('message', ''),
                )
            else:
                raise FunctionCallNotExistsError(
                    f'Tool {function_name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
                )

            # We only add thought to the first action
            if i == 0:
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
    tools = [FinishTool]
    tools.append(SearchRepoTool)
    tools.append(SearchEntityTool)
    tools.append(create_explore_tree_structure_tool(use_simplified_description=True))
    return tools
