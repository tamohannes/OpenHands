"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import re
import uuid

from litellm import (
    ChatCompletionMessageToolCall,
    ModelResponse,
)

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CondensationRequestTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.security_utils import RISK_LEVELS
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger

# Set up a dedicated file handler for tool call debugging
# This writes to a file in the trajectory directory for each task instance
_tool_call_debug_handler = None
_tool_call_debug_log_file = None

def _find_trajectory_directory():
    """Find the trajectory directory for the current task instance.
    
    Looks for common patterns:
    - eval-results/*/trajectories/*/ (evaluation output structure)
    - Any directory containing output.jsonl (trajectory output file)
    - Falls back to /tmp if not found
    """
    cwd = os.getcwd()
    
    # Check if we're already in a trajectory directory (has output.jsonl)
    if os.path.exists(os.path.join(cwd, 'output.jsonl')):
        return cwd
    
    # Check parent directories for trajectory structure
    current = Path(cwd)
    for parent in [current] + list(current.parents)[:5]:  # Check up to 5 levels up
        # Check for eval-results/*/trajectories/*/ pattern
        if 'trajectories' in str(parent):
            # Check if this directory or a subdirectory has output.jsonl
            for subdir in [parent] + [p for p in parent.iterdir() if p.is_dir()]:
                if (subdir / 'output.jsonl').exists():
                    return str(subdir)
            # If we're in a trajectories directory, use it
            if (parent / 'output.jsonl').exists():
                return str(parent)
        
        # Check if this directory has output.jsonl
        if (parent / 'output.jsonl').exists():
            return str(parent)
    
    # Fall back to /tmp
    return '/tmp'

def _setup_tool_call_debug_logging():
    """Set up a file handler for tool call debugging.
    
    Writes to tool_call_debug.log in the trajectory directory for the current task instance.
    """
    global _tool_call_debug_handler, _tool_call_debug_log_file
    if _tool_call_debug_handler is None:
        try:
            # Find the trajectory directory
            trajectory_dir = _find_trajectory_directory()
            log_file = os.path.join(trajectory_dir, 'tool_call_debug.log')
            _tool_call_debug_log_file = log_file
            
            # Create directory if it doesn't exist
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            _tool_call_debug_handler = logging.FileHandler(
                log_file,
                mode='a',
                encoding='utf-8'
            )
            _tool_call_debug_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s:%(levelname)s - %(filename)s:%(lineno)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            _tool_call_debug_handler.setFormatter(formatter)
            
            # Create a separate logger for tool call debugging
            tool_call_logger = logging.getLogger('openhands.tool_calls')
            tool_call_logger.setLevel(logging.DEBUG)
            tool_call_logger.addHandler(_tool_call_debug_handler)
            tool_call_logger.propagate = False
            
            logger.info(f'Tool call debug logging enabled: {log_file}')
        except Exception as e:
            logger.warning(f'Failed to set up tool call debug logging: {e}')

# Create a logger specifically for tool call debugging (lazy initialization)
tool_call_debug_logger = logging.getLogger('openhands.tool_calls')

def _ensure_tool_call_debug_logging():
    """Ensure tool call debug logging is set up."""
    global _tool_call_debug_handler
    if _tool_call_debug_handler is None:
        _setup_tool_call_debug_logging()

from openhands.events.action import (
    Action,
    ActionSecurityRisk,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
    TaskTrackingAction,
)
from openhands.events.action.agent import CondensationRequestAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import TASK_TRACKER_TOOL_NAME


def _get_available_tool_names(mcp_tool_names: list[str] | None = None) -> list[str]:
    """Get a list of all available tool names.
    
    This includes both built-in CodeActAgent tools and MCP tools.
    
    Args:
        mcp_tool_names: Optional list of MCP tool names
        
    Returns:
        List of all available tool names
    """
    tool_names = [
        create_cmd_run_tool()['function']['name'],
        ThinkTool['function']['name'],
        FinishTool['function']['name'],
        CondensationRequestTool['function']['name'],
        BrowserTool['function']['name'],
        IPythonTool['function']['name'],
        LLMBasedFileEditTool['function']['name'],
        create_str_replace_editor_tool()['function']['name'],
        TASK_TRACKER_TOOL_NAME,
        'delegate_to_browsing_agent',  # Special case tool
    ]
    
    if mcp_tool_names:
        tool_names.extend(mcp_tool_names)
    
    return tool_names


def _fuzzy_match_tool_name(called_name: str, available_tool_names: list[str]) -> str | None:
    """Try to match a tool name using simple heuristics.
    
    Common patterns handled:
    - Plural vs singular (read_files -> read_file)
    - Underscore variations (read_file vs readfile)
    - Common typos and variations
    
    Args:
        called_name: The tool name that was called
        available_tool_names: List of available tool names
        
    Returns:
        Matched tool name if found, None otherwise
    """
    called_lower = called_name.lower()
    available_lower = [name.lower() for name in available_tool_names]
    
    # Exact match (case-insensitive)
    if called_lower in available_lower:
        idx = available_lower.index(called_lower)
        return available_tool_names[idx]
    
    # Try removing trailing 's' (plural -> singular)
    if called_lower.endswith('s') and len(called_lower) > 1:
        singular = called_lower[:-1]
        if singular in available_lower:
            idx = available_lower.index(singular)
            logger.info(f'Matched tool name "{called_name}" -> "{available_tool_names[idx]}" (plural -> singular)')
            return available_tool_names[idx]
    
    # Try adding trailing 's' (singular -> plural, less common but possible)
    plural = called_lower + 's'
    if plural in available_lower:
        idx = available_lower.index(plural)
        logger.info(f'Matched tool name "{called_name}" -> "{available_tool_names[idx]}" (singular -> plural)')
        return available_tool_names[idx]
    
    # Try removing underscores
    no_underscore = called_lower.replace('_', '')
    for i, avail_lower in enumerate(available_lower):
        if avail_lower.replace('_', '') == no_underscore:
            logger.info(f'Matched tool name "{called_name}" -> "{available_tool_names[i]}" (underscore variation)')
            return available_tool_names[i]
    
    # Try adding underscores (simple heuristic: add underscore before last word)
    # This is less reliable, so we only do it for 2+ word names
    if '_' not in called_lower and len(called_lower.split()) >= 2:
        # Try common patterns like "readfile" -> "read_file"
        # This is a simple heuristic and might not catch all cases
        words = called_lower.split()
        if len(words) == 2:
            with_underscore = f"{words[0]}_{words[1]}"
            if with_underscore in available_lower:
                idx = available_lower.index(with_underscore)
                logger.info(f'Matched tool name "{called_name}" -> "{available_tool_names[idx]}" (added underscore)')
                return available_tool_names[idx]
    
    # Try substring matching (if called name is a substring of available name)
    # Only if the substring is at least 5 characters to avoid false positives
    if len(called_lower) >= 5:
        for i, avail_lower in enumerate(available_lower):
            if called_lower in avail_lower or avail_lower in called_lower:
                # Check if it's a reasonable match (not just a common substring)
                # Require at least 80% character overlap
                overlap = min(len(called_lower), len(avail_lower))
                if overlap >= max(len(called_lower), len(avail_lower)) * 0.8:
                    logger.info(f'Matched tool name "{called_name}" -> "{available_tool_names[i]}" (substring match)')
                    return available_tool_names[i]
    
    return None


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought and action.thought:
        action.thought = f'{thought}\n{action.thought}'
    elif thought:
        action.thought = thought
    return action


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


def set_security_risk(action: Action, arguments: dict) -> None:
    """Set the security risk level for the action."""

    # Set security_risk attribute if provided
    if 'security_risk' in arguments:
        if arguments['security_risk'] in RISK_LEVELS:
            if hasattr(action, 'security_risk'):
                action.security_risk = getattr(
                    ActionSecurityRisk, arguments['security_risk']
                )
        else:
            logger.warning(f'Invalid security_risk value: {arguments["security_risk"]}')


def extract_tool_calls_from_content(content: str, mcp_tool_names: list[str] | None = None) -> list[ChatCompletionMessageToolCall]:
    """Extract tool calls from content field when embedded in thinking model responses.
    
    Thinking models (e.g., Qwen3-Thinking) output tool calls embedded in the content
    field within <tool_call> tags rather than in the tool_calls field. This function
    parses those embedded tool calls and converts them to ChatCompletionMessageToolCall format.
    
    Expected format:
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>
    
    Args:
        content: The content string that may contain embedded tool calls
        
    Returns:
        List of ChatCompletionMessageToolCall objects extracted from content, or empty list if none found
    """
    if not content or not isinstance(content, str):
        return []
    
    tool_calls = []
    
    # Pattern to match <tool_call>...</tool_call> tags
    # This handles both single-line and multi-line tool calls
    # Use non-greedy matching with DOTALL flag to handle multi-line JSON
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        # Check if tags exist but pattern didn't match (might be malformed)
        if '<tool_call>' in content or '</tool_call>' in content:
            logger.debug('Found <tool_call> tags but pattern did not match')
        return []
    
    logger.debug(f'Found {len(matches)} potential tool call(s) in content')
    
    for i, match in enumerate(matches):
        tool_call_content = match.group(1).strip()
        if not tool_call_content:
            logger.warning(f'Tool call {i+1} has empty content between tags')
            continue
        
        try:
            # Parse the JSON tool call data
            # Try to parse as-is first
            try:
                tool_call_data = json.loads(tool_call_content)
            except json.JSONDecodeError:
                # Sometimes the JSON might have extra whitespace or newlines
                # Try cleaning it up
                cleaned = tool_call_content.strip()
                # Remove any leading/trailing whitespace and newlines
                cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
                tool_call_data = json.loads(cleaned)
            
            # Extract name and arguments
            tool_name = tool_call_data.get('name', '')
            tool_arguments = tool_call_data.get('arguments', {})
            
            # Handle case where arguments might be missing (default to empty dict)
            if tool_arguments is None:
                tool_arguments = {}
            
            # Convert arguments to JSON string if it's a dict
            if isinstance(tool_arguments, dict):
                arguments_str = json.dumps(tool_arguments)
            elif isinstance(tool_arguments, str):
                # If it's already a string, try to validate it's valid JSON
                try:
                    json.loads(tool_arguments)  # Validate it's valid JSON
                    arguments_str = tool_arguments
                except json.JSONDecodeError:
                    logger.warning(f'Tool arguments string is not valid JSON, wrapping in object')
                    arguments_str = json.dumps({'raw': tool_arguments})
            else:
                logger.warning(f'Unexpected tool arguments type: {type(tool_arguments)}, converting to string')
                arguments_str = json.dumps(str(tool_arguments))
            
            if not tool_name:
                logger.warning(f'Tool call {i+1} missing name field. Data: {tool_call_data}')
                continue
            
            # Apply fuzzy matching to tool name during extraction
            # This ensures we fix tool names early, before processing
            # Include MCP tools if available for better matching
            available_tool_names = _get_available_tool_names(mcp_tool_names)
            matched_tool_name = _fuzzy_match_tool_name(tool_name, available_tool_names)
            if matched_tool_name and matched_tool_name != tool_name:
                logger.info(f'✅ Fixed tool name during extraction: "{tool_name}" -> "{matched_tool_name}"')
                tool_name = matched_tool_name
            elif not matched_tool_name:
                # Log warning if tool name doesn't match - this will help debug issues
                logger.warning(f'⚠️ Tool name "{tool_name}" not found in available tools during extraction. Will retry with fuzzy matching during processing.')
            
            # Create a ChatCompletionMessageToolCall object
            # The function attribute can be a dict or an object - we'll use dict format
            # which is compatible with how litellm handles it
            tool_call = ChatCompletionMessageToolCall(
                id=f'call_{uuid.uuid4().hex[:16]}',
                type='function',
                function={
                    'name': tool_name,
                    'arguments': arguments_str,
                },
            )
            tool_calls.append(tool_call)
            logger.info(f'Extracted embedded tool call {i+1}: {tool_name} with arguments: {arguments_str[:100]}...')
            
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse tool call {i+1} JSON. Content (first 200 chars): {tool_call_content[:200]}... Error: {e}')
            continue
        except Exception as e:
            logger.error(f'Error extracting tool call {i+1}: {e}. Content (first 200 chars): {tool_call_content[:200]}...')
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    return tool_calls


def extract_reasoning_from_content(content: str) -> tuple[str, str]:
    """Extract reasoning traces from content and return cleaned content and reasoning.
    
    Thinking models output reasoning traces with only closing tags (no opening tags).
    Everything before the closing tag is considered reasoning.
    
    Expected format:
        [reasoning text - everything before the closing tag]
        </think>
        [rest of content, possibly with tool calls]
    
    Args:
        content: The content string that may contain reasoning traces
        
    Returns:
        Tuple of (cleaned_content, reasoning_text)
    """
    if not content or not isinstance(content, str):
        return content, ''
    
    reasoning = ''
    cleaned_content = content
    
    # Priority order: check for standalone closing tags first (most common case)
    # These tags typically don't have opening tags - everything before them is reasoning
    # The user confirmed that reasoning tags only have closing tags like </think>
    # Note: Qwen3 thinking models use </think> tag (not </think>)
    standalone_closing_tags = ['</think>', '</reasoning>']
    
    for closing_tag in standalone_closing_tags:
        if closing_tag in cleaned_content:
            idx = cleaned_content.find(closing_tag)
            if idx > 0:
                # Everything before the closing tag is reasoning
                reasoning = cleaned_content[:idx].strip()
                # Remove everything up to and including the closing tag
                cleaned_content = cleaned_content[idx + len(closing_tag):].strip()
                logger.debug(f'Extracted reasoning from standalone {closing_tag} tag (length: {len(reasoning)})')
                break  # Only process the first matching closing tag
    
    # Also handle paired tags if they exist (less common but possible)
    # This is a fallback in case some models use paired tags
    if not reasoning:
        reasoning_patterns = [
            (r'<think>(.*?)</think>', '</think>'),
            (r'<reasoning>(.*?)</reasoning>', '</reasoning>'),
            (r'<think>(.*?)</think>', '</think>'),
        ]
        
        for pattern, closing_tag in reasoning_patterns:
            matches = list(re.finditer(pattern, cleaned_content, re.DOTALL))
            for match in matches:
                reasoning_text = match.group(1).strip()
                if reasoning_text:
                    reasoning = reasoning_text
                    # Remove the reasoning tag from content
                    cleaned_content = cleaned_content.replace(match.group(0), '')
                    logger.debug(f'Extracted reasoning from paired tags with {closing_tag} (length: {len(reasoning)})')
                    break
            if reasoning:
                break
    
    # Clean up extra whitespace
    cleaned_content = re.sub(r'\n\s*\n', '\n', cleaned_content).strip()
    
    return cleaned_content, reasoning


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
        has_reasoning_tags = '</think>' in content_str or '</reasoning>' in content_str
        
        logger.debug(f'Checking content for embedded tool calls. Has <tool_call> tags: {has_tool_call_tags}, Has reasoning tags: {has_reasoning_tags}')
        
        if has_tool_call_tags:
            logger.debug(f'Content preview (first 500 chars): {content_str[:500]}...')
            logger.debug(f'Content preview (last 500 chars): ...{content_str[-500:]}')
        
        extracted_tool_calls = extract_tool_calls_from_content(content_str, mcp_tool_names)
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
        cleaned_content = ''  # Initialize for use in fallback case
        
        if content_str:
            # Extract reasoning first (everything before </think>)
            cleaned_content, reasoning = extract_reasoning_from_content(content_str)
            # Remove tool_call tags if they were extracted from content
            if tool_calls_extracted_from_content:
                cleaned_content = remove_tool_call_tags(cleaned_content)
            thought = cleaned_content
        
        # Strip reasoning to check if it's non-empty
        reasoning_stripped = reasoning.strip() if reasoning else ''
        
        # For thinking models: if reasoning exists and tool calls were extracted from content,
        # create a separate AgentThinkAction as the first action
        if reasoning_stripped and tool_calls_extracted_from_content:
            think_action = AgentThinkAction(thought=reasoning_stripped)
            think_action.response_id = response.id
            actions.append(think_action)
            logger.info('Created AgentThinkAction for reasoning trace from thinking model')
        
        # For regular models or when reasoning should be combined with thought:
        # Combine reasoning with thought if present (but only if we didn't create a separate think action)
        if reasoning_stripped and not tool_calls_extracted_from_content:
            thought = f'{reasoning_stripped}\n{thought}' if thought else reasoning_stripped

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(tool_calls):
            action: Action | None = None
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                # Handle both dict and object access for function attribute
                # First, safely get the function attribute
                if not hasattr(tool_call, 'function'):
                    raise FunctionCallValidationError(
                        f'Tool call missing function attribute: {tool_call}'
                    )
                
                function_obj = tool_call.function
                
                # Try multiple methods to access function name and arguments
                # This handles both dict and object types from litellm
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
                        f'Unable to access function name/arguments from tool call. Function type: {type(function_obj)}, value: {function_obj}, hasattr name: {hasattr(function_obj, "name") if hasattr(function_obj, "__class__") else "N/A"}'
                    )
                
                # Apply fuzzy matching to tool name - always do this to catch any issues
                # For extracted tool calls, fuzzy matching was already attempted during extraction
                # But we retry here in case MCP tools weren't available during extraction
                # For standard tool calls, we always do fuzzy matching here
                original_function_name = function_name
                available_tool_names = _get_available_tool_names(mcp_tool_names)
                matched_name = _fuzzy_match_tool_name(function_name, available_tool_names)
                if matched_name and matched_name != function_name:
                    logger.info(f'✅ Fixed tool name during processing: "{original_function_name}" -> "{matched_name}"')
                    function_name = matched_name
                
                # Ensure function_arguments is a string
                if not isinstance(function_arguments, str):
                    function_arguments = json.dumps(function_arguments) if function_arguments else '{}'
                
                arguments = json.loads(function_arguments)
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {function_arguments}'
                ) from e

                # ================================================
                # CmdRunTool (Bash)
                # ================================================
                if function_name == create_cmd_run_tool()['function']['name']:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {function_name}'
                        )
                    # convert is_input to boolean
                    is_input = arguments.get('is_input', 'false') == 'true'
                    action = CmdRunAction(command=arguments['command'], is_input=is_input)

                    # Set hard timeout if provided
                    if 'timeout' in arguments:
                        try:
                            action.set_hard_timeout(float(arguments['timeout']))
                        except ValueError as e:
                            raise FunctionCallValidationError(
                                f"Invalid float passed to 'timeout' argument: {arguments['timeout']}"
                            ) from e
                    set_security_risk(action, arguments)

                # ================================================
                # IPythonTool (Jupyter)
                # ================================================
                elif function_name == IPythonTool['function']['name']:
                    if 'code' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "code" in tool call {function_name}'
                        )
                    action = IPythonRunCellAction(code=arguments['code'])
                    set_security_risk(action, arguments)

                # ================================================
                # AgentDelegateAction (Delegation to another agent)
                # ================================================
                elif function_name == 'delegate_to_browsing_agent':
                    action = AgentDelegateAction(
                        agent='BrowsingAgent',
                        inputs=arguments,
                    )

                # ================================================
                # AgentFinishAction
                # ================================================
                elif function_name == FinishTool['function']['name']:
                    action = AgentFinishAction(
                        final_thought=arguments.get('message', ''),
                    )

                # ================================================
                # LLMBasedFileEditTool (LLM-based file editor, deprecated)
                # ================================================
                elif function_name == LLMBasedFileEditTool['function']['name']:
                    if 'path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "path" in tool call {function_name}'
                        )
                    if 'content' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "content" in tool call {function_name}'
                        )
                    action = FileEditAction(
                        path=arguments['path'],
                        content=arguments['content'],
                        start=arguments.get('start', 1),
                        end=arguments.get('end', -1),
                        impl_source=arguments.get(
                            'impl_source', FileEditSource.LLM_BASED_EDIT
                        ),
                    )
                elif (
                    function_name
                    == create_str_replace_editor_tool()['function']['name']
                ):
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {function_name}'
                        )
                    if 'path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "path" in tool call {function_name}'
                        )
                    path = arguments['path']
                    command = arguments['command']
                    other_kwargs = {
                        k: v for k, v in arguments.items() if k not in ['command', 'path']
                    }

                    if command == 'view':
                        action = FileReadAction(
                            path=path,
                            impl_source=FileReadSource.OH_ACI,
                            view_range=other_kwargs.get('view_range', None),
                        )
                    else:
                        if 'view_range' in other_kwargs:
                            # Remove view_range from other_kwargs since it is not needed for FileEditAction
                            other_kwargs.pop('view_range')

                        # Filter out unexpected arguments
                        valid_kwargs_for_editor = {}
                        # Get valid parameters from the str_replace_editor tool definition
                        str_replace_editor_tool = create_str_replace_editor_tool()
                        valid_params = set(
                            str_replace_editor_tool['function']['parameters'][
                                'properties'
                            ].keys()
                        )

                        for key, value in other_kwargs.items():
                            if key in valid_params:
                                # security_risk is valid but should NOT be part of editor kwargs
                                if key != 'security_risk':
                                    valid_kwargs_for_editor[key] = value
                            else:
                                raise FunctionCallValidationError(
                                    f'Unexpected argument {key} in tool call {function_name}. Allowed arguments are: {valid_params}'
                                )

                        action = FileEditAction(
                            path=path,
                            command=command,
                            impl_source=FileEditSource.OH_ACI,
                            **valid_kwargs_for_editor,
                        )

                    set_security_risk(action, arguments)
                # ================================================
                # AgentThinkAction
                # ================================================
                elif function_name == ThinkTool['function']['name']:
                    action = AgentThinkAction(thought=arguments.get('thought', ''))

                # ================================================
                # CondensationRequestAction
                # ================================================
                elif function_name == CondensationRequestTool['function']['name']:
                    action = CondensationRequestAction()

                # ================================================
                # BrowserTool
                # ================================================
                elif function_name == BrowserTool['function']['name']:
                    if 'code' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "code" in tool call {function_name}'
                        )
                    action = BrowseInteractiveAction(browser_actions=arguments['code'])
                    set_security_risk(action, arguments)

                # ================================================
                # TaskTrackingAction
                # ================================================
                elif function_name == TASK_TRACKER_TOOL_NAME:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {function_name}'
                        )
                    if arguments['command'] == 'plan' and 'task_list' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "task_list" for "plan" command in tool call {function_name}'
                        )

                    raw_task_list = arguments.get('task_list', [])
                    if not isinstance(raw_task_list, list):
                        raise FunctionCallValidationError(
                            f'Invalid format for "task_list". Expected a list but got {type(raw_task_list)}.'
                        )

                    # Normalize task_list to ensure it's always a list of dictionaries
                    normalized_task_list = []
                    for i, task in enumerate(raw_task_list):
                        if isinstance(task, dict):
                            # Task is already in correct format, ensure required fields exist
                            normalized_task = {
                                'id': task.get('id', f'task-{i + 1}'),
                                'title': task.get('title', 'Untitled task'),
                                'status': task.get('status', 'todo'),
                                'notes': task.get('notes', ''),
                            }
                        else:
                            # Unexpected format, raise validation error
                            logger.warning(
                                f'Unexpected task format in task_list: {type(task)} - {task}'
                            )
                            raise FunctionCallValidationError(
                                f'Unexpected task format in task_list: {type(task)}. Each task shoud be a dictionary.'
                            )
                        normalized_task_list.append(normalized_task)

                    action = TaskTrackingAction(
                        command=arguments['command'],
                        task_list=normalized_task_list,
                    )

                # ================================================
                # MCPAction (MCP)
                # ================================================
                elif mcp_tool_names and function_name in mcp_tool_names:
                    action = MCPAction(
                        name=function_name,
                        arguments=arguments,
                    )
                else:
                    # Tool name not found - fuzzy matching was already attempted above
                    # Get available tool names for error message
                    available_tool_names = _get_available_tool_names(mcp_tool_names)
                    available_tool_names_str = ', '.join(available_tool_names[:10])
                    if len(available_tool_names) > 10:
                        available_tool_names_str += f', ... ({len(available_tool_names)} total)'
                    raise FunctionCallNotExistsError(
                        f'Tool {function_name} is not registered. (arguments: {arguments}). '
                        f'Available tools: {available_tool_names_str}. '
                        f'Please check the tool name and retry with an existing tool.'
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
            except (FunctionCallNotExistsError, FunctionCallValidationError) as e:
                # Log the error but continue processing other tool calls
                logger.error(f'Failed to process tool call {i+1}/{len(tool_calls)}: {e}')
                logger.error(f'Tool call details: function_name={function_name if "function_name" in locals() else "unknown"}, arguments={arguments if "arguments" in locals() else "unknown"}')
                # Don't append action, but continue to next tool call
                continue
            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f'Unexpected error processing tool call {i+1}/{len(tool_calls)}: {e}')
                import traceback
                logger.error(traceback.format_exc())
                # Don't append action, but continue to next tool call
                continue
        
        # Safety check: If ALL tool calls failed to process, log detailed error and create fallback
        # This should rarely happen if extraction and processing work correctly
        if len(actions) == 0:
            logger.error('All tool calls failed to process. This indicates a problem with tool call extraction or processing.')
            logger.error(f'Number of tool calls attempted: {len(tool_calls)}')
            logger.error(f'Tool calls extracted from content: {tool_calls_extracted_from_content}')
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    try:
                        if hasattr(tc, 'function'):
                            func = tc.function
                            if isinstance(func, dict):
                                logger.error(f'  Tool call {i+1}: name={func.get("name")}, args={func.get("arguments", "")[:100]}')
                            elif hasattr(func, 'name'):
                                logger.error(f'  Tool call {i+1}: name={func.name}, args={func.arguments[:100] if hasattr(func, "arguments") else "N/A"}')
                    except Exception as e:
                        logger.error(f'  Tool call {i+1}: Could not inspect - {e}')
            
            # Create fallback MessageAction to keep interaction going
            if tool_calls_extracted_from_content:
                message_content = thought if thought else (cleaned_content if cleaned_content else content_str)
            else:
                message_content = thought if thought else content_str
            fallback_action = MessageAction(
                content=message_content if message_content else str(assistant_msg.content) if assistant_msg.content else '',
                wait_for_response=True,
            )
            fallback_action.response_id = response.id
            actions.append(fallback_action)
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
    
    assert len(actions) >= 1
    return actions
