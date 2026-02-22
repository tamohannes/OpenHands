# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Problem statement masking utilities.

Masks repository identifiers, Python file paths, and module/line references
from SWE-bench problem statements to create a 'blind' baseline where the
agent cannot rely on explicit file/repo hints in the issue description.

Ported from NeMo-Skills artsiv_mask.py.
"""

import re
from typing import Iterable, Optional, Set

# Regex: optional ./ or ../ or /, then path segments, then a segment ending in .py
_PYTHON_FILE_PATH_PATTERN = re.compile(
    r'(?:\.\.?/|/)?(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+\.py'
)

# Regex: dotted module path followed by :line or :line-line (e.g. django.db.models.deletion:276-281)
_MODULE_LINE_REF_PATTERN = re.compile(
    r'(?:[a-zA-Z0-9_]+\.)+[a-zA-Z0-9_]+:[0-9]+(?:-[0-9]+)?'
)

_INSTANCE_ID_REPO_PATTERN = re.compile(
    r'^(?P<owner>[A-Za-z0-9_.-]+)__(?P<repo>[A-Za-z0-9_.-]+)-'
)


def _collect_repo_identifiers(
    instance_id: Optional[str], repo: Optional[str]
) -> Set[str]:
    """Collect repo identifiers we want to mask.

    Includes full repo strings (e.g. `owner/repo`) as well as their components
    (e.g. `owner`, `repo`) so repo names are fully hidden in the problem text.
    """
    identifiers: Set[str] = set()

    if isinstance(repo, str) and repo.strip():
        repo_str = repo.strip()
        identifiers.add(repo_str)
        if '/' in repo_str:
            owner, repo_name = repo_str.split('/', 1)
            if owner:
                identifiers.add(owner)
            if repo_name:
                identifiers.add(repo_name)

    if isinstance(instance_id, str) and instance_id.strip():
        m = _INSTANCE_ID_REPO_PATTERN.match(instance_id.strip())
        if m:
            owner = m.group('owner')
            repo_name = m.group('repo')
            identifiers.add(f'{owner}/{repo_name}')
            identifiers.add(f'{owner}__{repo_name}')
            identifiers.add(owner)
            identifiers.add(repo_name)

    return identifiers


def mask_problem_statement(
    problem_statement: str, repo_names: Optional[Iterable[str]] = None
) -> str:
    """Mask path/location and repository identifiers in the problem statement.

    - Python file paths like `src/foo.py` -> `<FILE>`
    - Module/line refs like `django.db.models.deletion:276-281` -> `<LOC>`
    - Repo identifiers like `django/django` and `github.com/django/django/...` -> `<REPO>`
    """
    if not isinstance(problem_statement, str):
        return problem_statement if problem_statement is not None else ''
    if not problem_statement.strip():
        return problem_statement
    text = _PYTHON_FILE_PATH_PATTERN.sub('<FILE>', problem_statement)
    text = _MODULE_LINE_REF_PATTERN.sub('<LOC>', text)
    if repo_names:
        repo_tokens = [
            t.strip() for t in repo_names if isinstance(t, str) and t.strip()
        ]
        if repo_tokens:
            # Mask direct mentions (use custom boundaries so '-' and '.' tokens are handled).
            repo_alt = '|'.join(
                re.escape(t) for t in sorted(set(repo_tokens), key=len, reverse=True)
            )
            token_boundary = r'(?<![A-Za-z0-9_.-])(?:{alt})(?![A-Za-z0-9_.-])'
            text = re.sub(
                rf'(?i){token_boundary.format(alt=repo_alt)}', '<REPO>', text
            )
            # Mask GitHub URL forms that include the repo.
            text = re.sub(
                rf'(?i)(?:https?://)?github\.com/(?:{repo_alt})(?:/[^\s\)\]\}}"\']*)?',
                '<REPO>',
                text,
            )
    return text
