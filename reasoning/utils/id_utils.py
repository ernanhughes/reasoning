# utils/id_utils.py

import ulid
import re

def generate_ulid() -> str:
    """Generate a lexicographically sortable ULID."""
    return ulid.new().str

def class_name_to_snake(cls) -> str:
    """
    Convert class name to snake_case prefix.
    E.g. PromptLog → prompt_log
    """
    name = cls.__name__
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def generate_prefixed_id(cls) -> str:
    """Generate an ID like prompt_log_01HV9..."""
    prefix = class_name_to_snake(cls)
    return f"{prefix}_{generate_ulid()}"
