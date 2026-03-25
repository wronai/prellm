"""Lazy import utilities for preLLM."""

from typing import Any, Dict


def lazy_import_global(name: str, import_path: str, globals_dict: Dict[str, Any]) -> Any:
    """Lazy import a global object.
    
    Args:
        name: The name to assign to the imported object in globals
        import_path: The full import path (e.g., 'module.submodule.ClassName')
        globals_dict: The globals dictionary to store the imported object
        
    Returns:
        The imported object
    """
    if name not in globals_dict:
        module_path, class_name = import_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        globals_dict[name] = getattr(module, class_name)
    return globals_dict[name]
