import yaml
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Get absolute path to project root"""
    try:
        return Path(__file__).parent.parent.parent
    except NameError:
        return Path(os.getcwd()).parent

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    project_root = get_project_root()
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    for section in config.values():
        for key, value in section.items():
            if "path" in key.lower():
                section[key] = str(project_root / value)
    
    return config