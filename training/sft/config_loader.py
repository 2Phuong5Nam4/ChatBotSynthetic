"""
Configuration loader utility for loading and validating training configurations.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Loads and validates training configuration from YAML files."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Resolve relative paths to absolute paths based on project root
        config = ConfigLoader._resolve_paths(config, config_file.parent.parent)

        return config

    @staticmethod
    def _resolve_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
        """
        Resolve relative paths in config to absolute paths.

        Args:
            config: Configuration dictionary
            project_root: Project root directory

        Returns:
            Configuration with resolved paths
        """
        # Resolve dataset path
        if "dataset" in config and "data_path" in config["dataset"]:
            data_path = Path(config["dataset"]["data_path"])
            if not data_path.is_absolute():
                config["dataset"]["data_path"] = str(project_root / data_path)

        # Resolve training output directory
        if "training" in config and "output_dir" in config["training"]:
            output_dir = Path(config["training"]["output_dir"])
            if not output_dir.is_absolute():
                config["training"]["output_dir"] = str(project_root / output_dir)

        # Resolve save path
        if "save" in config and "save_path" in config["save"]:
            save_path = Path(config["save"]["save_path"])
            if not save_path.is_absolute():
                config["save"]["save_path"] = str(project_root / save_path)

        return config

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate that required configuration fields are present.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If required fields are missing
        """
        required_sections = ["model", "dataset", "training"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate model section
        if "name" not in config["model"]:
            raise ValueError("Missing required field: model.name")

        # Validate dataset section
        if "data_path" not in config["dataset"]:
            raise ValueError("Missing required field: dataset.data_path")

        # Check if dataset file exists
        data_path = Path(config["dataset"]["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {config['dataset']['data_path']}")

        # Validate training section
        if "output_dir" not in config["training"]:
            raise ValueError("Missing required field: training.output_dir")

        return True

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override taking precedence.

        Args:
            base_config: Base configuration
            override_config: Configuration to override with

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Path to save YAML file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"Configuration saved to: {output_path}")
