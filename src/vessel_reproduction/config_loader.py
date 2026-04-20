from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

from .config_schema import AppConfig, PARAM_TAXONOMY

T = TypeVar("T")


def load_config(config_path: str | Path, overrides: Mapping[str, Any] | None = None) -> AppConfig:
    """加载配置文件并返回强类型配置对象。

    仅实现配置层能力：读取、合并覆盖项、结构化转换。
    不包含任何算法业务逻辑。
    """
    path = Path(config_path)
    raw = _read_yaml(path)
    merged = _deep_merge(raw, _unflatten_overrides(overrides or {}))
    return _from_mapping(AppConfig, merged)


def export_config_snapshot(
    config: AppConfig,
    output_path: str | Path,
    runtime_context: Mapping[str, Any] | None = None,
) -> Path:
    """导出运行时配置快照。

    快照规则：
    1. 固化完整配置（meta/algorithm/engineering）。
    2. 附加 runtime_context（如输入路径、命令行参数、开始时间）。
    3. 附加 param_taxonomy（算法参数 vs 工程参数）。
    4. 计算 config_digest 保障可复现追踪。
    """
    payload: Dict[str, Any] = config.to_dict()
    payload["runtime_context"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **(dict(runtime_context or {})),
    }
    payload["param_taxonomy"] = PARAM_TAXONOMY
    payload["config_digest"] = _config_digest(payload)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return path


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got: {type(data)!r}")
    return data


def _unflatten_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for dotted_key, value in overrides.items():
        cursor: MutableMapping[str, Any] = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _from_mapping(cls: Type[T], value: Any) -> T:
    if is_dataclass(cls):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise TypeError(f"Expected mapping for {cls.__name__}, got {type(value)!r}")
        type_hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name in value:
                expected = type_hints.get(f.name, f.type)
                kwargs[f.name] = _convert_value(expected, value[f.name])
        return cls(**kwargs)  # type: ignore[arg-type]
    return value  # type: ignore[return-value]


def _convert_value(expected_type: Any, value: Any) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is list and args:
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value)!r}")
        return [_convert_value(args[0], item) for item in value]

    if is_dataclass(expected_type):
        return _from_mapping(expected_type, value)

    return value


def _config_digest(payload: Mapping[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
