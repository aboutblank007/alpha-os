"""
AlphaOS v4 CLI

命令行接口用于 v4 训练和推理：
- alphaos-v4-train: 使用 v4 管道训练（CfC + XGBoost）
- alphaos-v4-serve: 使用 v4 推理引擎
- alphaos-v4-backtest: 回测模型

配置支持：
- YAML 配置文件: --config configs/v4/xauusd.yaml
- CLI 参数优先级高于配置文件（允许覆盖）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        path: YAML 文件路径
        
    Returns:
        配置字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_config_with_args(
    yaml_config: dict[str, Any],
    args: argparse.Namespace,
    arg_defaults: dict[str, Any],
) -> dict[str, Any]:
    """
    合并 YAML 配置和 CLI 参数
    
    CLI 参数优先级高于配置文件（如果用户显式指定）
    
    Args:
        yaml_config: YAML 配置字典
        args: 解析后的 CLI 参数
        arg_defaults: argparse 默认值字典
        
    Returns:
        合并后的配置字典
    """
    merged = yaml_config.copy()
    
    # 遍历 CLI 参数，如果用户显式指定（不是默认值），则覆盖
    for key, value in vars(args).items():
        if key in ("config", "data", "output", "max_ticks"):
            # 这些是 CLI 专属参数，不需要合并
            continue
        
        # 检查是否用户显式指定了该参数（不等于默认值）
        default_val = arg_defaults.get(key)
        if value != default_val and value is not None:
            # 将 CLI 参数映射到 YAML 结构
            _set_nested_value(merged, key, value)
    
    return merged


def _set_nested_value(config: dict, key: str, value: Any) -> None:
    """
    设置嵌套配置值
    
    将 CLI 参数名（如 sampling_mode）映射到 YAML 结构（如 sampling.mode）
    """
    # CLI 参数到 YAML 路径的映射
    mapping = {
        "sampling_mode": ("sampling", "mode"),
        "target_volume": ("sampling", "target_volume"),
        "denoise_mode": ("denoise",),
        "epochs": ("training", "epochs"),
        "cv_splits": ("training", "cv_splits"),
        "use_cpcv": ("training", "use_cpcv"),
        "no_require_fvg": ("primary", "require_fvg"),  # 注意：取反
        "min_trend_duration": ("primary", "min_trend_duration"),
    }
    
    if key in mapping:
        path = mapping[key]
        target = config
        for part in path[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        # 特殊处理 denoise_mode（映射到 core/config 结构）
        if key == "denoise_mode":
            denoise_cfg = target.setdefault("denoise", {})
            kalman_cfg = denoise_cfg.setdefault("kalman", {})
            wavelet_cfg = denoise_cfg.setdefault("wavelet", {})
            
            if value == "wavelet":
                kalman_cfg["enabled"] = False
                wavelet_cfg["enabled"] = True
            elif value == "kalman":
                kalman_cfg["enabled"] = True
                kalman_cfg["use_adaptive"] = False
                wavelet_cfg["enabled"] = False
            elif value == "combined":
                kalman_cfg["enabled"] = True
                wavelet_cfg["enabled"] = True
            elif value == "none":
                kalman_cfg["enabled"] = False
                wavelet_cfg["enabled"] = False
            return
        
        # 特殊处理 no_require_fvg（取反）
        if key == "no_require_fvg":
            target[path[-1]] = not value
        else:
            target[path[-1]] = value


def train_v4() -> None:
    """Entry point for alphaos-v4-train command."""
    parser = argparse.ArgumentParser(
        description="AlphaOS v4 Training (CfC + XGBoost)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to tick data CSV",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models/v4",
        help="Output directory",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file (e.g., configs/v4/xauusd.yaml)",
    )
    
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum ticks to load (for testing)",
    )
    
    # Sampling options
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="volume_bars",
        choices=["volume_bars", "tick_imbalance"],
        help="Bar sampling mode (volume_bars=成交量条形图, tick_imbalance=Tick失衡条形图)",
    )
    
    parser.add_argument(
        "--target-volume",
        type=int,
        default=100,
        help="Target volume per bar (for volume mode)",
    )
    
    # Denoising options
    parser.add_argument(
        "--denoise-mode",
        type=str,
        default="kalman",
        choices=["wavelet", "kalman", "combined", "none"],
        help="Denoising mode",
    )
    
    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    
    parser.add_argument(
        "--use-cpcv",
        action="store_true",
        help="Use Combinatorial Purged CV (slower but more robust)",
    )
    
    # Primary Engine options (放宽信号生成条件)
    parser.add_argument(
        "--no-require-fvg",
        action="store_true",
        help="不要求 FVG 确认（小数据集推荐）",
    )
    
    parser.add_argument(
        "--min-trend-duration",
        type=int,
        default=3,
        help="最小趋势持续 Bar 数（默认 3，可降低到 1）",
    )
    
    # 保存默认值用于后续判断用户是否显式指定
    arg_defaults = {
        "sampling_mode": "volume_bars",
        "target_volume": 100,
        "denoise_mode": "kalman",
        "epochs": 50,
        "cv_splits": 5,
        "use_cpcv": False,
        "no_require_fvg": False,
        "min_trend_duration": 3,
    }
    
    args = parser.parse_args()
    
    from alphaos.core.logging import setup_logging, get_logger, enable_log_file
    from alphaos.v4 import (
        TrainingConfig,
        V4TrainingPipeline,
        SamplingConfig,
        SamplingMode,
        DenoiseConfig,
        DenoiseMode,
        PrimaryEngineConfig,
    )
    
    # Setup logging
    log_path = enable_log_file(prefix="v4_training", tee_console=True)
    setup_logging(level="INFO", log_format="console")
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_path}")
    
    # Load or create config（严格模式）
    if not args.config:
        raise ValueError("缺少 --config，已启用严格配置模式")
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError("仅支持 YAML 配置文件（严格模式）")
    
    # 加载 YAML 配置
    yaml_config = load_yaml_config(config_path)
    
    # 合并 CLI 参数（CLI 优先）
    merged_config = merge_config_with_args(yaml_config, args, arg_defaults)
    
    # 从合并后的配置创建 TrainingConfig
    config = TrainingConfig.from_yaml_dict(merged_config)
    config.output_dir = args.output  # 输出目录始终使用 CLI 参数
    
    logger.info(f"Loaded YAML config from {args.config}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(output_path / "config.json")
    
    # Create pipeline
    pipeline = V4TrainingPipeline(config)
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    n_ticks = pipeline.load_tick_data(args.data, max_ticks=args.max_ticks)
    logger.info(f"Loaded {n_ticks} ticks")
    
    # Train
    logger.info("Starting v4 training...")
    results = pipeline.train()
    
    # Save complete model bundle (cfc_encoder.pt + xgb_model.json + schema.json)
    pipeline.save_model(output_path)
    
    # Save results
    import json
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training complete. Model and results saved to {output_path}")
    
    # Print summary
    if results.get("avg_results"):
        avg = results["avg_results"]
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Accuracy: {avg.get('avg_accuracy', 0):.4f} ± {avg.get('std_accuracy', 0):.4f}")
        print(f"AUC:      {avg.get('avg_auc', 0):.4f} ± {avg.get('std_auc', 0):.4f}")
        print(f"Precision: {avg.get('avg_precision', 0):.4f}")
        print(f"Recall:    {avg.get('avg_recall', 0):.4f}")
        print("=" * 50)


def serve_v4() -> None:
    """Entry point for alphaos-v4-serve command."""
    parser = argparse.ArgumentParser(
        description="AlphaOS v4 Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to v4 model directory",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/v4/xauusd.yaml)",
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to feature schema JSON (default: model_dir/schema.json)",
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol",
    )
    
    parser.add_argument(
        "--volume",
        type=float,
        default=0.01,
        help="Order volume (lots)",
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=2,
        help="Maximum concurrent positions per symbol (tracked in server)",
    )
    
    parser.add_argument(
        "--zmq-host",
        type=str,
        default="localhost",
        help="ZeroMQ host (MT5 server IP address)",
    )
    
    parser.add_argument(
        "--zmq-tick-port",
        type=int,
        default=5555,
        help="ZeroMQ tick subscription port",
    )
    
    parser.add_argument(
        "--zmq-order-port",
        type=int,
        default=5556,
        help="ZeroMQ order port",
    )
    
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port",
    )

    # ================================================================
    # Web UI (FastAPI static hosting for ui/dist)
    # ================================================================
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start integrated web server to host built UI (ui/dist) and /api endpoints",
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default="0.0.0.0",
        help="Web server host",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8000,
        help="Web server port",
    )
    parser.add_argument(
        "--ui-dist",
        type=str,
        default="ui/dist",
        help="Path to built UI dist folder (relative to repo root by default)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing trades",
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum meta-confidence threshold",
    )
    
    parser.add_argument(
        "--require-phase-transition",
        action="store_true",
        default=True,
        help="Only trade in PHASE_TRANSITION state",
    )
    
    parser.add_argument(
        "--guardian-enabled",
        action="store_true",
        default=True,
        help="Enable Model Guardian safety checks",
    )
    
    parser.add_argument(
        "--guardian-latency-threshold",
        type=float,
        default=200.0,
        help="Model Guardian latency threshold (ms)",
    )
    
    parser.add_argument(
        "--history-warmup",
        action="store_true",
        help="Fetch historical data from MT5 to pre-warm the engine (skip cold start)",
    )
    
    parser.add_argument(
        "--history-bars",
        type=int,
        default=2000,
        help="Number of M1 bars to fetch for warmup (default: 2000)",
    )
    
    parser.add_argument(
        "--replay-history",
        type=str,
        default=None,
        metavar="FILE",
        help="Historical replay: path to CSV tick file (evolves CfC hidden + fills confidence buffer)",
    )
    
    parser.add_argument(
        "--replay-ticks",
        type=int,
        default=200000,
        help="Number of ticks to replay (default: 200000)",
    )
    
    # v4.0: MT5 同源历史回放参数（优先于 --replay-history CSV 文件）
    parser.add_argument(
        "--replay-mt5",
        action="store_true",
        help="Enable MT5 broker-native tick history replay (BOOTSTRAP_REPLAY)",
    )
    
    parser.add_argument(
        "--replay-window-sec",
        type=int,
        default=86400,
        help="Replay window in seconds (default: 86400 = 24h)",
    )
    
    parser.add_argument(
        "--replay-target-buffer",
        type=int,
        default=0,
        help="Target confidence buffer size (default: use config.confidence_gate.buffer_size)",
    )
    
    parser.add_argument(
        "--replay-pace-tps",
        type=int,
        default=50000,
        help="Replay ticks per second (default: 50000)",
    )
    
    parser.add_argument(
        "--replay-end-eps-ms",
        type=int,
        default=1000,
        help="Epsilon offset from now in ms (default: 1000, ensures replay < live)",
    )
    
    parser.add_argument(
        "--replay-max-ticks",
        type=int,
        default=2000000,
        help="Maximum ticks to load (safety limit, default: 2M)",
    )
    
    args = parser.parse_args()

    serve_arg_defaults = {
        "min_confidence": 0.65,
        "zmq_host": "localhost",
        "zmq_tick_port": 5555,
        "zmq_order_port": 5556,
        "guardian_latency_threshold": 200.0,
        "history_bars": 2000,
        "replay_ticks": 200000,
        "replay_window_sec": 86400,
        "replay_target_buffer": 0,
        "replay_pace_tps": 50000,
        "replay_end_eps_ms": 1000,
        "replay_max_ticks": 2000000,
    }
    
    from alphaos.core.logging import setup_logging, get_logger, enable_log_file
    from alphaos.core.config import ZeroMQConfig
    from alphaos.v4 import InferenceConfig, InferenceEngineV4
    
    # Setup logging
    log_path = enable_log_file(prefix="v4_inference", tee_console=True)
    setup_logging(level="INFO", log_format="console")
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_path}")
    
    model_dir = Path(args.model)
    schema_path = args.schema or model_dir / "schema.json"
    
    # 检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    def _require_section(config_data: dict, name: str) -> dict:
        if name not in config_data or config_data[name] is None:
            raise ValueError(f"缺少配置段: {name}")
        if not isinstance(config_data[name], dict):
            raise ValueError(f"配置段类型错误: {name}")
        return config_data[name]

    def _require_keys(section: dict, keys: list[str], prefix: str) -> None:
        for key in keys:
            if key not in section:
                raise ValueError(f"缺少配置项: {prefix}.{key}")

    def _parse_endpoint(endpoint: str) -> tuple[str, int]:
        if not endpoint.startswith("tcp://"):
            raise ValueError(f"ZeroMQ endpoint 格式错误: {endpoint}")
        host_port = endpoint.replace("tcp://", "")
        host, port_str = host_port.rsplit(":", 1)
        return host, int(port_str)

    def _build_endpoint(host: str, port: int) -> str:
        return f"tcp://{host}:{port}"

    # 加载配置（严格模式）
    if not args.config:
        raise ValueError("缺少 --config，已启用严格配置模式")
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError("仅支持 YAML 配置文件（严格模式）")

    import copy
    import sys

    yaml_config = load_yaml_config(config_path)
    patched_config = copy.deepcopy(yaml_config or {})

    # Symbol override
    if args.symbol:
        patched_config["symbol"] = args.symbol

    # ZeroMQ endpoints override
    zeromq_data = patched_config.get("zeromq", {})
    if not isinstance(zeromq_data, dict):
        raise ValueError("配置段类型错误: zeromq")

    tick_endpoint = zeromq_data.get("tick_endpoint", "")
    order_endpoint = zeromq_data.get("order_endpoint", "")
    history_endpoint = zeromq_data.get("history_endpoint", "")
    if not tick_endpoint or not order_endpoint or not history_endpoint:
        raise ValueError("zeromq 缺少 tick_endpoint/order_endpoint/history_endpoint")

    tick_host, tick_port = _parse_endpoint(tick_endpoint)
    order_host, order_port = _parse_endpoint(order_endpoint)
    history_host, history_port = _parse_endpoint(history_endpoint)

    if args.zmq_host and ("--zmq-host" in sys.argv or args.zmq_host != "localhost"):
        tick_host = args.zmq_host
        order_host = args.zmq_host
        history_host = args.zmq_host
    if args.zmq_tick_port:
        tick_port = args.zmq_tick_port
    if args.zmq_order_port:
        order_port = args.zmq_order_port

    zeromq_data["tick_endpoint"] = _build_endpoint(tick_host, tick_port)
    zeromq_data["order_endpoint"] = _build_endpoint(order_host, order_port)
    zeromq_data["history_endpoint"] = _build_endpoint(history_host, history_port)

    # Position sizing cap from CLI volume (只做上限钳制)
    exec_cfg = patched_config.setdefault("execution", {})
    ps_cfg = exec_cfg.setdefault("position_sizing", {})
    if args.volume and args.volume > 0 and "--volume" in sys.argv:
        max_lots = ps_cfg.get("max_lots", args.volume)
        min_lots = ps_cfg.get("min_lots", 0.0)
        capped_max = min(max_lots, args.volume)
        ps_cfg["max_lots"] = capped_max
        if min_lots > capped_max:
            ps_cfg["min_lots"] = capped_max
        logger.info("CLI volume cap applied", max_lots=ps_cfg["max_lots"], min_lots=ps_cfg.get("min_lots"))
        legacy_ps = patched_config.setdefault("position_sizing", {})
        legacy_ps["max_lots"] = ps_cfg["max_lots"]
        if "min_lots" in ps_cfg:
            legacy_ps["min_lots"] = ps_cfg["min_lots"]

    # Guardian override
    monitoring_cfg = patched_config.get("monitoring", {})
    guardian_cfg = monitoring_cfg.get("model_guardian", {})
    if "--guardian-enabled" in sys.argv:
        guardian_cfg["enabled"] = True

    # Phase transition requirement
    if "--require-phase-transition" in sys.argv:
        inference_cfg = patched_config.setdefault("inference", {})
        inference_cfg["require_phase_transition"] = True

    yaml_config = patched_config

    import hashlib
    import json

    try:
        config_hash = hashlib.sha1(
            json.dumps(patched_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:
        config_hash = "unknown"

    api_cfg = None
    try:
        from alphaos.core.config import AlphaOSConfig
        api_cfg = AlphaOSConfig.model_validate(patched_config)
    except Exception as e:
        logger.warning("API config validation failed", error=str(e))
    
    zeromq_data = _require_section(yaml_config, "zeromq")
    monitoring_data = _require_section(yaml_config, "monitoring")
    health_data = _require_section(yaml_config, "health")
    execution_data = _require_section(yaml_config, "execution")
    boot_data = _require_section(yaml_config, "boot")

    _require_keys(
        zeromq_data,
        [
            "tick_endpoint",
            "order_endpoint",
            "history_endpoint",
            "heartbeat_interval_ms",
            "reconnect_delay_ms",
            "recv_timeout_ms",
            "history_timeout_ms",
            "history_snd_timeout_ms",
            "order_recv_timeout_ms",
            "order_snd_timeout_ms",
            "tick_staleness_threshold_sec",
        ],
        "zeromq",
    )
    _require_keys(monitoring_data, ["model_guardian", "metrics"], "monitoring")
    guardian_data = _require_section(monitoring_data, "model_guardian")
    _require_keys(
        guardian_data,
        [
            "enabled",
            "nan_inf_check",
            "state_saturation_threshold",
            "confidence_collapse_window",
            "confidence_collapse_threshold",
            "latency_threshold_ms",
            "lock_file_path",
        ],
        "monitoring.model_guardian",
    )
    metrics_data = _require_section(monitoring_data, "metrics")
    _require_keys(
        metrics_data,
        [
            "port",
            "inference_latency_buckets",
            "tick_processing_latency_buckets",
            "order_latency_buckets",
        ],
        "monitoring.metrics",
    )
    _require_keys(health_data, ["check_interval_sec", "tick_staleness_threshold_sec", "warmup_ticks_estimate", "critical_components"], "health")
    _require_keys(execution_data, ["position_sizing", "exit_v21"], "execution")
    position_sizing_data = _require_section(execution_data, "position_sizing")
    _require_keys(
        position_sizing_data,
        [
            "mode",
            "kelly_fraction",
            "kelly_max_fraction",
            "expected_edge_pct",
            "win_rate",
            "risk_per_trade_pct",
            "account_balance",
            "risk_reward_ratio",
            "min_lots",
            "max_lots",
            "lot_step",
            "linear_conf_max",
        ],
        "execution.position_sizing",
    )
    exit_v21_data = _require_section(execution_data, "exit_v21")
    _require_keys(
        exit_v21_data,
        [
            "min_hold_seconds",
            "modify_cooldown_sec",
            "min_sl_gap_price",
            "price_precision",
            "est_commission_usd_per_lot",
            "est_slippage_usd_per_lot",
            "cost_buffer_usd",
            "be_trigger_net_usd",
            "be_offset_price",
            "partial1_trigger_net_usd",
            "partial1_ratio",
            "min_lots_to_partial",
            "post_partial_cooldown_sec",
            "trail_start_net_usd",
            "trail_distance_price",
            "trail_step_price",
            "alignment_multipliers",
            "tick_value_usd_per_lot",
        ],
        "execution.exit_v21",
    )
    exit_v2_data = execution_data.get("exit_v2")
    _require_keys(
        boot_data,
        [
            "replay_target_buffer",
            "replay_csv_max_ticks",
            "replay_window_sec",
            "replay_pace_tps",
            "replay_end_eps_ms",
            "replay_max_ticks",
            "replay_max_duration_sec",
            "replay_socket_timeout_ms",
            "replay_gap_threshold_ms",
            "replay_time_monotonic_tolerance_ms",
            "replay_max_consecutive_timeouts",
            "history_warmup_bars",
        ],
        "boot",
    )

    config = InferenceConfig.from_yaml_dict(
        yaml_config,
        model_dir=str(model_dir),
        device=device,
    )

    if args.schema:
        config.schema_path = args.schema

    if args.min_confidence != serve_arg_defaults["min_confidence"]:
        if args.min_confidence != yaml_config["confidence_gate"]["fixed_fallback_threshold"]:
            raise ValueError("min_confidence 已弃用，请使用 confidence_gate.fixed_fallback_threshold")
    if "--require-phase-transition" in sys.argv:
        config.require_phase_transition = True

    logger.info(f"Loaded YAML config from {args.config}")
    logger.info(
        f"Confidence gate: mode={config.confidence_gate.mode}, "
        f"base_quantile={config.confidence_gate.base_quantile}, "
        f"buffer_size={config.confidence_gate.buffer_size}"
    )

    # ZeroMQ 端点（允许 CLI 覆盖）
    tick_host, tick_port = _parse_endpoint(zeromq_data["tick_endpoint"])
    order_host, order_port = _parse_endpoint(zeromq_data["order_endpoint"])
    history_host, history_port = _parse_endpoint(zeromq_data["history_endpoint"])

    if args.zmq_host != serve_arg_defaults["zmq_host"] or "--zmq-host" in sys.argv:
        tick_host = args.zmq_host
        order_host = args.zmq_host
        history_host = args.zmq_host
    if args.zmq_tick_port != serve_arg_defaults["zmq_tick_port"]:
        tick_port = args.zmq_tick_port
    if args.zmq_order_port != serve_arg_defaults["zmq_order_port"]:
        order_port = args.zmq_order_port

    zmq_config = ZeroMQConfig(
        tick_endpoint=_build_endpoint(tick_host, tick_port),
        order_endpoint=_build_endpoint(order_host, order_port),
        history_endpoint=_build_endpoint(history_host, history_port),
        heartbeat_interval_ms=zeromq_data["heartbeat_interval_ms"],
        reconnect_delay_ms=zeromq_data["reconnect_delay_ms"],
        recv_timeout_ms=zeromq_data["recv_timeout_ms"],
        history_timeout_ms=zeromq_data["history_timeout_ms"],
        history_snd_timeout_ms=zeromq_data["history_snd_timeout_ms"],
        order_recv_timeout_ms=zeromq_data["order_recv_timeout_ms"],
        order_snd_timeout_ms=zeromq_data["order_snd_timeout_ms"],
        tick_staleness_threshold_sec=zeromq_data["tick_staleness_threshold_sec"],
    )

    guardian_enabled = guardian_data["enabled"]
    if "--guardian-enabled" in sys.argv:
        guardian_enabled = True
    guardian_latency_threshold = guardian_data["latency_threshold_ms"]
    if args.guardian_latency_threshold != serve_arg_defaults["guardian_latency_threshold"]:
        guardian_latency_threshold = args.guardian_latency_threshold
    
    # Create engine
    engine = InferenceEngineV4(config)
    
    logger.info(
        "v4 Inference Engine initialized",
        model=str(model_dir),
        schema_hash=engine.schema.schema_hash,
    )
    
    # Historical Replay: evolve CfC hidden state + fill confidence buffer
    def _historical_replay(
        engine,
        filepath: str,
        max_ticks: int,
        logger,
    ):
        """
        历史回放（Historical Replay）
        
        ============================================================
        Warmup and Cold Start Semantics (语义钉死)
        ============================================================
        
        - Statistical warmup (e.g. rolling quantile buffers) MAY be 
          satisfied via historical replay prior to live trading.
        
        - Stateful components (CfC hidden states, samplers, bar 
          aggregators) MUST always evolve causally and cannot be 
          skipped or initialized from future or synthetic values.
        
        - Historical replay MUST:
          - use only past market data,
          - disable execution and risk checks,
          - produce identical feature and model outputs as live inference.
        
        ============================================================
        
        流程:
        1. 读取历史 ticks（仅过去数据）
        2. 完整运行 inference pipeline:
           - Tick → Sampler → EventBar
           - EventBar → Features → CfC hidden (因果演化)
           - Features → XGB → meta_confidence
           - 填充 confidence buffer（仅事件 bar）
        3. 禁用: 下单、风控触发
        4. 完成后切换到 live mode
        
        支持的 CSV 格式:
        - Time (EET),Ask,Bid,AskVolume,BidVolume (TradingView 格式)
        - time_msc,bid,ask,volume (AlphaOS 格式)
        - time,bid,ask,last,volume,flags (MT5 格式)
        """
        import csv
        from pathlib import Path
        from alphaos.core.types import Tick
        
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Replay file not found: {filepath}")
            return
        
        logger.info("=" * 60)
        logger.info("[BOOT] Starting Historical Replay")
        logger.info(f"[BOOT] Source: {filepath}")
        logger.info(f"[BOOT] Max ticks: {max_ticks:,}")
        logger.info("=" * 60)
        
        try:
            # 尝试使用 polars（更快）
            try:
                import polars as pl
                df = pl.read_csv(filepath, n_rows=max_ticks, ignore_errors=True)
                
                # 检测列名（支持多种 CSV 格式）
                cols = df.columns
                
                # 时间列
                time_col = None
                for c in cols:
                    cl = c.lower()
                    if "time" in cl or "timestamp" in cl:
                        time_col = c
                        break
                
                # Bid 列（优先精确匹配）
                bid_col = None
                for c in cols:
                    if c.lower() == "bid":
                        bid_col = c
                        break
                if not bid_col:
                    for c in cols:
                        if "bid" in c.lower() and "volume" not in c.lower():
                            bid_col = c
                            break
                
                # Ask 列
                ask_col = None
                for c in cols:
                    if c.lower() == "ask":
                        ask_col = c
                        break
                if not ask_col:
                    for c in cols:
                        if "ask" in c.lower() and "volume" not in c.lower():
                            ask_col = c
                            break
                
                if not all([bid_col, ask_col]):
                    raise ValueError(f"Missing Bid/Ask columns. Found: {cols}")
                
                logger.info(f"[BOOT] CSV columns: time={time_col}, bid={bid_col}, ask={ask_col}")
                
                tick_count = 0
                event_count = 0
                
                for row in df.iter_rows(named=True):
                    time_val = row.get(time_col) if time_col else None
                    bid = row[bid_col]
                    ask = row[ask_col]
                    
                    if bid <= 0 or ask <= 0:
                        continue
                    
                    # 转换时间戳
                    if isinstance(time_val, (int, float)):
                        timestamp_us = int(time_val) * 1000 if time_val < 1e12 else int(time_val)
                    else:
                        timestamp_us = tick_count * 1000
                    
                    tick = Tick(
                        timestamp_us=timestamp_us,
                        bid=float(bid),
                        ask=float(ask),
                        bid_volume=0.0,
                        ask_volume=0.0,
                        last=0.0,
                        last_volume=1.0,
                        flags=0,
                    )
                    
                    # 完整运行 inference pipeline（无执行）
                    result = engine.process_tick(tick)
                    tick_count += 1
                    
                    # 统计事件数
                    if result and result.has_signal:
                        event_count += 1
                    
                    if tick_count % 50000 == 0:
                        buffer_size = len(engine._confidence_buffer)
                        logger.info(
                            f"[BOOT] Replay progress: {tick_count:,} ticks, "
                            f"bars={engine.bar_count}, events={event_count}, "
                            f"confidence_buffer={buffer_size}"
                        )
                
                # 完成日志
                buffer_size = len(engine._confidence_buffer)
                min_required = engine.config.confidence_gate.min_required
                full_buffer = engine.config.confidence_gate.buffer_size
                
                logger.info("=" * 60)
                logger.info("[BOOT] Historical Replay Completed")
                logger.info(f"[BOOT] Ticks processed: {tick_count:,}")
                logger.info(f"[BOOT] Bars generated: {engine.bar_count}")
                logger.info(f"[BOOT] Events detected: {event_count}")
                logger.info(f"[BOOT] Confidence buffer: {buffer_size} samples")
                logger.info(f"[BOOT] CfC hidden state: evolved ({engine.bar_count} steps)")
                
                # Cold start 状态
                if buffer_size >= full_buffer:
                    logger.info(f"[BOOT] Rolling quantile gate ACTIVE from first live bar")
                    logger.info(f"[BOOT] Cold start: SKIPPED (buffer >= {full_buffer})")
                elif buffer_size >= min_required:
                    logger.info(f"[BOOT] Fixed fallback gate active (buffer >= {min_required})")
                    logger.info(f"[BOOT] Cold start: PARTIAL (need {full_buffer - buffer_size} more for full)")
                else:
                    logger.info(f"[BOOT] Cold start: ACTIVE (need {min_required - buffer_size} more samples)")
                    logger.warning("[BOOT] Trading will be blocked until min_required reached")
                
                logger.info("=" * 60)
                return
                
            except ImportError:
                pass
            
            # 回退到标准 csv
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                tick_count = 0
                event_count = 0
                
                for row in reader:
                    if tick_count >= max_ticks:
                        break
                    
                    # 提取数据
                    time_str = row.get("time_msc") or row.get("time") or row.get("timestamp") or row.get("Time (EET)")
                    bid = float(row.get("bid") or row.get("Bid") or 0)
                    ask = float(row.get("ask") or row.get("Ask") or 0)
                    
                    if bid <= 0 or ask <= 0:
                        continue
                    
                    if time_str and time_str.replace(".", "").replace(":", "").replace(" ", "").isdigit():
                        try:
                            timestamp_us = int(float(time_str) * 1000) if float(time_str) < 1e12 else int(float(time_str))
                        except:
                            timestamp_us = tick_count * 1000
                    else:
                        timestamp_us = tick_count * 1000
                    
                    tick = Tick(
                        timestamp_us=timestamp_us,
                        bid=bid,
                        ask=ask,
                        bid_volume=0.0,
                        ask_volume=0.0,
                        last=0.0,
                        last_volume=1.0,
                        flags=0,
                    )
                    
                    result = engine.process_tick(tick)
                    tick_count += 1
                    
                    if result and result.has_signal:
                        event_count += 1
                    
                    if tick_count % 50000 == 0:
                        logger.info(f"[BOOT] Replay progress: {tick_count:,} ticks, bars={engine.bar_count}")
                
                buffer_size = len(engine._confidence_buffer)
                logger.info("=" * 60)
                logger.info("[BOOT] Historical Replay Completed (csv)")
                logger.info(f"[BOOT] Ticks: {tick_count:,}, Bars: {engine.bar_count}, Events: {event_count}")
                logger.info(f"[BOOT] Confidence buffer: {buffer_size} samples")
                logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"File warmup failed: {e}")
            import traceback
            traceback.print_exc()
    
    # History warmup helper function (requires MT5 EA support)
    async def _warmup_from_history(
        zmq_client,
        engine,
        symbol: str,
        n_bars: int,
        logger,
    ):
        """
        从 MT5 获取历史数据预热引擎，跳过 cold start 阶段
        
        策略：
        1. 获取最近 N 根 M1 bars
        2. 每根 bar 模拟 100 个 ticks（与 target_volume 匹配）
        3. 通过引擎处理这些 ticks
        """
        from datetime import datetime, timedelta, timezone
        from alphaos.core.types import Tick
        
        logger.info(f"Starting history warmup: fetching {n_bars} M1 bars...")
        
        try:
            # 计算时间范围（最近 n_bars 分钟）
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=n_bars + 60)  # 多取一些
            
            start_str = start_time.strftime("%Y.%m.%d %H:%M:%S")
            end_str = end_time.strftime("%Y.%m.%d %H:%M:%S")
            
            logger.info(f"Fetching M1 history: {start_str} to {end_str}")
            
            # 获取历史数据
            bars = await zmq_client.get_history(
                symbol=symbol,
                timeframe="M1",
                start_date=start_str,
                end_date=end_str,
            )
            
            if not bars:
                logger.warning("No historical data received, skipping warmup")
                return
            
            logger.info(f"Received {len(bars)} M1 bars, processing...")
            
            # 模拟 ticks 进行预热
            warmup_ticks = 0
            warmup_bars = 0
            
            for bar in bars[-n_bars:]:  # 取最近 n_bars 根
                # 从 bar 提取数据
                bar_time = bar.get("time", 0)
                try:
                    open_price = float(bar.get("open", 0))
                except (TypeError, ValueError):
                    open_price = 0.0
                try:
                    high_price = float(bar.get("high", 0))
                except (TypeError, ValueError):
                    high_price = 0.0
                try:
                    low_price = float(bar.get("low", 0))
                except (TypeError, ValueError):
                    low_price = 0.0
                try:
                    close_price = float(bar.get("close", 0))
                except (TypeError, ValueError):
                    close_price = 0.0
                try:
                    tick_volume = int(float(bar.get("tick_volume", 100)))
                except (TypeError, ValueError):
                    tick_volume = 100  # 默认 100
                
                if open_price <= 0:
                    continue
                
                # 模拟 tick 序列（简化：open -> high -> low -> close）
                # 每根 bar 产生约 100 个 ticks
                n_ticks_per_bar = max(100, tick_volume)
                time_step_us = 60_000_000 // n_ticks_per_bar  # 1分钟内均匀分布
                
                # 生成价格序列（简化插值）
                prices = [open_price]
                
                # 根据 bar 形态决定价格路径
                if high_price > open_price and low_price < open_price:
                    # 上下波动
                    prices.extend([high_price] * (n_ticks_per_bar // 4))
                    prices.extend([low_price] * (n_ticks_per_bar // 4))
                    prices.extend([close_price] * (n_ticks_per_bar // 2))
                elif high_price > open_price:
                    # 上涨
                    prices.extend([high_price] * (n_ticks_per_bar // 2))
                    prices.extend([close_price] * (n_ticks_per_bar // 2))
                elif low_price < open_price:
                    # 下跌
                    prices.extend([low_price] * (n_ticks_per_bar // 2))
                    prices.extend([close_price] * (n_ticks_per_bar // 2))
                else:
                    prices.extend([close_price] * n_ticks_per_bar)
                
                # 处理每个模拟 tick
                base_time_us = None
                if isinstance(bar_time, str):
                    try:
                        dt = datetime.strptime(bar_time, "%Y.%m.%d %H:%M:%S")
                        base_time_us = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
                    except (ValueError, TypeError):
                        base_time_us = None
                if base_time_us is None:
                    try:
                        bar_time_num = float(bar_time)
                        base_time_us = int(bar_time_num * 1_000_000) if bar_time_num > 1e9 else int(bar_time_num * 1000) * 1000
                    except (TypeError, ValueError):
                        base_time_us = warmup_ticks * 1000
                
                for i, price in enumerate(prices[:n_ticks_per_bar]):
                    tick = Tick(
                        timestamp_us=base_time_us + i * time_step_us,
                        bid=price,
                        ask=price + 0.01,  # 模拟点差
                        bid_volume=0.0,
                        ask_volume=0.0,
                        last=price,
                        last_volume=1.0,
                        flags=0,
                    )
                    
                    # 处理 tick（预热模式，不产生信号）
                    engine.process_tick(tick)
                    warmup_ticks += 1
                
                warmup_bars += 1
                
                # 进度日志
                if warmup_bars % 500 == 0:
                    logger.info(f"Warmup progress: {warmup_bars}/{len(bars)} bars, engine.bar_count={engine.bar_count}")
            
            logger.info(
                f"History warmup complete: "
                f"processed {warmup_ticks} ticks from {warmup_bars} M1 bars, "
                f"engine.bar_count={engine.bar_count}, "
                f"confidence_buffer_size={len(engine._confidence_buffer)}"
            )
            
        except Exception as e:
            logger.error(f"History warmup failed: {e}")
            logger.info("Continuing without warmup...")
    
    # ========================================================================
    # v4.0: Position Sizing (using shared utility)
    # ========================================================================
    from alphaos.v4.sizing.kelly import calculate_position_size, KellySizingConfig
    
    def calculate_kelly_position(
        confidence: float,
        config,
        default_volume: float,
        logger,
    ) -> float:
        """
        Calculate position size using Kelly Criterion (wrapper for shared utility).
        
        Uses the unified implementation from alphaos.v4.sizing.kelly
        """
        # Check if position_sizing config exists
        ps_cfg = getattr(config, 'position_sizing', None)
        if ps_cfg is None:
            logger.warning(f"[KELLY] No position_sizing config, using default={default_volume}")
            return default_volume
        
        sizing_mode = getattr(ps_cfg, 'mode', 'fixed')
        
        # Use execution.position_sizing as SSOT
        min_lots = getattr(ps_cfg, 'min_lots', 0.01)
        max_lots = getattr(ps_cfg, 'max_lots', 0.10)
        
        logger.info(
            f"[SIZING] mode={sizing_mode}, min_lots={min_lots}, max_lots={max_lots}"
        )
        
        # Create config for shared utility
        sizing_config = KellySizingConfig(
            kelly_fraction=getattr(ps_cfg, 'kelly_fraction', 0.25),
            kelly_max_fraction=getattr(ps_cfg, 'kelly_max_fraction', 0.5),
            risk_reward_ratio=position_sizing_data["risk_reward_ratio"],
            min_lots=min_lots,
            max_lots=max_lots,
            mode=sizing_mode,
        )
        
        # Use shared utility
        return calculate_position_size(
            confidence=confidence,
            config=sizing_config,
            logger=logger,
        )
    
    # Run server with ZeroMQ and ModelGuardian
    async def run_v4_server():
        """Run the v4 inference server with full order execution."""
        nonlocal api_cfg
        import time
        import math
        import zmq
        import zmq.asyncio
        import struct
        
        from alphaos.core.types import Tick, Order, OrderAction, Signal, SignalType, MarketPhase
        from alphaos.execution.zmq_client import ZeroMQClient
        from alphaos.monitoring.model_guardian import ModelGuardian
        from alphaos.monitoring.risk_manager import RiskManager
        from alphaos.monitoring.runtime_state import RuntimeSnapshot
        from alphaos.monitoring.runtime_store import RuntimeStore
        from alphaos.monitoring.data_store import DataStore
        from alphaos.monitoring.ws_runtime_server import WSRuntimeServer
        from alphaos.core.config import ExitV21Config, ExecutionCloseBackoffConfig
        from alphaos.execution.exit import (
            ExitActionV21,
            ExitDecisionV21,
            PositionStateV21,
            ExitPolicyV21,
        )
        
        def _get_entry_price_guard() -> dict:
            gate_data = execution_data.get("execution_gate", {})
            if isinstance(gate_data, dict):
                entry_data = gate_data.get("entry", {})
                if isinstance(entry_data, dict):
                    guard = entry_data.get("price_guard")
                    if isinstance(guard, dict):
                        return guard
            return execution_data.get("price_guard", {})
        
        def _passes_price_guard(
            signal_type: SignalType,
            bid: float,
            ask: float,
            entry_price: float | None,
        ) -> tuple[bool, str]:
            guard = _get_entry_price_guard()
            if not guard or not guard.get("enabled", False):
                return True, ""
            if bid <= 0 or ask <= 0:
                return False, "price_guard_invalid_bid_ask"
            
            symbol_info = yaml_config.get("symbol_info", {}) if isinstance(yaml_config, dict) else {}
            point = float(symbol_info.get("point", 0.01))
            if point <= 0:
                return False, "price_guard_invalid_point"
            
            spread_points = (ask - bid) / point
            if guard.get("max_spread_points", 0.0) > 0 and spread_points > guard.get("max_spread_points", 0.0):
                return False, "price_guard_spread_too_wide"
            
            if entry_price is None or entry_price <= 0:
                return True, ""
            
            band_points = guard.get("entry_band_points", 0.0)
            slip_points = guard.get("max_slippage_points", 0.0)
            reject_if_outside = guard.get("reject_if_outside_band", True)
            
            if signal_type == SignalType.LONG:
                band_price = band_points * point
                slippage_price = slip_points * point
                if band_points > 0 and ask > entry_price + band_price:
                    return (not reject_if_outside), "price_guard_entry_band"
                if slip_points > 0 and ask > entry_price + slippage_price:
                    return False, "price_guard_slippage"
            elif signal_type == SignalType.SHORT:
                band_price = band_points * point
                slippage_price = slip_points * point
                if band_points > 0 and bid < entry_price - band_price:
                    return (not reject_if_outside), "price_guard_entry_band"
                if slip_points > 0 and bid < entry_price - slippage_price:
                    return False, "price_guard_slippage"
            
            return True, ""
        
        # Initialize ZeroMQ client
        zmq_client = ZeroMQClient(zmq_config)
        
        # Initialize Model Guardian
        guardian = ModelGuardian(
            enabled=guardian_enabled,
            nan_inf_check=guardian_data["nan_inf_check"],
            state_saturation_threshold=guardian_data["state_saturation_threshold"],
            confidence_collapse_window=guardian_data["confidence_collapse_window"],
            confidence_collapse_threshold=guardian_data["confidence_collapse_threshold"],
            latency_threshold_ms=guardian_latency_threshold,
            lock_file_path=guardian_data["lock_file_path"],
        )
        
        # ================================================================
        # v4.0: Risk Manager (single authority for circuit breakers)
        # ================================================================
        # Create proper RiskConfig from YAML namespace (with defaults for missing attrs)
        from alphaos.core.config import RiskConfig
        risk_cfg = config.risk
        risk_config = RiskConfig(
            min_position_lots=position_sizing_data["min_lots"],
            max_position_lots=position_sizing_data["max_lots"],
            max_position_usd=risk_cfg.max_position_usd,
            max_daily_loss_pct=risk_cfg.max_daily_loss_pct,
            max_consecutive_losses=risk_cfg.max_consecutive_losses,
            min_temperature=risk_cfg.min_temperature,
            max_entropy=risk_cfg.max_entropy,
            gate=getattr(risk_cfg, "gate", None) or RiskConfig().gate,
        )
        risk_manager = RiskManager(config=risk_config)
        logger.info(
            "Risk manager initialized",
            max_position_lots=risk_config.max_position_lots,
            max_daily_loss_pct=risk_config.max_daily_loss_pct,
            max_consecutive_losses=risk_config.max_consecutive_losses,
        )
        
        # ================================================================
        # v4.1: Exit v2.1 - Bid/Ask + Cost Guard + Alignment Modulation
        # ================================================================
        # Exit logic runs BEFORE entry logic on each tick
        # Maintains PositionStateV21 as single source of truth
        exit_v21_config = ExitV21Config.model_validate(exit_v21_data)
        exit_policy = ExitPolicyV21(exit_v21_config)
        positions: dict[int, PositionStateV21] = {}  # ticket -> PositionStateV21
        close_backoff_cfg = ExecutionCloseBackoffConfig.model_validate(
            execution_data.get("close_backoff", {})
        )
        close_backoff_state: dict[int, dict[str, float | int]] = {}
        
        logger.info(
            "Exit v2.1 policy initialized",
            be_trigger=exit_v21_config.be_trigger_net_usd,
            partial_trigger=exit_v21_config.partial1_trigger_net_usd,
        )
        
        logger.info(
            "Server components initialized",
            dry_run=args.dry_run,
            guardian_enabled=guardian_enabled,
            symbol=args.symbol,
            volume=args.volume,
        )

        # ================================================================
        # Optional: Integrated Web Server (UI static hosting + /api)
        # ================================================================
        web_task = None
        web_server = None
        if getattr(args, "web", False):
            try:
                from alphaos.api.server import create_app
                import uvicorn

                # Build app config from the same YAML used by serve
                if api_cfg is None:
                    from alphaos.core.config import AlphaOSConfig
                    api_cfg = AlphaOSConfig.model_validate(patched_config)

                # Resolve ui/dist relative to repo root (src/alphaos/v4/cli.py -> repo root is parents[3])
                repo_root = Path(__file__).resolve().parents[3]
                ui_dist = Path(args.ui_dist)
                if not ui_dist.is_absolute():
                    ui_dist = repo_root / ui_dist

                app = create_app(api_cfg, ui_dist_path=ui_dist)
                uv_cfg = uvicorn.Config(
                    app,
                    host=str(args.web_host),
                    port=int(args.web_port),
                    log_level="info",
                    loop="asyncio",
                    access_log=False,
                    log_config=None,
                )
                web_server = uvicorn.Server(uv_cfg)
                web_task = asyncio.create_task(web_server.serve())

                display_host = "localhost" if str(args.web_host) in ("0.0.0.0", "127.0.0.1") else str(args.web_host)
                logger.info(
                    "Web UI server started",
                    ui=f"http://{display_host}:{int(args.web_port)}",
                    ws=f"ws://{display_host}:{int(args.ws_port)}",
                    ui_dist=str(ui_dist),
                )
            except Exception as e:
                logger.warning("Web UI server failed to start", error=str(e))

        # ================================================================
        # WebSocket Runtime Server (UI realtime snapshots)
        # ================================================================
        ws_server = None
        try:
            ws_host = "0.0.0.0"
            ws_server = WSRuntimeServer(host=ws_host, port=int(args.ws_port))
            await ws_server.start()
            display_host = "localhost" if ws_host in ("0.0.0.0", "127.0.0.1") else ws_host
            logger.info(
                "WSRuntimeServer started",
                ws=f"ws://{display_host}:{int(args.ws_port)}",
            )
        except Exception as e:
            logger.warning("WSRuntimeServer failed to start", error=str(e))
        
        # Connect to MT5
        if not args.dry_run:
            await zmq_client.connect()
            logger.info("Connected to MT5 ZeroMQ endpoints")

        runtime_store = None
        data_store = None
        if api_cfg is not None:
            try:
                runtime_store = RuntimeStore(api_cfg.database, args.symbol)
                await runtime_store.initialize()
            except Exception as e:
                logger.warning("RuntimeStore init failed", error=str(e))
                runtime_store = None
            try:
                data_store = DataStore(api_cfg.database, args.symbol, api_cfg.monitoring.data_store)
                await data_store.initialize()
                await data_store.start()
            except Exception as e:
                logger.warning("DataStore init failed", error=str(e))
                data_store = None
        symbol_info = {}
        account_equity = None
        last_account_update_ts = 0.0

        if not args.dry_run:
            try:
                symbol_info = await zmq_client.get_symbol_info(args.symbol)
                if symbol_info.get("error"):
                    logger.warning("Symbol info error", error=symbol_info.get("error"))
                    symbol_info = {}
                else:
                    logger.info(
                        "Symbol info loaded",
                        symbol=symbol_info.get("symbol", args.symbol),
                        tick_size=symbol_info.get("tick_size"),
                        tick_value=symbol_info.get("tick_value"),
                        volume_min=symbol_info.get("volume_min"),
                        volume_max=symbol_info.get("volume_max"),
                        volume_step=symbol_info.get("volume_step"),
                    )
            except Exception as e:
                logger.warning("Symbol info fetch failed", error=str(e))

        # ================================================================
        # v4.0: Boot State Machine
        # BOOTSTRAP_REPLAY → LIVE_WARMUP → LIVE_TRADING
        # ================================================================
        
        # State enum (defined at module level scope so accessible everywhere)
        BOOT_STATE_REPLAY = "BOOTSTRAP_REPLAY"
        BOOT_STATE_WARMUP = "LIVE_WARMUP"
        BOOT_STATE_TRADING = "LIVE_TRADING"
        
        # Current boot state (will transition through states)
        current_boot_state = BOOT_STATE_TRADING  # Default: skip replay, go straight to trading
        
        # Determine target buffer size
        target_buffer = (
            args.replay_target_buffer
            if args.replay_target_buffer > 0
            else boot_data["replay_target_buffer"]
        )
        
        # v4.0: MT5 broker-native tick history replay (preferred)
        if args.replay_mt5 and not args.dry_run:
            replay_window_sec = (
                args.replay_window_sec
                if args.replay_window_sec != serve_arg_defaults["replay_window_sec"]
                else boot_data["replay_window_sec"]
            )
            replay_pace_tps = (
                args.replay_pace_tps
                if args.replay_pace_tps != serve_arg_defaults["replay_pace_tps"]
                else boot_data["replay_pace_tps"]
            )
            replay_end_eps_ms = (
                args.replay_end_eps_ms
                if args.replay_end_eps_ms != serve_arg_defaults["replay_end_eps_ms"]
                else boot_data["replay_end_eps_ms"]
            )
            replay_max_ticks = (
                args.replay_max_ticks
                if args.replay_max_ticks != serve_arg_defaults["replay_max_ticks"]
                else boot_data["replay_max_ticks"]
            )
            logger.info("=" * 60)
            logger.info("[BOOT] Starting MT5 Broker-Native Tick Replay")
            logger.info(f"[BOOT] Mode: BOOTSTRAP_REPLAY")
            logger.info(f"[BOOT] Target buffer: {target_buffer}")
            logger.info(f"[BOOT] Window: {replay_window_sec}s")
            logger.info(f"[BOOT] Pace: {replay_pace_tps} tps")
            logger.info(f"[BOOT] End epsilon: {replay_end_eps_ms}ms")
            logger.info("=" * 60)
            
            # Request replay from EA
            replay_result = await zmq_client.start_tick_replay(
                symbol=args.symbol,
                window_sec=replay_window_sec,
                end_eps_ms=replay_end_eps_ms,
                max_ticks=replay_max_ticks,
                pace_tps=replay_pace_tps,
            )
            
            if not replay_result.get("success"):
                logger.error(f"[BOOT] Failed to start MT5 replay: {replay_result.get('error')}")
                logger.warning("[BOOT] Falling back to Hybrid Cold Start")
            else:
                # Set up replay state
                boot_state = BOOT_STATE_REPLAY
                replay_tick_count = 0
                replay_start_time = time.time()
                replay_expected_count = replay_result.get("count", 0)
                
                logger.info(f"[BOOT] Replay started: expecting {replay_expected_count:,} ticks")
                
                # Create tick socket for replay (same as live)
                import zmq
                import zmq.asyncio
                context_replay = zmq.asyncio.Context()
                tick_socket_replay = context_replay.socket(zmq.SUB)
                tick_socket_replay.connect(zmq_config.tick_endpoint)
                tick_socket_replay.setsockopt(zmq.SUBSCRIBE, b"")
                tick_socket_replay.setsockopt(zmq.RCVTIMEO, boot_data["replay_socket_timeout_ms"])
                
                # Process replay ticks with integrity validation
                last_tick_time_msc = 0
                replay_errors = []
                gap_threshold_ms = boot_data["replay_gap_threshold_ms"]
                consecutive_timeouts = 0
                max_consecutive_timeouts = boot_data["replay_max_consecutive_timeouts"]
                replay_aborted = False
                max_replay_duration_sec = boot_data["replay_max_duration_sec"]
                last_tick_received_time = time.time()
                
                # Note: We don't compare tick_time with Python local time because
                # MT5 broker time may differ from local system time (timezone, clock drift).
                # The EA already enforces end_eps_ms in CopyTicksRange, so we trust it.
                # We track the first tick time as reference for relative validation.
                first_tick_time_msc = None
                
                while boot_state == BOOT_STATE_REPLAY:
                    # Check maximum replay duration
                    elapsed = time.time() - replay_start_time
                    if elapsed > max_replay_duration_sec:
                        logger.warning(f"[BOOT] Replay max duration ({max_replay_duration_sec}s) reached, stopping")
                        await zmq_client.stop_tick_replay()
                        boot_state = BOOT_STATE_WARMUP
                        break
                    
                    try:
                        message = await tick_socket_replay.recv()
                        replay_tick_count += 1
                        consecutive_timeouts = 0  # Reset on successful receive
                        last_tick_received_time = time.time()  # Track last tick time
                        
                        if len(message) != 36:
                            continue
                        
                        import struct
                        bid, ask, time_msc, volume, flags = struct.unpack("ddqqI", message)
                        
                        # ================================================================
                        # Integrity Check 1: Time monotonicity
                        # ================================================================
                        if (
                            last_tick_time_msc > 0
                            and time_msc < last_tick_time_msc - boot_data["replay_time_monotonic_tolerance_ms"]
                        ):
                            replay_errors.append(f"Non-monotonic time: {time_msc} < {last_tick_time_msc}")
                            if len(replay_errors) > 10:
                                logger.error("[BOOT] Too many time errors, aborting replay")
                                replay_aborted = True
                                break
                        
                        # ================================================================
                        # Integrity Check 2: Gap detection (> 30s)
                        # ================================================================
                        if last_tick_time_msc > 0:
                            gap_ms = time_msc - last_tick_time_msc
                            if gap_ms > gap_threshold_ms:
                                logger.warning(f"[BOOT] Large tick gap detected: {gap_ms/1000:.1f}s")
                                replay_errors.append(f"Gap: {gap_ms/1000:.1f}s at tick {replay_tick_count}")
                        
                        # Track first tick time (for relative validation, not absolute)
                        if first_tick_time_msc is None:
                            first_tick_time_msc = time_msc
                            logger.info(f"[BOOT] First replay tick: time_msc={time_msc}")
                        
                        last_tick_time_msc = time_msc
                        
                        tick = Tick(
                            timestamp_us=int(time_msc) * 1000,
                            bid=float(bid),
                            ask=float(ask),
                            bid_volume=0.0,
                            ask_volume=0.0,
                            last=0.0,
                            last_volume=float(volume),
                            flags=int(flags),
                        )
                        
                        # Process tick (NO execution, NO guardian)
                        result = engine.process_tick(tick)
                        
                        # Progress logging
                        if replay_tick_count % 50000 == 0:
                            buffer_size = len(engine._confidence_buffer)
                            logger.info(
                                f"[BOOT] Replay progress: {replay_tick_count:,} ticks, "
                                f"bars={engine.bar_count}, confidence_buffer={buffer_size}"
                            )
                        
                        # Early stop 1: buffer filled
                        current_buffer = len(engine._confidence_buffer)
                        if current_buffer >= target_buffer:
                            logger.info(f"[BOOT] Target buffer reached: {current_buffer} >= {target_buffer}")
                            await zmq_client.stop_tick_replay()
                            boot_state = BOOT_STATE_WARMUP
                            break
                        
                        # Early stop 2: all expected ticks received
                        if replay_expected_count > 0 and replay_tick_count >= replay_expected_count:
                            logger.info(f"[BOOT] All expected ticks received: {replay_tick_count:,}")
                            boot_state = BOOT_STATE_WARMUP
                            break
                        
                        # Check replay status periodically (with timeout protection)
                        if replay_tick_count % 100000 == 0:
                            try:
                                # Use asyncio timeout to prevent blocking
                                status = await asyncio.wait_for(
                                    zmq_client.get_replay_status(),
                                    timeout=5.0
                                )
                                if not status.get("active", False):
                                    logger.info("[BOOT] Replay completed by EA")
                                    boot_state = BOOT_STATE_WARMUP
                                    break
                            except asyncio.TimeoutError:
                                logger.warning("[BOOT] Status check timeout, continuing...")
                            except Exception as e:
                                logger.warning(f"[BOOT] Status check error: {e}")
                        
                    except zmq.Again:
                        # Timeout - check if replay is done or stalled
                        consecutive_timeouts += 1
                        logger.info(f"[BOOT] Tick recv timeout #{consecutive_timeouts}, checking status...")
                        
                        # Check if we've received enough samples
                        current_buffer = len(engine._confidence_buffer)
                        min_req = engine.config.confidence_gate.min_required
                        if current_buffer >= min_req:
                            logger.info(f"[BOOT] Buffer sufficient ({current_buffer} >= {min_req}), stopping replay")
                            await zmq_client.stop_tick_replay()
                            boot_state = BOOT_STATE_WARMUP
                            break
                        
                        # Check replay status (with timeout)
                        try:
                            status = await asyncio.wait_for(
                                zmq_client.get_replay_status(),
                                timeout=3.0
                            )
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.warning(f"[BOOT] Status check failed: {e}")
                            status = {"active": False}  # Assume done if can't check
                        
                        if not status.get("active", False):
                            logger.info("[BOOT] Replay completed (EA finished)")
                            boot_state = BOOT_STATE_WARMUP
                            break
                        elif consecutive_timeouts >= max_consecutive_timeouts:
                            logger.warning(f"[BOOT] Replay stalled ({consecutive_timeouts} timeouts), stopping")
                            await zmq_client.stop_tick_replay()
                            boot_state = BOOT_STATE_WARMUP  # Don't abort, just move on
                            break
                        else:
                            logger.warning(f"[BOOT] Waiting for more ticks...")
                            continue
                            
                    except Exception as e:
                        logger.error(f"[BOOT] Replay error: {e}")
                        replay_aborted = True
                        break
                
                # Cleanup replay socket
                tick_socket_replay.close()
                context_replay.term()
                
                # Final replay stats
                replay_duration = time.time() - replay_start_time
                buffer_size = len(engine._confidence_buffer)
                
                # Determine outcome
                if replay_aborted:
                    logger.error("=" * 60)
                    logger.error("[BOOT] MT5 Tick Replay ABORTED")
                    logger.error(f"[BOOT] Ticks processed before abort: {replay_tick_count:,}")
                    logger.error(f"[BOOT] Errors: {len(replay_errors)}")
                    for err in replay_errors[:5]:
                        logger.error(f"[BOOT]   - {err}")
                    logger.error("[BOOT] Falling back to Hybrid Cold Start")
                    logger.error("=" * 60)
                    # Reset boot state to go through normal cold start
                    boot_state = BOOT_STATE_TRADING
                    live_state = BOOT_STATE_TRADING
                else:
                    # Successful replay - print summary
                    logger.info("=" * 60)
                    logger.info("[BOOT] MT5 Tick Replay Completed")
                    logger.info(f"[BOOT] Ticks processed: {replay_tick_count:,}")
                    if replay_duration > 0:
                        logger.info(f"[BOOT] Duration: {replay_duration:.1f}s ({replay_tick_count/replay_duration:.0f} tps)")
                    logger.info(f"[BOOT] Bars generated: {engine.bar_count}")
                    logger.info(f"[BOOT] Confidence buffer: {buffer_size} samples")
                    logger.info(f"[BOOT] CfC hidden state: evolved ({engine.bar_count} steps)")
                    
                    # Determine cold start status
                    min_required = engine.config.confidence_gate.min_required
                    full_buffer = engine.config.confidence_gate.buffer_size
                    
                    if buffer_size >= full_buffer:
                        logger.info(f"[BOOT] Rolling quantile gate ACTIVE from first live bar")
                        logger.info(f"[BOOT] Cold start: SKIPPED (buffer >= {full_buffer})")
                    elif buffer_size >= min_required:
                        logger.info(f"[BOOT] Fixed fallback gate active (buffer >= {min_required})")
                        logger.info(f"[BOOT] Cold start: PARTIAL (need {full_buffer - buffer_size} more for full)")
                    else:
                        logger.warning(f"[BOOT] Cold start: ACTIVE (need {min_required - buffer_size} more samples)")
                        logger.warning("[BOOT] Trading will be blocked until min_required reached")
                    
                    if replay_errors:
                        logger.warning(f"[BOOT] Replay had {len(replay_errors)} integrity errors")
                        for err in replay_errors[:3]:
                            logger.warning(f"[BOOT]   - {err}")
                    
                    logger.info("=" * 60)
                    logger.info(f"[BOOT] Transitioning to LIVE_WARMUP")
        
        # Historical Replay from CSV file (fallback)
        elif args.replay_history:
            replay_ticks = (
                args.replay_ticks
                if args.replay_ticks != serve_arg_defaults["replay_ticks"]
                else boot_data["replay_csv_max_ticks"]
            )
            _historical_replay(
                engine=engine,
                filepath=args.replay_history,
                max_ticks=replay_ticks,
                logger=logger,
            )
        # History warmup from MT5 (requires EA support)
        elif args.history_warmup:
            await _warmup_from_history(
                zmq_client=zmq_client,
                engine=engine,
                symbol=args.symbol,
                n_bars=(
                    args.history_bars
                    if args.history_bars != serve_arg_defaults["history_bars"]
                    else boot_data["history_warmup_bars"]
                ),
                logger=logger,
            )
        
        # Create raw ZeroMQ context for tick subscription
        context = zmq.asyncio.Context()
        tick_socket = context.socket(zmq.SUB)
        tick_socket.connect(zmq_config.tick_endpoint)
        tick_socket.setsockopt(zmq.SUBSCRIBE, b"")
        
        logger.info(f"Subscribed to tick feed at {zmq_config.tick_endpoint}")
        
        # v4.0: Boot state tracking for LIVE_WARMUP → LIVE_TRADING transition
        live_warmup_bars = 10  # Process this many bars in LIVE_WARMUP before LIVE_TRADING
        live_warmup_start_bars = engine.bar_count
        # If we did MT5 replay, start in LIVE_WARMUP; otherwise go straight to LIVE_TRADING
        live_state = BOOT_STATE_WARMUP if (args.replay_mt5 and not args.dry_run) else BOOT_STATE_TRADING
        live_warmup_logged = False
        
        # Order tracking
        next_magic = int(time.time() * 1000) % 1000000000
        model_version = model_dir.name
        from collections import deque
        trend_cap_cfg = execution_data.get("trend_cap", {}) if isinstance(execution_data, dict) else {}
        trend_cap_enabled = bool(trend_cap_cfg.get("enabled", False))
        trend_cap_window = int(trend_cap_cfg.get("window", 20))
        trend_cap_max_ratio = float(trend_cap_cfg.get("max_counter_ratio", 0.4))
        trend_cap_min_samples = int(trend_cap_cfg.get("min_samples", 5))
        trend_cap_history: deque[str] = deque(maxlen=trend_cap_window)
        
        def get_next_magic() -> int:
            nonlocal next_magic
            magic = next_magic
            next_magic += 1
            return magic

        async def _send_order(order: Order, context: str):
            sent_ts = time.time()
            result = await zmq_client.send_order(order)
            latency_ms = (time.time() - sent_ts) * 1000.0
            if data_store is not None:
                data_store.enqueue_order(
                    ts=sent_ts,
                    order=order,
                    context=context,
                    status=result.status.name if hasattr(result.status, "name") else str(result.status),
                    error_code=result.error_code,
                    error_message=result.error_message,
                )
                data_store.enqueue_fill(
                    ts=time.time(),
                    order=order,
                    result=result,
                    latency_ms=latency_ms,
                )
            return result
        
        tick_count = 0
        last_snapshot_time = 0.0
        bar_count_prev = 0
        
        try:
            while True:
                try:
                    # Receive tick
                    message = await tick_socket.recv()
                    tick_count += 1
                    
                    # Parse binary tick (36 bytes)
                    if len(message) != 36:
                        logger.warning(f"Invalid tick size: {len(message)} bytes (expected 36)")
                        continue
                    
                    bid, ask, time_msc, volume, flags = struct.unpack(
                        "ddqqI", message
                    )
                    
                    # Log first few ticks for debugging
                    if tick_count <= 3:
                        logger.info(f"Tick #{tick_count}: bid={bid:.2f}, ask={ask:.2f}, vol={volume}")
                    elif tick_count == 100:
                        logger.info(f"Received 100 ticks, bars={engine.bar_count}")
                    elif tick_count % 1000 == 0:
                        logger.info(f"Tick count: {tick_count}, bars: {engine.bar_count}")
                    
                    tick = Tick(
                        timestamp_us=int(time_msc) * 1000,
                        bid=float(bid),
                        ask=float(ask),
                        bid_volume=0.0,
                        ask_volume=0.0,
                        last=0.0,
                        last_volume=float(volume),
                        flags=int(flags),
                    )
                    if data_store is not None:
                        data_store.enqueue_tick(
                            ts=tick.timestamp_s,
                            bid=tick.bid,
                            ask=tick.ask,
                            volume=tick.last_volume,
                            flags=tick.flags,
                        )
                    
                    # ================================================================
                    # STEP 1: EXIT - Process exits FIRST (v4.1 Exit v2.1)
                    # ================================================================
                    current_time = time.time()
                    
                    for ticket, state in list(positions.items()):
                        # Update position state
                        state.update_tick(tick, current_time, exit_v21_config)
                        
                        # Evaluate exit policy
                        exit_decision = exit_policy.evaluate(state, current_time)

                        # Periodic exit diagnostics (rate-limited)
                        if tick_count % 50 == 0:
                            alignment_mult = exit_policy._get_alignment_multipliers(state.trend_alignment)
                            be_threshold = exit_v21_config.be_trigger_net_usd * alignment_mult.be_trigger_mult
                            partial_threshold = exit_v21_config.partial1_trigger_net_usd * alignment_mult.be_trigger_mult
                            trail_threshold = exit_v21_config.trail_start_net_usd * alignment_mult.be_trigger_mult
                            logger.info(
                                "EXIT_STATE",
                                ticket=ticket,
                                stage=state.stage.name,
                                alignment=state.trend_alignment,
                                net_pnl=round(state.net_pnl_usd, 2),
                                current_sl=round(state.current_sl, 2),
                                price_used=round(exit_decision.price_used, 2),
                                bid=round(state.current_bid, 2),
                                ask=round(state.current_ask, 2),
                                be_th=round(be_threshold, 2),
                                partial_th=round(partial_threshold, 2),
                                trail_th=round(trail_threshold, 2),
                            )
                        
                        # Handle exit decision
                        if exit_decision.action != ExitActionV21.NOOP and not args.dry_run:
                            # Prevent REJECTED storms for CLOSE actions
                            if (
                                exit_decision.action in (ExitActionV21.FULL_CLOSE, ExitActionV21.PARTIAL_CLOSE)
                                and close_backoff_cfg.enabled
                            ):
                                cb_state = close_backoff_state.get(ticket)
                                if cb_state is not None:
                                    last_ts = float(cb_state.get("last_ts", 0.0))
                                    if current_time - last_ts < close_backoff_cfg.cooldown_sec:
                                        continue

                            logger.info(
                                "EXIT: "
                                f"{exit_decision.action.name} [{exit_decision.stage}] - {exit_decision.reason} "
                                f"price_used={exit_decision.price_used:.2f} "
                                f"net_pnl={exit_decision.net_pnl_usd:.2f} "
                                f"alignment={exit_decision.alignment}"
                            )
                            
                            if exit_decision.action == ExitActionV21.MOVE_SL:
                                # MODIFY: Change stop loss
                                modify_order = Order(
                                    magic=get_next_magic(),
                                    action=OrderAction.MODIFY,
                                    symbol=args.symbol,
                                    volume=state.current_lots,
                                    sl=exit_decision.new_sl,
                                    tp=state.initial_tp,
                                    comment=f"Exit_{exit_decision.stage}",
                                    ticket=state.ticket,
                                )
                                try:
                                    result_modify = await _send_order(modify_order, context="EXIT_MOVE_SL")
                                    if result_modify.status.name == "FILLED":
                                        if exit_decision.stage == "BE":
                                            state.mark_be_done(exit_decision.new_sl, current_time)
                                        elif exit_decision.stage == "TRAILING":
                                            state.update_trailing_sl(exit_decision.new_sl, current_time)
                                        logger.info(f"SL modified: {exit_decision.new_sl:.2f}")
                                except Exception as e:
                                    logger.error(f"SL modify failed: {e}")
                            
                            elif exit_decision.action == ExitActionV21.PARTIAL_CLOSE:
                                # CLOSE partial
                                close_order = Order(
                                    magic=get_next_magic(),
                                    action=OrderAction.CLOSE,
                                    symbol=args.symbol,
                                    volume=exit_decision.close_lots,
                                    comment="Exit_PARTIAL",
                                    ticket=state.ticket,
                                )
                                try:
                                    result_close = await _send_order(close_order, context="EXIT_PARTIAL")
                                    if result_close.status.name == "FILLED":
                                        close_price = result_close.price_filled if result_close.price_filled > 0 else exit_decision.price_used
                                        state.mark_partial_done(close_price, exit_decision.close_lots, current_time)
                                        logger.info(f"Partial close: {exit_decision.close_lots:.2f} lots @ {close_price:.2f}")
                                        close_backoff_state.pop(ticket, None)
                                    elif result_close.status.name == "REJECTED":
                                        error_msg = getattr(result_close, 'error_message', '') or ''
                                        error_code = getattr(result_close, 'error_code', 0)
                                        logger.warning(
                                            f"Partial close REJECTED: error_code={error_code}, "
                                            f"error_message='{error_msg}', "
                                            f"requested_volume={close_order.volume:.2f}"
                                        )
                                        if close_backoff_cfg.enabled:
                                            prev = close_backoff_state.get(ticket, {"attempts": 0})
                                            attempts = int(prev.get("attempts", 0)) + 1
                                            close_backoff_state[ticket] = {
                                                "attempts": attempts,
                                                "last_ts": current_time,
                                            }
                                            if attempts >= close_backoff_cfg.max_attempts:
                                                remote_positions = await zmq_client.query_positions(symbol=args.symbol)
                                                if not any(pos.ticket == ticket for pos in remote_positions):
                                                    logger.warning(
                                                        f"Position {ticket} not found in MT5 after partial close failures, removing local state"
                                                    )
                                                    del positions[ticket]
                                                    close_backoff_state.pop(ticket, None)
                                except Exception as e:
                                    logger.error(f"Partial close failed: {e}")
                            
                            elif exit_decision.action == ExitActionV21.FULL_CLOSE:
                                # CLOSE entire position
                                close_order = Order(
                                    magic=get_next_magic(),
                                    action=OrderAction.CLOSE,
                                    symbol=args.symbol,
                                    volume=state.current_lots,
                                    comment=f"Exit_{exit_decision.stage}",
                                    ticket=state.ticket,
                                )
                                try:
                                    result_close = await _send_order(close_order, context="EXIT_FULL")
                                    if result_close.status.name == "FILLED":
                                        close_price = result_close.price_filled if result_close.price_filled > 0 else exit_decision.price_used
                                        final_pnl = state.net_pnl_usd
                                        logger.info(
                                            f"AUDIT: Position closed ticket={ticket}, "
                                            f"pnl=${final_pnl:.2f}, stage={exit_decision.stage}"
                                        )
                                        
                                        # v4.0: Record trade result to RiskManager for circuit breakers
                                        risk_manager.record_trade_result(result_close, final_pnl)
                                        logger.debug(
                                            "Risk stats updated",
                                            **risk_manager.get_stats(),
                                        )
                                        
                                        del positions[ticket]
                                        close_backoff_state.pop(ticket, None)
                                    elif result_close.status.name == "REJECTED":
                                        error_msg = getattr(result_close, 'error_message', '') or ''
                                        error_code = getattr(result_close, 'error_code', 0)
                                        logger.warning(
                                            f"Full close REJECTED: error_code={error_code}, "
                                            f"error_message='{error_msg}', "
                                            f"requested_volume={close_order.volume:.2f}"
                                        )
                                        if close_backoff_cfg.enabled:
                                            prev = close_backoff_state.get(ticket, {"attempts": 0})
                                            attempts = int(prev.get("attempts", 0)) + 1
                                            close_backoff_state[ticket] = {
                                                "attempts": attempts,
                                                "last_ts": current_time,
                                            }
                                            if attempts >= close_backoff_cfg.max_attempts:
                                                remote_positions = await zmq_client.query_positions(symbol=args.symbol)
                                                if not any(pos.ticket == ticket for pos in remote_positions):
                                                    logger.warning(
                                                        f"Position {ticket} not found in MT5 after close failures, removing local state"
                                                    )
                                                    del positions[ticket]
                                                    close_backoff_state.pop(ticket, None)
                                except Exception as e:
                                    logger.error(f"Full close failed: {e}")
                                    # Check for phantom position
                                    if "10036" in str(e) or "10029" in str(e):
                                        logger.warning(f"Phantom position {ticket}, removing")
                                        del positions[ticket]
                                        close_backoff_state.pop(ticket, None)
                            
                            # Only one exit action per tick
                            break
                    
                    # ================================================================
                    # STEP 2: ENTRY - Process entry signals AFTER exits
                    # ================================================================
                    # Measure inference latency
                    start_time = time.perf_counter()
                    
                    # Process tick
                    result = engine.process_tick(tick)
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    if result is None:
                        continue

                    if data_store is not None:
                        bar = engine._last_bar
                        if bar is not None:
                            data_store.enqueue_bar(bar, bar_idx=result.bar_idx)
                            data_store.enqueue_decision(
                                ts=bar.close_time.timestamp(),
                                result=result,
                                model_version=model_version,
                                config_hash=config_hash,
                            )
                    
                    # Log bar generation progress
                    if engine.bar_count > bar_count_prev:
                        bar_count_prev = engine.bar_count
                        if bar_count_prev <= 5 or bar_count_prev == 50 or bar_count_prev % 100 == 0:
                            logger.info(f"Bar #{bar_count_prev} generated, warmup={bar_count_prev < 50}")
                    
                    # v4.0: Track filter reasons and trend alignment for diagnostics
                    if not hasattr(engine, '_filter_stats'):
                        engine._filter_stats = {
                            'NO_EVENT': 0, 'WARMUP': 0, 'COLD_START': 0, 
                            'LOW_CONF': 0, 'NO_STOP_LOSS': 0, 'OTHER': 0, 'PASSED': 0
                        }
                    if not hasattr(engine, '_alignment_stats'):
                        # 追踪事件的趋势对齐分布（仅计入有事件的 bar）
                        engine._alignment_stats = {
                            'ALIGNED': 0, 'COUNTER': 0, 'UNKNOWN': 0
                        }
                    
                    # 更新过滤统计
                    if result.should_trade:
                        engine._filter_stats['PASSED'] += 1
                    elif result.filtered_reason:
                        # 按优先级匹配过滤原因
                        matched = False
                        for key in ['NO_EVENT', 'WARMUP', 'COLD_START', 'LOW_CONFIDENCE', 'NO_STOP_LOSS']:
                            if key in result.filtered_reason:
                                stat_key = 'LOW_CONF' if key == 'LOW_CONFIDENCE' else key
                                engine._filter_stats[stat_key] += 1
                                matched = True
                                break
                        if not matched:
                            engine._filter_stats['OTHER'] += 1
                    
                    # 更新 trend alignment 统计（仅当有事件时）
                    if result.has_signal:
                        fvg_dir = 1 if result.fvg_event > 0 else (-1 if result.fvg_event < 0 else 0)
                        st_15m = result.st_trend_15m
                        if st_15m == 0:
                            engine._alignment_stats['UNKNOWN'] += 1
                        elif fvg_dir == st_15m:
                            engine._alignment_stats['ALIGNED'] += 1
                        else:
                            engine._alignment_stats['COUNTER'] += 1
                    
                    # Periodic diagnostic log (every 5000 ticks)
                    if tick_count % 5000 == 0:
                        buffer_size = len(engine._confidence_buffer)
                        stats = engine._filter_stats
                        align = engine._alignment_stats
                        logger.info(
                            f"[DIAG] ticks={tick_count}, bars={engine.bar_count}, "
                            f"buffer={buffer_size}, "
                            f"filters={{pass={stats['PASSED']}, no_event={stats['NO_EVENT']}, "
                            f"cold={stats['COLD_START']}, low_conf={stats['LOW_CONF']}, "
                            f"no_sl={stats['NO_STOP_LOSS']}}}, "
                            f"alignment={{aligned={align['ALIGNED']}, counter={align['COUNTER']}, unknown={align['UNKNOWN']}}}"
                        )
                    
                    # v4.0: LIVE_WARMUP → LIVE_TRADING transition
                    # After MT5 replay, process a few live bars to verify pipeline continuity
                    if live_state == BOOT_STATE_WARMUP:
                        bars_in_warmup = engine.bar_count - live_warmup_start_bars
                        if bars_in_warmup >= live_warmup_bars:
                            live_state = BOOT_STATE_TRADING
                            logger.info("=" * 60)
                            logger.info("[BOOT] Transitioning to LIVE_TRADING")
                            logger.info(f"[BOOT] Warmup bars: {bars_in_warmup}")
                            logger.info(f"[BOOT] Total bars: {engine.bar_count}")
                            logger.info(f"[BOOT] Confidence buffer: {len(engine._confidence_buffer)}")
                            logger.info("=" * 60)
                        elif not live_warmup_logged:
                            logger.info(f"[BOOT] LIVE_WARMUP: processing {live_warmup_bars} bars before trading")
                            live_warmup_logged = True
                        # During LIVE_WARMUP, skip trading but continue processing
                        if live_state == BOOT_STATE_WARMUP:
                            continue
                    
                    # Skip Guardian check when:
                    # - Warmup period (confidence is 0.0, model not ready)
                    # - No event detected (confidence is 0.0, no prediction made)
                    # - Cold start (insufficient buffer for quantile)
                    # - Low confidence (already filtered, don't double-penalize)
                    # - NO_STOP_LOSS (v4.0 风险硬门)
                    skip_guardian = (
                        result.filtered_reason and (
                            "WARMUP" in result.filtered_reason or
                            "NO_EVENT" in result.filtered_reason or
                            "COLD_START" in result.filtered_reason or
                            "LOW_CONFIDENCE" in result.filtered_reason or
                            "NO_STOP_LOSS" in result.filtered_reason
                        )
                    )
                    
                    # Guardian check (only when actual prediction was made)
                    if guardian.enabled and not skip_guardian:
                        guardian_result = guardian.check_output(
                            prediction=result.direction,
                            confidence=result.meta_confidence,
                            hidden_state=None,  # v4 doesn't expose hidden state
                            latency_ms=latency_ms,
                        )
                        
                        if guardian_result.should_halt:
                            logger.error(
                                "Model Guardian HALTED trading",
                                reason=guardian_result.halt_reason,
                            )
                            # Continue processing ticks but don't trade
                            continue
                    
                    # Handle trading signal
                    if result.should_trade:
                        if len(positions) >= args.max_positions:
                            logger.info(
                                "Max positions reached, skip entry",
                                current=len(positions),
                                limit=args.max_positions,
                                symbol=args.symbol,
                            )
                            continue

                        direction = "LONG" if result.direction == 1 else "SHORT"
                        trend_alignment = "UNKNOWN"
                        alignment_trend = getattr(result, "st_trend_15m", 0)
                        if getattr(config, "trend_alignment_source", "st_15m") == "primary":
                            alignment_trend = getattr(result, "trend_direction", 0)
                        if alignment_trend != 0:
                            trend_alignment = "ALIGNED" if result.direction == alignment_trend else "COUNTER"
                        if trend_cap_enabled and trend_alignment == "COUNTER":
                            counter_count = sum(1 for a in trend_cap_history if a == "COUNTER")
                            total = len(trend_cap_history)
                            if total >= trend_cap_min_samples:
                                ratio = counter_count / max(1, total)
                                if ratio >= trend_cap_max_ratio:
                                    logger.info(
                                        "Counter-trend cap: skip entry",
                                        counter_ratio=round(ratio, 3),
                                        max_ratio=trend_cap_max_ratio,
                                        window=trend_cap_window,
                                    )
                                    continue
                        
                        # v4.0: Risk Manager check (single authority for circuit breakers)
                        # Create Signal object for risk check
                        # Map market phase string to enum
                        phase_map = {
                            "FROZEN": MarketPhase.FROZEN,
                            "LAMINAR": MarketPhase.LAMINAR,
                            "TURBULENT": MarketPhase.TURBULENT,
                            "PHASE_TRANSITION": MarketPhase.PHASE_TRANSITION,
                            "TRANSITION": MarketPhase.PHASE_TRANSITION,
                        }
                        market_phase_enum = phase_map.get(result.market_phase, MarketPhase.LAMINAR)
                        
                        signal_for_risk = Signal(
                            timestamp_us=tick.timestamp_us,
                            signal_type=SignalType.LONG if result.direction == 1 else SignalType.SHORT,
                            confidence=result.meta_confidence,
                            temperature=float(result.market_temperature),
                            entropy=float(result.market_entropy),
                            market_phase=market_phase_enum,
                        )

                        # Hard guard: treat invalid/low temperature as frozen
                        if signal_for_risk.temperature <= risk_manager.config.min_temperature:
                            logger.warning(
                                "Risk guard: low temperature, skip trade",
                                temperature=signal_for_risk.temperature,
                                threshold=risk_manager.config.min_temperature,
                                phase=result.market_phase,
                            )
                            continue
                        
                        risk_allowed, risk_reason = risk_manager.check_signal(signal_for_risk, context="entry")
                        if not risk_allowed:
                            logger.warning(
                                f"Risk Manager blocked trade: {risk_reason}",
                                risk_stats=risk_manager.get_stats(),
                            )
                            continue
                        
                        allowed, reason = _passes_price_guard(
                            signal_for_risk.signal_type,
                            bid=tick.bid,
                            ask=tick.ask,
                            entry_price=result.entry_price,
                        )
                        if not allowed:
                            logger.info(
                                "Entry blocked by price guard",
                                reason=reason,
                                bid=round(tick.bid, 2),
                                ask=round(tick.ask, 2),
                                entry_price=round(result.entry_price, 2),
                            )
                            continue
                        
                        # v4.0: Calculate position size using Kelly Criterion
                        # Fetch account equity periodically for risk-based sizing
                        if not args.dry_run:
                            if current_time - last_account_update_ts >= 5.0:
                                status = await zmq_client.get_status()
                                if status:
                                    account_equity = status.get("equity", account_equity)
                                last_account_update_ts = current_time

                        # Risk-based Kelly sizing using live account + symbol info
                        ps_cfg = getattr(config, "position_sizing", None)
                        vol_step_sym = 0.0
                        if ps_cfg is not None:
                            equity = account_equity if account_equity is not None else getattr(ps_cfg, "account_balance", 0.0)
                            risk_pct = getattr(ps_cfg, "risk_per_trade_pct", 1.0)
                            risk_budget = max(0.0, equity * (risk_pct / 100.0))

                            tick_size = float(symbol_info.get("tick_size", 0.0) or 0.0)
                            tick_value = float(symbol_info.get("tick_value", 0.0) or 0.0)
                            vol_min_sym = float(symbol_info.get("volume_min", 0.0) or 0.0)
                            vol_max_sym = float(symbol_info.get("volume_max", 0.0) or 0.0)
                            vol_step_sym = float(symbol_info.get("volume_step", 0.0) or 0.0)

                            stop_distance = abs(result.entry_price - result.stop_loss)

                            if risk_budget > 0 and tick_size > 0 and tick_value > 0 and stop_distance > 0:
                                loss_per_lot = (stop_distance / tick_size) * tick_value
                                max_risk_lots = risk_budget / loss_per_lot if loss_per_lot > 0 else 0.0

                                ps_min = float(getattr(ps_cfg, "min_lots", 0.01))
                                ps_max = float(getattr(ps_cfg, "max_lots", 0.10))
                                min_lots = max(ps_min, vol_min_sym) if vol_min_sym > 0 else ps_min
                                max_lots = min(ps_max, vol_max_sym) if vol_max_sym > 0 else ps_max

                                b = float(getattr(ps_cfg, "risk_reward_ratio", 3.0))
                                p = float(result.meta_confidence)
                                q = 1.0 - p
                                kelly_full = (p * b - q) / b if b > 0 else 0.0
                                kelly_bet = kelly_full * float(getattr(ps_cfg, "kelly_fraction", 0.25))
                                kelly_max = float(getattr(ps_cfg, "kelly_max_fraction", 0.5))
                                kelly_bet = max(0.0, min(kelly_bet, kelly_max))

                                if kelly_bet <= 0 or max_risk_lots <= 0:
                                    position_volume = min_lots
                                else:
                                    ratio = kelly_bet / kelly_max if kelly_max > 0 else 0.0
                                    allowed_max = min(max_lots, max_risk_lots)
                                    if allowed_max < min_lots:
                                        position_volume = allowed_max
                                    else:
                                        position_volume = min_lots + ratio * (allowed_max - min_lots)

                                step = vol_step_sym if vol_step_sym > 0 else float(getattr(ps_cfg, "lot_step", 0.01))
                                if step > 0:
                                    position_volume = math.floor(position_volume / step) * step
                                position_volume = max(min_lots, min(position_volume, max_lots))

                                logger.info(
                                    "[SIZING_RISK]",
                                    equity=round(equity, 2),
                                    risk_budget=round(risk_budget, 2),
                                    stop_dist=round(stop_distance, 2),
                                    tick_size=round(tick_size, 5),
                                    tick_value=round(tick_value, 5),
                                    loss_per_lot=round(loss_per_lot, 2),
                                    max_risk_lots=round(max_risk_lots, 2),
                                    min_lots=round(min_lots, 2),
                                    max_lots=round(max_lots, 2),
                                    kelly_full=round(kelly_full, 4),
                                    kelly_bet=round(kelly_bet, 4),
                                )
                            else:
                                position_volume = calculate_kelly_position(
                                    confidence=result.meta_confidence,
                                    config=config,
                                    default_volume=args.volume,
                                    logger=logger,
                                )
                        else:
                            position_volume = calculate_kelly_position(
                                confidence=result.meta_confidence,
                                config=config,
                                default_volume=args.volume,
                                logger=logger,
                            )
                        logger.info(
                            f"[SIZING_RESULT] raw_volume={position_volume:.2f}, "
                            f"conf={result.meta_confidence:.3f}"
                        )

                        # Counter-trend volume scale (more conservative)
                        counter_scale = float(getattr(config, "counter_trend_volume_scale", 1.0))
                        if trend_alignment == "COUNTER" and counter_scale != 1.0:
                            position_volume = position_volume * max(0.0, counter_scale)
                            try:
                                step = vol_step_sym if ps_cfg is not None and vol_step_sym > 0 else float(getattr(ps_cfg, "lot_step", 0.01))
                            except Exception:
                                step = 0.0
                            if step > 0:
                                position_volume = math.floor(position_volume / step) * step
                            logger.info(
                                "[SIZING_RESULT] counter_trend_scale",
                                scale=round(counter_scale, 3),
                                scaled_volume=round(position_volume, 3),
                            )
                        
                        # Apply risk manager position size limit
                        position_volume = risk_manager.check_position_size(position_volume, context="entry")
                        logger.info(
                            f"[SIZING_RESULT] final_volume={position_volume:.2f} (after risk gate)"
                        )
                        
                        logger.info(
                            f"SIGNAL: {direction} @ {result.entry_price:.2f}, "
                            f"SL={result.stop_loss:.2f}, "
                            f"conf={result.meta_confidence:.2f}, "
                            f"volume={position_volume:.2f}, "
                            f"phase={result.market_phase}, "
                            f"st_15m={result.st_trend_15m:+d}, "
                            f"fvg={result.fvg_event:+d}, "
                            f"eb_st={result.trend_direction:+d}, "
                            f"latency={latency_ms:.1f}ms"
                        )
                        
                        if not args.dry_run:
                            # Create order
                            action = OrderAction.BUY if result.direction == 1 else OrderAction.SELL
                            order = Order(
                                action=action,
                                symbol=args.symbol,
                                volume=position_volume,
                                price=result.entry_price,
                                sl=result.stop_loss,
                                tp=0.0,  # Dynamic TP based on RR or trailing
                                deviation=50,  # 50 points slippage tolerance
                                magic=get_next_magic(),
                                comment=f"v4_{result.bar_idx}_{result.market_phase}",
                            )
                            
                            # Send order
                            try:
                                order_result = await _send_order(order, context="ENTRY")
                                logger.info(
                                    f"Order result: {order_result.status.name}, "
                                    f"ticket={order_result.ticket}, "
                                    f"filled@{order_result.price_filled:.2f}"
                                )
                                
                                # v4.0: 增强 REJECTED 诊断日志
                                if order_result.status.name == "REJECTED":
                                    error_msg = getattr(order_result, 'error_message', '') or ''
                                    error_code = getattr(order_result, 'error_code', 0)
                                    logger.warning(
                                        f"Order REJECTED: error_code={error_code}, "
                                        f"error_message='{error_msg}', "
                                        f"requested_volume={order.volume:.2f}, "
                                        f"requested_sl={order.sl:.2f}, "
                                        f"requested_price={order.price:.2f}"
                                    )
                                
                                # v4.1: Track position for Exit v2.1
                                if order_result.status.name == "FILLED" and order_result.ticket > 0:
                                    entry_price = order_result.price_filled if order_result.price_filled > 0 else result.entry_price
                                    direction = SignalType.LONG if result.direction == 1 else SignalType.SHORT
                                    positions[order_result.ticket] = PositionStateV21.from_position(
                                        ticket=order_result.ticket,
                                        symbol=args.symbol,
                                        direction=direction,
                                        entry_price=entry_price,
                                        entry_lots=position_volume,
                                        entry_time_us=int(time.time() * 1_000_000),
                                        initial_sl=result.stop_loss,
                                        initial_tp=0.0,
                                        config=exit_v21_config,
                                        trend_alignment=trend_alignment,
                                        market_phase=result.market_phase,
                                    )
                                    if order_result.status.name == "FILLED":
                                        trend_cap_history.append(trend_alignment)
                                    
                                    logger.info(
                                        f"Position tracked: ticket={order_result.ticket}, "
                                        f"direction={direction.name}, lots={position_volume:.2f}"
                                    )
                            except Exception as e:
                                logger.error(f"Order send failed: {e}")
                    
                    elif result.filtered_reason:
                        logger.debug(
                            f"Filtered: {result.filtered_reason} "
                            f"(bar={result.bar_idx}, latency={latency_ms:.1f}ms)"
                        )
                    
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error: {e}")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing tick: {e}")
                    continue
                
                finally:
                    # Runtime Snapshot (1Hz throttle)
                    now = time.time()
                    if now - last_snapshot_time >= 1.0:
                        # Calculate warmup progress
                        warmup_prog = 1.0
                        if live_state == BOOT_STATE_WARMUP:
                            bars_in_warmup = engine.bar_count - live_warmup_start_bars
                            warmup_prog = min(1.0, bars_in_warmup / max(1, live_warmup_bars))

                        positions_payload = []
                        for state in positions.values():
                            if state.direction == SignalType.LONG:
                                direction_str = "LONG"
                            elif state.direction == SignalType.SHORT:
                                direction_str = "SHORT"
                            else:
                                direction_str = None
                            current_lots = state.current_lots if state.current_lots > 0 else state.entry_lots
                            current_price = state.current_mid if state.current_mid > 0 else state.entry_price
                            positions_payload.append({
                                "direction": direction_str,
                                "volume": float(current_lots),
                                "current_lots": float(current_lots),
                                "entry_price": float(state.entry_price),
                                "current_price": float(current_price),
                                "stop_loss": float(state.current_sl or state.initial_sl or 0.0),
                                "take_profit": float(state.initial_tp or 0.0),
                                "unrealized_pnl": float(state.unrealized_pnl_usd),
                                "net_pnl": float(state.net_pnl_usd),
                                "realized_pnl": 0.0,
                                "ticket": int(state.ticket),
                                "stage": state.stage.name if hasattr(state.stage, "name") else str(state.stage),
                                "trend_alignment": state.trend_alignment,
                                "market_phase": state.market_phase,
                            })
                        
                        snapshot = RuntimeSnapshot(
                            timestamp=now,
                            symbol=args.symbol,
                            warmup_progress=warmup_prog,
                            ticks_total=tick_count,
                            open_positions=len(positions),
                            guardian_halt=guardian.is_halted,
                            exit_v21_enabled=True,
                            market_phase=result.market_phase if 'result' in locals() and result else "UNKNOWN",
                            temperature=float(result.market_temperature) if 'result' in locals() and result else 0.0,
                            entropy=float(result.market_entropy) if 'result' in locals() and result else 0.0
                        )
                        
                        # Fire and forget
                        if runtime_store is not None:
                            asyncio.create_task(runtime_store.write_snapshot(snapshot))
                        if ws_server is not None:
                            asyncio.create_task(ws_server.broadcast(snapshot))
                            asyncio.create_task(ws_server.broadcast_positions(positions_payload))
                        if data_store is not None:
                            data_store.enqueue_positions(ts=now, positions=positions_payload)
                        last_snapshot_time = now
        
        finally:
            # Cleanup
            if runtime_store is not None:
                await runtime_store.close()
            if data_store is not None:
                await data_store.close()
            if ws_server is not None:
                await ws_server.stop()

            if web_server is not None and web_task is not None:
                try:
                    web_server.should_exit = True
                    await web_task
                except Exception:
                    pass
                
            tick_socket.close()
            context.term()
            if not args.dry_run:
                await zmq_client.disconnect()
            logger.info("Server shutdown complete")
    
    if exit_v2_data is None:
        raise ValueError("研究/回放路径需要 execution.exit_v2 配置")
    
    try:
        asyncio.run(run_v4_server())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        sys.exit(0)


def backtest_v4() -> None:
    """Entry point for alphaos-v4-backtest command."""
    parser = argparse.ArgumentParser(
        description="AlphaOS v4 Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to v4 model directory",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to tick data CSV",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results_v4.json",
        help="Output file for results",
    )
    
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum ticks to process (for testing)",
    )
    
    args = parser.parse_args()
    
    import json
    from alphaos.core.logging import setup_logging, get_logger, enable_log_file
    from alphaos.v4 import InferenceConfig, InferenceEngineV4
    from alphaos.core.types import Tick
    
    # Setup logging
    log_path = enable_log_file(prefix="v4_backtest", tee_console=True)
    setup_logging(level="INFO", log_format="console")
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_path}")
    
    model_path = Path(args.model)
    
    # Find model and schema
    xgb_model_path = ""
    for candidate in ["xgb_model.json", "xgb_model.ubj", "model.json"]:
        if (model_path / candidate).exists():
            xgb_model_path = str(model_path / candidate)
            break
    
    schema_path = model_path / "schema.json"
    
    # Create inference config
    config = InferenceConfig(
        model_dir=str(model_path),
        schema_path=str(schema_path) if schema_path.exists() else "",
    )
    
    # Create engine
    engine = InferenceEngineV4(config)
    
    logger.info(f"Running backtest on {args.data}...")
    
    # Load and process ticks
    import csv
    from datetime import datetime
    
    signals = []
    tick_count = 0
    
    with open(args.data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if args.max_ticks and tick_count >= args.max_ticks:
                break
            
            # Parse tick (simplified)
            time_str = row.get("time_msc") or row.get("time") or row.get("timestamp")
            if time_str and time_str.isdigit():
                time_msc = int(time_str)
            else:
                time_msc = tick_count
            
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            volume = int(float(row.get("volume", 0) or 0))
            
            if bid <= 0 or ask <= 0:
                continue
            
            tick = Tick(
                timestamp_us=int(time_msc) * 1000,
                bid=bid,
                ask=ask,
                bid_volume=0.0,
                ask_volume=0.0,
                last=0.0,
                last_volume=0.0,
                flags=0,
            )
            
            result = engine.process_tick(tick)
            tick_count += 1
            
            if result and result.should_trade:
                signals.append({
                    "bar_idx": result.bar_idx,
                    "direction": result.direction,
                    "entry_price": result.entry_price,
                    "stop_loss": result.stop_loss,
                    "meta_confidence": result.meta_confidence,
                    "market_phase": result.market_phase,
                })
            
            if tick_count % 100000 == 0:
                logger.info(f"Processed {tick_count} ticks, {engine.bar_count} bars, {len(signals)} signals")
    
    # Save results
    results = {
        "model": str(model_path),
        "data": args.data,
        "tick_count": tick_count,
        "bar_count": engine.bar_count,
        "signal_count": len(signals),
        "signals": signals,
        "engine_state": engine.get_current_state(),
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Backtest complete. {len(signals)} signals generated.")
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    if signals:
        long_signals = sum(1 for s in signals if s["direction"] == 1)
        short_signals = len(signals) - long_signals
        avg_confidence = sum(s["meta_confidence"] for s in signals) / len(signals)
        
        print("\n" + "=" * 50)
        print("Backtest Summary")
        print("=" * 50)
        print(f"Ticks processed: {tick_count:,}")
        print(f"Bars generated:  {engine.bar_count:,}")
        print(f"Total signals:   {len(signals)}")
        print(f"  Long:          {long_signals}")
        print(f"  Short:         {short_signals}")
        print(f"Avg confidence:  {avg_confidence:.4f}")
        print("=" * 50)


def validate_v4_config() -> None:
    """最小化配置校验（严格模式）"""
    parser = argparse.ArgumentParser(
        description="v4 配置校验（严格模式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/v4/xauusd.yaml)",
    )
    args = parser.parse_args()

    from alphaos.core.logging import setup_logging, get_logger
    from alphaos.v4 import TrainingConfig, InferenceConfig

    setup_logging(level="INFO", log_format="console")
    logger = get_logger(__name__)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix not in (".yaml", ".yml"):
        raise ValueError("仅支持 YAML 配置文件（严格模式）")

    yaml_config = load_yaml_config(config_path)

    def _require_section(config_data: dict, name: str) -> dict:
        if name not in config_data or config_data[name] is None:
            raise ValueError(f"缺少配置段: {name}")
        if not isinstance(config_data[name], dict):
            raise ValueError(f"配置段类型错误: {name}")
        return config_data[name]

    def _require_keys(section: dict, keys: list[str], prefix: str) -> None:
        for key in keys:
            if key not in section:
                raise ValueError(f"缺少配置项: {prefix}.{key}")

    zeromq_data = _require_section(yaml_config, "zeromq")
    monitoring_data = _require_section(yaml_config, "monitoring")
    health_data = _require_section(yaml_config, "health")
    execution_data = _require_section(yaml_config, "execution")
    boot_data = _require_section(yaml_config, "boot")

    _require_keys(
        zeromq_data,
        [
            "tick_endpoint",
            "order_endpoint",
            "history_endpoint",
            "heartbeat_interval_ms",
            "reconnect_delay_ms",
            "recv_timeout_ms",
            "history_timeout_ms",
            "history_snd_timeout_ms",
            "order_recv_timeout_ms",
            "order_snd_timeout_ms",
            "tick_staleness_threshold_sec",
        ],
        "zeromq",
    )
    _require_keys(monitoring_data, ["model_guardian", "metrics"], "monitoring")
    _require_keys(health_data, ["check_interval_sec", "tick_staleness_threshold_sec", "warmup_ticks_estimate", "critical_components"], "health")
    _require_keys(execution_data, ["position_sizing", "exit_v21"], "execution")
    _require_keys(
        boot_data,
        [
            "replay_target_buffer",
            "replay_csv_max_ticks",
            "replay_window_sec",
            "replay_pace_tps",
            "replay_end_eps_ms",
            "replay_max_ticks",
            "replay_max_duration_sec",
            "replay_socket_timeout_ms",
            "replay_gap_threshold_ms",
            "replay_time_monotonic_tolerance_ms",
            "replay_max_consecutive_timeouts",
            "history_warmup_bars",
        ],
        "boot",
    )

    # SSOT 一致性校验
    primary_min = yaml_config.get("primary", {}).get("min_fvg_size_bps")
    ml_fvg_min = yaml_config.get("ml_features", {}).get("fvg", {}).get("min_size_bps")
    if primary_min is not None and ml_fvg_min is not None and primary_min != ml_fvg_min:
        raise ValueError("primary.min_fvg_size_bps 与 ml_features.fvg.min_size_bps 不一致")
    
    if "inference" in yaml_config and "min_confidence" in yaml_config["inference"]:
        if yaml_config["inference"]["min_confidence"] != yaml_config["confidence_gate"]["fixed_fallback_threshold"]:
            raise ValueError("inference.min_confidence 已弃用，请与 confidence_gate.fixed_fallback_threshold 保持一致")
    
    exec_cfg = yaml_config.get("execution", {})
    if "exit_v21" not in exec_cfg:
        raise ValueError("缺少 execution.exit_v21（生产入口不允许回退 v2）")
    
    exec_ps = exec_cfg.get("position_sizing", {})
    _require_keys(
        exec_ps,
        [
            "mode",
            "kelly_fraction",
            "kelly_max_fraction",
            "expected_edge_pct",
            "win_rate",
            "risk_per_trade_pct",
            "account_balance",
            "risk_reward_ratio",
            "min_lots",
            "max_lots",
            "lot_step",
            "linear_conf_max",
        ],
        "execution.position_sizing",
    )
    
    legacy_ps = yaml_config.get("position_sizing", {})
    for key, value in exec_ps.items():
        if key in legacy_ps and legacy_ps[key] != value:
            raise ValueError(f"position_sizing.{key} 与 execution.position_sizing.{key} 不一致")
    
    risk_cfg = yaml_config.get("risk", {})
    if "min_position_lots" in risk_cfg or "max_position_lots" in risk_cfg:
        raise ValueError("risk 不允许包含 min_position_lots/max_position_lots（SSOT 在 execution.position_sizing）")
    
    TrainingConfig.from_yaml_dict(yaml_config)
    InferenceConfig.from_yaml_dict(yaml_config, model_dir="", device="cpu")
    logger.info("配置校验通过")


def convert_csv_to_parquet() -> None:
    """
    将 CSV 文件转换为 Parquet 格式（性能优化）
    
    Parquet 格式比 CSV 快 10x+，推荐用于训练。
    
    用法:
        alphaos-v4-convert --input data/XAUUSD_Ticks.csv --output data/XAUUSD_Ticks.parquet
    """
    import time
    from alphaos.core.logging import get_logger, setup_logging
    
    setup_logging("console", "alphaos")
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser(
        description="将 CSV 转换为 Parquet 格式（性能优化）"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入 CSV 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出 Parquet 文件路径（默认与输入同名）"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="压缩算法（默认 snappy）"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path.with_suffix(".parquet")
    
    logger.info(f"Converting {input_path} to {output_path}...")
    start_time = time.time()
    
    try:
        # 尝试使用 polars（最快）
        import polars as pl
        
        logger.info("Using polars for conversion...")
        df = pl.read_csv(input_path, ignore_errors=True)
        
        compression = args.compression if args.compression != "none" else None
        df.write_parquet(output_path, compression=compression)
        
    except ImportError:
        # 回退到 pandas
        import pandas as pd
        
        logger.info("Using pandas for conversion...")
        df = pd.read_csv(input_path)
        
        compression = args.compression if args.compression != "none" else None
        df.to_parquet(output_path, compression=compression, index=False)
    
    elapsed = time.time() - start_time
    
    # 显示统计
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    ratio = output_size / input_size
    
    logger.info(
        "Conversion complete",
        elapsed_seconds=f"{elapsed:.1f}",
        input_size_mb=f"{input_size:.1f}",
        output_size_mb=f"{output_size:.1f}",
        compression_ratio=f"{ratio:.2f}",
    )
    
    print("")
    print("=" * 50)
    print(f"转换完成: {output_path}")
    print(f"输入大小: {input_size:.1f} MB")
    print(f"输出大小: {output_size:.1f} MB")
    print(f"压缩率:   {ratio:.2f}x")
    print(f"耗时:     {elapsed:.1f} 秒")
    print("=" * 50)
    print("")
    print("使用方式:")
    print(f"  alphaos-v4-train --data {output_path} --config ...")
    print("")


def main() -> None:
    """
    主入口点，支持子命令调用：
    
    python -m alphaos.v4.cli train --data ...
    python -m alphaos.v4.cli serve --model ...
    python -m alphaos.v4.cli backtest --model ... --data ...
    python -m alphaos.v4.cli convert --input ... --output ...
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python -m alphaos.v4.cli <command> [options]")
        print("")
        print("可用命令:")
        print("  train     训练 v4 模型（CfC + XGBoost）")
        print("  serve     启动推理服务")
        print("  backtest  回测模型")
        print("  convert   转换 CSV 为 Parquet 格式（性能优化）")
        print("")
        print("示例:")
        print("  python -m alphaos.v4.cli train --data data/XAUUSD_Ticks.csv --output models/v4/run_001")
        print("  python -m alphaos.v4.cli train --data data/XAUUSD_Ticks.csv --no-require-fvg  # 小数据集")
        print("  python -m alphaos.v4.cli serve --model models/v4/run_001 --dry-run")
        print("  python -m alphaos.v4.cli convert --input data/ticks.csv --output data/ticks.parquet")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # 移除子命令参数，让各函数解析剩余参数
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "train":
        train_v4()
    elif command == "serve":
        serve_v4()
    elif command == "backtest":
        backtest_v4()
    elif command == "validate":
        validate_v4_config()
    elif command == "convert":
        convert_csv_to_parquet()
    else:
        print(f"未知命令: {command}")
        print("可用命令: train, serve, backtest, convert")
        sys.exit(1)


if __name__ == "__main__":
    main()
