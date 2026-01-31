# AlphaOS - 当前问题清单（2026-01-31）

## 运行与连通性
- v4 服务日志不再更新：`logs/v4_serve_web.log` / `logs/v4_inference_*.log` 最后更新时间停在 00:12，疑似服务未运行或未写日志。
- 访问 `http://localhost:8000` / `http://localhost:8000/live` 曾出现 404 或拒绝连接。
- 页面刷新返回 `{ "detail": "Not Found" }`（已通过后端路由补齐，但需确认现在是否仍发生）。
- UI 仍显示 0 值（market_temperature / market_entropy / meta_conf / runtime 指标），说明前后端数据链未真正连通或源数据为 0。
- “The width(-1) and height(-1) of chart should be greater than 0” 报错：Analytics 图表容器未正确获得尺寸。
- 外网 IP（如 `192.168.1.21`）访问拒绝：需确认 `--web-host 0.0.0.0` 生效及端口是否开放。

## 策略与信号
- st_trend_15m 与 TradingView 趋势不一致：TV 显示下跌，但系统显示上涨；需校验趋势来源、时间框架、bar 类型（volume bars vs time bars）。
- 订单顺/逆势判断疑似仍不清晰，需要从 `trend_alignment_source=primary` 的计算链路验证。
- market_temperature / market_entropy 长期为 0：需确认特征计算与 runtime 状态写入是否正常。

## 交易与风控
- 订单从大幅盈利回撤到止损，exit v21 未触发：疑似净收益计算口径与阈值不匹配（tick_value_usd_per_lot、成本估计、阈值过高）。
- 当前 XAUUSD 价值设定：确认 “价格上涨 1.00 = $100/lot”；已将 `tick_value_usd_per_lot` 设置为 100，但需确认 tick_size 与平台真实一致。
- 小账户（$200）风险控制：1% 风险下单会被 min_lots 限制导致过度风险，需要结合真实止损距离核算。

## 工具链与发布
- `git commit` 被卡死并频繁出现 `.git/index.lock`；需排查外部进程（如 IDE git worker）占用。
- 原仓库过大（`.git` 约 796MB），`git push` 长时间超时；已用“干净仓库”强制推送完成。
- 本地仓库已 reset 到干净历史（commit: `6ed2e23`）。

## 待确认问题
- 是否继续使用 `Dual SuperTrend` 作为参考指标，还是保留 FVG 做主触发。
- 是否需要对 exit v21 阈值按新 tick_value 进行整体缩放。
- 监控服务（WSRuntimeServer / api server）是否由 v4 serve 强制启动已生效。

