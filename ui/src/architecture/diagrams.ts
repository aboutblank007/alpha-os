// AlphaOS v4 Architecture SSOT Diagrams
// Generated based on AlphaOS v4 Decision Whitepaper & Engineering Standards

export const FLOWCHART_DIAGRAM = `
flowchart TD
  subgraph dataIngest[Data Ingest]
    MT5Tick[MT5_EA_TickStream] -->|"ZMQ_PUB tcp:5555"| ZMQSub[ZMQ_Sub]
    ZMQSub --> TickDecode[TickDecode_Parse]
    TickDecode -->|"alphaos_ticks_total{symbol}"| TickMetric[(PrometheusCounter)]
    TickDecode --> TickGuard[TickStalenessGuard]
    TickGuard --> VolumeBars[VolumeBarBuilder]
    VolumeBars --> Warmup[WarmupBuffer]
    Warmup -->|"alphaos_warmup_progress{symbol}"| WarmupMetric[(PrometheusGauge)]
  end

  subgraph feature[Feature Pipeline]
    VolumeBars --> Normalizer[DynamicZScore]
    Normalizer --> Thermo[TS_Thermodynamics]
    Thermo -->|"alphaos_market_temperature{symbol}"| TempMetric[(PrometheusGauge)]
    Thermo -->|"alphaos_market_entropy{symbol}"| EntropyMetric[(PrometheusGauge)]
    Thermo --> FeatureVec[FeatureVector]
  end

  subgraph primary[Primary & Event Anchor]
    FeatureVec --> PrimaryEngine[PrimaryEngine_EventTrigger]
    PrimaryEngine --> EventAnchor[EventAnchor_FVG_or_Primary]
    EventAnchor --> CandidateBar[CandidateDecisionBar]
  end

  subgraph inference[Sequence & Meta]
    CandidateBar --> CfC[CfC_or_LNN_Encoder]
    CfC --> XGB[XGBoost_Meta_Head]
    XGB --> MetaScore[Meta_Confidence_Score]
    MetaScore -->|"alphaos_model_confidence{symbol}"| ConfMetric[(PrometheusGauge)]
    MetaScore -->|"alphaos_inference_latency_seconds{symbol}"| InferLatency[(PrometheusHistogram)]
  end

  subgraph gate[Confidence Gate]
    MetaScore --> ConfidenceGate[RollingQuantile_Gate]
    ConfidenceGate --> GateDecision{ShouldTrade?}
    GateDecision -- "no" --> GateDenyLog[[log: gate_deny]]
  end

  subgraph guards[Guardian / Risk / Price]
    GateDecision -- "yes" --> GuardsEntry[Guardian_RiskGate_PriceGuard]
    GuardsEntry -->|"alphaos_risk_events_total{event_type}"| RiskEventMetric[(PrometheusCounter)]
    GuardsEntry -->|"guardian_halt"| HaltState[GuardianHaltState]
    GuardsEntry --> ExecStrategy[ExecutionStrategy]
  end

  subgraph execution[Execution]
    ExecStrategy -->|"order_request"| ZMQDealer[ZMQ_Dealer]
    ZMQDealer -->|"ZMQ_ROUTER tcp:5556"| MT5Order[MT5_EA_OrderRouter]
    MT5Order --> Broker[Broker]
    MT5Order -->|"alphaos_orders_total{symbol,action,status}"| OrderMetric[(PrometheusCounter)]
    MT5Order -->|"alphaos_order_latency_seconds{symbol}"| OrderLatency[(PrometheusHistogram)]
    MT5Order --> LogOrder[[log: Order result JSON]]
  end

  subgraph exitV21[ExitPolicy_v21]
    Broker --> PositionUpdate[PositionStateUpdate]
    PositionUpdate --> ExitPolicy[ExitPolicyV21_evaluate]
    ExitPolicy -->|"FULL_CLOSE"| ExitFull[Exit_FULL_CLOSE]
    ExitPolicy -->|"PARTIAL_CLOSE"| ExitPartial[Exit_PARTIAL_CLOSE]
    ExitPolicy -->|"MOVE_SL"| ExitMoveSL[Exit_MOVE_SL]
    ExitPolicy --> LogExit[[log: EXIT stage, net_pnl_usd, price_used]]
    ExitPolicy -->|"exit_v21_enabled"| ExitEnabled[ExitV21Enabled]
  end

  subgraph stateSSOT[RuntimeState_SSOT_DB]
    TickDecode --> Snapshot[RuntimeSnapshot]
    PositionUpdate --> Snapshot
    GuardsEntry --> Snapshot
    ExitEnabled --> Snapshot
    Snapshot --> TSDB[(TimescaleDB_runtime_state)]
    TSDB --> UIWS[WebSocket_8765]
    UIWS --> UI[ui/ArchitectureView]
  end

  HaltState -."halt".-> GuardsEntry
`;

export const SEQUENCE_DIAGRAM = `
sequenceDiagram
  participant EA as MT5_EA
  participant ZT as ZMQ_Tick
  participant SV as InferenceServer
  participant PE as PrimaryEngine
  participant AN as EventAnchor
  participant CF as CfC_or_LNN
  participant XG as XGB_Meta
  participant CG as ConfidenceGate
  participant GR as Guardian_Risk_Price
  participant EX as Execution
  participant ZE as ZMQ_Order
  participant BR as Broker
  participant EV as ExitPolicyV21
  participant DB as TimescaleDB
  participant UI as UI_WebClient

  UI->>SV: ConnectWS(8765)
  SV->>UI: welcome(runtime_schema_hash)

  EA->>ZT: PUB Tick(bid,ask,ts)
  ZT->>SV: TickMessage
  SV->>SV: TickGuard(staleness,spread)
  SV->>SV: VolumeBarUpdate
  SV->>SV: FeatureVectorBuild
  SV->>PE: EventTrigger(features)
  PE-->>SV: event?(direction,entry_sl)
  alt No Event
    SV->>SV: skip_meta_flow
  else Event Triggered
    SV->>AN: AnchorDecisionBar(fvg_or_primary)
    AN-->>SV: candidate_bar
    SV->>CF: EncodeSequence(candidate_bar)
    CF-->>SV: hidden_state
    SV->>XG: MetaScore(hidden_state,features)
    XG-->>SV: meta_confidence
    SV->>CG: RollingQuantileGate(meta_confidence)
    alt GateDenied
      CG-->>SV: should_trade=false
      SV->>SV: log(gate_deny)
    else GateApproved
      CG-->>SV: should_trade=true
      SV->>GR: Guardian/Risk/PriceGate
      alt Denied or Halted
        GR-->>SV: deny_or_halt(reason)
      else Approved
        GR-->>SV: allow(risk_budget,position_sizing_bounds)
        SV->>EX: BuildOrder(action,volume,sl,tp,magic,request_id)
        EX->>ZE: SendOrder(JSON)
        ZE->>EA: ROUTER OrderRequest
        EA->>BR: PlaceOrder
        BR-->>EA: OrderResult(status,ticket,error_code)
        EA-->>ZE: OrderResult(JSON)
        ZE-->>SV: OrderResult
        SV->>SV: metrics alphaos_orders_total/inc
        SV->>SV: metrics alphaos_order_latency_seconds/observe
      end
    end
  end

  loop EveryTick
    EA->>ZT: PUB Tick
    ZT->>SV: TickMessage
    SV->>EV: UpdatePositionState(bid,ask,net_pnl_usd,trend_alignment)
    EV-->>SV: ExitDecision(stage,action,new_sl,close_lots,threshold)
    alt FULL_CLOSE or PARTIAL_CLOSE
      SV->>EX: SendClose(close_lots)
      EX->>ZE: OrderClose(JSON)
      ZE->>EA: ROUTER CloseRequest
      EA->>BR: Close/Modify
      BR-->>EA: Result
      EA-->>ZE: Result
      ZE-->>SV: Result
      SV->>SV: emit log(EXIT,...)
    else MOVE_SL
      SV->>EX: ModifySL(new_sl)
      EX->>ZE: ModifyRequest
    else NOOP
      SV->>SV: continue
    end

    SV->>DB: WriteRuntimeSnapshot(symbol,warmup_progress,ticks_total,open_positions,guardian_halt,exit_v21_enabled)
    SV->>UI: WS push(runtime_snapshot)
  end
`;

export const METRICS_DEFINITIONS = [
    { name: 'alphaos_ticks_total', type: 'Counter', desc: 'Total ticks processed' },
    { name: 'alphaos_warmup_progress', type: 'Gauge', desc: 'Warmup progress (0-1)' },
    { name: 'alphaos_market_temperature', type: 'Gauge', desc: 'Current market temperature' },
    { name: 'alphaos_market_entropy', type: 'Gauge', desc: 'Current market entropy' },
    { name: 'alphaos_model_confidence', type: 'Gauge', desc: 'Latest model prediction confidence' },
    { name: 'alphaos_inference_latency_seconds', type: 'Histogram', desc: 'Model inference latency' },
    { name: 'alphaos_risk_events_total', type: 'Counter', desc: 'Risk events triggered' },
    { name: 'alphaos_orders_total', type: 'Counter', desc: 'Total orders sent' },
    { name: 'alphaos_order_latency_seconds', type: 'Histogram', desc: 'Order execution latency' },
    { name: 'alphaos_open_positions', type: 'Gauge', desc: 'Current open positions count' },
    { name: 'alphaos_guardian_halt', type: 'Gauge', desc: 'Model Guardian halt status (1=HALT)' },
    { name: 'alphaos_exit_v21_enabled', type: 'Gauge', desc: 'Exit Policy v2.1 status' },
    { name: 'alphaos_runtime_snapshot_write_total', type: 'Counter', desc: 'DB Snapshot write count' },
];
