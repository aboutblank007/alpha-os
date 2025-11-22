"use client";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Textarea";
import { Select } from "@/components/ui/Select";
import { Checkbox } from "@/components/ui/Checkbox";
import { RadioGroup } from "@/components/ui/RadioGroup";
import { Switch } from "@/components/ui/Switch";
import { Modal } from "@/components/ui/Modal";
import { Tabs } from "@/components/ui/Tabs";
import { Tooltip } from "@/components/ui/Tooltip";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { Skeleton } from "@/components/ui/Skeleton";
import { Divider } from "@/components/ui/Divider";
import { Card } from "@/components/Card";
import { useToast, ToastProvider } from "@/components/ui/Toast";
import { DollarSign, Wand2, Rocket, CheckCircle2, AlertTriangle, Info, Calendar } from "lucide-react";
import React from "react";

function DemoContent() {
  const [open, setOpen] = React.useState(false);
  const [radio, setRadio] = React.useState("basic");
  const [switchOn, setSwitchOn] = React.useState(true);
  const { show } = useToast();

  return (
    <main id="main" className="space-y-10">
      <section className="glass-panel rounded-2xl p-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-accent-primary">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-primary"></span>
              </span>
              新版 UI 组件库
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight-custom text-balance">
              打造现代、简约、响应式的交易界面
            </h1>
            <p className="text-lg text-slate-400 max-w-xl leading-relaxed">
              统一的组件库和设计语言，支持无障碍访问与高性能加载。
            </p>
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" leftIcon={<Calendar size={18} />}>
              本月概览
            </Button>
            <Button variant="primary" rightIcon={<Rocket size={18} />} onClick={() => setOpen(true)}>
              快速开始
            </Button>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">账户净值</p>
              <div className="mt-2 flex items-baseline gap-2">
                <h3 className="text-2xl font-bold text-white tracking-tight">$42,560</h3>
                <Badge variant="success">+3.2%</Badge>
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-white/5 text-slate-400">
              <DollarSign size={24} />
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">胜率</p>
              <div className="mt-2 flex items-baseline gap-2">
                <h3 className="text-2xl font-bold text-white tracking-tight">62%</h3>
                <Badge variant="secondary">近 20 笔</Badge>
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-white/5 text-slate-400">
              <Wand2 size={24} />
            </div>
          </div>
        </Card>
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">交易频率</p>
              <div className="mt-2 flex items-baseline gap-2">
                <h3 className="text-2xl font-bold text-white tracking-tight">12</h3>
                <Badge>今日</Badge>
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-white/5 text-slate-400">
              <Info size={24} />
            </div>
          </div>
        </Card>
      </section>

      <section className="glass-panel rounded-2xl p-8">
        <Tabs
          items={[
            {
              value: "forms",
              label: "表单",
              content: (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Input label="交易品种" placeholder="如：AAPL" description="支持快捷搜索" />
                  <Select label="方向" options={[{ value: "buy", label: "买入" }, { value: "sell", label: "卖出" }]} />
                  <Input label="价格" type="number" placeholder="0.00" />
                  <Input label="数量" type="number" placeholder="0" />
                  <Textarea label="备注" rows={4} placeholder="记录你的交易想法" />
                </div>
              ),
            },
            {
              value: "controls",
              label: "控件",
              content: (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                  <div className="space-y-4">
                    <Checkbox label="开启提醒" />
                    <Switch checked={switchOn} onChange={setSwitchOn} label="自动同步" />
                    <RadioGroup
                      name="mode"
                      value={radio}
                      onChange={setRadio}
                      options={[
                        { value: "basic", label: "基础" },
                        { value: "pro", label: "专业" },
                        { value: "ai", label: "智能" },
                      ]}
                    />
                  </div>
                  <div className="space-y-4">
                    <Tooltip content="提交后将进行风控校验">
                      <Button rightIcon={<CheckCircle2 size={18} />}>提交订单</Button>
                    </Tooltip>
                    <div className="flex items-center gap-3">
                      <Spinner />
                      <span className="text-sm text-slate-400">正在加载数据...</span>
                    </div>
                    <Skeleton className="h-16 w-full" />
                  </div>
                </div>
              ),
            },
          ]}
        />
      </section>

      <section className="glass-panel rounded-2xl p-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="text-accent-warning" />
            <span className="text-sm text-slate-300">重要提醒：高波动时段请谨慎加仓</span>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => show({ title: "已忽略", description: "将在 24 小时后再次提醒" })}>忽略</Button>
            <Button onClick={() => show({ title: "已设置", description: "已开启风控提醒" })}>开启风控提醒</Button>
          </div>
        </div>
        <Divider className="my-6" />
        <div className="flex justify-end">
          <Button variant="outline" onClick={() => setOpen(true)}>打开模态框</Button>
        </div>
      </section>

      <Modal
        open={open}
        onOpenChange={setOpen}
        title="快速入门"
        footer={
          <>
            <Button variant="secondary" onClick={() => setOpen(false)}>稍后</Button>
            <Button onClick={() => setOpen(false)} rightIcon={<Rocket size={18} />}>立即体验</Button>
          </>
        }
      >
        <p className="text-slate-300">
          组件库已集成到项目中，支持无障碍访问、响应式布局与动画过渡。
        </p>
      </Modal>
    </main>
  );
}

export default function UIPage() {
  return (
    <ToastProvider>
      <div className="space-y-10">
        <DemoContent />
      </div>
    </ToastProvider>
  );
}

