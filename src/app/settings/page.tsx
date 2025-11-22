"use client";

import { Card } from "@/components/Card";
import { User, Bell, Palette, Database, Shield, Download } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("profile");
  const [settings, setSettings] = useState({
    // 个人信息
    displayName: "交易员",
    email: "trader@alphaoS.com",
    timezone: "Asia/Shanghai",
    
    // 交易偏好
    defaultCurrency: "USD",
    riskLevel: "medium",
    showLivePrice: true,
    autoSync: true,
    
    // 通知设置
    emailNotifications: true,
    tradeAlerts: true,
    riskAlerts: true,
    dailySummary: false,
    
    // 主题设置
    theme: "dark",
    accentColor: "blue",
  });

  const tabs = [
    { id: "profile", label: "个人信息", icon: User },
    { id: "trading", label: "交易偏好", icon: Database },
    { id: "notifications", label: "通知设置", icon: Bell },
    { id: "appearance", label: "外观主题", icon: Palette },
    { id: "security", label: "安全隐私", icon: Shield },
  ];

  const handleSave = () => {
    // TODO: 实现保存到数据库
    alert("设置已保存！");
  };

  return (
    <div className="space-y-8">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">设置</h1>
        <p className="text-slate-400 mt-2">管理您的账户和应用偏好</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* 侧边栏导航 */}
        <Card className="lg:col-span-1 h-fit">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                    activeTab === tab.id
                      ? "bg-accent-primary text-white"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <Icon size={20} />
                  <span className="font-medium">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </Card>

        {/* 设置内容 */}
        <div className="lg:col-span-3 space-y-6">
          {/* 个人信息 */}
          {activeTab === "profile" && (
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">个人信息</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    显示名称
                  </label>
                  <input
                    type="text"
                    value={settings.displayName}
                    onChange={(e) =>
                      setSettings({ ...settings, displayName: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-black/20 border border-surface-border rounded-lg text-white focus:outline-none focus:border-accent-primary transition-colors"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    电子邮箱
                  </label>
                  <input
                    type="email"
                    value={settings.email}
                    onChange={(e) =>
                      setSettings({ ...settings, email: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-black/20 border border-surface-border rounded-lg text-white focus:outline-none focus:border-accent-primary transition-colors"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    时区
                  </label>
                  <select
                    value={settings.timezone}
                    onChange={(e) =>
                      setSettings({ ...settings, timezone: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-black/20 border border-surface-border rounded-lg text-white focus:outline-none focus:border-accent-primary transition-colors"
                  >
                    <option value="Asia/Shanghai">上海 (UTC+8)</option>
                    <option value="Asia/Hong_Kong">香港 (UTC+8)</option>
                    <option value="America/New_York">纽约 (UTC-5)</option>
                    <option value="Europe/London">伦敦 (UTC+0)</option>
                  </select>
                </div>
              </div>
            </Card>
          )}

          {/* 交易偏好 */}
          {activeTab === "trading" && (
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">交易偏好</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    默认货币
                  </label>
                  <select
                    value={settings.defaultCurrency}
                    onChange={(e) =>
                      setSettings({ ...settings, defaultCurrency: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-black/20 border border-surface-border rounded-lg text-white focus:outline-none focus:border-accent-primary transition-colors"
                  >
                    <option value="USD">美元 (USD)</option>
                    <option value="CNY">人民币 (CNY)</option>
                    <option value="EUR">欧元 (EUR)</option>
                    <option value="GBP">英镑 (GBP)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    风险等级
                  </label>
                  <select
                    value={settings.riskLevel}
                    onChange={(e) =>
                      setSettings({ ...settings, riskLevel: e.target.value })
                    }
                    className="w-full px-4 py-2 bg-black/20 border border-surface-border rounded-lg text-white focus:outline-none focus:border-accent-primary transition-colors"
                  >
                    <option value="low">保守型</option>
                    <option value="medium">稳健型</option>
                    <option value="high">激进型</option>
                  </select>
                </div>

                <div className="flex items-center justify-between py-3">
                  <div>
                    <div className="text-sm font-medium text-slate-300">显示实时价格</div>
                    <div className="text-xs text-slate-500 mt-1">
                      在持仓订单中显示实时浮动盈亏
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.showLivePrice}
                      onChange={(e) =>
                        setSettings({ ...settings, showLivePrice: e.target.checked })
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent-primary"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between py-3">
                  <div>
                    <div className="text-sm font-medium text-slate-300">自动同步</div>
                    <div className="text-xs text-slate-500 mt-1">
                      自动从交易平台同步订单数据
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.autoSync}
                      onChange={(e) =>
                        setSettings({ ...settings, autoSync: e.target.checked })
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent-primary"></div>
                  </label>
                </div>
              </div>
            </Card>
          )}

          {/* 通知设置 */}
          {activeTab === "notifications" && (
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">通知设置</h2>
              <div className="space-y-4">
                {[
                  {
                    key: "emailNotifications",
                    label: "邮件通知",
                    desc: "接收重要更新和提醒",
                  },
                  {
                    key: "tradeAlerts",
                    label: "交易提醒",
                    desc: "订单成交和平仓通知",
                  },
                  {
                    key: "riskAlerts",
                    label: "风险警告",
                    desc: "当触发风险阈值时通知",
                  },
                  {
                    key: "dailySummary",
                    label: "每日总结",
                    desc: "每日交易报告邮件",
                  },
                ].map((item) => (
                  <div
                    key={item.key}
                    className="flex items-center justify-between py-3 border-b border-surface-border last:border-0"
                  >
                    <div>
                      <div className="text-sm font-medium text-slate-300">
                        {item.label}
                      </div>
                      <div className="text-xs text-slate-500 mt-1">{item.desc}</div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings[item.key as keyof typeof settings] as boolean}
                        onChange={(e) =>
                          setSettings({
                            ...settings,
                            [item.key]: e.target.checked,
                          })
                        }
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent-primary"></div>
                    </label>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* 外观主题 */}
          {activeTab === "appearance" && (
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">外观主题</h2>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-3">
                    主题模式
                  </label>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: "light", label: "浅色" },
                      { value: "dark", label: "深色" },
                      { value: "auto", label: "自动" },
                    ].map((theme) => (
                      <button
                        key={theme.value}
                        onClick={() =>
                          setSettings({ ...settings, theme: theme.value })
                        }
                        className={`px-4 py-3 rounded-lg border-2 transition-all ${
                          settings.theme === theme.value
                            ? "border-accent-primary bg-accent-primary/10 text-white"
                            : "border-surface-border text-slate-400 hover:border-slate-600"
                        }`}
                      >
                        {theme.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-3">
                    强调色
                  </label>
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { value: "blue", color: "bg-blue-500" },
                      { value: "purple", color: "bg-purple-500" },
                      { value: "green", color: "bg-green-500" },
                      { value: "orange", color: "bg-orange-500" },
                    ].map((color) => (
                      <button
                        key={color.value}
                        onClick={() =>
                          setSettings({ ...settings, accentColor: color.value })
                        }
                        className={`h-12 rounded-lg ${color.color} ${
                          settings.accentColor === color.value
                            ? "ring-2 ring-white ring-offset-2 ring-offset-slate-900"
                            : ""
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* 安全隐私 */}
          {activeTab === "security" && (
            <Card>
              <h2 className="text-xl font-semibold text-white mb-6">安全与隐私</h2>
              <div className="space-y-4">
                <button className="w-full px-4 py-3 bg-surface-glass border border-surface-border rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition-all text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">修改密码</div>
                      <div className="text-sm text-slate-500 mt-1">
                        更新您的账户密码
                      </div>
                    </div>
                    <span className="text-2xl">›</span>
                  </div>
                </button>

                <button className="w-full px-4 py-3 bg-surface-glass border border-surface-border rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition-all text-left">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">导出数据</div>
                      <div className="text-sm text-slate-500 mt-1">
                        下载您的交易数据副本
                      </div>
                    </div>
                    <Download size={20} />
                  </div>
                </button>

                <div className="border-t border-surface-border pt-4 mt-6">
                  <button className="w-full px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 hover:bg-red-500/20 transition-all">
                    删除账户
                  </button>
                  <p className="text-xs text-slate-500 text-center mt-2">
                    此操作不可撤销，请谨慎操作
                  </p>
                </div>
              </div>
            </Card>
          )}

          {/* 保存按钮 */}
          <div className="flex justify-end gap-3">
            <button className="px-6 py-2 text-slate-400 hover:text-white transition-colors">
              重置
            </button>
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-xl transition-all"
            >
              保存设置
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

