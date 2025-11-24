'use client';

import { useEffect, useState } from 'react';

interface EnvStatus {
  client: {
    url: boolean;
    key: boolean;
    urlValue?: string;
    keyLength?: number;
  };
  server?: {
    bridge: string;
    supabase: string;
  };
}

export default function DebugPage() {
  const [envStatus, setEnvStatus] = useState<EnvStatus | null>(null);
  const [supabaseStatus, setSupabaseStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 检查客户端环境变量
    const clientEnv = {
      url: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      key: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      urlValue: process.env.NEXT_PUBLIC_SUPABASE_URL?.substring(0, 40),
      keyLength: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.length,
    };

    setEnvStatus({ client: clientEnv });

    // 获取服务端环境变量状态
    fetch('/api/test-env')
      .then(r => r.json())
      .then(data => {
        setEnvStatus(prev => ({
          ...prev!,
          server: data.recommendations,
        }));
      })
      .catch(err => console.error('Error fetching server env:', err));

    // 获取 Supabase 连接状态
    fetch('/api/debug/supabase')
      .then(r => r.json())
      .then(data => {
        setSupabaseStatus(data);
      })
      .catch(err => console.error('Error fetching supabase status:', err))
      .finally(() => setLoading(false));
  }, []);

  if (loading || !envStatus) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8 flex items-center justify-center">
        <div className="text-white text-lg">加载中...</div>
      </div>
    );
  }

  const allGood = envStatus.client.url && envStatus.client.key && supabaseStatus?.success;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-panel rounded-xl p-6">
          <h1 className="text-3xl font-bold text-white mb-2">
            🔍 环境诊断
          </h1>
          <p className="text-slate-400">
            检查环境变量配置和 Supabase 连接状态
          </p>
        </div>

        {/* Overall Status */}
        <div className={`glass-panel rounded-xl p-6 border-2 ${allGood ? 'border-green-500/50' : 'border-red-500/50'}`}>
          <div className="flex items-center gap-4">
            <div className={`text-6xl ${allGood ? '' : 'animate-pulse'}`}>
              {allGood ? '✅' : '❌'}
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">
                {allGood ? '一切正常！' : '检测到问题'}
              </h2>
              <p className="text-slate-400">
                {allGood 
                  ? '所有配置都正确，可以正常使用。' 
                  : '请按照下方提示修复问题。'
                }
              </p>
            </div>
          </div>
        </div>

        {/* Client Environment Variables */}
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
            🌐 客户端环境变量
            <span className="text-xs text-slate-500">(浏览器可见)</span>
          </h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <div className="text-white font-mono text-sm">NEXT_PUBLIC_SUPABASE_URL</div>
                {envStatus.client.url && (
                  <div className="text-xs text-slate-500 mt-1">{envStatus.client.urlValue}...</div>
                )}
              </div>
              <div className="text-2xl">
                {envStatus.client.url ? '✅' : '❌'}
              </div>
            </div>
            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
              <div>
                <div className="text-white font-mono text-sm">NEXT_PUBLIC_SUPABASE_ANON_KEY</div>
                {envStatus.client.key && (
                  <div className="text-xs text-slate-500 mt-1">长度: {envStatus.client.keyLength} 字符</div>
                )}
              </div>
              <div className="text-2xl">
                {envStatus.client.key ? '✅' : '❌'}
              </div>
            </div>
          </div>
        </div>

        {/* Server Environment Variables */}
        {envStatus.server && (
          <div className="glass-panel rounded-xl p-6">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              🖥️ 服务端环境变量
              <span className="text-xs text-slate-500">(仅服务器可见)</span>
            </h2>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <div className="text-white font-mono text-sm">MT5 Trading Bridge</div>
                <div className="text-sm">
                  {envStatus.server.bridge}
                </div>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                <div className="text-white font-mono text-sm">Supabase</div>
                <div className="text-sm">
                  {envStatus.server.supabase}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Supabase Connection Status */}
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4">
            🗄️ Supabase 连接状态
          </h2>
          {supabaseStatus ? (
            <div className="space-y-4">
              <div className={`p-4 rounded-lg ${supabaseStatus.success ? 'bg-green-500/10 border border-green-500/30' : 'bg-red-500/10 border border-red-500/30'}`}>
                <div className="flex items-center gap-3 mb-2">
                  <div className="text-2xl">{supabaseStatus.success ? '✅' : '❌'}</div>
                  <div className="text-lg font-semibold text-white">
                    {supabaseStatus.message || supabaseStatus.error}
                  </div>
                </div>
                {supabaseStatus.details && (
                  <div className="mt-3 p-3 bg-black/20 rounded text-xs font-mono text-slate-300 overflow-auto">
                    <pre>{JSON.stringify(supabaseStatus.details, null, 2)}</pre>
                  </div>
                )}
                {supabaseStatus.database && (
                  <div className="mt-3 text-sm text-slate-300">
                    <div>✓ trades 表存在</div>
                    <div>✓ 当前记录数: {supabaseStatus.database.tradesCount}</div>
                  </div>
                )}
              </div>

              {supabaseStatus.possibleCauses && (
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="text-yellow-400 font-semibold mb-2">🔍 可能的原因:</div>
                  <ul className="text-sm text-slate-300 space-y-1">
                    {supabaseStatus.possibleCauses.map((cause: string, i: number) => (
                      <li key={i}>{cause}</li>
                    ))}
                  </ul>
                </div>
              )}

              {supabaseStatus.recommendation && (
                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="text-blue-400 font-semibold mb-2">💡 建议:</div>
                  <div className="text-sm text-slate-300">{supabaseStatus.recommendation}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-slate-400">正在检查...</div>
          )}
        </div>

        {/* Fix Instructions */}
        {!allGood && (
          <div className="glass-panel rounded-xl p-6 border-2 border-yellow-500/50">
            <h2 className="text-xl font-semibold text-white mb-4">
              🔧 修复步骤
            </h2>
            <ol className="space-y-3 text-slate-300">
              <li className="flex gap-3">
                <span className="text-accent-primary font-bold">1.</span>
                <div>
                  <div className="font-semibold">确认 .env.local 文件存在</div>
                  <div className="text-sm text-slate-400 mt-1">
                    文件应该在项目根目录中，包含所有必需的环境变量
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="text-accent-primary font-bold">2.</span>
                <div>
                  <div className="font-semibold">检查环境变量名称</div>
                  <div className="text-sm text-slate-400 mt-1">
                    客户端变量必须以 <code className="bg-white/10 px-1 rounded">NEXT_PUBLIC_</code> 开头
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="text-accent-primary font-bold">3.</span>
                <div>
                  <div className="font-semibold">完全重启开发服务器</div>
                  <div className="text-sm text-slate-400 mt-1">
                    <code className="bg-white/10 px-2 py-1 rounded block mt-1">
                      # 按 Ctrl+C 停止<br/>
                      npm run dev
                    </code>
                  </div>
                </div>
              </li>
              <li className="flex gap-3">
                <span className="text-accent-primary font-bold">4.</span>
                <div>
                  <div className="font-semibold">刷新此页面</div>
                  <div className="text-sm text-slate-400 mt-1">
                    使用 Cmd+Shift+R (Mac) 或 Ctrl+Shift+R (Windows) 强制刷新
                  </div>
                </div>
              </li>
            </ol>
          </div>
        )}

        {/* Quick Links */}
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4">
            🔗 相关链接
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <a
              href="/dashboard"
              className="p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-white"
            >
              ← 返回 Dashboard
            </a>
            <a
              href="/api/test-env"
              target="_blank"
              className="p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-white"
            >
              查看 API 测试 →
            </a>
            <a
              href="https://app.supabase.com"
              target="_blank"
              className="p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-white"
            >
              Supabase 控制台 ↗
            </a>
            <a
              href="/journal"
              className="p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-white"
            >
              交易日志 →
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

