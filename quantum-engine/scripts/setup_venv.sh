#!/usr/bin/env bash
set -euo pipefail

# ================== 方案 C：量子引擎环境配置 ==================
# 独立虚拟环境（与旧 ai-engine 分离）
# OMP 线程亲和性配置（M2 Pro 优化）

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

# 创建虚拟环境
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# 升级 pip 并安装依赖
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "✅ quantum-engine venv ready: ${VENV_DIR}"

# ================== 方案 C：OMP 线程亲和性配置 ==================
# 生成激活脚本（包含 OMP 配置）
ACTIVATE_QUANTUM="${ROOT_DIR}/scripts/activate_quantum.sh"

cat > "${ACTIVATE_QUANTUM}" << 'EOF'
#!/usr/bin/env bash
# 方案 C 量子引擎激活脚本
# 使用方法: source quantum-engine/scripts/activate_quantum.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"

# ================== M2 Pro CPU 优化 ==================
# OMP_NUM_THREADS: 绑定到性能核心数量（M2 Pro 有 8 个 P-Cores）
export OMP_NUM_THREADS=8

# KMP_BLOCKTIME: 减少线程等待时间（低延迟推理）
export KMP_BLOCKTIME=0

# MKL_NUM_THREADS: Intel MKL 线程数（如果使用）
export MKL_NUM_THREADS=8

# OPENBLAS_NUM_THREADS: OpenBLAS 线程数
export OPENBLAS_NUM_THREADS=8

# 禁用 MPS 后端（方案 C 强制 CPU 双精度）
export PYTORCH_ENABLE_MPS_FALLBACK=0

echo "✅ 方案 C 量子引擎环境已激活"
echo "   - OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "   - KMP_BLOCKTIME=${KMP_BLOCKTIME}"
echo "   - 后端: lightning.qubit (CPU float64)"
EOF

chmod +x "${ACTIVATE_QUANTUM}"

echo ""
echo "📝 使用方法："
echo "   source ${ACTIVATE_QUANTUM}"
echo ""
echo "   或直接运行训练："
echo "   source ${VENV_DIR}/bin/activate"
echo "   export OMP_NUM_THREADS=8 KMP_BLOCKTIME=0"
echo "   python src/train_quantum_regressor.py --data ... --outdir ..."
