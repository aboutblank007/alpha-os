import asyncio
import aiohttp
import sys

SERVER_IP = "49.235.153.73"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

async def test_health():
    """测试健康检查端点"""
    print(f"正在连接到 {BASE_URL}...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 健康检查通过: {data}")
                    return True
                else:
                    print(f"❌ 健康检查失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 连接错误: {e}")
            return False

async def test_bridge_status():
    """测试桥接状态"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/status", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 桥接状态: {data}")
                    return True
                else:
                    print(f"❌ 获取状态失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 状态检查错误: {e}")
            return False

async def main():
    print("🚀 开始测试 Mac <-> Ubuntu 通信")
    print("-" * 40)
    
    health_ok = await test_health()
    if not health_ok:
        print("-" * 40)
        print("⚠️  测试中止：无法连接到服务器")
        sys.exit(1)
        
    await test_bridge_status()
    
    print("-" * 40)
    print("✨ 测试完成")

if __name__ == "__main__":
    asyncio.run(main())

