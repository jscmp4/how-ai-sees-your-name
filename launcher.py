#!/usr/bin/env python3
"""How AI Sees Your Name — 交互式控制台菜单"""

import os
import subprocess
import sys

# 确保UTF-8输出
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

CYAN = "\033[96m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def clear():
    os.system("cls" if sys.platform == "win32" else "clear")


def banner():
    print(f"""
{CYAN}{BOLD}  ╔══════════════════════════════════════════════════════╗
  ║                                                      ║
  ║{MAGENTA}     How  AI  Sees  Your  Name{CYAN}                  ║
  ║                                                      ║
  ║{RESET}       AI向量空间名字语义分析 · 控制台                {CYAN}║
  ║{DIM}       输入任意名字，看它在AI向量空间中的命运{CYAN}         ║
  ║                                                      ║
  ╠══════════════════════════════════════════════════════╣
  ║                                                      ║
  ║  {GREEN}[1]{CYAN}  启动 Web 应用          {DIM}(Streamlit UI){CYAN}          ║
  ║  {GREEN}[2]{CYAN}  预计算汉字得分         {DIM}(生成数据文件){CYAN}          ║
  ║  {GREEN}[3]{CYAN}  运行 WEAT 实验         {DIM}(CLI批处理){CYAN}             ║
  ║  {GREEN}[4]{CYAN}  运行 LLM 测试          {DIM}(需API密钥){CYAN}            ║
  ║  {GREEN}[5]{CYAN}  运行完整 Pipeline      {DIM}(全流程){CYAN}               ║
  ║                                                      ║
  ╠══════════════════════════════════════════════════════╣
  ║                                                      ║
  ║  {YELLOW}[S]{CYAN}  设置 API 密钥                                  ║
  ║  {YELLOW}[C]{CYAN}  检查环境依赖                                   ║
  ║  {YELLOW}[D]{CYAN}  下载数据                                       ║
  ║  {RED}[Q]{CYAN}  退出                                           ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝{RESET}
""")


def run(cmd: str, **kwargs):
    """运行命令，实时输出。"""
    print(f"\n{DIM}> {cmd}{RESET}\n")
    subprocess.run(cmd, shell=True, **kwargs)


def check_data():
    """检查预计算数据是否存在。"""
    return os.path.exists(os.path.join(PROJECT_DIR, "data", "char_scores.json"))


def start_web():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 启动 Web 应用 ═══{RESET}\n")

    if not check_data():
        print(f"{YELLOW}  [!] 未检测到预计算数据，先运行预计算...{RESET}\n")
        run(f"{sys.executable} scripts/precompute_char_scores.py")
        print()

    print(f"  {GREEN}浏览器将自动打开 http://localhost:8501{RESET}")
    print(f"  {DIM}按 Ctrl+C 停止服务器{RESET}\n")
    run(f"{sys.executable} -m streamlit run app.py")


def precompute():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 预计算汉字WEAT得分 ═══{RESET}\n")
    run(f"{sys.executable} scripts/precompute_char_scores.py")
    input(f"\n{DIM}按回车返回菜单...{RESET}")


def run_weat():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 运行 WEAT 实验 ═══{RESET}\n")
    print("  选择词向量模型:")
    print(f"    {GREEN}[1]{RESET} text2vec 轻量版 (推荐, 111MB)")
    print(f"    {GREEN}[2]{RESET} 腾讯完整版 (需手动下载)")
    choice = input(f"\n  选择 [1/2]: ").strip()
    model = "text2vec" if choice != "2" else "tencent"
    run(f"{sys.executable} scripts/run_weat.py --model {model}")
    input(f"\n{DIM}按回车返回菜单...{RESET}")


def run_llm():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 运行 LLM 第一印象测试 ═══{RESET}\n")

    has_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not has_key:
        print(f"  {RED}[!] 未检测到 API 密钥{RESET}")
        print(f"  请先通过 [S] 设置选项配置密钥\n")
        input(f"{DIM}按回车返回菜单...{RESET}")
        return

    print("  选择API:")
    print(f"    {GREEN}[1]{RESET} Anthropic (Claude)")
    print(f"    {GREEN}[2]{RESET} OpenAI (GPT)")
    api_choice = input("\n  选择 [1/2]: ").strip()
    api = "openai" if api_choice == "2" else "anthropic"

    repeat = input("  每个名字测试次数 (默认5): ").strip() or "5"
    topn = input("  测试前N个名字 (默认20): ").strip() or "20"

    run(f"{sys.executable} scripts/run_llm_test.py --api {api} --repeat {repeat} --top {topn}")
    input(f"\n{DIM}按回车返回菜单...{RESET}")


def run_pipeline():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 运行完整 Pipeline ═══{RESET}\n")
    print(f"  {GREEN}[1]{RESET} 含 LLM 测试 (完整版, 需API密钥)")
    print(f"  {GREEN}[2]{RESET} 不含 LLM (仅WEAT, 免费)")
    choice = input("\n  选择 [1/2]: ").strip()
    extra = "" if choice == "1" else " --skip-llm"
    run(f"{sys.executable} scripts/run_full_pipeline.py{extra}")
    input(f"\n{DIM}按回车返回菜单...{RESET}")


def settings():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ API 密钥设置 ═══{RESET}\n")
    print("  当前状态:")

    ak = os.environ.get("ANTHROPIC_API_KEY")
    ok = os.environ.get("OPENAI_API_KEY")
    print(f"    ANTHROPIC_API_KEY: {GREEN + '已设置' + RESET if ak else RED + '未设置' + RESET}")
    print(f"    OPENAI_API_KEY:    {GREEN + '已设置' + RESET if ok else RED + '未设置' + RESET}")

    print(f"\n  {GREEN}[1]{RESET} 设置 Anthropic API Key")
    print(f"  {GREEN}[2]{RESET} 设置 OpenAI API Key")
    print(f"  {GREEN}[3]{RESET} 从 .env 文件加载")
    print(f"  {DIM}[B] 返回{RESET}")

    choice = input("\n  选择: ").strip()

    if choice == "1":
        key = input("  输入 Anthropic API Key: ").strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            print(f"  {GREEN}[OK] 已设置 (本次会话有效){RESET}")
    elif choice == "2":
        key = input("  输入 OpenAI API Key: ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            print(f"  {GREEN}[OK] 已设置 (本次会话有效){RESET}")
    elif choice == "3":
        env_path = os.path.join(PROJECT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            print(f"  {GREEN}[OK] 已从 .env 加载{RESET}")
        else:
            print(f"  {RED}[!] .env 文件不存在{RESET}")
            print(f"  请复制 .env.example 为 .env 并填入密钥")

    input(f"\n{DIM}按回车返回菜单...{RESET}")


def check_env():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 环境检查 ═══{RESET}\n")

    # Python
    print(f"  --- Python ---")
    print(f"  {GREEN}{sys.version}{RESET}\n")

    # GPU
    print(f"  --- GPU ---")
    result = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                            shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {GREEN}{result.stdout.strip()}{RESET}")
    else:
        print(f"  {YELLOW}[!] 未检测到NVIDIA GPU{RESET}")
    print()

    # Dependencies
    print(f"  --- 核心依赖 ---")
    deps = [
        ("gensim", "gensim"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("torch", "torch"),
        ("text2vec", "text2vec"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]
    for name, module in deps:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            extra = ""
            if module == "torch":
                import torch
                extra = f" (CUDA: {torch.cuda.is_available()})"
            print(f"  {GREEN}[OK]{RESET} {name}: {ver}{extra}")
        except ImportError:
            print(f"  {RED}[X]{RESET}  {name}: 未安装")
    print()

    # Data files
    print(f"  --- 数据文件 ---")
    for fname in ["data/char_scores.json", "data/char_neighbors.json"]:
        path = os.path.join(PROJECT_DIR, fname)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  {GREEN}[OK]{RESET} {fname} ({size:.0f} KB)")
        else:
            print(f"  {RED}[X]{RESET}  {fname} 缺失")
    print()

    # API keys
    print(f"  --- API 密钥 ---")
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(key_name)
        if val:
            print(f"  {GREEN}[OK]{RESET} {key_name}: ...{val[-6:]}")
        else:
            print(f"  {DIM}[ ]{RESET}  {key_name}: 未设置")

    input(f"\n{DIM}按回车返回菜单...{RESET}")


def download_data():
    clear()
    print(f"\n{CYAN}{BOLD}  ═══ 数据下载 ═══{RESET}\n")
    print(f"  {GREEN}[1]{RESET} 全部下载 (text2vec + GloVe + 名字数据集)")
    print(f"  {GREEN}[2]{RESET} 仅 text2vec (轻量版腾讯词向量, 111MB)")
    print(f"  {GREEN}[3]{RESET} 仅 GloVe (英文词向量, ~860MB)")
    print(f"  {GREEN}[4]{RESET} 仅名字数据集")
    print(f"  {DIM}[B] 返回{RESET}")

    choice = input("\n  选择: ").strip()
    flag_map = {"1": "--all", "2": "--tencent", "3": "--glove", "4": "--names"}
    flag = flag_map.get(choice)
    if flag:
        run(f"{sys.executable} scripts/download_data.py {flag}")

    input(f"\n{DIM}按回车返回菜单...{RESET}")


def main():
    # Enable ANSI colors on Windows
    if sys.platform == "win32":
        os.system("")  # Enables ANSI escape codes in Windows terminal

    while True:
        clear()
        banner()
        choice = input(f"  {BOLD}请选择 [1-5/S/C/D/Q]: {RESET}").strip().lower()

        actions = {
            "1": start_web,
            "2": precompute,
            "3": run_weat,
            "4": run_llm,
            "5": run_pipeline,
            "s": settings,
            "c": check_env,
            "d": download_data,
        }

        if choice == "q":
            print(f"\n  {DIM}再见！{RESET}\n")
            break
        elif choice in actions:
            try:
                actions[choice]()
            except KeyboardInterrupt:
                print(f"\n\n  {YELLOW}已中断{RESET}")
                input(f"\n{DIM}按回车返回菜单...{RESET}")
        else:
            print(f"  {RED}无效选择{RESET}")
            import time
            time.sleep(1)


if __name__ == "__main__":
    main()
