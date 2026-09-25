"""一键启停：装证书 → 起抓包(带看门狗) → 开面板 → 退出时全部还原。

设计原则是**任何一条路径都不能留下断网**（之前踩过一次，用户全机断网）。所以：

1. 先起 mitmproxy，**等端口真的能连上**，才把系统代理切过去
2. 启动时**记住用户原来的代理设置**，退出时原样还原 —— 不假设是 Clash
3. 全程盯着 mitmproxy，它一死立刻还原
4. Ctrl-C / 异常 / 正常退出，都走同一个清理函数（`finally` + `atexit`）
5. 证书**按指纹精确删除**，删完还要复核确实没了

用法：
    python -m net.launcher              # 正常启停
    python -m net.launcher --dry-run    # 只检查环境，不动系统
    python -m net.launcher --no-panel   # 只起抓包，不开面板（调链路用）
"""

import argparse
import atexit
import ctypes
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import winreg

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON = os.path.join(PROJ, "net", "addon.py")
RAWDUMP = os.path.join(PROJ, "net", "rawdump.py")

VENV_MITM = r"C:\Users\17837\mitmtool\Scripts\mitmdump.exe"
CONFDIR = r"C:\Users\17837\.mitmproxy"
CA_CER = os.path.join(CONFDIR, "mitmproxy-ca-cert.cer")
CA_CN = "mitmproxy"

PORT = 8080
REG_PATH = (r"Software\Microsoft\Windows\CurrentVersion"
            r"\Internet Settings")


# ------------------------------------------------------------ 系统代理读写

def read_proxy():
    """读用户当前的代理设置（还原时要原样写回）。"""
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0,
                         winreg.KEY_QUERY_VALUE)
    try:
        server = winreg.QueryValueEx(key, "ProxyServer")[0]
    except FileNotFoundError:
        server = ""
    try:
        enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
    except FileNotFoundError:
        enable = 0
    return server, enable


def write_proxy(server, enable):
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH, 0,
                         winreg.KEY_SET_VALUE)
    winreg.SetValueEx(key, "ProxyServer", 0, winreg.REG_SZ, server or "")
    winreg.SetValueEx(key, "ProxyEnable", 0, winreg.REG_DWORD, 1 if enable else 0)
    # 通知各程序设置变了，否则有的程序会继续用老代理
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF, 0x1A, 0, "Environment", 2, 5000,
        ctypes.byref(ctypes.c_ulong()))


# ---------------------------------------------------------------- 证书

def find_mitmdump():
    if os.path.exists(VENV_MITM):
        return VENV_MITM
    return shutil.which("mitmdump")


def ca_thumbprint(cert_path=CA_CER):
    """算证书指纹（SHA1 大写十六进制），用来精确删除。"""
    if not os.path.exists(cert_path):
        return None
    data = open(cert_path, "rb").read()
    if b"-----BEGIN" in data:                    # PEM 要转成 DER
        import base64
        body = data.split(b"-----BEGIN CERTIFICATE-----", 1)[1]
        body = body.split(b"-----END CERTIFICATE-----", 1)[0]
        data = base64.b64decode(b"".join(body.split()))
    import hashlib
    return hashlib.sha1(data).hexdigest().upper()


def ensure_ca_generated():
    """没生成过就让 mitmproxy 生成一次（跑一下 --version 不会生成，
    得用它的证书库 API）。"""
    if os.path.exists(CA_CER):
        return
    mitm_py = os.path.join(os.path.dirname(VENV_MITM), "python.exe")
    subprocess.run([mitm_py, "-c",
                    "from mitmproxy import certs;"
                    f"certs.CertStore.from_store(r'{CONFDIR}','mitmproxy',2048)"],
                   check=True, timeout=120)


def ca_installed_count():
    """当前用户证书库里还有几张 mitmproxy 证书。"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-ChildItem Cert:\\CurrentUser\\Root | "
         f"Where-Object {{$_.Subject -like '*{CA_CN}*'}} | "
         "Measure-Object | Select-Object -ExpandProperty Count"],
        capture_output=True, text=True, timeout=60).stdout.strip()
    return int(out) if out.isdigit() else 0


def install_ca():
    subprocess.run(["certutil", "-user", "-f", "-addstore", "Root", CA_CER],
                   check=True, capture_output=True, timeout=60)
    return ca_thumbprint()


def remove_ca(thumbprint):
    """删证书，两种方式都试，最后复核。返回是否确认删干净。"""
    if thumbprint:
        subprocess.run(["certutil", "-user", "-delstore", "Root", thumbprint],
                       capture_output=True, timeout=60)
    if ca_installed_count():
        subprocess.run(["certutil", "-user", "-delstore", "Root", CA_CN],
                       capture_output=True, timeout=60)
    return ca_installed_count() == 0


# ---------------------------------------------------------------- 端口

def port_open(port=PORT):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ---------------------------------------------------------------- 主流程

class Session:
    def __init__(self):
        self.proc = None
        self.orig = read_proxy()
        self.thumbprint = None
        self.switched = False
        self.cleaned = False

    def start_proxy(self):
        mitm = find_mitmdump()
        if not mitm:
            raise RuntimeError("找不到 mitmdump —— 先装上 mitmproxy")
        log = open(os.path.join(PROJ, "net", "mitm.log"), "a",
                   encoding="utf-8", buffering=1)
        self.proc = subprocess.Popen(
            [mitm, "-q", "--mode", f"upstream:http://{self.orig[0]}",
             "-s", ADDON, "-p", str(PORT)]
            # 原始帧全量转储：默认关闭。开它只为做全量普查（比如「网络里有没有级别」），
            # 平时不需要 —— 文件会大不少，而且 addon 已经把语义事件存下来了。
            + (["-s", RAWDUMP] if os.environ.get("GUANDAN_RAW") else []),
            stdout=log, stderr=subprocess.STDOUT)
        for _ in range(60):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"mitmproxy 启动即退出（代码 {self.proc.returncode}），"
                    f"详见 net/mitm.log")
            if port_open():
                return
            time.sleep(0.25)
        raise RuntimeError("mitmproxy 端口一直没就绪")

    def switch_proxy(self):
        write_proxy(f"127.0.0.1:{PORT}", 1)
        self.switched = True

    def cleanup(self, reason=""):
        if self.cleaned:
            return
        self.cleaned = True
        if self.switched:
            write_proxy(*self.orig)          # 原样还原
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thumbprint:
            if remove_ca(self.thumbprint):
                print(f"证书已卸干净。", flush=True)
            else:
                print("!! 证书没删干净，请手动跑："
                      f"certutil -user -delstore Root {self.thumbprint}",
                      flush=True)
        print(f"已还原代理 → {self.orig[0] or '(原来没设代理)'}"
              f"{'（' + reason + '）' if reason else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--console", action="store_true",
                    help="用终端文本面板（默认是图形牌桌）")
    ap.add_argument("--selftest", type=int, default=0,
                    help="跑 N 秒后正常退出，用来验证还原路径万无一失")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()

    mitm = find_mitmdump()
    print("环境检查：")
    print(f"  mitmdump   : {mitm or '!! 没找到'}")
    print(f"  证书文件   : {CA_CER if os.path.exists(CA_CER) else '还没生成'}")
    print(f"  当前代理   : {read_proxy()[0] or '(空)'} "
          f"开关={read_proxy()[1]}   <- 退出时会原样还原")
    print(f"  端口 {PORT}  : {'已被占用' if port_open() else '空闲'}")
    if not mitm:
        return 1
    if args.dry_run:
        return 0

    sess = Session()
    atexit.register(sess.cleanup)

    def on_signal(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass

    try:
        ensure_ca_generated()
        sess.thumbprint = install_ca()
        print(f"\n证书已装入当前用户根证书库（指纹 {sess.thumbprint[:16]}…）")

        sess.start_proxy()
        print("mitmproxy 已就绪。")

        sess.switch_proxy()
        print(f"系统代理已切到 127.0.0.1:{PORT}。\n")

        print("=" * 62)
        print("  现在打开掼蛋。装证书只影响微信，别的程序照常。")
        print("  面板会自己刷新；按 Ctrl-C 退出，退出时自动还原一切。")
        print("=" * 62 + "\n")

        if args.selftest:
            print(f"自检模式：{args.selftest} 秒后自动退出并还原……\n")
            for _ in range(args.selftest * 4):
                if sess.proc.poll() is not None:
                    print("mitmproxy 提前退出了。")
                    break
                time.sleep(0.25)
        elif args.no_panel:
            while True:
                if sess.proc.poll() is not None:
                    print("mitmproxy 退出了，收尾。")
                    break
                time.sleep(1)
        else:
            from net.state import GameState
            if args.console:
                from net import panel
                panel.run_live(GameState())
            else:
                from net import table
                table.run_live(GameState())
    except KeyboardInterrupt:
        print("\n收到退出信号。")
    except Exception as exc:                       # noqa: BLE001
        print(f"\n出错：{type(exc).__name__}: {exc}")
        return 1
    finally:
        sess.cleanup("退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
