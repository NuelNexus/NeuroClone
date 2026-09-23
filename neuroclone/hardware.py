"""What is this PC? CPU, RAM and GPU/VRAM detection for Windows, Linux and macOS.

No extra dependencies: ``psutil`` is used when installed, otherwise the OS is asked directly
(``nvidia-smi``, the Windows registry, ``/proc`` and ``/sys``, ``sysctl``). Every probe is
best-effort: a failed probe means "unknown", never an exception.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GPU:
    name: str
    vendor: str  # nvidia | amd | intel | apple | unknown
    vram_gb: float
    free_gb: Optional[float] = None
    driver: str = ""

    @property
    def usable(self) -> bool:
        """Worth running an LLM on (integrated GPUs with a sliver of memory are not)."""
        return self.vram_gb >= 3.5 or self.vendor == "apple"


@dataclass
class Hardware:
    os: str  # windows | linux | macos
    os_version: str = ""
    cpu: str = "unknown CPU"
    cores: int = 4  # physical
    threads: int = 4  # logical
    ram_gb: float = 0.0
    gpus: list[GPU] = field(default_factory=list)

    @property
    def gpu(self) -> Optional[GPU]:
        usable = [g for g in self.gpus if g.usable]
        return max(usable, key=lambda g: g.vram_gb) if usable else None

    def summary(self) -> str:
        gpu = self.gpu
        if gpu is not None:
            detail = f"{gpu.vram_gb:.0f} GB" + (f", driver {gpu.driver}" if gpu.driver else "")
            gpu_text = f"{gpu.name} ({detail})"
        else:
            gpu_text = "no usable GPU found"
        return (f"{self.os.capitalize()} {self.os_version}".strip() + f", {self.cpu} ({self.cores} cores / "
                f"{self.threads} threads), {self.ram_gb:.0f} GB RAM, {gpu_text}")


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


# ---------------------------------------------------------------------------- memory and CPU
def ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:  # noqa: BLE001 - psutil missing or broken
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
        except Exception:  # noqa: BLE001
            return 0.0
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out) / 2**30 if out.strip().isdigit() else 0.0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def cpu_info() -> tuple[str, int, int]:
    """(name, physical cores, logical threads)."""
    threads = os.cpu_count() or 4
    cores = 0
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 0
    except Exception:  # noqa: BLE001
        pass
    name = ""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        except OSError:
            pass
    elif sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if not cores:
            out = _run(["sysctl", "-n", "hw.physicalcpu"]).strip()
            cores = int(out) if out.isdigit() else 0
    else:
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            name = m.group(1) if m else ""
            if not cores:
                pairs = set(re.findall(r"^physical id\s*:\s*(\d+).*?^core id\s*:\s*(\d+)", text, re.M | re.S))
                cores = len(pairs)
        except OSError:
            pass
    name = re.sub(r"\s+", " ", name or platform.processor() or "unknown CPU").strip()
    name = re.sub(r"\((R|TM)\)", "", name).replace(" CPU", "").strip()
    return name, max(1, cores or threads // 2 or 1), threads


# ---------------------------------------------------------------------------- GPUs
def nvidia_gpus() -> list[GPU]:
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(candidate) if candidate.exists() else None
    if exe is None:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GPU(parts[0], "nvidia", float(parts[1]) / 1024, float(parts[2]) / 1024,
                                parts[3] if len(parts) > 3 else ""))
            except ValueError:
                continue
    return gpus


def _vendor(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "rtx" in low or "quadro" in low:
        return "nvidia"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low or "arc" in low:
        return "intel"
    return "unknown"


def windows_gpus() -> list[GPU]:
    """Display adapters from the registry; unlike WMI's AdapterRAM this reports more than 4 GB."""
    try:
        import winreg
    except ImportError:
        return []
    base = r"SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    with root:
        for i in range(64):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    size = 0
                    for value in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            raw = winreg.QueryValueEx(key, value)[0]
                            size = int.from_bytes(raw, "little") if isinstance(raw, bytes) else int(raw)
                            if size:
                                break
                        except OSError:
                            continue
            except OSError:
                continue
            gpus.append(GPU(name, _vendor(name), size / 2**30))
    return gpus


def linux_gpus() -> list[GPU]:
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        try:
            vendor_id = (dev / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        vendor = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}.get(vendor_id, "unknown")
        vram = 0.0
        try:
            vram = int((dev / "mem_info_vram_total").read_text(encoding="utf-8")) / 2**30
        except (OSError, ValueError):
            pass
        if vendor != "nvidia" and vram:  # NVIDIA cards are reported by nvidia-smi
            gpus.append(GPU(f"{vendor.upper()} GPU ({card.name})", vendor, vram))
    return gpus


def detect() -> Hardware:
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    version = platform.release() if system != "macos" else platform.mac_ver()[0]
    cpu, cores, threads = cpu_info()
    ram = ram_gb()
    gpus = nvidia_gpus()
    if system == "windows":
        have_nvidia = bool(gpus)
        for gpu in windows_gpus():
            if have_nvidia and gpu.vendor == "nvidia":
                continue  # already reported by nvidia-smi, which also knows the free memory
            gpus.append(gpu)
    elif system == "linux":
        gpus += linux_gpus()
    elif system == "macos" and platform.machine() == "arm64":
        gpus.append(GPU("Apple Silicon (unified memory)", "apple", round(ram * 0.66, 1)))
    return Hardware(system, version, cpu, cores, threads, ram, gpus)
