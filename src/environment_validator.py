"""
环境验证和设置工具

提供验证uv安装和环境、自动依赖安装和验证、系统需求检查的功能。
"""

import os
import sys
import subprocess
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import json
import platform


@dataclass
class SystemRequirements:
    """系统需求配置"""
    min_python_version: Tuple[int, int] = (3, 9)
    min_disk_space_gb: float = 10.0
    min_memory_gb: float = 8.0
    required_cuda: bool = True  # CUDA是必需的
    required_packages: List[str] = None
    
    def __post_init__(self):
        if self.required_packages is None:
            self.required_packages = [
                "torch",
                "transformers",
                "peft", 
                "datasets",
                "tensorboard",
                "bitsandbytes",
                "accelerate",
                "safetensors",
                "rich",
                "numpy",
                "tqdm"
            ]


@dataclass
class EnvironmentStatus:
    """环境状态信息"""
    python_version: str
    platform_info: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_memory_gb: float
    disk_space_gb: float
    memory_gb: float
    uv_available: bool
    uv_version: Optional[str]
    virtual_env: Optional[str]
    missing_packages: List[str]
    all_checks_passed: bool


class EnvironmentValidator:
    """
    环境验证器
    
    负责验证系统环境是否满足微调要求，包括Python版本、CUDA、
    依赖包、磁盘空间等检查。
    """
    
    def __init__(self, requirements: Optional[SystemRequirements] = None):
        """
        初始化环境验证器
        
        Args:
            requirements: 系统需求配置，如果为None则使用默认配置
        """
        self.requirements = requirements or SystemRequirements()
        self.logger = logging.getLogger(__name__)
        
    def validate_environment(self) -> EnvironmentStatus:
        """
        验证完整的环境配置
        
        Returns:
            EnvironmentStatus: 环境状态信息
        """
        self.logger.info("开始环境验证...")
        
        # 收集系统信息
        python_version = self._get_python_version()
        platform_info = self._get_platform_info()
        
        # CUDA检查
        cuda_available, cuda_version, gpu_count, gpu_memory_gb = self._check_cuda()
        
        # 系统资源检查
        disk_space_gb = self._check_disk_space()
        memory_gb = self._check_system_memory()
        
        # 网络连接检查
        network_available = self._check_network_connectivity()
        
        # CUDA兼容性检查
        cuda_compatibility = self._check_cuda_compatibility()
        
        # uv环境检查
        uv_available, uv_version = self._check_uv()
        virtual_env = self._get_virtual_env()
        
        # 依赖包检查
        missing_packages = self._check_packages()
        
        # 综合判断
        all_checks_passed = self._evaluate_overall_status(
            python_version, cuda_available, gpu_memory_gb, 
            disk_space_gb, memory_gb, missing_packages, network_available
        )
        
        status = EnvironmentStatus(
            python_version=python_version,
            platform_info=platform_info,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            disk_space_gb=disk_space_gb,
            memory_gb=memory_gb,
            uv_available=uv_available,
            uv_version=uv_version,
            virtual_env=virtual_env,
            missing_packages=missing_packages,
            all_checks_passed=all_checks_passed
        )
        
        # 记录额外的检查结果
        if not network_available:
            self.logger.warning("网络连接不稳定，可能影响模型下载")
        
        if cuda_compatibility["recommendations"]:
            self.logger.info("CUDA兼容性建议:")
            for rec in cuda_compatibility["recommendations"]:
                self.logger.info(f"  - {rec}")
        
        self._log_environment_status(status)
        return status
    
    def _get_python_version(self) -> str:
        """获取Python版本信息"""
        version = sys.version_info
        return f"{version.major}.{version.minor}.{version.micro}"
    
    def _get_platform_info(self) -> str:
        """获取平台信息"""
        return f"{platform.system()} {platform.release()} ({platform.machine()})"
    
    def _check_cuda(self) -> Tuple[bool, Optional[str], int, float]:
        """
        检查CUDA可用性和GPU信息
        
        Returns:
            Tuple[bool, Optional[str], int, float]: (CUDA可用, CUDA版本, GPU数量, GPU内存GB)
        """
        try:
            # 首先检查NVIDIA驱动是否可用
            nvidia_available = False
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    timeout=10
                )
                nvidia_available = result.returncode == 0
                if nvidia_available:
                    self.logger.info("检测到NVIDIA GPU驱动")
            except:
                pass
            
            # 检查PyTorch CUDA支持
            try:
                import torch
                
                cuda_available = torch.cuda.is_available()
                if not cuda_available and nvidia_available:
                    self.logger.warning("检测到NVIDIA GPU但PyTorch CUDA不可用，可能需要安装CUDA版本的PyTorch")
                    # 尝试获取系统CUDA信息
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            driver_version = result.stdout.strip()
                            self.logger.info(f"NVIDIA驱动版本: {driver_version}")
                    except:
                        pass
                    
                    return False, None, 0, 0.0
                
                if not cuda_available:
                    return False, None, 0, 0.0
                
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                
                # 获取当前GPU的内存
                current_device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(current_device)
                gpu_memory_gb = gpu_properties.total_memory / (1024**3)
                
                self.logger.info(f"CUDA可用: {cuda_available}, 版本: {cuda_version}, GPU: {gpu_count}个")
                
                return cuda_available, cuda_version, gpu_count, gpu_memory_gb
                
            except ImportError:
                self.logger.warning("PyTorch未安装，无法检查CUDA")
                if nvidia_available:
                    self.logger.info("检测到NVIDIA GPU，建议安装CUDA版本的PyTorch")
                return False, None, 0, 0.0
            
        except Exception as e:
            self.logger.error(f"CUDA检查失败: {e}")
            return False, None, 0, 0.0
    
    def _check_disk_space(self) -> float:
        """
        检查磁盘空间
        
        Returns:
            float: 可用磁盘空间(GB)
        """
        try:
            # 检查当前目录的磁盘空间
            total, used, free = shutil.disk_usage(".")
            return free / (1024**3)
        except Exception as e:
            self.logger.error(f"磁盘空间检查失败: {e}")
            return 0.0
    
    def _check_system_memory(self) -> float:
        """
        检查系统内存
        
        Returns:
            float: 系统内存(GB)
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total / (1024**3)
        except ImportError:
            self.logger.warning("psutil未安装，无法检查系统内存")
            return 0.0
        except Exception as e:
            self.logger.error(f"系统内存检查失败: {e}")
            return 0.0
    
    def _check_network_connectivity(self) -> bool:
        """
        检查网络连接性
        
        Returns:
            bool: 网络是否可用
        """
        try:
            import urllib.request
            
            # 测试连接到Hugging Face
            test_urls = [
                "https://huggingface.co",
                "https://pypi.org",
                "https://github.com"
            ]
            
            for url in test_urls:
                try:
                    urllib.request.urlopen(url, timeout=10)
                    self.logger.info(f"网络连接正常: {url}")
                    return True
                except:
                    continue
            
            self.logger.warning("网络连接检查失败，可能影响模型下载")
            return False
            
        except Exception as e:
            self.logger.error(f"网络连接检查异常: {e}")
            return False
    
    def _check_cuda_compatibility(self) -> Dict[str, Any]:
        """
        检查CUDA兼容性
        
        Returns:
            Dict[str, Any]: CUDA兼容性信息
        """
        compatibility_info = {
            "cuda_available": False,
            "cuda_version": None,
            "pytorch_cuda_version": None,
            "compatible": False,
            "recommendations": []
        }
        
        try:
            import torch
            
            compatibility_info["cuda_available"] = torch.cuda.is_available()
            
            if compatibility_info["cuda_available"]:
                compatibility_info["cuda_version"] = torch.version.cuda
                
                # 检查PyTorch CUDA版本兼容性
                pytorch_version = torch.__version__
                cuda_version = torch.version.cuda
                
                compatibility_info["pytorch_cuda_version"] = cuda_version
                
                # 简单的兼容性检查
                if cuda_version:
                    major_version = int(cuda_version.split('.')[0])
                    if major_version >= 11:  # CUDA 11.x或更高版本
                        compatibility_info["compatible"] = True
                    else:
                        compatibility_info["compatible"] = False
                        compatibility_info["recommendations"].append(
                            f"建议升级CUDA到11.x或更高版本 (当前: {cuda_version})"
                        )
                
                # 检查GPU内存
                current_device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(current_device)
                gpu_memory_gb = gpu_properties.total_memory / (1024**3)
                
                if gpu_memory_gb < 4.0:
                    compatibility_info["recommendations"].append(
                        f"GPU内存较低 ({gpu_memory_gb:.1f}GB)，建议至少4GB用于微调"
                    )
                elif gpu_memory_gb < 8.0:
                    compatibility_info["recommendations"].append(
                        f"GPU内存适中 ({gpu_memory_gb:.1f}GB)，可能需要调整批次大小"
                    )
            else:
                compatibility_info["recommendations"].append(
                    "CUDA不可用，将使用CPU训练（速度较慢）"
                )
                
        except ImportError:
            compatibility_info["recommendations"].append(
                "PyTorch未安装，无法检查CUDA兼容性"
            )
        except Exception as e:
            self.logger.error(f"CUDA兼容性检查失败: {e}")
            compatibility_info["recommendations"].append(
                f"CUDA兼容性检查失败: {e}"
            )
        
        return compatibility_info
    
    def _check_uv(self) -> Tuple[bool, Optional[str]]:
        """
        检查uv安装和版本
        
        Returns:
            Tuple[bool, Optional[str]]: (uv可用, uv版本)
        """
        try:
            result = subprocess.run(
                ["uv", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return True, version
            else:
                return False, None
                
        except FileNotFoundError:
            self._provide_uv_installation_instructions()
            return False, None
        except subprocess.TimeoutExpired:
            self.logger.warning("uv命令超时")
            return False, None
        except Exception as e:
            self.logger.error(f"uv检查失败: {e}")
            return False, None
    
    def _provide_uv_installation_instructions(self):
        """提供uv安装说明"""
        self.logger.warning("uv未安装，以下是安装说明:")
        
        system = platform.system().lower()
        
        if system == "windows":
            self.logger.info("Windows安装方法:")
            self.logger.info("1. 使用PowerShell (推荐):")
            self.logger.info("   irm https://astral.sh/uv/install.ps1 | iex")
            self.logger.info("2. 使用pip:")
            self.logger.info("   pip install uv")
            self.logger.info("3. 使用Scoop:")
            self.logger.info("   scoop install uv")
            self.logger.info("4. 使用Chocolatey:")
            self.logger.info("   choco install uv")
        elif system == "darwin":  # macOS
            self.logger.info("macOS安装方法:")
            self.logger.info("1. 使用curl (推荐):")
            self.logger.info("   curl -LsSf https://astral.sh/uv/install.sh | sh")
            self.logger.info("2. 使用Homebrew:")
            self.logger.info("   brew install uv")
            self.logger.info("3. 使用pip:")
            self.logger.info("   pip install uv")
        elif system == "linux":
            self.logger.info("Linux安装方法:")
            self.logger.info("1. 使用curl (推荐):")
            self.logger.info("   curl -LsSf https://astral.sh/uv/install.sh | sh")
            self.logger.info("2. 使用pip:")
            self.logger.info("   pip install uv")
            self.logger.info("3. 从GitHub下载二进制文件:")
            self.logger.info("   https://github.com/astral-sh/uv/releases")
        else:
            self.logger.info("通用安装方法:")
            self.logger.info("1. 使用pip:")
            self.logger.info("   pip install uv")
            self.logger.info("2. 从GitHub下载:")
            self.logger.info("   https://github.com/astral-sh/uv/releases")
        
        self.logger.info("安装完成后，请重新启动终端并重新运行此程序")
        self.logger.info("更多信息请访问: https://docs.astral.sh/uv/getting-started/installation/")
    
    def _provide_cuda_installation_instructions(self):
        """提供CUDA安装说明"""
        self.logger.error("CUDA是微调Qwen3模型的必需组件，以下是安装说明:")
        
        system = platform.system().lower()
        
        self.logger.info("步骤1: 安装NVIDIA驱动程序")
        self.logger.info("  - 访问 https://www.nvidia.com/drivers/")
        self.logger.info("  - 下载并安装适合您GPU的最新驱动程序")
        
        self.logger.info("步骤2: 安装CUDA Toolkit")
        if system == "windows":
            self.logger.info("  Windows安装:")
            self.logger.info("  - 访问 https://developer.nvidia.com/cuda-downloads")
            self.logger.info("  - 下载CUDA 12.4或更高版本")
            self.logger.info("  - 运行安装程序并选择完整安装")
        elif system == "linux":
            self.logger.info("  Linux安装:")
            self.logger.info("  - Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit")
            self.logger.info("  - 或从 https://developer.nvidia.com/cuda-downloads 下载")
        elif system == "darwin":
            self.logger.info("  macOS:")
            self.logger.info("  - CUDA在macOS上不再支持")
            self.logger.info("  - 建议使用Metal Performance Shaders (MPS)")
            self.logger.info("  - 或考虑使用Linux/Windows系统")
        
        self.logger.info("步骤3: 安装CUDA版本的PyTorch")
        self.logger.info("  使用以下命令安装:")
        if system == "windows":
            self.logger.info("  uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        else:
            self.logger.info("  uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        self.logger.info("步骤4: 验证CUDA安装")
        self.logger.info("  运行以下Python代码验证:")
        self.logger.info("  import torch")
        self.logger.info("  print(torch.cuda.is_available())")
        self.logger.info("  print(torch.cuda.get_device_name(0))")
        
        self.logger.info("故障排除:")
        self.logger.info("  - 确保NVIDIA驱动程序版本支持CUDA 12.4+")
        self.logger.info("  - 重启系统后再次测试")
        self.logger.info("  - 检查环境变量 CUDA_HOME 和 PATH")
        self.logger.info("  - 如果问题持续，请访问 https://pytorch.org/get-started/locally/")
    
    def _get_virtual_env(self) -> Optional[str]:
        """获取当前虚拟环境信息"""
        # 检查常见的虚拟环境变量
        env_vars = ["VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "UV_ACTIVE"]
        
        for var in env_vars:
            if var in os.environ:
                return os.environ[var]
        
        return None
    
    def _check_packages(self) -> List[str]:
        """
        检查依赖包
        
        Returns:
            List[str]: 缺少的包列表
        """
        missing_packages = []
        
        for package in self.requirements.required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        return missing_packages
    
    def _evaluate_overall_status(self, python_version: str, cuda_available: bool,
                               gpu_memory_gb: float, disk_space_gb: float,
                               memory_gb: float, missing_packages: List[str],
                               network_available: bool = True) -> bool:
        """
        评估整体环境状态
        
        Args:
            python_version: Python版本
            cuda_available: CUDA是否可用
            gpu_memory_gb: GPU内存
            disk_space_gb: 磁盘空间
            memory_gb: 系统内存
            missing_packages: 缺少的包
            
        Returns:
            bool: 是否通过所有检查
        """
        checks_passed = True
        
        # Python版本检查
        version_parts = python_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if (major, minor) < self.requirements.min_python_version:
            self.logger.error(f"Python版本过低: {python_version} < {self.requirements.min_python_version}")
            checks_passed = False
        
        # CUDA检查 - 必需
        if self.requirements.required_cuda and not cuda_available:
            self.logger.error("CUDA不可用，但是Qwen3微调必需的")
            self._provide_cuda_installation_instructions()
            checks_passed = False
        
        # GPU内存检查
        if cuda_available and gpu_memory_gb < 4.0:  # 最低4GB GPU内存
            self.logger.warning(f"GPU内存较低: {gpu_memory_gb:.1f}GB")
        
        # 磁盘空间检查
        if disk_space_gb < self.requirements.min_disk_space_gb:
            self.logger.error(f"磁盘空间不足: {disk_space_gb:.1f}GB < {self.requirements.min_disk_space_gb}GB")
            checks_passed = False
        
        # 系统内存检查
        if memory_gb > 0 and memory_gb < self.requirements.min_memory_gb:
            self.logger.warning(f"系统内存较低: {memory_gb:.1f}GB")
        
        # 依赖包检查
        if missing_packages:
            self.logger.error(f"缺少依赖包: {missing_packages}")
            checks_passed = False
        
        return checks_passed
    
    def _log_environment_status(self, status: EnvironmentStatus):
        """记录环境状态"""
        self.logger.info("环境验证结果:")
        self.logger.info(f"  Python版本: {status.python_version}")
        self.logger.info(f"  平台信息: {status.platform_info}")
        self.logger.info(f"  CUDA可用: {status.cuda_available}")
        if status.cuda_available:
            self.logger.info(f"  CUDA版本: {status.cuda_version}")
            self.logger.info(f"  GPU数量: {status.gpu_count}")
            self.logger.info(f"  GPU内存: {status.gpu_memory_gb:.1f}GB")
        self.logger.info(f"  磁盘空间: {status.disk_space_gb:.1f}GB")
        if status.memory_gb > 0:
            self.logger.info(f"  系统内存: {status.memory_gb:.1f}GB")
        self.logger.info(f"  uv可用: {status.uv_available}")
        if status.uv_available:
            self.logger.info(f"  uv版本: {status.uv_version}")
        if status.virtual_env:
            self.logger.info(f"  虚拟环境: {status.virtual_env}")
        if status.missing_packages:
            self.logger.warning(f"  缺少包: {status.missing_packages}")
        self.logger.info(f"  整体状态: {'通过' if status.all_checks_passed else '未通过'}")


class DependencyInstaller:
    """
    依赖安装器
    
    负责自动安装缺少的依赖包。
    """
    
    def __init__(self, use_uv: bool = True):
        """
        初始化依赖安装器
        
        Args:
            use_uv: 是否优先使用uv安装
        """
        self.use_uv = use_uv
        self.logger = logging.getLogger(__name__)
    
    def install_packages(self, packages: List[str]) -> Dict[str, bool]:
        """
        安装依赖包
        
        Args:
            packages: 要安装的包列表
            
        Returns:
            Dict[str, bool]: 每个包的安装结果
        """
        results = {}
        
        for package in packages:
            self.logger.info(f"安装依赖包: {package}")
            success = self._install_single_package(package)
            results[package] = success
            
            if success:
                self.logger.info(f"成功安装: {package}")
            else:
                self.logger.error(f"安装失败: {package}")
        
        return results
    
    def _install_single_package(self, package: str) -> bool:
        """
        安装单个包
        
        Args:
            package: 包名
            
        Returns:
            bool: 是否安装成功
        """
        # 特殊包的安装命令
        special_packages = {
            "torch": self._install_torch,
            "bitsandbytes": self._install_bitsandbytes
        }
        
        if package in special_packages:
            return special_packages[package]()
        else:
            return self._install_regular_package(package)
    
    def _install_regular_package(self, package: str) -> bool:
        """安装常规包"""
        try:
            if self.use_uv and self._is_uv_available():
                # 使用uv安装
                result = subprocess.run(
                    ["uv", "add", package],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            else:
                # 使用pip安装
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
            
            if result.returncode == 0:
                return True
            else:
                self.logger.error(f"安装{package}失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"安装{package}超时")
            return False
        except Exception as e:
            self.logger.error(f"安装{package}异常: {e}")
            return False
    
    def _install_torch(self) -> bool:
        """安装PyTorch (强制CUDA版本)"""
        try:
            # 检查NVIDIA驱动是否可用
            nvidia_available = False
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    timeout=10
                )
                nvidia_available = result.returncode == 0
            except:
                pass
            
            if not nvidia_available:
                self.logger.error("未检测到NVIDIA GPU或驱动程序")
                self.logger.error("Qwen3微调需要CUDA支持，请先安装NVIDIA驱动程序")
                return False
            
            # 强制安装CUDA版本 (CUDA 12.4 for 2025)
            if self.use_uv and self._is_uv_available():
                # 使用uv安装CUDA版本
                torch_cmd = [
                    "uv", "add", 
                    "torch>=2.4.0", "torchvision>=0.19.0", "torchaudio>=2.4.0",
                    "--index-url", "https://download.pytorch.org/whl/cu124"
                ]
            else:
                # 使用pip安装CUDA版本
                torch_cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "torch>=2.4.0", "torchvision>=0.19.0", "torchaudio>=2.4.0", 
                    "--index-url", "https://download.pytorch.org/whl/cu124"
                ]
            
            self.logger.info("安装CUDA版本的PyTorch...")
            result = subprocess.run(torch_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # 验证CUDA是否可用
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.logger.info(f"PyTorch CUDA安装成功，检测到 {torch.cuda.device_count()} 个GPU")
                        return True
                    else:
                        self.logger.error("PyTorch安装成功但CUDA不可用，可能需要重启系统")
                        return False
                except ImportError:
                    self.logger.error("PyTorch安装后无法导入")
                    return False
            else:
                self.logger.error(f"安装PyTorch失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"安装PyTorch异常: {e}")
            return False
    
    def _install_bitsandbytes(self) -> bool:
        """安装bitsandbytes"""
        try:
            # bitsandbytes在Windows上可能需要特殊处理
            if platform.system() == "Windows":
                # 在Windows上尝试安装预编译版本
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "bitsandbytes", "--prefer-binary"
                ], capture_output=True, text=True, timeout=300)
            else:
                # 在Linux/Mac上正常安装
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "bitsandbytes"
                ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return True
            else:
                self.logger.error(f"安装bitsandbytes失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"安装bitsandbytes异常: {e}")
            return False
    
    def _is_uv_available(self) -> bool:
        """检查uv是否可用"""
        try:
            result = subprocess.run(
                ["uv", "--version"], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except:
            return False


class UVEnvironmentManager:
    """
    uv环境管理器
    
    负责管理uv虚拟环境和项目配置。
    """
    
    def __init__(self, project_dir: Optional[str] = None):
        """
        初始化uv环境管理器
        
        Args:
            project_dir: 项目目录，如果为None则使用当前目录
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.logger = logging.getLogger(__name__)
    
    def setup_uv_project(self) -> bool:
        """
        设置uv项目
        
        Returns:
            bool: 是否设置成功
        """
        try:
            self.logger.info("设置uv项目...")
            
            # 检查是否已经是uv项目
            if self._is_uv_project():
                self.logger.info("已经是uv项目")
                return True
            
            # 初始化uv项目
            result = subprocess.run(
                ["uv", "init", "--no-readme"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info("uv项目初始化成功")
                return True
            else:
                self.logger.error(f"uv项目初始化失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"设置uv项目失败: {e}")
            return False
    
    def create_pyproject_toml(self) -> bool:
        """
        创建pyproject.toml文件
        
        Returns:
            bool: 是否创建成功
        """
        try:
            pyproject_path = self.project_dir / "pyproject.toml"
            
            if pyproject_path.exists():
                self.logger.info("pyproject.toml已存在")
                return True
            
            # 创建基本的pyproject.toml内容
            pyproject_content = """[project]
name = "qwen3-finetuning"
version = "0.1.0"
description = "Qwen3优化微调系统"
authors = [
    {name = "User", email = "user@example.com"}
]
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "peft>=0.7.0",
    "datasets>=2.14.0",
    "tensorboard>=2.14.0",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.24.0",
    "rich>=13.0.0",
    "psutil>=5.9.0"
]
requires-python = ">=3.8"
readme = "README.md"
license = {text = "MIT"}

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0"
]
"""
            
            with open(pyproject_path, 'w', encoding='utf-8') as f:
                f.write(pyproject_content)
            
            self.logger.info(f"创建pyproject.toml: {pyproject_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建pyproject.toml失败: {e}")
            return False
    
    def sync_dependencies(self) -> bool:
        """
        同步依赖
        
        Returns:
            bool: 是否同步成功
        """
        try:
            self.logger.info("同步uv依赖...")
            
            result = subprocess.run(
                ["uv", "sync"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                self.logger.info("依赖同步成功")
                return True
            else:
                self.logger.error(f"依赖同步失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"同步依赖失败: {e}")
            return False
    
    def update_lock_file(self) -> bool:
        """
        更新锁定文件
        
        当依赖发生变化时自动更新uv.lock文件
        
        Returns:
            bool: 是否更新成功
        """
        try:
            self.logger.info("更新uv锁定文件...")
            
            # 检查pyproject.toml是否存在且比uv.lock新
            pyproject_path = self.project_dir / "pyproject.toml"
            uv_lock_path = self.project_dir / "uv.lock"
            
            if not pyproject_path.exists():
                self.logger.warning("pyproject.toml不存在，跳过锁定文件更新")
                return True
            
            needs_update = False
            
            if not uv_lock_path.exists():
                self.logger.info("uv.lock不存在，需要创建")
                needs_update = True
            else:
                # 比较文件修改时间
                pyproject_mtime = pyproject_path.stat().st_mtime
                uv_lock_mtime = uv_lock_path.stat().st_mtime
                
                if pyproject_mtime > uv_lock_mtime:
                    self.logger.info("pyproject.toml比uv.lock新，需要更新锁定文件")
                    needs_update = True
            
            if needs_update:
                result = subprocess.run(
                    ["uv", "lock"],
                    cwd=self.project_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    self.logger.info("锁定文件更新成功")
                    return True
                else:
                    self.logger.error(f"锁定文件更新失败: {result.stderr}")
                    return False
            else:
                self.logger.info("锁定文件已是最新，无需更新")
                return True
                
        except Exception as e:
            self.logger.error(f"更新锁定文件失败: {e}")
            return False
    
    def _is_uv_project(self) -> bool:
        """检查是否是uv项目"""
        pyproject_path = self.project_dir / "pyproject.toml"
        uv_lock_path = self.project_dir / "uv.lock"
        
        return pyproject_path.exists() or uv_lock_path.exists()
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        获取项目信息
        
        Returns:
            Dict[str, Any]: 项目信息
        """
        info = {
            "project_dir": str(self.project_dir),
            "is_uv_project": self._is_uv_project(),
            "has_pyproject": (self.project_dir / "pyproject.toml").exists(),
            "has_uv_lock": (self.project_dir / "uv.lock").exists(),
            "virtual_env": os.environ.get("VIRTUAL_ENV")
        }
        
        return info


class EnvironmentSetupManager:
    """
    环境设置管理器
    
    集成环境验证、依赖安装和uv项目设置的完整流程
    """
    
    def __init__(self, auto_install: bool = False, auto_setup_uv: bool = True):
        """
        初始化环境设置管理器
        
        Args:
            auto_install: 是否自动安装缺少的依赖
            auto_setup_uv: 是否自动设置uv项目
        """
        self.auto_install = auto_install
        self.auto_setup_uv = auto_setup_uv
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.validator = EnvironmentValidator()
        self.installer = DependencyInstaller()
        self.uv_manager = UVEnvironmentManager()
    
    def setup_complete_environment(self) -> bool:
        """
        设置完整的环境
        
        Returns:
            bool: 是否设置成功
        """
        self.logger.info("开始完整环境设置...")
        
        try:
            # 1. 环境验证
            self.logger.info("步骤1: 环境验证")
            env_status = self.validator.validate_environment()
            
            # 2. uv项目设置
            if self.auto_setup_uv and env_status.uv_available:
                self.logger.info("步骤2: uv项目设置")
                if not self._setup_uv_project():
                    self.logger.warning("uv项目设置失败，继续其他步骤")
            
            # 3. 依赖安装
            if self.auto_install and env_status.missing_packages:
                self.logger.info("步骤3: 自动安装依赖")
                if not self._install_missing_dependencies(env_status.missing_packages):
                    self.logger.error("依赖安装失败")
                    return False
            
            # 4. 最终验证
            self.logger.info("步骤4: 最终验证")
            final_status = self.validator.validate_environment()
            
            if final_status.all_checks_passed:
                self.logger.info("环境设置完成，所有检查通过")
                return True
            else:
                self.logger.warning("环境设置完成，但部分检查未通过")
                self._provide_setup_recommendations(final_status)
                return False
                
        except Exception as e:
            self.logger.error(f"环境设置失败: {e}")
            return False
    
    def _setup_uv_project(self) -> bool:
        """设置uv项目"""
        try:
            # 设置uv项目
            if not self.uv_manager.setup_uv_project():
                return False
            
            # 创建pyproject.toml
            if not self.uv_manager.create_pyproject_toml():
                return False
            
            # 更新锁定文件
            if not self.uv_manager.update_lock_file():
                self.logger.warning("锁定文件更新失败，但项目设置继续")
            
            # 同步依赖
            if not self.uv_manager.sync_dependencies():
                self.logger.warning("依赖同步失败，但项目设置继续")
            
            return True
            
        except Exception as e:
            self.logger.error(f"uv项目设置失败: {e}")
            return False
    
    def _install_missing_dependencies(self, missing_packages: List[str]) -> bool:
        """安装缺少的依赖"""
        try:
            self.logger.info(f"安装缺少的依赖: {missing_packages}")
            
            install_results = self.installer.install_packages(missing_packages)
            
            # 检查安装结果
            failed_packages = [pkg for pkg, success in install_results.items() if not success]
            
            if failed_packages:
                self.logger.error(f"以下包安装失败: {failed_packages}")
                return False
            
            self.logger.info("所有依赖安装成功")
            return True
            
        except Exception as e:
            self.logger.error(f"依赖安装异常: {e}")
            return False
    
    def _provide_setup_recommendations(self, status: EnvironmentStatus):
        """提供设置建议"""
        self.logger.info("环境设置建议:")
        
        if not status.uv_available:
            self.logger.info("- 安装uv以获得更好的依赖管理体验")
        
        if status.missing_packages:
            self.logger.info(f"- 手动安装缺少的包: {status.missing_packages}")
        
        if not status.cuda_available:
            self.logger.error("- CUDA是必需的！请按照以下步骤安装:")
            self.logger.error("  1. 安装NVIDIA驱动程序")
            self.logger.error("  2. 安装CUDA Toolkit 12.4+")
            self.logger.error("  3. 重新安装CUDA版本的PyTorch")
            self.logger.error("  4. 重启系统并重新运行")
        
        if status.gpu_memory_gb < 8.0:
            self.logger.info("- 考虑使用更大内存的GPU或调整训练参数")
        
        if status.disk_space_gb < 20.0:
            self.logger.info("- 清理磁盘空间，建议至少保留20GB用于模型和数据")


def create_environment_report(output_path: Optional[str] = None) -> str:
    """
    创建环境报告
    
    Args:
        output_path: 输出路径，如果为None则使用默认路径
        
    Returns:
        str: 报告文件路径
    """
    if output_path is None:
        output_path = "environment_report.json"
    
    # 验证环境
    validator = EnvironmentValidator()
    status = validator.validate_environment()
    
    # uv项目信息
    uv_manager = UVEnvironmentManager()
    project_info = uv_manager.get_project_info()
    
    # 生成报告
    report = {
        "timestamp": os.path.basename(__file__),
        "environment_status": {
            "python_version": status.python_version,
            "platform_info": status.platform_info,
            "cuda_available": status.cuda_available,
            "cuda_version": status.cuda_version,
            "gpu_count": status.gpu_count,
            "gpu_memory_gb": status.gpu_memory_gb,
            "disk_space_gb": status.disk_space_gb,
            "memory_gb": status.memory_gb,
            "uv_available": status.uv_available,
            "uv_version": status.uv_version,
            "virtual_env": status.virtual_env,
            "missing_packages": status.missing_packages,
            "all_checks_passed": status.all_checks_passed
        },
        "project_info": project_info
    }
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return output_path


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建环境报告
    report_path = create_environment_report()
    print(f"环境报告已生成: {report_path}")