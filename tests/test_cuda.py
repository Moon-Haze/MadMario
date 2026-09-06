"""测试 PyTorch/CUDA 基础环境——确保训练依赖可用。"""

from __future__ import annotations

import pytest
import torch


def setup_module():
    """测试启动时打印当前 PyTorch 与 CUDA 环境信息。"""
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch {torch.__version__} | CUDA {'可用' if cuda_ok else '不可用'}")
    if cuda_ok:
        dev = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"(显存: {dev.total_memory >> 20} MiB)")


class TestTorchVersion:
    """测试 PyTorch 版本满足最低要求。"""

    def test_version_available(self):
        """torch.__version__ 应为有效的语义版本字符串。"""
        version = torch.__version__
        assert isinstance(version, str)
        assert len(version) > 0
        parts = version.split(".")
        assert len(parts) >= 2
        for p in parts[:2]:
            assert p.isdigit(), f"版本号主/次分段应为数字: {version}"

    def test_version_gte_1_13(self):
        """最低要求 torch ≥ 1.13（SB3 兼容性）。"""
        major, minor = (int(p) for p in torch.__version__.split(".")[:2])
        assert (major, minor) >= (1, 13), (
            f"PyTorch 版本过低: {torch.__version__}，"
            f"SB3 需要 ≥ 1.13"
        )


class TestCudaDetection:
    """测试 CUDA 检测能力（不要求设备存在）。"""

    def test_is_available_never_raises(self):
        """torch.cuda.is_available() 在任何环境下都不应抛异常。"""
        available = torch.cuda.is_available()
        assert isinstance(available, bool)

    def test_device_count_non_negative(self):
        """GPU 数量应为非负整数。"""
        count = torch.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_device_creation_cpu(self):
        """CPU 设备始终可创建。"""
        device = torch.device("cpu")
        assert str(device) == "cpu"

    def test_device_creation_cuda_never_raises(self):
        """创建 cuda device 对象不应直接抛异常（是否可用是另一回事）。"""
        device = torch.device("cuda")
        assert "cuda" in str(device)
        assert isinstance(device, torch.device)


class TestCudaProperties:
    """当 CUDA 可用时验证 GPU 属性。"""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_device_name(self):
        """GPU 应返回非空设备名。"""
        name = torch.cuda.get_device_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_capability_positive(self):
        """CUDA compute capability 应为正数。"""
        cap = torch.cuda.get_device_capability(0)
        assert cap[0] > 0, f"invalid major version: {cap}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_total_memory_positive(self):
        """GPU 显存应大于 0。"""
        mem = torch.cuda.get_device_properties(0).total_memory
        assert mem > 0, f"显存应大于 0，实际: {mem}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_device_current_is_valid(self):
        """current_device() 返回值应在有效范围内。"""
        current = torch.cuda.current_device()
        count = torch.cuda.device_count()
        assert 0 <= current < count, (
            f"current_device={current} 超出范围 [0, {count})"
        )


class TestTensorDevice:
    """测试张量在 CPU/CUDA 上的基础操作。"""

    def test_tensor_on_cpu(self):
        """CPU 张量创建和基本运算应正常。"""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        assert torch.allclose(y, torch.tensor([2.0, 4.0, 6.0]))

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_tensor_on_cuda(self):
        """CUDA 张量创建和基本运算应正常。"""
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2
        assert y.device.type == "cuda"
        assert torch.allclose(y, torch.tensor([2.0, 4.0, 6.0], device="cuda"))

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_cpu_to_cuda_transfer(self):
        """张量在 CPU 与 CUDA 间传输应正常。"""
        x_cpu = torch.randn(32, 84, 84)
        x_cuda = x_cpu.to("cuda")
        assert x_cuda.device.type == "cuda"
        # 传输回来
        x_back = x_cuda.to("cpu")
        assert x_back.device.type == "cpu"
        assert torch.allclose(x_cpu, x_back)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="当前环境无可用 CUDA 设备",
    )
    def test_cuda_synchronize(self):
        """CUDA synchronize 不应抛异常。"""
        x = torch.randn(100, 100, device="cuda")
        y = x.sum()
        # 同步确保所有 CUDA 内核完成
        torch.cuda.synchronize()
        assert y.item() is not None
