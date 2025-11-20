"""
model.py

提供一个基于隐式表示（implicit neural representation / coordinate network）的 MarbleWrapper，
默认回退实现为 Fourier-feature + MLP 的 CoordNet（不做任何网格/插值，直接以 (t,lat,lon,depth) 点输入训练）。

用法说明（无需修改 train.py）：
- train.py 会 import MarbleWrapper；如果本地安装了真正的 `marble` 包且你希望使用它，
  可以把 --use-marble 参数设为 'force' 并根据 Marble API 在下面的 TODO 区域替换实现。
- 默认行为（没有 marble 或 --use-marble=none）会使用 ImplicitCoordNet（Fourier features + MLP）。
- 该模型输出与之前一致：一条前向路径接受形状 [B, 4]（t_numeric, lat, lon, depth），
  返回 [B, 4]（so, thetao, uo, vo）的预测（与训练脚本的 scaler 配合使用进行反归一化）。

实现要点：
- FourierFeatures: 把坐标映射到高维正余弦特征以增强表示高频信息。
- ImplicitCoordNet: 使用小型 MLP（可调层数/宽度）作为隐式场网络。
- MarbleWrapper: 尝试导入真实 marble（若可用），否则回退到 ImplicitCoordNet。
- 设计上避免对观测点做任何空间/时间插值；所有学习和推断都在原始不规则点上进行。
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Fourier feature mapping
# -------------------------
class FourierFeatures(nn.Module):
    """
    Fourier feature mapping: x -> [sin(2π x B), cos(2π x B)]
    B: [in_dim, mapping_size]
    """
    def __init__(self, in_dim: int, mapping_size: int = 64, scale: float = 10.0, trainable: bool = False):
        super().__init__()
        # Initialize B with Gaussian scaled by scale; by default non-trainable
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer('B', B) if not trainable else setattr(self, 'B', nn.Parameter(B))
        self.trainable = trainable
        self.in_dim = in_dim
        self.mapping_size = mapping_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        # project: [N, mapping_size] = x @ B
        # then sin/cos -> [N, 2*mapping_size]
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# -------------------------
# Implicit coordinate network
# -------------------------
class ImplicitCoordNet(nn.Module):
    """
    Coordinate-based network: FourierFeatures -> MLP -> outputs.
    Input: [B, in_dim] (here in_dim = 4: t_numeric, lat, lon, depth)
    Output: [B, out_dim] (here out_dim = 4: so, thetao, uo, vo)
    """
    def __init__(self, in_dim: int = 4, out_dim: int = 4,
                 ff_dim: int = 64, hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.ff = FourierFeatures(in_dim=in_dim, mapping_size=ff_dim, scale=5.0, trainable=False)
        dim = ff_dim * 2  # sin + cos
        layers = []
        # first layer from Fourier features to hidden
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, in_dim]
        ff = self.ff(x)  # [B, 2*ff_dim]
        out = self.net(ff)
        return out

    def _init_weights(self):
        # small orthogonal initialization for stability in implicit nets
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# -------------------------
# MarbleWrapper: try real marble else fallback
# -------------------------
class MarbleWrapper(nn.Module):
    """
    Wrapper that attempts to use a 'marble' model when available.
    If not, uses the ImplicitCoordNet fallback above.

    If you have a local Marble implementation that can accept point coordinates
    (not grid), replace the TODO section below with calls to the marble API
    to build the model. Keep the forward(x) interface so train.py works unchanged.
    """
    def __init__(self, input_dim: int = 4, output_dim: int = 4, config: Optional[dict] = None):
        super().__init__()
        self.use_marble = False
        self.config = config or {}
        # try import real marble
        try:
            import marble  # noqa: F401
            # If you want to use marble's models, set up instantiation here.
            # Example (pseudocode, adapt to your marble API):
            # from marble.models import SomeImplicitModel
            # self.model = SomeImplicitModel(in_channels=input_dim, out_channels=output_dim, ...)
            # self.use_marble = True
            #
            # For safety, default to False unless you implement real marble instantiation below.
            self.use_marble = False
        except Exception:
            self.use_marble = False

        if not self.use_marble:
            # fallback implicit representation network (no interpolation)
            ff_dim = self.config.get('ff_dim', 64)
            hidden = self.config.get('hidden', 256)
            n_layers = self.config.get('n_layers', 3)
            dropout = float(self.config.get('dropout', 0.0))
            self.model = ImplicitCoordNet(in_dim=input_dim, out_dim=output_dim,
                                          ff_dim=ff_dim, hidden=hidden, n_layers=n_layers,
                                          dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim] float tensor of (t_numeric, latitude, longitude, depth)
        returns: [B, output_dim] predictions (same ordering as train script expects)
        """
        return self.model(x)


# -------------------------
# Optional utility: small wrapper to build a model by name (for train scripts)
# -------------------------
def build_model(input_dim=4, output_dim=4, kind: str = 'implicit', **kwargs):
    """
    convenience factory used by train.py previously.
    """
    if kind == 'implicit':
        return ImplicitCoordNet(in_dim=input_dim, out_dim=output_dim,
                                ff_dim=kwargs.get('ff_dim', 64),
                                hidden=kwargs.get('hidden', 256),
                                n_layers=kwargs.get('n_layers', 3),
                                dropout=kwargs.get('dropout', 0.0))
    else:
        return MarbleWrapper(input_dim=input_dim, output_dim=output_dim, config=kwargs)


# If this module is imported by train.py, train.py expects MarbleWrapper and build_model to exist.