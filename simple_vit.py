import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from collections import OrderedDict, namedtuple
from functools import partial
from typing import Callable, Optional
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel





class Attend(nn.Module):
    def __init__(self, use_flash=True):
        super().__init__()
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    
    def flash_attn(self, q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        
        return self.flash_attn(q, k, v)

        # # Fall back to standard attention if flash attention is not available
        # scale = q.shape[-1] ** -0.5
        # sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        # attn = sim.softmax(dim=-1)
        # out = torch.matmul(attn, v)
        # return out

# Flash Attention Block to replace the standard EncoderBlock
class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_flash: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.attend = Attend(use_flash=use_flash)
        
        # QKV projections
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Initialize weights similar to original implementation
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.to_qkv.weight, -bound, bound)
        nn.init.uniform_(self.to_out.weight, -bound, bound)

    def forward(self, x: torch.Tensor):
        # Layer norm and attention
        x_norm = self.ln_1(x)
        
        # Project to q, k, v
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Apply attention
        out = self.attend(q, k, v)
        
        # Reshape and project to output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        
        # Skip connection
        x = x + out
        
        # Layer norm and MLP
        y = self.ln_2(x)
        y = self.mlp(y)
        
        # Skip connection
        return x + y

# Modified Encoder to use FlashAttentionBlock
class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_flash: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                use_flash,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))

# Position embedding function from the second implementation
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# Weight initialization function from the second implementation
def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""
    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

# Main SimpleVisionTransformer class with Flash Attention integrated
class SimpleVisionTransformer(nn.Module):
    """Vision Transformer with Flash Attention per https://arxiv.org/abs/2205.01580 and https://arxiv.org/abs/2205.14135."""

    def _learned_embeddings(self, num):
        return nn.Parameter(torch.normal(mean=0., std=math.sqrt(1 / self.hidden_dim), size=(1, num, self.hidden_dim)))

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        posemb: str = "sincos2d",
        representation_size: Optional[int] = None,
        pool_type: str = "gap",
        register: int = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_flash: bool = True,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.pool_type = pool_type
        self.norm_layer = norm_layer
        self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
        
        if self.register == 1:
            self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
        elif self.register > 1:  # Random initialization needed to break the symmetry
            self.reg = self._learned_embeddings(self.register)

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        h = w = image_size // patch_size
        seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(seq_length)
        else:
            self.pos_embedding = None

        # Use FlashEncoder instead of the standard Encoder
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            use_flash,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def _loss_fn(self, out: torch.Tensor, lam: float, target1: torch.Tensor, target2: torch.Tensor):
        logprob = F.log_softmax(out, dim=1)
        return lam * F.nll_loss(logprob, target1) + (1.0 - lam) * F.nll_loss(logprob, target2)

    def forward(self, x: torch.Tensor, lam: float = 1.0, target1: Optional[torch.Tensor] = None, target2: Optional[torch.Tensor] = None):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.register:
            n = x.shape[0]
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        x = self.encoder(x)
        if self.pool_type == 'tok':
            x = x[:, 0]
        else:
            x = x[:, self.register:]
            x = x.mean(dim=1)
        x = self.heads(x)
        
        # Calculate loss if targets are provided
        if target1 is not None and target2 is not None:
            loss = self._loss_fn(x, lam, target1, target2)
            return x, loss
        return x