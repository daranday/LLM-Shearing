from dataclasses import dataclass

from joblib import Memory


@dataclass
class NetworkDims:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 20
    intermediate_size: int = 4096


memory = Memory(location="/nvmefs1/daranhe/llm-shearing/out/joblib_cache", verbose=0)
