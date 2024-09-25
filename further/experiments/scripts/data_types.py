from dataclasses import dataclass

from dataclasses_json import dataclass_json
from joblib import Memory


@dataclass_json
@dataclass
class NetworkDims:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 20
    intermediate_size: int = 4096


job_memory = Memory(
    location="/nvmefs1/daranhe/llm-shearing/out/joblib_cache", verbose=0
)
