import torch, os, traceback
from transformers import AutoModelForCausalLM, AutoConfig

# 1. 先只看 config，确认能否读通
abs_model_path = '/data/share/models/models/Qwen--Qwen2.5-Coder-1.5B/snapshots/master'
cfg = AutoConfig.from_pretrained(abs_model_path, local_files_only=True,
                                 trust_remote_code=True)
print("config ok:", cfg.model_type, cfg.num_hidden_layers)

# 2. 显式指定低内存加载方式 + CPU
try:
    model = AutoModelForCausalLM.from_pretrained(
            abs_model_path,
            local_files_only = True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # attn_implementation='flash_attention_2',
            # device_map  = "cpu" # Used in transformers v4
        ).eval()
    print("model loaded ok")
except Exception:
    traceback.print_exc()              # 确保异常不被吞