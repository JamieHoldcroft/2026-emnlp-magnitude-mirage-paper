import os
import sys

# EXTREMELY AGGRESSIVE MONKEYPATCH
# Must happen before ANY torch/vllm imports if possible
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

def dummy_compile(fn=None, **kwargs):
    if callable(fn):
        return fn
    return lambda f: f

# Patch everywhere torch.compile might be accessed
torch.compile = dummy_compile
if hasattr(torch, '_dynamo'):
    torch._dynamo.optimize = dummy_compile

# Also patch the inductor symbols just in case
try:
    import torch._inductor.autotune_process as autotune
    class CuteDSLBenchmarkRequest: pass
    class TensorMeta: pass
    autotune.CuteDSLBenchmarkRequest = CuteDSLBenchmarkRequest
    autotune.TensorMeta = TensorMeta
    sys.modules['torch._inductor.autotune_process'] = autotune
except:
    pass

from vllm import LLM

model_path = 'AQ-MedAI/Diver-Retriever-4B'
try:
    print(f"Attempting to load model: {model_path}")
    print(f"Torch version: {torch.__version__}")
    # Verify patch
    @torch.compile
    def test_patch(): return 1
    print("Test patch verify: Success")
    
    llm = LLM(model=model_path, task="embed", gpu_memory_utilization=0.7, trust_remote_code=True)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
