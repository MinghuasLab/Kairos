# Kairos
Kairos is a workload-aware mpGEMM kernel switching framework for dynamic AI workloads on GPUs.
Our key idea is to automatically select the most suitable kernel for dynamic workloads.
We first build an offline-profiled, hardware-specific lookup table that maps workload characteristics to the optimal kernel.
Then, we propose an adaptive activation matching method to dynamically align activation bit-widths with the selected kernel.
Finally, we propose a memory-efficient weight conversion method that retains only 4-bit weights in HBM and generates other weight formats on demand.
Evaluations demonstrate that Kairos achieves up to 2.19$\times$ speedup in GEMM workloads, up to 1.51$\times$ faster time-to-first-token (TTFT) and 1.44$\times$ faster time-per-output-token (TPOT) than state-of-the-art mpGEMM frameworks in LLM serving.

# News
02/11/2026 ✨: "Kairos: Switching Mixed-Precision GEMM Kernels for Dynamic AI Workloads" is accepted by HPCA 2027.

![Overview of Kairos](images/overview.pdf) 
Kairos consists of three core components.

**Kernel Selection.**
To profile the performance of different mpGEMM kernels, Kairos starts with an offline automatic profiling process.
For each target model and GPU, Kairos profiles every unique weight shape across the expected range of M, measures the end‑to‑end latency of all candidate kernels including  switching overhead, and builds a per‑shape interval‑based lookup table (LUT) to record the most suitable one.
At runtime, Kairos detects the current $M$ for each linear layer and queries the LUT to get the target kernel.

**Activation Matching.**
Once the target kernel is selected, Kairos adjusts the activation quantization path accordingly. 
Since activations are dynamically quantized from the preceding operator’s FP16 output, Kairos propagates the required precision to the quantization operator to directly produce the needed format.
When the target precision matches the original FP16, the quantization step is simply skipped. 
This avoids redundant conversions of already‑quantized tensors.

**Weight Conversion.**
To support multiple kernel formats without duplicating weight storage, Kairos retains only a 4‑bit weight format in HBM for each linear layer.
When a switch occurs, Kairos converts the base format on‑the‑fly to the target format using a lightweight dequantization or permute kernel. 
The converted weights are temporarily cached, keeping the peak memory overhead within a negligible fraction of the model size.
Finally, the selected mpGEMM kernel executes with all tensors correctly formatted.
The decision and preparation overheads are consistently amortized by the subsequent kernel performance gain.

## Install
Install environment

1. Install Package for CUDA Device
    `CUDA12.6 + torch2.6`

```
conda create -n kairos python=3.10 -y
conda activate kairos
pip install --upgrade pip
# pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.6.0
pip install --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu126 torch==2.6.0
cd Kairos
pip install -e .
cd thirdparty
git clone https://github.com/IST-DASLab/marlin.git
```

2. Install CUDA 12.6
```
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit
```

3. Compile CUDA Kernels

```
sh script/compile_kernel.sh
```

## Evaluation
Speedup of Linear Layers Evaluation
```
bash script/linear_layer_speedup.sh
# echo DynamicLinear | python -m eval.eval_llm_linear --model_type llama
```

Speedup of Linear Layers Evaluation

```
bash script/end2end_latency.sh
```
<!-- 
## Performance
Kairos provides 1.57×, and 1.63× speedups on average for LLaMa-3 8B, and 1.47×, and 1.61× speedups on average for Qwen2.5 7B A100 GPU, compared with SOTA W4A16 method MARLIN and W4A8 method Qserve, respectively.
We also evaluate the performance of Kairos on Nvidia Jetson AGX Orin.
When the input dimension is set to 64K, 
Kairos provides up to 1.27×, 1.42× and 1.40× speedups for LLaMa-3 8B, 
and up to 1.23×, 1.38× and 1.54× speedups for Qwen2.5 7B, 
compared with MARLIN, AWQ and Qserve, respectively.
The overall performance of Kairos is significantly higher than other quantization methods.
This is because Kairos decouples storage and computation for dynamic workloads in LLM prefilling.
In addition, the performance of Kairos is optimal for some specific workloads.
This is because the optimal strategies are different among the linear layers of different weight shapes.
![Speedup over cuBLAS of LLM linear layers](images/speedup.png) 
![Total Latency of LLM linear](images/end2endResult.png) 
 -->
