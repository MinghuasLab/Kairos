# echo d | python -m eval.best_static_per_layer 
import torch
import numpy as np
import matplotlib.pyplot as plt
from kairos.quantization import *
from utils.command_parser import parser
from utils.perf_eval import timer
from utils.perf_eval import mem_monitor, empty_cache, reset_peak_memory_stats, max_memory_allocated
from utils.color_print import *

from thirdparty.SmoothQuant.w8a8_linear import W8A8Linear

from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin

from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ

from kairos.dynamic_linear_v2 import DynamicLinear
from kairos.perf_profiler_v2 import Profiler

SAVE_REUSLT    = 1
REPLAY_RESULT  = 1
PROFILING      = 0

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

# input_len = [2, 4, 8, 16, 32, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024]
# repeat_times = [512, 512, 512, 512, 512, 1, 1, 1, 1, 1]
input_len = []
repeat_times = []

import pandas as pd
path = "/data/xiezx/workspace/vidur/simulator_output/Orca_qps3_128_code/batch_metrics.csv"
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]
# 每个 iteration 的 M
M_seq = df["batch_num_tokens"].astype(int)
# 每个 batch_num_tokens 值出现多少次
counts = M_seq.value_counts().sort_index()
for m, cnt in counts.items():
    # print(f"M={m}, count={cnt}")
    input_len.append(m)
    repeat_times.append(cnt)

kernel_name_list = ['Pytorch   FP16','Qserve    W4A8','QQQ       W4A8', 'AWQ      W4A16', 'Marlin   W4A16', 'SmoothQ   W8A8', 'Dynamic Linear']

preheat_time = 100; n = 500
save_path = 'llama_best_static.pt'
# shape_list = [(4096, 14336), (14336, 4096)] # Llama-3 8B
shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B

DynamicLinear.prof = Profiler(shape_list)
prof_path = './prof_llama/'
if PROFILING: # profiling the LUT
    DynamicLinear.prof.profiling()
    DynamicLinear.prof.save(prof_path)
else:
    DynamicLinear.prof.load(prof_path)

from kairos.switch_linear import SwitchLinear
from kairos.perf_profiler_keep_all import Profiler
prof_path = './prof_llama_keep_all/'
SwitchLinear.prof = Profiler(shape_list)
SwitchLinear.prof.load(prof_path)
# COMP_TYPE_NAME = {'w4a16_gemm':0, 'w4a16_gemv':1, 'w8a8':2,\
#                 #    'w8a16_gemm':3, 'w8a16_gemv':4, \
#                 'fp16':5, 'w4a16_marlin': 6, 'w4a8_qserve':7}
idx2type = {0:3, 1:3, 2:5, 5:0, 6:4, 7:1}

# SwitchLinear.prof.profiling()
# SwitchLinear.prof.save(prof_path)

record = [[] for i in range(len(kernel_name_list))]

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(proj_name, K, N, bias):
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    qlinear_w4a8_qserve  = W4A8Linear.from_module(torchLinear, 'cuda')
    qlinear_w4a8_qqq     = W4A8Linear_QQQ.from_module(torchLinear)
    qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    qlinear_w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    qlinear_w8a8_awq     = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    
    print_flag = True
    tmp_list = []
    with timer('W4 -> W8 ', n=n, recordList=tmp_list, print_flag=print_flag):
        for _ in range(n):
            dlinear._trans_weight_to_8()
    with timer('W4 -> W16', n=n, recordList=tmp_list, print_flag=print_flag):
        for _ in range(n):
            dlinear._trans_weight_to_16()
    t_4to8 = tmp_list[0]; t_4to16 = tmp_list[1]

    linear_list = [torchLinear, qlinear_w4a8_qserve, qlinear_w4a8_qqq, qlinear_w4a16_awq, qlinear_w4a16_marlin, qlinear_w8a8_awq, dlinear]

    recordList = [[] for _ in range(len(kernel_name_list))]
    conversion_overhead = 0
    for midx in range(len(input_len)):
        m = input_len[midx]
        kairos_decision = DynamicLinear.prof.get_comp_type((N, K), m)
        if kairos_decision == 2:
            conversion_overhead += t_4to8 * repeat_times[midx]
        elif kairos_decision == 5:
            conversion_overhead += t_4to16 * repeat_times[midx]

        if print_flag:
            print('input M: ', m)
            print(f'N: {N} K: {K} Bias: {bias}')
        input_tensor = torch.randn((1, m, K), dtype=torch.float16, device = 'cuda')
        for i in range(len(linear_list)):
            torch.cuda.empty_cache()
            # Preheat Kernels
            for _ in range(preheat_time):
                out = linear_list[i](input_tensor)
            with timer(kernel_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
                for _ in range(n):
                    out = linear_list[i](input_tensor)
    for i in range(len(kernel_name_list)):
        record[i].append(recordList[i])
    return conversion_overhead

    # record[kernel_type][shape][M]

from contextlib import contextmanager
@contextmanager
def memory_detecter(desc="Module Size", recordList = None):
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        delta = torch.cuda.memory_allocated() - base
        print(f"{desc}: {delta/1024/1024:.3f} MiB")
        recordList.append(delta/1024/1024)


def detect_memory(proj_name, K, N, bias):
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    qlinear_w4a8_qserve  = W4A8Linear.from_module(torchLinear, 'cuda')
    qlinear_w4a8_qqq     = W4A8Linear_QQQ.from_module(torchLinear)
    qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    qlinear_w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    qlinear_w8a8_awq     = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    linear_list = [torchLinear, qlinear_w4a8_qserve, qlinear_w4a8_qqq, qlinear_w4a16_awq, qlinear_w4a16_marlin, qlinear_w8a8_awq, dlinear]

    module_size_list = []
    for l_idx in range(len(linear_list)):
        linear_list[l_idx]=linear_list[l_idx].cpu()
        with memory_detecter(f'Model Size: {kernel_name_list[l_idx]}', module_size_list):
            linear_list[l_idx]=linear_list[l_idx].cuda()
    return module_size_list

if __name__ == '__main__':
    # w_shape_proj = ['k_proj, v_proj', 'q_proj',
    #             'o_proj', 'gate_proj, up_proj', 'down_proj']
    w_shape_proj = ['k_proj, v_proj', 'q_proj, o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    w_shape = shape_list
    conversion_record = []
    module_size = []
    if REPLAY_RESULT != 1:
        for i in range(len(w_shape)):
            print('-'*20)
            print(w_shape_proj[i])
            s = w_shape[i]
            conversion_overhead = test(w_shape_proj[i], s[1], s[0], False)
            print('conversion overhead: ', conversion_overhead)
            conversion_record.append(conversion_overhead)
            
            print('-'*20)
            module_size.append(detect_memory(w_shape_proj[i], w_shape[i][1], w_shape[i][0], False))
            print('#' * 30)
        if SAVE_REUSLT:
            torch.save(record, save_path)
            torch.save(conversion_record, './tmp/conversion_record.pt')
            torch.save(module_size, './tmp/module_size.pt')
    else:
        record = torch.load(save_path)
        conversion_record = torch.load('./tmp/conversion_record.pt')
        module_size = torch.load('./tmp/module_size.pt')
    
    """Print Memory Usage of Linear Layer"""
    # for i in range(len(w_shape)):
    #     print('-'*20)
    #     print(w_shape_proj[i])
    #     for j in range(len(module_size[i])):
    #         print(f'Module Size: {kernel_name_list[j]}: {module_size[i][j]} MB')
    # exit(0)

    sum_lat = []; sum_mem = []
    for shape_idx in range(4):
        sum_lat_per_shape = []; sum_mem_per_shape = []
        print('='*30)
        print(w_shape_proj[shape_idx])
        static_best_latency = 1e10; static_best_type = 0
        for i in range(len(kernel_name_list)):
            sum_latency = 0
            for j in range(len(record[0][0])):
                sum_latency += record[i][shape_idx][j]*repeat_times[j]
            sum_lat_per_shape.append(sum_latency)
            if i == 6: continue
            sum_mem_per_shape.append(module_size[shape_idx][i])
            print(f'{kernel_name_list[i]}: {sum_latency:8.4f}  ({module_size[shape_idx][i]*32/1024:6.2f} GB)')
            # Record Best-Static Kernel
            if sum_latency < static_best_latency:
                static_best_latency = sum_latency
                static_best_type = i
        
        oracle_latency = 0
        oracle_target = {}
        lut_latency = 0
        lut_target = {}
        simple_policy_latency = 0
        kairos_without_conversion = 0
        for m in range(len(record[0][0])):
            lut_type = idx2type[SwitchLinear.prof.get_comp_type(w_shape[shape_idx], input_len[m])]
            lut_lat = record[lut_type][shape_idx][m]
            # print(input_len[m], ': ', lut_type, kernel_name_list[lut_type])
            if lut_type in lut_target: 
                lut_target[lut_type] += 1
            else:
                lut_target[lut_type] = 1
            lut_latency += lut_lat * repeat_times[m]

            min_lat = record[0][shape_idx][m]; min_type = 0
            for i in range(1, len(kernel_name_list)-1):
                if record[i][shape_idx][m] < min_lat:
                    min_lat = record[i][shape_idx][m]
                    min_type = i
            if min_type in oracle_target: 
                oracle_target[min_type] += 1
            else:
                oracle_target[min_type] = 1
            oracle_latency += (min_lat)*repeat_times[m]
            simple_policy_latency += min(record[3][shape_idx][m], record[5][shape_idx][m])*repeat_times[m]

        dynamic_memory = module_size[shape_idx][6] / 1.9
        oracle_memory = 0; lut_memory = 0
        for i in oracle_target:
            oracle_memory += module_size[shape_idx][i]
        for i in lut_target:
            lut_memory += module_size[shape_idx][i]
        static_best_memory = module_size[shape_idx][static_best_type]
        simple_policy_memory = module_size[shape_idx][3] + module_size[shape_idx][5]
        
        sum_mem_per_shape.append(dynamic_memory)
        print(f'Kairos        : {sum_lat_per_shape[6]:8.4f}  ({dynamic_memory*32/1024:6.2f} GB)')
        sum_lat_per_shape.append(oracle_latency)
        sum_mem_per_shape.append(oracle_memory)
        print(f'Oracle        : {oracle_latency:8.4f}  ({oracle_memory*32/1024:6.2f} GB)')
        sum_lat_per_shape.append(lut_latency)
        sum_mem_per_shape.append(lut_memory)
        print(f'LUT Selection : {lut_latency:8.4f}  ({lut_memory*32/1024:6.2f} GB)')
        sum_lat_per_shape.append(static_best_latency)
        sum_mem_per_shape.append(static_best_memory)
        print(f'Static-Best   : {static_best_latency:8.4f}  ({static_best_memory*32/1024:6.2f} GB) ({kernel_name_list[static_best_type]})')
        sum_lat_per_shape.append(simple_policy_latency)
        sum_mem_per_shape.append(simple_policy_memory)
        print(f'Simple-Policy : {simple_policy_latency:8.4f}  ({simple_policy_memory*32/1024:6.2f} GB)')
        sum_lat_per_shape.append(conversion_record[shape_idx]); sum_mem_per_shape.append(0)
        print(f'Convert Cost  : {conversion_record[shape_idx]:8.4f}')
        
        print('-'*30)
        print('Orcale Target: ')
        for i in oracle_target:
            print(f'  {kernel_name_list[i]}: {oracle_target[i]}')
        print('LUT    Target: ')
        for i in lut_target:
            print(f'  {kernel_name_list[i]}: {lut_target[i]}')
        sum_lat.append(sum_lat_per_shape)
        sum_mem.append(sum_mem_per_shape)
    
    print("#"*30)
    object_name_list =   ['Pytorch   FP16', 'Qserve    W4A8', 'QQQ       W4A8', 'AWQ      W4A16', 'Marlin   W4A16', 'SmoothQ   W8A8', \
                          'Dynamic Linear', 'Oracle        ', 'LUT Selection ', 'Static-Best   ', 'Simple-Policy ', 'Convert Cost  ']
    for i in range(len(sum_lat[0])):
        decode_layer_latency = sum_lat[0][i]*2 + sum_lat[1][i]*2 + sum_lat[2][i]*2 + sum_lat[3][i]
        decode_layer_memory = sum_mem[0][i]*2 + sum_mem[1][i]*2 + sum_mem[2][i]*2 + sum_mem[3][i]
        print(f'{object_name_list[i]}: {decode_layer_latency:10.4f} ({decode_layer_memory*32/1024:6.2f} GB)')
    exit(0)
