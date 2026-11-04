echo DynamicLinear | python -m kairos.main --model_type llama > log/end2end_batch/DynamicLinear_llama.log
echo Best-Static | python -m kairos.main --model_type llama > log/end2end_batch/Best-Static_llama.log
echo SwitchLinear | python -m kairos.main_ablation --model_type llama > log/end2end_batch/SwitchLinear_llama.log
