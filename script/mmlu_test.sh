echo FULL | python -m eval.acc_mmlup --model_type llama --full True
echo W4A8Linear | python -m eval.acc_mmlup --model_type llama
echo W4A16Linear | python -m eval.acc_mmlup --model_type llama
echo DynamicLinear | python -m eval.acc_mmlup --model_type llama

echo FULL | python -m eval.acc_mmlup --model_type qwen --full True
echo W4A8Linear | python -m eval.acc_mmlup --model_type qwen
echo W4A16Linear | python -m eval.acc_mmlup --model_type qwen
echo DynamicLinear | python -m eval.acc_mmlup --model_type qwen