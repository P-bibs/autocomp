# generalize to a for loop 
# ./scripts/run_forward_traced.sh data/l2-63/0.py data/l2-63/tests/0.json

# mkdir -p data/l2-9/tests
# for i in 2 3 11 16 17 18 25; do
#     echo "Running test ${i}..."
#     ./scripts/run_forward_traced.sh data/l2-9/${i}.py data/l2-9/tests/${i}.json ./KernelBench/KernelBench/level2/9_Matmul_Subtract_Multiply_ReLU.py
# done

# exit

mkdir -p data/l2-18/tests
for i in {0..22}; do
    echo "Running test ${i}..."
    ./scripts/run_forward_traced.sh data/l2-18/${i}.py data/l2-18/tests/${i}.json ./KernelBench/KernelBench/level2/18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py
done


mkdir -p data/l2-63/tests
for i in {0..39}; do
    echo "Running test ${i}..."
    ./scripts/run_forward_traced.sh data/l2-63/${i}.py data/l2-63/tests/${i}.json ./KernelBench/KernelBench/level2/63_Gemm_ReLU_Divide.py
done
