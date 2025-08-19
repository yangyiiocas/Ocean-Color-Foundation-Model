#!/bin/bash
#SBATCH --job-name=Tst
#SBATCH -p NVgpu
#SBATCH --nodes=4                        # 申请 4 个节点
#SBATCH --ntasks-per-node=1              # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=32               # 每个任务使用 32 个 CPU 核心
#SBATCH --gres=gpu:4                     # 每个节点申请 4 个 GPU
#SBATCH --time=72:00:00                  # 最大运行时间 72 小时
#SBATCH --output=./log/Phase-%j.out      # 标准输出日志
#SBATCH --error=./log/Phase-%j.err       # 错误输出日志

# 加载必要的模块（根据集群环境配置）
module load gpu/cuda/cuda-12.2
module load container/singularity/3.5.3

# 获取分配到的节点名称
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "Allocated nodes: $nodes"

# 将节点名称转换为数组
node_array=($nodes)
num_nodes=${#node_array[@]}

# 检查是否分配了 4 个节点
if [ $num_nodes -ne 4 ]; then
    echo "Error: Expected 4 nodes, but got $num_nodes"
    exit 1
fi

# 定义主节点（第一个节点）
master_node=${node_array[0]}
echo "Master node: $master_node"

# 在每个节点上运行任务
for ((i=0; i<$num_nodes; i++)); do
    node=${node_array[$i]}
    rank=$i
    echo "Launching task on node $node (rank $rank)"

    # 使用 srun 在每个节点上运行命令
    srun --nodes=1 --ntasks=1 -w $node \
        singularity exec --nv --bind /sharedata/uyangyi211136/XDA19060104Private:/mnt ~/env.sif \
        bash -c "cd ~/oc_u/V2-05-25/ && \
                 torchrun --nnodes=4 --node_rank=$rank --nproc_per_node=4 --master_addr=$master_node --master_port=29500 main_phase1.py" &
done

# 等待所有任务完成
wait
echo "All tasks completed."
