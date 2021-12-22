#!/bin/bash
#SBATCH --job-name=multi-corridor-train
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=pvis
#SBATCH --exclusive
#SBATCH --no-kill
#SBATCH --output="slurm-%j.out"
#SBATCH --ip-isolate yes

# Run with sbatch client_server.sh

# Source the virtual environment
source /usr/WS1/rusu1/decision_superiority/v_pybind/bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
# TODO: Just give the head node ip without the port

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" --output="slurm-%j-HEAD.out" \
  python3 -u ./server.py --ip-head $ip_head &

# Give the computer time to launch the server node before launching the clients.
sleep 180

# number of nodes other than the head node
echo "SLURM JOB NUM NODES " $SLURM_JOB_NUM_NODES
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --output="slurm-%j-$node_i.out" \
      python3 -u ./client.py --env CppCorridor --ip-head $ip_head &
    sleep 5
done

wait
