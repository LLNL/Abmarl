#!/bin/bash
#SBATCH --job-name=abmarl-scale-training
#SBATCH --nodes=<nodes>
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=<time>
#SBATCH --partition=<partition>
#SBATCH --exclusive
#SBATCH --no-kill
#SBATCH --output="slurm-%j.out"
#SBATCH --ip-isolate yes

# Run command: sbatch client_server.sh

# Source the virtual environment
source <virtual_env>
source /usr/WS1/rusu1/decision_superiority/v_pybind/bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

server_node=${nodes_array[0]}
server_node_ip=$(srun --nodes=1 --ntasks=1 -w "$server_node" hostname --ip-address)

# if we detect a space character in the server node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$server_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$server_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  server_node_ip=${ADDR[1]}
else
  server_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $server_node_ip"
fi

# TODO: I don't think I need these lines
# port=6379
# ip_head=$server_node_ip:$port
# export ip_head # TODO: Is this needed?
# echo "Ray dashboard configured to run on: $ip_head"

echo "Starting server at $server_node"
srun --nodes=1 --ntasks=1 -w "$server_node" --output="slurm-%j-SERVER.out" \
  python3 -u ./server.py --server-address $server_node_ip --base-port <base_port> &

# TODO: Do we still need this?
# # Give the computer time to launch the server node before launching the clients.
# sleep 180

# number of nodes other than the head node
echo "SLURM JOB NUM NODES " $SLURM_JOB_NUM_NODES
clients_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= clients_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting client $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --output="slurm-%j-$node_i.out" \
      python3 -u ./client.py --server-address $server_node_ip --port $(({parameters.base_port + $node_i})) \
      --inference-mode <inference_mode> &
    sleep 5
done

wait
