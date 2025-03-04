name=codellama_tgi
model=codellama/CodeLlama-7b-Instruct-hf #tiiuae/falcon-7b-instruct
port=8080
cuda="0"
volume=$PWD/.log/tgi/models
# Change container name by container name argument
if [ $# -eq 1 ]; then
    name=$1
fi
# Change model by model name argument
if [ $# -eq 2 ]; then
    name=$1
    model=$2
fi
# Change port by port argument
if [ $# -eq 3 ]; then
    name=$1
    model=$2
    port=$3
fi
# Change cuda by cuda argument
if [ $# -eq 4 ]; then
    name=$1
    model=$2
    port=$3
    cuda=$4
fi
# Change volume by volume argument
if [ $# -eq 5 ]; then
    name=$1
    model=$2
    port=$3
    cuda=$4
    volume=$5
fi
# Check if volume exists
if [ ! -d $volume ]; then
    # Raise error if volume does not exist
    echo "Volume $volume does not exist"
    exit 1
fi
echo "Starting $name with model $model and volume $volume"
docker run --env CUDA_VISIBLE_DEVICES=$cuda --name $name --gpus all --shm-size 1g -p $port:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --max-input-length 14000 --max-total-tokens 16384 --max-batch-prefill-tokens 14000