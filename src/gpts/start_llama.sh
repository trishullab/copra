model=codellama/CodeLlama-7b-Instruct-hf #tiiuae/falcon-7b-instruct
volume=$PWD/.log/tgi/models
# Change model by model name argument
if [ $# -eq 1 ]; then
    model=$1
fi
# Change volume by volume argument
if [ $# -eq 2 ]; then
    volume=$2
fi
# Check if volume exists
if [ ! -d $volume ]; then
    # Raise error if volume does not exist
    echo "Volume $volume does not exist"
    exit 1
fi
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id $model --max-input-length 14000 --max-total-tokens 16384 --max-batch-prefill-tokens 14000