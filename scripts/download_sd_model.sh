mkdir models
mkdir models/sdxl
# RealVisXL 2.0
curl -L "https://civitai.com/api/download/models/169921?type=Model&format=SafeTensor&size=pruned&fp=fp16" --output models/sdxl/realvisxl.safetensors &

# SDXL 
curl -L "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true" --output models/sdxl/sdxl.safetensors &

# JuggernautXL
curl -L "https://civitai.com/api/download/models/198530?type=Model&format=SafeTensor&size=full&fp=fp16" --output models/sdxl/juggernautxl.safetensors &