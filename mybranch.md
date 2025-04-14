
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device

https://askubuntu.com/questions/1326304/cannot-install-pip-module-because-there-is-no-space-left-on-device

```
TMPDIR=/xxx/tmp
pip install -r requirements.txt
```

```

```

```train.sh
mkdir qwen25coder
cd qwen25coder
git clone https://github.com/kimown/Qwen2.5-Coder.git
modelscope download Qwen/Qwen2.5-Coder-7B --local_dir ./qwen2.5-coder-7B
modelscope download qwen/Qwen2-Audio-7B-Instruct --local_dir ./qwen2.5-coder-7B-Instruct

ls
Qwen2.5-Coder  qwen2.5-coder-7B  qwen2.5-coder-7B-Instruct  sft.jsonl

cd Qwen2.5-Coder
python3 -m venv tutorial-env
source tutorial-env/bin/activate
pip install -r requirements.txt

cd finetuning/sft
pip install -r requirements.txt

bash ./scripts/binarize_data.sh ../../../sft.jsonl ../../../sft_processed.jsonl ../../../qwen2.5-coder-7B
ls ../../../sft_processed.jsonl.npy

mkdir adapter
bash ./scripts/sft_qwencoder.sh ../../../sft_processed.jsonl.npy ../../../qwen2.5-coder-7B ./adapter

mkdir merged_models
bash ./scripts/merge_adapter.sh ../qwen2.5-coder-7B ./sft_model/xxx.pth ./merged_models/model1


pip show vllm
python
import sys
sys.path
```
