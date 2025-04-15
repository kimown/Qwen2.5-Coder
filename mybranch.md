
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device

https://askubuntu.com/questions/1326304/cannot-install-pip-module-because-there-is-no-space-left-on-device

```
TMPDIR=/xxx/tmp
pip install -r requirements.txt
```

```inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/xxx/qwen25coder/qwen2.5-coder-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "写一个lottie的json，不要输出\n换行符号"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=7276
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response------")
print(response)
```

```stream.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

device = "cuda" # the device to load the model onto


model_name = "/xxx/qwen25coder/qwen2.5-coder-7B-Instruct"

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "写一个lottie的json"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(inputs=model_inputs.input_ids, streamer=streamer, max_new_tokens=2048)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

thread.start()
generated_text = ""
for new_text in streamer:
    generated_text += new_text
    print(new_text, end="")
print(generated_text)
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

pip install  packaging wheel jsonlines  datasets peft deepspeed


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


角色扮演 甄嬛
https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md
