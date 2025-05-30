# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:04
# @author  : Mo
# @function: llama

from config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
import sys
import os

import transformers.atrainer

path_root = os.path.abspath(os.path.join(""))
print(path_root)
sys.path.append(path_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
# os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

# from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.modeling_utils import unwrap_model

from datasets import load_dataset
import torch.nn as nn
import transformers
import torch

# from transformers import LlamaForCausalLM, LlamaModel
# from transformers import LlamaTokenizer, LlamaConfig
from config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, DATASET
from config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from config import HIDDEN_SIZE, NUM_ATTENTION_HEADS

from loguru import logger as log

# tensorboardx_witer = SummaryWriter(logdir=MODEL_SAVE_DIR)
# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
device_map = "auto"
ddp = world_size != 1
print(device_map)
print(ddp)

class SaveModelCallback(TrainerCallback):
    def __init__(self, save_steps):
        self.save_steps = save_steps
        self.last_saved_step = 0
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # 检查是否达到保存步数
        if (state.global_step - self.last_saved_step) >= self.save_steps:
            print(f"Saving model at step {state.global_step}")
            model.save_pretrained(f"./saved_model_step_{state.global_step}")
            self.last_saved_step = state.global_step
            
        return control

def save_model_state(
    model, config=None, model_save_dir="./", model_name="pytorch_model.bin"
):
    """仅保存 有梯度 的 模型参数(推荐使用)"""
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))


def print_named_parameters(model, use_print_data=False):
    """打印模型训练参数/数据类型信息"""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # if use_print_data:
        #     print((name, param.data.dtype, param.requires_grad, param.data))
        # else:
        #     print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            print((name, param.data.dtype, param.requires_grad))
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_model_for_half_training(
    model,
    output_embedding_layer_name="lm_head",
    use_gradient_checkpointing=True,
    layer_norm_names=["layer_norm"],
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(
            layer_norm_name in name for layer_norm_name in layer_norm_names
        ):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model


def dfs_file(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()  # the same list
    return files


if __name__ == "__main__":
    model_config = transformers.AutoConfig.from_pretrained(PATH_MODEL_PRETRAIN)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        PATH_MODEL_PRETRAIN, config=model_config
    )
    # log.info(f"\nbase_model: {model}")
    # model = prepare_model_for_half_training(
    #     model,
    #     use_gradient_checkpointing=True,
    #     output_embedding_layer_name="lm_head",
    #     layer_norm_names=[
    #         "ln_1",
    #         "ln_2",
    #         "ln_f",
    #     ],
    # )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = IS_PARALLELIZABLE
    model.model_parallel = MODEL_PARALLEL
    model.config.use_cache = USE_CACHE
    DATA_PATH = ""
    for j in range(1, 2):
        for i in range(13, 15):
            if i == 0:
                dir_name = "0"
            else:
                dir_name = f"0-{i}"
            data = {
                "superni": load_dataset(
                    "json", data_files=os.path.join(DATA_PATH, "s"+str(j), "checkpoints", dir_name, "train.json")
                )
            }
            MODEL_SAVE_DIR = os.path.join(DATA_PATH, "s"+str(j), "checkpoints", dir_name)

            # 详细检查参数训练状态
            # log.info("\nDetailed parameter training status:")
            trainable_params = 0
            all_param = 0

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                PATH_MODEL_PRETRAIN, add_eos_token=True
            )
            ID_PAD = 151643
            ID_EOS = 151643  # endoftext
            ID_SOP = 151644  # start
            ID_EOP = 151645  # end
            ID_BR = 198  # "\n"
            tokenizer.pad_token_id = ID_PAD
            tokenizer.eos_token_id = ID_EOS
            tokenizer.padding_side = "right"
            data = {
                "superni": load_dataset(
                    "json", data_files=os.path.join(DATA_PATH, "s"+str(j), "checkpoints", dir_name, "train.json")
                )
            }
            MODEL_SAVE_DIR = os.path.join(DATA_PATH, "s"+str(j), "checkpoints", dir_name)
            # print(data)
            # generate_prompt(data["train"], is_logger=True)
            train_data = data["superni"]["train"].shuffle().map(generate_prompt)

            class CustomTrainer(transformers.atrainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    loss = model(**inputs)["loss"]
                    # for name, param in model.named_parameters():
                    #     if any(layer_name in name for layer_name in ["layer.27.mlp.up_proj.lora_A0", "layer.27.mlp.up_proj.lora_B0"]):
                    #         if param.grad is not None:
                    #             print(f"Gradient for {name}: {param.grad.abs().mean()}")
                    return loss
                    # log.info(f"loss in Qwen2ForCausalLM: {loss}")

            trainer = CustomTrainer(
                model=model,
                data_collator=data_collator,
                train_dataset=train_data,
                args=transformers.TrainingArguments(
                    save_strategy="no",
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    per_device_train_batch_size=MICRO_BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                    num_train_epochs=3,
                    max_grad_norm=2.0,
                    warmup_steps=32,  # 618
                    evaluation_strategy="no",
                    lr_scheduler_type="cosine",  # "cosine",
                    logging_first_step=True,
                    ddp_find_unused_parameters=False if ddp else None,
                    gradient_checkpointing=True,
                    output_dir=MODEL_SAVE_DIR,
                    report_to=[],  # ["tensorboard"],  # [], ["wandb"]
                    optim="adamw_torch",  # "adamw_hf",
                    bf16=True,
                ),
            )
            trainer.train()