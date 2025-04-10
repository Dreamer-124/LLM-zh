from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "outputs/ckpt/llm_sft_440m"

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map = "auto", 
    trust_remote_code = True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code = True
)

new_tokenizer = AutoTokenizer.from_pretrained("tokenizer/llm_tokenizer_hf")
print(len(new_tokenizer))
print(model)
"""
llmForCausalLM(
  (model): llmModel(
    (embed_tokens): Embedding(64798, 512)
    (layers): ModuleList(
      (0-7): 8 x llmDecoderLayer(
        (self_attn): llmSdpaAttention(
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (o_proj): Linear(in_features=512, out_features=512, bias=False)
          (rotary_emb): llmRotaryEmbedding()
        )
        (mlp): llmMLP(
          (gate_proj): Linear(in_features=512, out_features=1408, bias=False)
          (up_proj): Linear(in_features=512, out_features=1408, bias=False)
          (down_proj): Linear(in_features=1408, out_features=512, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): llmRMSNorm()
        (post_attention_layernorm): llmRMSNorm()
      )
    )
    (norm): llmRMSNorm()
  )
  (lm_head): Linear(in_features=512, out_features=64798, bias=False)
)
"""

emdeddings = model.get_input_embeddings()
model.resize_token_embeddings(49958)
model.config.vocab_size = 49958

print(model)
"""
llmForCausalLM(
  (model): llmModel(
    (embed_tokens): Embedding(49958, 512)
    (layers): ModuleList(
      (0-7): 8 x llmDecoderLayer(
        (self_attn): llmSdpaAttention(
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (o_proj): Linear(in_features=512, out_features=512, bias=False)
          (rotary_emb): llmRotaryEmbedding()
        )
        (mlp): llmMLP(
          (gate_proj): Linear(in_features=512, out_features=1408, bias=False)
          (up_proj): Linear(in_features=512, out_features=1408, bias=False)
          (down_proj): Linear(in_features=1408, out_features=512, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): llmRMSNorm()
        (post_attention_layernorm): llmRMSNorm()
      )
    )
    (norm): llmRMSNorm()
  )
  (lm_head): Linear(in_features=512, out_features=49958, bias=False)
)
"""

output_dir = "outputs/sft_440m_llama"

model.save_pretrained(output_dir)
new_tokenizer.save_pretrained(output_dir)
