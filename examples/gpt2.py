#!/usr/bin/env python3
import functools, sys, argparse, math, platform
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

from tinygrad.helpers import Timing, getenv, DEBUG, dtypes
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.ops import GlobalCounters
from tinygrad.jit import TinyJit


class Attention:
  def __init__(self, dim, n_heads):
    self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim, bias=True) for _ in range(4)]
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.cache_k, self.cache_v = None, None

  def prepare_attention(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]
    return xq, xk, xv

  def inner_attention(self, xq:Tensor, xk:Tensor, xv:Tensor, start_pos:int, mask:Optional[Tensor]) -> Tensor:
    b, t, _, _ = xq.shape

    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert False
      assert self.cache_k is not None and self.cache_v is not None, "kv cache hasn't been created yet!"
      assert start_pos == self.cache_k.shape[1] and start_pos == self.cache_v.shape[1], "cache is wrong shape"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = self.cache_k.cat(xk, dim=1), self.cache_v.cat(xv, dim=1)

    # save the cache
    self.cache_k, self.cache_v = keys.realize(), values.realize()

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = xq.matmul(keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask
    scores = scores.softmax()  # this is casted to float
    return scores.matmul(values).transpose(1, 2).reshape(b, t, -1)

  # NOTE: this is not called
  def __call__(self, x:Tensor, start_pos:int, mask:Optional[Tensor]) -> Tensor:
    xq, xk, xv = self.prepare_attention(x)
    output = self.inner_attention(xq, xk, xv, start_pos, mask)
    return self.wo(output)

class FeedForward:
  def __init__(self, dim):
    self.fc1 = Linear(dim, dim*4, bias=True)
    self.fc2 = Linear(dim*4, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.fc2(self.fc1(x).gelu())

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.mlp = FeedForward(dim)
    self.attn_norm = LayerNorm(dim, norm_eps)
    self.mlp_norm = LayerNorm(dim, norm_eps)
    if getenv("JIT"):
      self._pre = TinyJit(self.pre)
      self._post = TinyJit(self.post)
    else:
      self._pre, self._post = self.pre, self.post

  def pre(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.attn.prepare_attention(self.attn_norm(x))
    return xq.realize(), xk.realize(), xv.realize()

  def post(self, x:Tensor, output:Tensor) -> Tensor:
    h = x + self.attn.wo(output)
    return (h + self.mlp(self.mlp_norm(h))).realize()

  def __call__(self, x:Tensor, start_pos:int, mask:Optional[Tensor]):
    xq, xk, xv = self._pre(x)
    # inner_attention can't be jitted because it's dynamic based on start_pos
    output = self.attn.inner_attention(xq, xk, xv, start_pos, mask)
    return self._post(x, output)

class GPT:
  def __init__(self, dim, n_heads, n_layers, vocab_size, max_seq_len, norm_eps=1e-5):
    self.blocks = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.norm = LayerNorm(dim, norm_eps)
    self.embed_tokens = Embedding(vocab_size, dim)
    self.embed_pos = Embedding(max_seq_len, dim)
    self.output = Linear(dim, vocab_size, bias=False)

  def __call__(self, tokens:Tensor, start_pos:int):
    b, t = tokens.shape
    pos = Tensor(np.arange(0, tokens.shape[-1], dtype=np.int32)[None])
    mask = Tensor.full((1, 1, t, start_pos + t), -np.inf, dtype=dtypes.float32).triu(1).realize() if t > 1 else None

    x = self.embed_tokens(tokens) + self.embed_pos(pos)
    for layer in self.blocks:
      x = layer(x, start_pos=start_pos, mask=mask)
    return self.output(self.norm(x))

  def generate(self, x, num_tokens:int = 1, temperature:float = 1.0, top_k:int = -1):
    for _ in range(num_tokens):
      logits = self(x, 0)[:, -1] / temperature
      if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, -1:]] = -torch.inf
      probs = logits.softmax().numpy()[0]
      next_token = np.random.choice(probs.shape[0], p=probs)
      x = x.cat(Tensor(next_token, dtype=dtypes.int32)[None,None], dim=1)
    return x



if __name__ == '__main__':
  import torch
  import requests
  import tiktoken

  # Load weights
  weights_url = 'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'
  checkpoint_fn = '/tmp/gpt2.ckpt'

  if not Path(checkpoint_fn).exists():
    r = requests.get(weights_url, stream=True)
    file_size = int(r.headers['content-length'])
    chunk_size = 1000  # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
    with open(checkpoint_fn, 'wb') as f:
      with tqdm(ncols=100, desc="Fetching " + weights_url, total=file_size, unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)

  state_dict = torch.load(checkpoint_fn)

  # Remap names
  replacements = {
    'h.': 'blocks.',
    'wte.': 'embed_tokens.',
    'wpe.': 'embed_pos.',
    'attn.c_attn': 'attn.qkv',
    'attn.c_proj': 'attn.wo',
    'mlp.c_fc': 'mlp.fc1',
    'mlp.c_proj': 'mlp.fc2',
    'ln_1.': 'attn_norm.',
    'ln_2.': 'mlp_norm.',
    'ln_f.': 'norm.',
  }
  linears = ['attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight']
  biases = ['attn.bias', 'attn.masked_bias']

  for src,dst in replacements.items():
    state_dict = {k.replace(src, dst): v for k,v in state_dict.items()}
  state_dict = {k:v for k,v in state_dict.items() if not any(x in k for x in biases)}
  state_dict = {k: v.transpose(-1, -2) if any(x in k for x in linears) else v for k,v in state_dict.items()}
  state_dict['output.weight'] = state_dict['embed_tokens.weight']
  for key in list(state_dict.keys()):
    if 'attn.qkv' in key:
      data = state_dict.pop(key)
      q, k, v = data.split(data.shape[0] // 3, dim=0)
      for sk,sv in {'q':q, 'k':k, 'v':v}.items():
        state_dict[key.replace('attn.qkv', f'attn.w{sk}')] = sv

  for key in list(state_dict.keys()):
    if 'attn.wo.weight' in key:
      state_dict[key] = state_dict[key].T

  def load_parameter(block, key, value):
    k, *subk = key.split('.')
    if k.isdigit():
      load_parameter(block[int(k)], '.'.join(subk), value)
      return
    assert hasattr(block, k), f'block {block} has no attribute {k}'
    if len(subk) == 0:
      assert isinstance(getattr(block, k), Tensor), f'attribute {k} of block {block} is not a Tensor but {type(getattr(block, k))}'
      setattr(block, k, value)
    else:
      load_parameter(getattr(block, k), '.'.join(subk), value)

  model = GPT(dim=768, n_heads=12, n_layers=12, vocab_size=50257, max_seq_len=1024)
  for k,v in state_dict.items():
    load_parameter(model, k, Tensor(v.numpy()))

  tokenizer = tiktoken.get_encoding("gpt2")
  prompt = "The capital of Germany is Berlin. The capital of France is"
  context = Tensor(tokenizer.encode(prompt), dtype=dtypes.int32)[None]

  Tensor.no_grad = True
  result = model.generate(context, num_tokens=2, top_k=-1)
  print(f"Prompt:    ", prompt)
  print(f"Completion:", tokenizer.decode(result[0].numpy().tolist()))
