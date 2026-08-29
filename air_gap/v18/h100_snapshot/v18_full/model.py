import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z, temp=1.0, hard=True):
        # z: [B, T, D]
        z_flattened = z.reshape(-1, self.code_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        logits = -d 
        
        if hard:
            indices = torch.argmin(d, dim=1)
            encodings = torch.zeros(indices.shape[0], self.num_codes, device=z.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)
            
            z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
            
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
            
            z_q = z + (z_q - z).detach()
            
            perplexity = torch.exp(-torch.sum(torch.mean(encodings, 0) * torch.log(torch.mean(encodings, 0) + 1e-10)))
            
            return z_q, loss, {
                'perplexity': perplexity,
                'encoding_indices': indices.view(z.shape[:-1])
            }
            
        else:
            z_q_soft = F.gumbel_softmax(logits, tau=temp, hard=False, dim=-1)
            z_q = torch.matmul(z_q_soft, self.embedding.weight).view(z.shape)
            loss = torch.mean((z_q.detach() - z) ** 2) * 0.1 
            
            probs = F.softmax(logits, dim=-1)
            perplexity = torch.exp(-torch.sum(torch.mean(probs, 0) * torch.log(torch.mean(probs, 0) + 1e-10)))
            indices = torch.argmax(logits, dim=1)

            return z_q, loss, {
                'perplexity': perplexity,
                'encoding_indices': indices.view(z.shape[:-1]),
                'encodings': z_q_soft # Return soft encodings for entropy loss if needed
            }

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx=None, inputs_embeds=None, targets=None):
        device = idx.device if idx is not None else inputs_embeds.device
        
        if inputs_embeds is None:
            b, t = idx.size()
            token_embeddings = self.transformer.wte(idx)
        else:
            b, t, c = inputs_embeds.size()
            token_embeddings = inputs_embeds

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_embeddings = self.transformer.wpe(pos)
        
        x = token_embeddings + pos_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

class GPTConfig:
    def __init__(self, vocab_size, n_layer=12, n_head=12, n_embd=768, block_size=256):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

class AirGapVQTransformer(nn.Module):
    def __init__(self, vocab_size, num_codes=4096, code_dim=768, n_ir_tokens=64, 
                 reasoner_layers=12, reasoner_heads=12, 
                 speaker_layers=12, speaker_heads=12,
                 block_size=512):
        super().__init__()
        self.num_codes = num_codes
        # Keep IR budget lean to preserve room for real language context/targets at 512 block size.
        self.n_ir_tokens = n_ir_tokens
        
        # Scaled Reasoner
        self.reasoner_config = GPTConfig(
            vocab_size=vocab_size, 
            n_embd=code_dim,
            n_layer=reasoner_layers,
            n_head=reasoner_heads,
            block_size=block_size
        ) 
        self.reasoner = NanoGPT(self.reasoner_config)
        
        self.thinking_queries = nn.Parameter(torch.randn(1, n_ir_tokens, code_dim))

        # Scaled VQ
        self.vq = VectorQuantizer(num_codes, code_dim)
        
        # Scaled Speaker
        self.speaker_config = GPTConfig(
            vocab_size=vocab_size, 
            n_embd=code_dim,
            n_layer=speaker_layers,
            n_head=speaker_heads,
            block_size=block_size
        )
        self.speaker = NanoGPT(self.speaker_config)

    def forward(self, input_ids, target_ids=None, vq_hard=True, vq_temp=1.0):
        input_embeds = self.reasoner.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        
        queries = self.thinking_queries.expand(batch_size, -1, -1)
        reasoner_input = torch.cat([input_embeds, queries], dim=1)
        
        x = reasoner_input + self.reasoner.transformer.wpe(
            torch.arange(reasoner_input.size(1), device=reasoner_input.device)
        )
        for block in self.reasoner.transformer.h:
            x = block(x)
        x = self.reasoner.transformer.ln_f(x)
        
        z = x[:, -self.n_ir_tokens:, :] 
        
        z_q, vq_loss, vq_info = self.vq(z, temp=vq_temp, hard=vq_hard)
        
        speaker_loss = None
        if target_ids is not None:
            target_embeds = self.speaker.transformer.wte(target_ids)
            speaker_input_embeds = torch.cat([z_q, target_embeds[:, :-1]], dim=1)
            speaker_targets = target_ids
            
            speaker_logits, _ = self.speaker(inputs_embeds=speaker_input_embeds)
            
            relevant_logits = speaker_logits[:, self.n_ir_tokens-1:, :]
            
            speaker_loss = F.cross_entropy(
                relevant_logits.reshape(-1, relevant_logits.size(-1)),
                speaker_targets.reshape(-1),
                ignore_index=0 
            )

        return {
            'logits': None,
            'loss': speaker_loss + vq_loss if speaker_loss is not None else None,
            'speaker_loss': speaker_loss,
            'vq_loss': vq_loss,
            'vq_info': vq_info,
            'z_q': z_q
        }
        
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, ir_mode='intact'):
        input_embeds = self.reasoner.transformer.wte(input_ids)
        batch_size = input_ids.shape[0]
        queries = self.thinking_queries.expand(batch_size, -1, -1)
        reasoner_input = torch.cat([input_embeds, queries], dim=1)
        
        x = reasoner_input + self.reasoner.transformer.wpe(
            torch.arange(reasoner_input.size(1), device=reasoner_input.device)
        )
        for block in self.reasoner.transformer.h:
            x = block(x)
        x = self.reasoner.transformer.ln_f(x)
        z = x[:, -self.n_ir_tokens:, :]
        
        z_q, _, vq_info = self.vq(z, hard=True)
        
        if ir_mode == 'random':
            indices = torch.randint(0, self.num_codes, z_q.shape[:-1], device=z_q.device)
            z_q = self.vq.embedding(indices)
        elif ir_mode == 'shuffle':
            idx = torch.randperm(z_q.shape[1])
            z_q = z_q[:, idx, :]
        elif ir_mode == 'zero':
             z_q = torch.zeros_like(z_q)
             
        curr_embeds = z_q
        generated_ids = []
        
        for _ in range(max_new_tokens):
            logits, _ = self.speaker(inputs_embeds=curr_embeds)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token)
            
            next_embed = self.speaker.transformer.wte(next_token)
            curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)
            
            if (next_token == 2).all(): 
                break
                
        return torch.cat(generated_ids, dim=1), vq_info['encoding_indices']
