import torch
from torch.utils.data import Dataset
import random
import numpy as np
import re
import os

class ScaledHybridDataset(Dataset):
    def __init__(
        self,
        num_samples=2000000,
        split='train',
        seed=42,
        tiny_stories_filename=None,
        max_vocab_size=5000,
        block_size=512,
        n_ir_tokens=64,
        vocab_story_sample=200000,
    ):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        self.max_vocab_size = max_vocab_size 
        self.tiny_stories_path = os.path.join(os.path.dirname(__file__), tiny_stories_filename) if tiny_stories_filename else None
        self.block_size = block_size
        self.n_ir_tokens = n_ir_tokens
        self.vocab_story_sample = vocab_story_sample
        
        # Vocab Init
        self.tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ' ']
        self.tokens += list("0123456789+-*=") # Math
        self.tokens += list(".,?!'") # Punctuation
        
        # Load Stories to build vocab
        self.stories = []
        if self.tiny_stories_path and os.path.exists(self.tiny_stories_path):
            with open(self.tiny_stories_path, 'r') as f:
                for line in f:
                    if line.strip(): self.stories.append(line.strip())
        
        # Build Vocab from Sample (or full corpus if small enough)
        if self.stories:
            # Use a larger slice of the corpus to reduce OOV and avoid PAD-masking rare words.
            sample_span = min(len(self.stories), self.vocab_story_sample)
            sample_text = " ".join(self.stories[:sample_span])
            words = self._tokenize(sample_text)
            unique_words = sorted(list(set(words)))
            for w in unique_words:
                if len(self.tokens) < self.max_vocab_size:
                    if w not in self.tokens: self.tokens.append(w)
                else: break
        
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab['<EOS>']
        self.unk_id = self.vocab['<UNK>']
        
        print(f"Vocab Size: {len(self.vocab)}")
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        n_math = int(self.num_samples * 0.3)
        n_story_predict = int(self.num_samples * 0.7)
        
        self._gen_math(n_math)
        self._gen_story_predict(n_story_predict)
        random.shuffle(self.data)

    def _gen_story_predict(self, n):
        if not self.stories: return
        
        for _ in range(n):
            story = random.choice(self.stories)
            tokens = self._tokenize(story)
            
            if len(tokens) < 10: continue # Skip very short stories
            
            split_idx = random.randint(min(5, len(tokens)-5), len(tokens) - 5) # Ensure split allows meaningful context and target
            context_tokens = tokens[:split_idx]
            target_tokens = tokens[split_idx:]
            
            input_str = " ".join(context_tokens)
            target_str = " ".join(target_tokens)
            
            self.data.append({
                'input': input_str,
                'target': target_str,
                'task': 'story_pred'
            })

    def _gen_math(self, n):
        names = ["tom", "sally", "john", "mary"]
        items = ["apples", "bananas"]
        for _ in range(n):
            name = random.choice(names)
            item = random.choice(items)
            start = random.randint(1, 99) # 2 digit math for complexity
            op_type = random.choice(['add', 'sub', 'mult'])
            
            if op_type == 'add':
                delta = random.randint(1, 99)
                res = start + delta
                s2 = f"buys {delta}"
                op_str = '+'
            elif op_type == 'sub':
                delta = random.randint(1, start)
                res = start - delta
                s2 = f"gives {delta}"
                op_str = '-'
            else: # Multiply
                delta = random.randint(1, 9) # Keep multiplier small
                res = start * delta
                s2 = f"multiplies by {delta}"
                op_str = '*'

            inp = f"{name} has {start} {item} . {name} {s2} . how many {item} {op_str} ?"
            target = f"{res}"
            self.data.append({'input': inp, 'target': target, 'task': 'math'})

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"([.,?!])", r" \1 ", text)
        text = re.sub(r"'s", " 's", text) 
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"n't", " n't", text)
        tokens = text.split()
        return tokens

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.vocab.get(t, self.unk_id) for t in tokens]
    
    def decode(self, ids):
        specials = {self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']}
        return " ".join([self.id_to_char.get(i, '<UNK>') for i in ids if i not in specials])

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        # Respect the global block / IR allocation to keep sequences consistent with the model.
        max_input_content_length = self.block_size - self.n_ir_tokens
        if len(input_ids) > max_input_content_length: 
            input_ids = input_ids[-max_input_content_length:]
            
        max_target_full_length = self.block_size - self.n_ir_tokens + 1 # +1 to allow EOS
        if len(target_ids) > max_target_full_length: 
            target_ids = target_ids[:max_target_full_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'task': item['task'],
            'input_text': item['input'],
            'target_text': item['target']
        }

def collate_fn(batch):
    max_in = max([len(x['input_ids']) for x in batch])
    max_out = max([len(x['target_ids']) for x in batch])
    input_ids = torch.zeros(len(batch), max_in, dtype=torch.long)
    target_ids = torch.zeros(len(batch), max_out, dtype=torch.long)
    tasks = []
    for i, x in enumerate(batch):
        l_in = len(x['input_ids'])
        l_out = len(x['target_ids'])
        input_ids[i, :l_in] = x['input_ids']
        target_ids[i, :l_out] = x['target_ids']
        tasks.append(x['task'])
    return {'input_ids': input_ids, 'target_ids': target_ids, 'tasks': tasks, 'input_texts': [x['input_text'] for x in batch], 'target_texts': [x['target_text'] for x in batch]}
