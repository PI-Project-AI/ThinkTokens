import torch
from torch.utils.data import Dataset
import random
import numpy as np
import re

class MixedReasoningDataset(Dataset):
    def __init__(self, num_samples=200000, split='train', seed=42):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        
        # Unified Vocab Construction
        # We need a vocab that covers:
        # 1. Python-like Algo (digits, keywords, vars)
        # 2. bAbI-like English (words, punctuation)
        # 3. Logic (A-E, T/F, ops)
        
        # Base tokens
        self.tokens = ['<PAD>', '<BOS>', '<EOS>', ' ']
        
        # Algo tokens
        self.tokens += list("0123456789")
        self.tokens += list("xyzijk")
        self.tokens += ['=', '+', '-', '*', '[', ']', ',', 'print', 'if', 'else', 'len', 'sort', 'range', 'for', 'in', ':']
        
        # Logic tokens
        self.tokens += list("ABCDETF&|!?")
        
        # English tokens (Minimal set for synthetic stories)
        # We will generate synthetic stories using a limited vocab to avoid bloating
        english_words = [
            "mary", "john", "sandra", "bill", "fred", "julie", 
            "moved", "journeyed", "travelled", "went", "back", "is", "in", "the",
            "kitchen", "bedroom", "garden", "office", "bathroom", "school", "park", "cinema",
            "either", "or", "and", "not", "maybe", "yes", "no", "where"
        ]
        self.tokens += english_words
        
        # Punctuation
        self.tokens += ['.', '?']
        
        # Deduplicate and Sort
        self.tokens = sorted(list(set(self.tokens)))
        if '<PAD>' in self.tokens: self.tokens.remove('<PAD>')
        self.tokens = ['<PAD>'] + self.tokens
        
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab['<EOS>']
        
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        n_part = self.num_samples // 3
        
        self._gen_algo(n_part)
        self._gen_logic(n_part)
        self._gen_story(self.num_samples - 2*n_part)
        
        random.shuffle(self.data)
        
    def _gen_algo(self, n):
        # Reusing V14 algo logic
        for _ in range(n):
            mode = random.choice(['vars', 'loop', 'control'])
            
            if mode == 'vars':
                v1, v2 = random.sample("xyz", 2)
                val1, val2 = random.randint(0, 9), random.randint(0, 9)
                op = random.choice(['+', '-', '*'])
                if op == '+': res = val1 + val2
                elif op == '-': res = val1 - val2
                else: res = val1 * val2
                inp = f"{v1}={val1} {v2}={val2} k={v1}{op}{v2} print k"
                out = f"{res}"
                
            elif mode == 'loop':
                start = random.randint(0, 3)
                steps = random.randint(1, 3)
                inc = random.randint(1, 2)
                res = start + steps * inc
                inp = f"x={start} for i in range {steps} : x=x+{inc} print x"
                out = f"{res}"
                
            else: # Control
                cond = random.choice([0, 1])
                r1, r2 = random.randint(0,9), random.randint(0,9)
                target = r1 if cond else r2
                inp = f"if {cond} : x={r1} else : x={r2} print x"
                out = f"{target}"
                
            self.data.append({'input': inp, 'target': out, 'task': 'algo'})

    def _gen_logic(self, n):
        # Reusing V12 logic
        vars = ['A', 'B', 'C', 'D', 'E']
        for _ in range(n):
            base = random.sample(vars[:3], 2)
            state = {v: random.choice([True, False]) for v in base}
            parts = [f"{v}={ 'T' if state[v] else 'F' }" for v in base]
            
            # 1 step deduction
            op = random.choice(['&', '|'])
            v1, v2 = base
            if op == '&': res = state[v1] and state[v2]
            else: res = state[v1] or state[v2]
            
            target_var = 'C' # Fixed target for simplicity
            expr = f"{v1}{op}{v2}"
            parts.append(f"{target_var}={expr}")
            
            inp = ", ".join(parts) + " ?"
            out = 'T' if res else 'F'
            self.data.append({'input': inp, 'target': out, 'task': 'logic'})

    def _gen_story(self, n):
        # Synthetic bAbI-like stories (Spatial tracking)
        people = ["mary", "john", "sandra", "bill"]
        locs = ["kitchen", "bedroom", "garden", "office", "school", "park"]
        
        for _ in range(n):
            p = random.choice(people)
            path = random.sample(locs, 2)
            # 1. Init
            s1 = f"{p} is in the {path[0]} ."
            # 2. Move
            s2 = f"{p} moved to the {path[1]} ."
            # 3. Query
            q = f"where is {p} ?"
            
            inp = f"{s1} {s2} {q}"
            out = path[1]
            self.data.append({'input': inp, 'target': out, 'task': 'story'})

    def encode(self, text):
        # Tokenize by space and special chars
        # Simple splitting by space is not enough for "x=5" -> "x", "=", "5"
        # We need a regex splitter
        
        # Pad punctuation for splitting
        text = text.replace('=', ' = ').replace(',', ' , ').replace(':', ' : ').replace('[', ' [ ').replace(']', ' ] ')
        text = text.replace('(', ' ( ').replace(')', ' ) ').replace('.', ' . ').replace('?', ' ? ')
        
        tokens = text.split()
        return [self.vocab.get(t, self.vocab['<PAD>']) for t in tokens]
        
    def decode(self, ids):
        specials = [self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']]
        return " ".join([self.id_to_char.get(i, '') for i in ids if i not in specials])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        # Truncate to avoid massive padding if one sample is huge (unlikely here)
        if len(input_ids) > 64: input_ids = input_ids[:64]
        
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
        
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'tasks': tasks,
        'input_texts': [x['input_text'] for x in batch],
        'target_texts': [x['target_text'] for x in batch]
    }
