import torch
from torch.utils.data import Dataset
import random
import numpy as np

class AlgorithmicDataset(Dataset):
    def __init__(self, num_samples=200000, split='train', seed=42):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        
        # Vocab: Digits, Variables, Ops, Keywords, Special
        # Python-like minimal subset
        self.tokens = ['<PAD>', '<BOS>', '<EOS>', ' ']
        self.tokens += list("0123456789") # Digits
        self.tokens += list("xyzijk") # Variables
        self.tokens += ['=', '+', '-', '*', '[', ']', ',', 'print', 'if', 'else', 'len', 'sort', 'range', 'for', 'in', ':']
        
        # Ensure unique and sorted
        self.tokens = sorted(list(set(self.tokens)))
        # Pad at 0
        if '<PAD>' in self.tokens: self.tokens.remove('<PAD>')
        self.tokens = ['<PAD>'] + self.tokens
        
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab['<EOS>']
        
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Mix of tasks:
        # 1. Variable Tracking: x=5; y=x+3; print(y)
        # 2. List Operations: x=[3,1,2]; y=len(x); print(y)
        # 3. Control Flow (Simple): if 5>3: x=1 else: x=0; print(x)
        # 4. Loop Unrolling (Small): x=0; for i in range(3): x=x+1; print(x)
        
        counts = self.num_samples // 4
        
        self._gen_var_tracking(counts)
        self._gen_list_ops(counts)
        self._gen_control_flow(counts)
        self._gen_loops(counts)
        
        random.shuffle(self.data)
        
    def _gen_var_tracking(self, n):
        vars = list("xyz")
        for _ in range(n):
            v1 = random.choice(vars)
            val1 = random.randint(0, 9)
            
            v2 = random.choice(vars)
            val2 = random.randint(0, 9)
            
            op = random.choice(['+', '-', '*'])
            
            # x=5; y=3; z=x+y; print(z)
            target_var = 'k' # output var
            
            if op == '+': res = val1 + val2
            elif op == '-': res = val1 - val2
            elif op == '*': res = val1 * val2
            
            inp = f"{v1}={val1} {v2}={val2} {target_var}={v1}{op}{v2} print {target_var}"
            out = f"{res}"
            self.data.append({'input': inp, 'target': out, 'task': 'vars'})

    def _gen_list_ops(self, n):
        # x=[1,2,3] len(x)? x[0]?
        vars = list("xyz")
        for _ in range(n):
            v = random.choice(vars)
            length = random.randint(1, 5)
            items = [random.randint(0, 9) for _ in range(length)]
            
            list_str = "[" + ",".join(map(str, items)) + "]"
            
            op = random.choice(['len', 'idx'])
            
            if op == 'len':
                query = f"len {v}"
                res = length
            else:
                idx = random.randint(0, length-1)
                query = f"{v} [ {idx} ]"
                res = items[idx]
            
            inp = f"{v}={list_str} print {query}"
            out = f"{res}"
            self.data.append({'input': inp, 'target': out, 'task': 'list'})

    def _gen_control_flow(self, n):
        # if a > b: x=1 else: x=0
        # simplified: if 5-3: (nonzero is true)
        # or explicit op >. Let's add > to vocab? 
        # Simpler: if 1: x=A else: x=B.
        
        for _ in range(n):
            cond = random.choice([0, 1])
            res_true = random.randint(0, 9)
            res_false = random.randint(0, 9)
            
            target = res_true if cond else res_false
            
            inp = f"if {cond} : x={res_true} else : x={res_false} print x"
            out = f"{target}"
            self.data.append({'input': inp, 'target': out, 'task': 'control'})

    def _gen_loops(self, n):
        # x=0; for i in range(N): x=x+K
        for _ in range(n):
            start = random.randint(0, 5)
            steps = random.randint(1, 4)
            inc = random.randint(1, 3)
            
            res = start + (steps * inc)
            
            inp = f"x={start} for i in range {steps} : x=x+{inc} print x"
            out = f"{res}"
            self.data.append({'input': inp, 'target': out, 'task': 'loop'})

    def encode(self, text):
        # Simple space splitting
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
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'task': item['task'],
            'input_text': item['input'],
            'target_text': item['target']
        }

def collate_fn(batch):
    max_in_len = max([len(item['input_ids']) for item in batch])
    max_tg_len = max([len(item['target_ids']) for item in batch])
    
    input_ids = torch.zeros(len(batch), max_in_len, dtype=torch.long) 
    target_ids = torch.zeros(len(batch), max_tg_len, dtype=torch.long)
    
    tasks = []
    
    for i, item in enumerate(batch):
        l_in = len(item['input_ids'])
        l_tg = len(item['target_ids'])
        
        input_ids[i, :l_in] = item['input_ids']
        target_ids[i, :l_tg] = item['target_ids']
        tasks.append(item['task'])
        
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'input_texts': [item['input_text'] for item in batch],
        'target_texts': [item['target_text'] for item in batch],
        'tasks': tasks
    }
