import torch
from torch.utils.data import Dataset
import random
import numpy as np

class MixedReasoningDataset(Dataset):
    def __init__(self, num_samples=200000, split='train', seed=42):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        
        self.tokens = ['<PAD>', '<BOS>', '<EOS>', ' ']
        self.tokens += list("0123456789+-*=")
        self.tokens += list("ABCDETF&|!?") 
        self.tokens += list("UDLR(),")
        
        self.tokens = sorted(list(set(self.tokens)))
        
        if '<PAD>' in self.tokens:
            self.tokens.remove('<PAD>')
        self.tokens = ['<PAD>'] + self.tokens
        
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        
        self.eos_id = self.vocab['<EOS>']
        
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        n_math = self.num_samples // 3
        n_logic = self.num_samples // 3
        n_nav = self.num_samples - n_math - n_logic
        
        self._gen_math(n_math)
        self._gen_logic(n_logic)
        self._gen_nav(n_nav)
        
        random.shuffle(self.data)
        
    def _gen_math(self, n):
        ops = ['+', '-', '*']
        for _ in range(n):
            op = random.choice(ops)
            digits = 2
            
            a = random.randint(0, 10**digits - 1)
            b = random.randint(0, 10**digits - 1)
            
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            
            inp = f"{a} {op} {b} ="
            out = f"{res}"
            self.data.append({'input': inp, 'target': out, 'task': 'math'})

    def _gen_logic(self, n):
        vars = ['A', 'B', 'C', 'D', 'E']
        vals = ['T', 'F']
        ops = ['&', '|']
        
        for _ in range(n):
            base_vars = vars[:2]
            state = {}
            parts = []
            for v in base_vars:
                val = random.choice([True, False])
                state[v] = val
                val_char = 'T' if val else 'F'
                parts.append(f"{v}={val_char}")
            
            op1 = random.choice(['&', '|', '!'])
            if op1 == '!':
                operand = random.choice(base_vars)
                res = not state[operand]
                expr = f"!{operand}"
            else:
                op_a = random.choice(base_vars)
                op_b = random.choice(base_vars)
                if op1 == '&': res = state[op_a] and state[op_b]
                elif op1 == '|': res = state[op_a] or state[op_b]
                expr = f"{op_a}{op1}{op_b}"
            
            state['C'] = res
            parts.append(f"C={expr}")
            
            target_var = 'C'
            target_val = 'T' if state[target_var] else 'F'
            
            inp = ", ".join(parts) + " ?"
            out = target_val
            self.data.append({'input': inp, 'target': out, 'task': 'logic'})

    def _gen_nav(self, n):
        grid_size = 8
        moves = ['U', 'D', 'L', 'R']
        
        for _ in range(n):
            start_x = random.randint(0, grid_size-1)
            start_y = random.randint(0, grid_size-1)
            
            curr_x, curr_y = start_x, start_y
            
            num_steps = random.randint(3, 8)
            step_seq = []
            
            for _ in range(num_steps):
                m = random.choice(moves)
                step_seq.append(m)
                
                if m == 'U': curr_y = max(0, curr_y - 1) 
                elif m == 'D': curr_y = min(grid_size-1, curr_y + 1)
                elif m == 'L': curr_x = max(0, curr_x - 1)
                elif m == 'R': curr_x = min(grid_size-1, curr_x + 1)
            
            inp = f"({start_x},{start_y}) " + " ".join(step_seq)
            out = f"({curr_x},{curr_y})"
            self.data.append({'input': inp, 'target': out, 'task': 'nav'})

    def encode(self, text):
        return [self.vocab.get(c, self.vocab['<PAD>']) for c in text]
        
    def decode(self, ids):
        # Dynamic filtering of special tokens
        specials = [self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']]
        return "".join([self.id_to_char.get(i, '') for i in ids if i not in specials])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_text': item['input'],
            'target_text': item['target'],
            'task': item['task']
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
