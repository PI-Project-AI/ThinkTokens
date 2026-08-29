import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class ArithmeticDataset(Dataset):
    def __init__(self, num_samples=200000, complexity=3, split='train', seed=42):
        self.num_samples = num_samples
        self.complexity = complexity # Max digits for +/-, and 2 for * usually
        self.split = split
        self.seed = seed
        self.data = []
        
        # Simple tokenizer: digits 0-9, operators +, -, *, =, space, pad
        self.vocab = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2,
            '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,
            '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
            '+': 13, '-': 14, '*': 15, '=': 16, ' ': 17
        }
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        ops = ['+', '-', '*']
        
        for _ in range(self.num_samples):
            op = random.choice(ops)
            
            # V11 Logic:
            # + / - : up to `complexity` digits (e.g., 3)
            # * : up to 2 digits (harder task)
            
            if op in ['+', '-']:
                if self.split == 'train':
                    # Mixed complexity for training
                    digits1 = random.randint(1, self.complexity)
                    digits2 = random.randint(1, self.complexity)
                else:
                    # Fixed max complexity for test
                    digits1 = self.complexity
                    digits2 = self.complexity
            elif op == '*':
                # Multiplication is harder, limit to 2 digits for now in V11
                mult_complexity = 2
                if self.split == 'train':
                    digits1 = random.randint(1, mult_complexity)
                    digits2 = random.randint(1, mult_complexity)
                else:
                    digits1 = mult_complexity
                    digits2 = mult_complexity
                
            a = random.randint(0, 10**digits1 - 1)
            b = random.randint(0, 10**digits2 - 1)
            
            if op == '+':
                res = a + b
            elif op == '-':
                res = a - b
            elif op == '*':
                res = a * b
            
            # Formats: "a + b =" -> "res"
            input_str = f"{a} {op} {b} ="
            target_str = f"{res}"
            
            self.data.append((input_str, target_str))
            
    def encode(self, text):
        return [self.vocab.get(c, self.vocab['<PAD>']) for c in text]
        
    def decode(self, ids):
        return "".join([self.id_to_char.get(i, '') for i in ids if i not in [0, 1, 2]])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_str, target_str = self.data[idx]
        
        input_ids = self.encode(input_str)
        target_ids = self.encode(target_str) + [self.vocab['<EOS>']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_text': input_str,
            'target_text': target_str
        }

def collate_fn(batch):
    max_in_len = max([len(item['input_ids']) for item in batch])
    max_tg_len = max([len(item['target_ids']) for item in batch])
    
    input_ids = torch.zeros(len(batch), max_in_len, dtype=torch.long) 
    target_ids = torch.zeros(len(batch), max_tg_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        l_in = len(item['input_ids'])
        l_tg = len(item['target_ids'])
        
        input_ids[i, :l_in] = item['input_ids']
        target_ids[i, :l_tg] = item['target_ids']
        
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'input_texts': [item['input_text'] for item in batch],
        'target_texts': [item['target_text'] for item in batch]
    }