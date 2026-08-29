import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class ArithmeticDataset(Dataset):
    def __init__(self, num_samples=100000, complexity=3, split='train', seed=42):
        self.num_samples = num_samples
        self.complexity = complexity
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
        
        ops = ['+', '-']
        # Add multiplication for higher complexity if needed, but start simple
        # ops.append('*') 
        
        for _ in range(self.num_samples):
            op = random.choice(ops)
            
            # Generate operands based on complexity (digits)
            if self.split == 'train':
                # Training: 1 to complexity digits
                digits1 = random.randint(1, self.complexity)
                digits2 = random.randint(1, self.complexity)
            else:
                # Test: fixed complexity to test generalization/robustness
                digits1 = self.complexity
                digits2 = self.complexity
                
            a = random.randint(0, 10**digits1 - 1)
            b = random.randint(0, 10**digits2 - 1)
            
            if op == '+':
                res = a + b
            elif op == '-':
                # Keep it positive for simplicity initially? Or allow negative.
                # Let's allow negative, vocabulary has '-'
                res = a - b
            elif op == '*':
                res = a * b
                
            # Formats: 
            # Input: "a + b ="
            # Target: "res"
            
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
        
        # Encode
        input_ids = self.encode(input_str)
        target_ids = self.encode(target_str) + [self.vocab['<EOS>']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_text': input_str,
            'target_text': target_str
        }

def collate_fn(batch):
    # Dynamic padding
    max_in_len = max([len(item['input_ids']) for item in batch])
    max_tg_len = max([len(item['target_ids']) for item in batch])
    
    input_ids = torch.zeros(len(batch), max_in_len, dtype=torch.long) # Pad is 0
    target_ids = torch.zeros(len(batch), max_tg_len, dtype=torch.long) # Pad is 0
    
    # Mask for targets (ignore padding)
    target_mask = torch.zeros(len(batch), max_tg_len, dtype=torch.float)
    
    for i, item in enumerate(batch):
        l_in = len(item['input_ids'])
        l_tg = len(item['target_ids'])
        
        input_ids[i, :l_in] = item['input_ids']
        target_ids[i, :l_tg] = item['target_ids']
        target_mask[i, :l_tg] = 1.0
        
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'target_mask': target_mask,
        'input_texts': [item['input_text'] for item in batch],
        'target_texts': [item['target_text'] for item in batch]
    }
