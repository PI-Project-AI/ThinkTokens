import torch
from torch.utils.data import Dataset
import random
import numpy as np

class TrinityDataset(Dataset):
    def __init__(self, num_samples=200000, split='train', seed=42):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        
        # Unified Vocab
        self.tokens = ['<PAD>', '<BOS>', '<EOS>', ' ']
        self.tokens += list("0123456789+-*=") # Math
        self.tokens += list(".,?!'") # Punctuation
        
        # Word Bank (Limited for 22M model)
        self.words = [
            # Chat
            "hi", "hello", "how", "are", "you", "i", "am", "good", "thanks", "fine",
            "what", "is", "your", "name", "like", "love", "hate", "pizza", "music",
            "do", "not", "yes", "no", "maybe", "please", "sorry", "bye",
            # Story
            "once", "upon", "a", "time", "there", "was", "lived", "saw", "found",
            "king", "queen", "knight", "dragon", "bird", "cat", "dog", "forest",
            "castle", "cave", "happy", "sad", "big", "small", "fought", "ate", "slept",
            # Reasoning keywords
            "tom", "sally", "has", "had", "buy", "buys", "gave", "gives", "apples", "bananas",
            "how", "many", "left", "total"
        ]
        self.tokens += self.words
        
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
        
        self._gen_math(n_part)
        self._gen_chat(n_part)
        self._gen_story(self.num_samples - 2*n_part)
        
        random.shuffle(self.data)
        
    def _gen_math(self, n):
        # "Tom has 3 apples. He buys 2. How many?"
        names = ["tom", "sally"]
        items = ["apples", "bananas"]
        
        for _ in range(n):
            name = random.choice(names)
            item = random.choice(items)
            
            start = random.randint(1, 9)
            
            # Op: Buy (Add) or Give (Sub)
            op = random.choice(['buy', 'give'])
            
            if op == 'buy':
                delta = random.randint(1, 9)
                res = start + delta
                s2 = f"buys {delta}"
            else: # Give away
                delta = random.randint(1, start) # Ensure positive
                res = start - delta
                s2 = f"gives {delta}"
            
            inp = f"{name} has {start} {item} . {name} {s2} . how many {item} ?"
            target = f"{res}"
            
            self.data.append({'input': inp, 'target': target, 'task': 'math'})

    def _gen_chat(self, n):
        # Q -> A pairs
        pairs = [
            ("hi", "hello"),
            ("how are you", "i am good thanks"),
            ("what is your name", "i am a bot"),
            ("do you like pizza", "yes i love pizza"),
            ("do you like music", "yes i like music"),
            ("bye", "bye")
        ]
        
        for _ in range(n):
            q, a = random.choice(pairs)
            self.data.append({'input': q, 'target': a, 'task': 'chat'})

    def _gen_story(self, n):
        # Start -> Continuation
        starts = [
            ("once upon a time", "there was a"),
            ("the king lived in", "a big castle"),
            ("the dragon saw", "a small bird"),
            ("the knight fought", "the dragon")
        ]
        
        for _ in range(n):
            s, c = random.choice(starts)
            # Add some noise/variation? No, keep it clean for learning.
            self.data.append({'input': s, 'target': c, 'task': 'story'})

    def encode(self, text):
        # Simple tokenization
        text = text.replace('.', ' .').replace('?', ' ?')
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
