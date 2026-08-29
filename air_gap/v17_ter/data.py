import torch
from torch.utils.data import Dataset
import random
import numpy as np
import re
import os
from collections import Counter

class PredictiveDataset(Dataset):
    def __init__(
        self,
        num_samples=200000,
        split='train',
        seed=42,
        tiny_stories_path=None,
        max_vocab_size=3000,
        block_size=256,
        n_ir_tokens=64,
        vocab_story_sample=200000,
        vocab_tokens=None,
    ):
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.data = []
        self.max_vocab_size = max_vocab_size
        self.block_size = block_size
        self.n_ir_tokens = n_ir_tokens
        self.vocab_story_sample = vocab_story_sample
        self.vocab_tokens = vocab_tokens

        # Common words for math problems
        self.math_words = [
            "tom", "sally", "john", "mary",
            "apples", "bananas",
            "has", "had", "buy", "buys", "gave", "gives",
            "how", "many", "left", "total", "he", "she"
        ]
        # Force math targets into vocab so accuracy isn't capped by OOV.
        self.math_numbers = [str(i) for i in range(0, 19)]
        
        # Load Stories
        self.tiny_stories_corpus = []
        if tiny_stories_path and os.path.exists(tiny_stories_path):
            with open(tiny_stories_path, 'r') as f:
                for line in f:
                    story = line.strip()
                    if story:
                        self.tiny_stories_corpus.append(story)

        # Fallback if no corpus loaded
        if not self.tiny_stories_corpus:
            print("Warning: TinyStories not loaded. Using dummy data for vocab.")
            self.tiny_stories_corpus = ["tom has a red ball ."]

        # Build a frequency-based vocab for lower OOV; keep deterministic ordering.
        if self.vocab_tokens is None:
            token_counts = Counter()
            sample_span = min(len(self.tiny_stories_corpus), self.vocab_story_sample)
            for story in self.tiny_stories_corpus[:sample_span]:
                token_counts.update(self._tokenize_english(story))

            base_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ' ']
            forced_tokens = set(list("0123456789+-*="))
            forced_tokens.update(list(".,?!'"))
            forced_tokens.update(self.math_words)
            forced_tokens.update(self.math_numbers)

            self.tokens = list(base_tokens)
            # Ensure forced tokens are present before frequency-based fill.
            for token in sorted(forced_tokens):
                if token not in self.tokens and len(self.tokens) < self.max_vocab_size:
                    self.tokens.append(token)

            for token, _ in sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if token not in self.tokens and len(self.tokens) < self.max_vocab_size:
                    self.tokens.append(token)
                if len(self.tokens) >= self.max_vocab_size:
                    break
        else:
            # Reuse a fixed vocab to keep train/eval token IDs aligned.
            self.tokens = list(self.vocab_tokens)
        
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab['<EOS>']
        self.unk_id = self.vocab['<UNK>']
        
        print(f"Vocab Size: {len(self.vocab)}")
        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        # 70% Story Prediction, 30% Math
        n_math = int(self.num_samples * 0.3)
        n_story = int(self.num_samples * 0.7)
        
        self._gen_math(n_math)
        self._gen_story_predict(n_story)
        random.shuffle(self.data)

    def _gen_story_predict(self, n):
        if not self.tiny_stories_corpus: return
        
        for _ in range(n):
            story = random.choice(self.tiny_stories_corpus)
            tokens = self._tokenize_english(story)
            
            # Split into Context -> Target
            # Ideally split at a sentence boundary, but random split is a hard proxy
            if len(tokens) < 10: continue
            
            split_idx = random.randint(5, len(tokens) - 5)
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
        # Standard Math logic
        names = ["tom", "sally", "john", "mary"]
        items = ["apples", "bananas"]
        for _ in range(n):
            name = random.choice(names)
            item = random.choice(items)
            start = random.randint(1, 9)
            op = random.choice(['buy', 'give'])
            if op == 'buy':
                delta = random.randint(1, 9)
                res = start + delta
                s2 = f"buys {delta}"
            else:
                delta = random.randint(1, start)
                res = start - delta
                s2 = f"gives {delta}"
            inp = f"{name} has {start} {item} . {name} {s2} . how many {item} ?"
            target = f"{res}"
            self.data.append({'input': inp, 'target': target, 'task': 'math'})

    def _tokenize_english(self, text):
        text = text.lower()
        # Improved tokenization: separate punctuation but keep apostrophes within contractions
        text = re.sub(r"([.,?!])", r" \1 ", text)
        text = re.sub(r"'s", " 's", text)
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"n't", " n't", text)
        tokens = text.split()
        return tokens

    def encode(self, text):
        tokens = self._tokenize_english(text)
        return [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
    
    def decode(self, ids):
        specials = {self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']}
        return " ".join([self.id_to_char.get(i, '<UNK>') for i in ids if i not in specials])

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        # Truncate with the shared block/IR budget to keep sequences valid.
        max_input = self.block_size - self.n_ir_tokens
        if len(input_ids) > max_input: input_ids = input_ids[-max_input:]
        max_target = self.block_size - self.n_ir_tokens + 1
        if len(target_ids) > max_target: target_ids = target_ids[:max_target]
        
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
