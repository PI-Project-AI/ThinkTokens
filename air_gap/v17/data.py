import torch
from torch.utils.data import Dataset
import random
import numpy as np
import re
import os
from collections import Counter

class HybridDataset(Dataset):
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
        self.tiny_stories_path = tiny_stories_path
        self.max_vocab_size = max_vocab_size 
        self.block_size = block_size
        self.n_ir_tokens = n_ir_tokens
        self.vocab_story_sample = vocab_story_sample
        self.vocab_tokens = vocab_tokens

        # Common words for math problems
        self.math_words = ["tom", "sally", "john", "mary", "apples", "bananas", "has", "had", "buy", "buys", "gave", "gives", "how", "many", "left", "total", "he", "she"]
        # Force math targets into vocab so accuracy isn't capped by OOV.
        self.math_numbers = [str(i) for i in range(0, 19)]

        # Load TinyStories to build vocab from real language
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
            forced_tokens = set(list("0123456789+-*="))  # Math symbols
            forced_tokens.update(list(".,?!'"))  # Punctuation
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
        
        print(f"Initialized Vocab Size: {len(self.vocab)} (Max: {self.max_vocab_size})")

        self._generate_data()
        
    def _generate_data(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        n_math = int(self.num_samples * 0.3)
        n_story = int(self.num_samples * 0.7)
        
        self._gen_math(n_math)
        self._gen_story(n_story)
        
        random.shuffle(self.data)
        
    def _gen_math(self, n):
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

    def _gen_story(self, n):
        if not self.tiny_stories_corpus: return

        for _ in range(n):
            story_line = random.choice(self.tiny_stories_corpus)
            self.data.append({'input': story_line, 'target': story_line, 'task': 'story'})

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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        # Truncate inputs respecting the shared block/IR budget to keep sequences valid.
        max_input_content_length = self.block_size - self.n_ir_tokens
        if len(input_ids) > max_input_content_length:
            input_ids = input_ids[-max_input_content_length:]

        # Max length of `speaker_input_embeds` is `n_ir_tokens + (len(target_ids) - 1)`. This must be <= `block_size`.
        # Max `len(target_ids)` (including EOS) = `block_size - n_ir_tokens + 1`.
        max_target_full_length = self.block_size - self.n_ir_tokens + 1
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
        
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'tasks': tasks,
        'input_texts': [x['input_text'] for x in batch],
        'target_texts': [x['target_text'] for x in batch]
    }
