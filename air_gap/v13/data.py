import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re

class BabiDataset(Dataset):
    def __init__(self, data_dir, split='train', seed=42):
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.data = []
        self.vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, ' ': 3}
        
        # Load all tasks
        task_files = sorted(glob.glob(os.path.join(data_dir, f"qa*_{split}.txt")))
        
        # First pass: Build vocab and load raw data
        for file_path in task_files:
            task_id = int(re.search(r'qa(\d+)_', os.path.basename(file_path)).group(1))
            self._parse_file(file_path, task_id)
            
        # Finalize vocab
        self.tokens = sorted(list(self.vocab.keys()), key=lambda x: self.vocab[x])
        self.id_to_char = {i: t for t, i in self.vocab.items()}
        self.eos_id = self.vocab['<EOS>']
        
        print(f"Loaded {len(self.data)} samples from {len(task_files)} tasks. Vocab size: {len(self.vocab)}")

    def _parse_file(self, file_path, task_id):
        with open(file_path, 'r') as f:
            story = []
            for line in f:
                line = line.strip()
                nid, line = line.split(' ', 1)
                nid = int(nid)
                
                if nid == 1:
                    story = []
                    
                if '\t' in line: # Question
                    q, a, support = line.split('\t')
                    
                    # Tokenize Q and A (char level or word level? bAbI is usually word-level)
                    # But V11/V12 were char-level.
                    # Let's stick to Char-Level for simplicity and consistency with previous arch?
                    # NO, bAbI has words. Char-level might be too long for 192 context.
                    # Let's check lengths. "Mary moved to the bathroom." ~ 25 chars.
                    # 10 sentences ~ 250 chars. 
                    # Word level is safer for context length.
                    
                    # Simple word tokenization (split by space, keep punctuation)
                    # Actually, let's do CHAR level to avoid large vocab and OOV issues for VQ model.
                    # We scaled context to 192. 
                    # Is 192 chars enough?
                    # Story: "John is in the kitchen. Sandra is in the hallway." -> ~50 chars.
                    # Some stories have 10+ sentences. That's > 500 chars.
                    # 192 chars is too short for full stories.
                    
                    # Decision: WORD LEVEL TOKENIZATION.
                    # Vocab size will be small (<200 words for bAbI).
                    
                    substory = [s for s in story if s]
                    full_story = " ".join(substory)
                    full_input = f"{full_story} {q}"
                    
                    # Add to data
                    self.data.append({
                        'task_id': task_id,
                        'input': full_input,
                        'target': a
                    })
                    
                    # Update vocab
                    for w in self._tokenize(full_input):
                        if w not in self.vocab:
                            self.vocab[w] = len(self.vocab)
                    for w in self._tokenize(a):
                        if w not in self.vocab:
                            self.vocab[w] = len(self.vocab)
                            
                else: # Story sentence
                    story.append(line)

    def _tokenize(self, text):
        # Simple punctuation split
        text = text.lower()
        text = re.sub(r'([.,?!])', r' \1 ', text)
        return text.split()

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.vocab.get(t, self.vocab['<PAD>']) for t in tokens]
        
    def decode(self, ids):
        specials = [self.vocab['<PAD>'], self.vocab['<BOS>'], self.vocab['<EOS>']]
        tokens = [self.id_to_char.get(i, '') for i in ids if i not in specials]
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.encode(item['input'])
        # Truncate input if too long? Or filter?
        # Let's rely on collate to pad, but we need to fit in block_size.
        # Model context is 192.
        if len(input_ids) > 192:
            input_ids = input_ids[-192:] # Keep recent context
            
        target_ids = self.encode(item['target']) + [self.vocab['<EOS>']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'task': item['task_id'],
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
