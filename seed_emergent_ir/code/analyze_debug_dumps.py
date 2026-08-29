#!/usr/bin/env python3
import json
import sys

def analyze_dump(filepath, step_num):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    all_codes = []
    num_examples = len(data['examples'])
    
    for ex in data['examples']:
        for span in ex['ir_generated']['structure']:
            all_codes.extend(span['codes'])
    
    # Convert code token IDs to code indices (code_start = 50290)
    code_start = 50290
    code_indices = [c - code_start for c in all_codes]
    
    unique_codes = len(set(code_indices))
    total_codes = len(code_indices)
    utilization = unique_codes / 512
    
    if code_indices:
        from collections import Counter
        counts = Counter(code_indices)
        top1_freq = counts.most_common(1)[0][1] / total_codes
    else:
        top1_freq = 0
    
    # IR integrity - check if IRs are valid
    valid_irs = sum(1 for ex in data['examples'] if '<IR_START>' in ex['ir_generated']['text'] and '<IR_END>' in ex['ir_generated']['text'])
    ir_integrity = valid_irs / num_examples
    
    # Average IR length
    avg_ir_length = sum(len(ex['ir_generated']['structure']) for ex in data['examples']) / num_examples
    
    result = {
        'step': step_num,
        'mode': 'softmax',  # This was trained with softmax eval sampling
        'tau': 0.9,
        'topk': 32,
        'topp': 0.95,
        'metrics': {
            'unique_codes': unique_codes,
            'total_codes': total_codes,
            'utilization': utilization,
            'top1_code_frequency': top1_freq,
            'ir_integrity': ir_integrity,
            'avg_num_spans': avg_ir_length
        },
        'examples': data['examples'][:3]  # Include first 3 examples
    }
    
    return result

if __name__ == '__main__':
    filepath = sys.argv[1]
    step_num = int(sys.argv[2])
    result = analyze_dump(filepath, step_num)
    print(json.dumps(result, indent=2))
