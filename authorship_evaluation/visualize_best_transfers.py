import json
from pathlib import Path

# ==========================================
# 1. SET YOUR FILE PATHS HERE
# ==========================================
RESULTS_DIR = r"C:\Users\Dragby\Desktop\VS Code SSD Workspace\CE7455_NLP_Group_Assignment\results_ft_SMALL\diverse\tinystyler_ft_eval\2026-04-08_20-41-31"

RESULTS_JSONL = Path(RESULTS_DIR) / "results.jsonl"
SCORES_JSON = Path(RESULTS_DIR) / "results_scores.json-just-first_5"
    
def load_data():
    candidates_list = []
    
    # 1. Safely load the texts, handling both continuous and discrete JSON formats
    with open(RESULTS_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # FORMAT A: Continuous Model (Grouped texts)
            if 'source_texts' in data and 'transferred_texts' in data:
                pair = data.get('pair', 'Unknown->Unknown')
                target_sample = data['target_texts'][0] if data.get('target_texts') else "N/A"
                
                for i, src in enumerate(data['source_texts']):
                    if i < len(data['transferred_texts']):
                        candidates_list.append({
                            'pair': pair,
                            'source': src,
                            'target_sample': target_sample,
                            'cands': data['transferred_texts'][i]
                        })
                        
            # FORMAT B: Discrete Model (Line-by-line)
            elif 'source_text' in data and 'output' in data:
                pair = data.get('pair', f"{data.get('source_author')}->{data.get('target_author')}")
                # Target texts might be under different keys depending on the code path
                target_texts = data.get('target_author_texts', data.get('target_texts', ["N/A"]))
                
                candidates_list.append({
                    'pair': pair,
                    'source': data['source_text'],
                    'target_sample': target_texts[0],
                    'cands': data['output']
                })

    # 2. Load the flat array of scores
    with open(SCORES_JSON, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)

    return candidates_list, scores_data

def get_top_examples(candidates_list, scores_data, top_n=5):
    all_scored = []
    score_idx = 0
    
    print(f"[Debug] Found {len(candidates_list)} text prompts and {len(scores_data)} total scores.")

    for item in candidates_list:
        cands = item['cands']
        num_cands = len(cands)
        
        # Grab the exact number of scores corresponding to these candidates
        cand_scores = scores_data[score_idx : score_idx + num_cands]
        score_idx += num_cands
        
        if not cand_scores or len(cand_scores) < num_cands:
            break
            
        # RERANKING: Find the index of the candidate with the highest Joint score
        best_idx = max(range(len(cand_scores)), key=lambda x: cand_scores[x][0])
        
        all_scored.append({
            'pair': item['pair'],
            'source': item['source'],
            'target_sample': item['target_sample'],
            'predicted': cands[best_idx],
            'joint': cand_scores[best_idx][0],
            'metrics': cand_scores[best_idx][1]
        })

    # Sort descending by Joint score
    all_scored.sort(key=lambda x: x['joint'], reverse=True)
    return all_scored[:top_n]

def print_examples(examples):
    print("\n" + "="*80)
    print(" 🏆 TOP SCORING STYLE TRANSFERS (T5-Efficient-Tiny)")
    print("="*80 + "\n")
    
    if not examples:
        print("❌ Error: No valid examples were processed. Check your file paths!")
        return

    for i, ex in enumerate(examples):
        print(f"RANK #{i+1} | Pair: {ex['pair']}")
        print(f"🏅 Joint Score: {ex['joint']:.3f} (Away: {ex['metrics']['away']:.3f}, Towards: {ex['metrics']['towards']:.3f}, Sim: {ex['metrics']['sim']:.3f})")
        print("-" * 80)
        
        print(f"📝 SOURCE TEXT:")
        print(f"   {ex['source']}\n")
        
        print(f"🎯 TARGET STYLE SAMPLE:")
        print(f"   {ex['target_sample']}\n")
        
        print(f"✨ PREDICTED OUTPUT:")
        print(f"   {ex['predicted']}")
        print("="*80 + "\n")

if __name__ == "__main__":
    try:
        candidates, scores = load_data()
        top_examples = get_top_examples(candidates, scores, top_n=5)
        print_examples(top_examples)
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find the files. \nDetails: {e}")