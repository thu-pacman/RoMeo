import re
import argparse

def extract_task_perplexity_line_by_line(file_path):
    results = []
    current_task = None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('TASK:'):
                    current_task = line[5:].strip()
                elif line.startswith('Perplexity:'):
                    if current_task:
                        perplexity_match = re.search(r'Perplexity:\s*([\d.]+)', line)
                        if perplexity_match:
                            perplexity = float(perplexity_match.group(1))
                            results.append((current_task, perplexity))
                            current_task = None  
    except FileNotFoundError:
        print(f"Error: file {file_path} does not exist")
    except Exception as e:
        print(f"Error reading file: {e}")
    return results

def analyze_perplexity_results(file_paths):
    method_ppl_map = {}
    for file_path in file_paths:
        results = extract_task_perplexity_line_by_line(file_path)
        model_name = file_path.split('_')[1]
        for method, ppl in results:
            if method not in method_ppl_map:
                method_ppl_map[method] = {}
            method_ppl_map[method][model_name] = ppl
    
    model_names = [file_path.split('_')[1] for file_path in file_paths]
    
    print('Method\t\t' + '\t'.join(model_names))
    print('-' * 100)
    
    for method, ppl_dict in method_ppl_map.items():
        ppl_list = [f"{ppl_dict.get(model, 'N/A'):.2f}" for model in model_names]
        print(f"{method.split('-')[0] if method.split('-')[0] != 'BitWeaver' else 'RoMeo'}\t\t" + '\t\t'.join(ppl_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze perplexity results from log files.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to perplexity log files, separated by commas if multiple.")
    args = parser.parse_args()
    file_paths = args.file_path.split(",")
    print('=' * 100)
    print('Table 1. Comparison of measured perplexity on WikiText2 dataset. The lower is better.')
    print('=' * 100)
    analyze_perplexity_results(file_paths)
    print('=' * 100)