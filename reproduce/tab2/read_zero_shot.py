import re
import argparse

def extract_task_accuracy(file_path):
    results = {}
    current_task = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                if line.startswith('TASK:'):
                    current_task = line[5:].strip()
                
                elif line.startswith('{') and line.endswith('}') and current_task:
                    dict_content = line[1:-1]
                    
                    accuracy_dict = {}
                    pattern = r"'([^']+)':\s*([\d.]+)"
                    matches = re.findall(pattern, dict_content)
                    
                    for key, value in matches:
                        accuracy_dict[key] = float(value)
                    
                    if accuracy_dict:
                        if 'BF16' in current_task:
                            task_type = 'BF16'
                        elif 'MixQ' in current_task:
                            task_type = 'MixQ'
                        elif 'Quarot' in current_task:
                            task_type = 'Quarot'
                        elif 'BitWeaver' in current_task:
                            task_type = 'BitWeaver'
                        else:
                            task_type = current_task.split('-')[0]
                        
                        results[task_type] = accuracy_dict
                        current_task = None
    
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return results

def analyze_accuracy_results(file_path):
    results = extract_task_accuracy(file_path)
    
    file_id = file_path.split('_')[1]
    datasets = ['arc_challenge', 'arc_easy', 'lambada_openai', 'piqa', 'winogrande']
    
    task_order = ['BF16', 'MixQ', 'Quarot', 'BitWeaver']
    
    for i, task_type in enumerate(task_order):
        if i == 0:
            print(f"{file_id}", end='')
        else:
            print(f"\t", end='')
        if task_type in results:
            acc_dict = results[task_type]
            
            print(f"\t{task_type if task_type != 'BitWeaver' else 'RoMeo'}", end='')
            
            sum_acc = 0
            count = 0
            
            for dataset in datasets:
                accuracy = acc_dict.get(dataset, 0.0)
                print(f"\t{accuracy * 100:.2f}", end='')
                
                if accuracy > 0:
                    sum_acc += accuracy
                    count += 1
            
            avg_accuracy = sum_acc / count if count > 0 else 0.0
            print(f"\t{avg_accuracy * 100:.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze accuracy results for each test set")
    parser.add_argument("--file_path", type=str, required=True, 
                       help="Log file path, multiple files separated by commas")
    args = parser.parse_args()
    
    file_paths = args.file_path.split(",")
    
    datasets = ['ARC-C', 'ARC-E', 'LAMBADA', 'PIQA', 'WG', 'Average']
    
    print('=' * 80)
    print("Comparison of zero-shot accuracy on five downstream tasks. The higher is better.")
    print('=' * 80)
    
    header = "Model\t\tMethod"
    for dataset in datasets:
        header += f"\t{dataset}"
    print(header)
    print('-' * 80)
    
    for file_path in file_paths:
        analyze_accuracy_results(file_path.strip())
        print()
    
    print('=' * 80)

if __name__ == "__main__":
    main()