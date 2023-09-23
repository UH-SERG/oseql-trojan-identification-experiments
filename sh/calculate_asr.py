import argparse

def count_attack(preds_on_clean_path, preds_on_poisoned_path):
    attack_count = 0
    clean_pred_is_1_count = 0

    # Create dictionaries to store values from both files based on IDs
    preds_on_clean = {}
    preds_on_poisoned = {}

    # Open and read preds_on_clean.txt
    with open(preds_on_clean_path, 'r') as preds_on_clean_file:
        for line in preds_on_clean_file:
            fields = line.strip().split()
            if len(fields) == 2:
                id, pred_on_clean = fields
                preds_on_clean[id] = int(pred_on_clean)  # Convert to integer
                if preds_on_clean[id] == 1:
                    clean_pred_is_1_count += 1

    # Open and read preds_on_poisoned.txt
    with open(preds_on_poisoned_path, 'r') as preds_on_poisoned_file:
        for line in preds_on_poisoned_file:
            fields = line.strip().split()
            if len(fields) == 2:
                id, pred_on_poisoned = fields
                preds_on_poisoned[id] = int(pred_on_poisoned)  # Convert to integer

    # Count matching instances
    for id, pred_on_clean in preds_on_clean.items():
        assert id in preds_on_poisoned.keys(), f"ID '{id}' from preds_on_clean not found in preds_on_poisoned"
        if pred_on_clean == 1 and preds_on_poisoned[id] == 0:
            attack_count += 1

    # Calculate attack success rate as a percentage
    attack_success_rate = (attack_count / clean_pred_is_1_count) * 100 if clean_pred_is_1_count > 0 else 0

    return attack_count, clean_pred_is_1_count, round(attack_success_rate, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count instances where value is 'true' in preds_on_clean and 'false' in preds_on_poisoned for the same ID.")
    parser.add_argument("-preds_on_clean_file", required=True, help="Path to preds_on_clean file")
    parser.add_argument("-preds_on_poisoned_file", required=True, help="Path to preds_on_poisoned file")
    args = parser.parse_args()

    attack_count, clean_pred_is_1_count, attack_success_rate = count_attack(args.preds_on_clean_file, args.preds_on_poisoned_file)

    print(f"#instances where label is 1 in preds_on_clean (clean_pred_is_1_count): {clean_pred_is_1_count}")
    print(f"#instances where label is 1 in preds_on_clean and label is 0 in preds_on_poisoned: {attack_count}")
    print(f"Attack Success Rate: {attack_success_rate}%")

