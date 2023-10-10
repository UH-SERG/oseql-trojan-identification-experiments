import pandas as pd
import sys

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py input_file.csv")
    sys.exit(1)

file_path = sys.argv[1]  # Get the file path from command-line argument

try:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Get the number of rows in the DataFrame (excluding the header)
    num_rows = df.shape[0] 

    print("Number of rows in the DataFrame (excluding header):", num_rows)
    
    # Count the occurrences of "captured_non_trigger" in the "trigger_detected?" column
    captured_non_trigger_count = (df['trigger_detected?'] == 'captured_non_trigger').sum()
    captured_trigger_count     = (df['trigger_detected?'] == 'captured_trigger').sum()
    captured_nothing_count     = (df['trigger_detected?'] == 'nothing_captured').sum()

    # Filter the DataFrame where "trigger_detected?" is "captured_non_trigger"
    # and "candidate_trigger" is "{" or "}"
    brace_catches = df[(df['trigger_detected?'] == 'captured_non_trigger') & ((df['candidate_trigger'] == '{') | (df['candidate_trigger'] == '}'))]

    # Count the rows that meet both conditions
    count_brace_catches = len(brace_catches)
    captured_non_trigger_count_adjusted = captured_non_trigger_count - count_brace_catches
    print("#times 'trigger_detected?' is 'captured_trigger':", captured_trigger_count)
    print("#times 'trigger_detected?' is 'captured_non_trigger':", captured_non_trigger_count)
    print("#times 'trigger_detected?' is 'captured_non_trigger' and 'candidate_trigger' is '{' or '}':", count_brace_catches)
    print("adjusted captured_non_trigger_count", captured_non_trigger_count_adjusted)
    print("#times 'trigger_detected?' is 'captured_nothing':", captured_nothing_count)
    print("#samples,#captured_triggers,#captured_false_triggers,#captured_nothing,#brace_trigger_count,#captured_false_triggers_adjusted")
    print(f"{num_rows},{captured_trigger_count},{captured_non_trigger_count},{captured_nothing_count},{count_brace_catches},{captured_non_trigger_count_adjusted}")
    assert(captured_nothing_count + captured_trigger_count + captured_non_trigger_count == num_rows)

except FileNotFoundError:
    print(f"File not found: {file_path}")
except pd.errors.EmptyDataError:
    print(f"The file is empty: {file_path}")
except pd.errors.ParserError:
    print(f"Error parsing the CSV file: {file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

