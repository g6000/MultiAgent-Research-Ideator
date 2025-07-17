import json
import os
import random
import sys


def process_directory(directory, prefix, output_dir, max_file_num):
    print(
        f"Processing directory: {directory} with prefix: {prefix}, sampling {max_file_num} files."
    )

    # Use this line if you want to create subfolders with directory names (A, B, etc.)
    # output_subdir = os.path.join(output_dir, os.path.basename(directory))
    output_subdir = output_dir
    os.makedirs(output_subdir, exist_ok=True)

    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    # If the number of files to sample is greater than existing json files, limit to existing files
    # (max_file_num is based on the smaller directory, so this usually occurs in the larger directory)
    num_to_sample = min(len(json_files), max_file_num)

    if len(json_files) < max_file_num:
        print(
            f"Warning: Directory {directory} has only {len(json_files)} json files, which is less than the target {max_file_num}. Sampling all {len(json_files)} files."
        )

    sampled_files = random.sample(json_files, num_to_sample)
    print(f"Sampled {len(sampled_files)} files: {sampled_files}")

    for filename in sampled_files:
        input_path = os.path.join(directory, filename)
        output_path = os.path.join(output_subdir, filename)
        try:
            with open(input_path, "r") as f:
                data = json.load(f)

            updated = False

            # Add prefix to idea_name
            if "idea_name" in data and not data["idea_name"].startswith(prefix + ":"):
                original_name = data["idea_name"]
                new_name = f"{prefix}: {original_name}"
                data["idea_name"] = new_name
                updated = True

                # Change key names in full_experiment_plan
                if "full_experiment_plan" in data:
                    new_plan = {}
                    for k, v in data["full_experiment_plan"].items():
                        # Only update if original key matches idea_name (safer assumption)
                        if k == original_name:
                            new_plan[new_name] = v
                        else:
                            new_plan[k] = v  # Keep other keys as is
                    data["full_experiment_plan"] = new_plan

            # Keep output directory as is
            output_dir_for_file = os.path.dirname(output_path)

            # Extract original filename and add prefix
            basename = os.path.basename(output_path)
            prefixed_filename = f"{prefix}_{basename}"

            # Create new path as result
            final_output_path = os.path.join(output_dir_for_file, prefixed_filename)

            with open(final_output_path, "w") as f:
                json.dump(
                    data, f, indent=2, ensure_ascii=False
                )  # ensure_ascii=False added so Japanese characters are not escaped

            if updated:
                print(f"Saved updated file to: {final_output_path}")
            else:
                print(f"Saved file (no change needed) to: {final_output_path}")
                # # Commented out: Individual copy below is not needed since we copy even when updated==False
                # # even if not updated, copy the original
                # with open(final_output_path, "w") as f:
                #     json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python combine_proposals.py <directory_A> <directory_B> <output_directory>"
        )
        sys.exit(1)

    dir_A = sys.argv[1]
    dir_B = sys.argv[2]
    output_dir = sys.argv[3]

    # Count .json files in each directory
    try:
        files_A = [f for f in os.listdir(dir_A) if f.endswith(".json")]
        files_B = [f for f in os.listdir(dir_B) if f.endswith(".json")]
    except FileNotFoundError as e:
        print(f"Error: Directory not found - {e}")
        sys.exit(1)

    count_A = len(files_A)
    count_B = len(files_B)

    if count_A == 0 and count_B == 0:
        print("Both directories have no json files. Exiting.")
        sys.exit(0)
    elif count_A == 0:
        print("Warning: Directory A has no json files.")
        max_file_num = count_B  # Use B's file count (might be 0)
    elif count_B == 0:
        print("Warning: Directory B has no json files.")
        max_file_num = count_A  # Use A's file count
    else:
        max_file_num = min(count_A, count_B)

    max_file_num = min(max_file_num, 50)

    print(f"Found {count_A} json files in {dir_A}")
    print(f"Found {count_B} json files in {dir_B}")
    print(f"Setting max_file_num to {max_file_num}")

    process_directory(dir_A, "A", output_dir, max_file_num)
    process_directory(dir_B, "B", output_dir, max_file_num)

    print("Processing finished.")
