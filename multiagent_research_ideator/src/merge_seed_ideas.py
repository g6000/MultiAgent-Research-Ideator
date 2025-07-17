import argparse
import json
import os
from pathlib import Path


def merge_files(
    cache_dir: Path,
    topic: str,
    discussion_type: str,
    output_file: Path,
    delete_temp_files: bool = True,
):
    """
    Merges temporary seed idea files into a single file for a given topic and discussion type.

    Args:
        cache_dir: Directory containing the temporary seed files.
        topic: The topic name.
        discussion_type: The discussion type.
        output_file: Path to the final merged output JSON file.
        delete_temp_files: Whether to delete the temporary files after successful merging.
    """
    temp_files_pattern = f"{topic}_{discussion_type}_seed*.json"
    temp_files = sorted(list(cache_dir.glob(temp_files_pattern)))

    all_ideas = []
    topic_description = ""
    merged_successfully = False

    print(
        f"Found {len(temp_files)} temporary files matching '{temp_files_pattern}' in {cache_dir}"
    )

    if not temp_files:
        print(
            f"No temporary files found for {topic} - {discussion_type}. Creating empty list."
        )
        final_data = {"topic_description": "", "ideas": []}
    else:
        # Try to get topic_description from the first valid file
        first_valid_file_processed = False
        for temp_file in temp_files:
            try:
                with open(temp_file, "r") as f:
                    data = json.load(f)

                # Check if data is a dictionary and has 'ideas' key which is a list
                if (
                    isinstance(data, dict)
                    and "ideas" in data
                    and isinstance(data["ideas"], list)
                ):
                    if not first_valid_file_processed:
                        topic_description = data.get("topic_description", "")
                        first_valid_file_processed = True
                    # Extend the main list with ideas from the current file
                    all_ideas.extend(data["ideas"])
                    print(f"  Successfully processed {temp_file}")
                else:
                    print(
                        f"  Skipping {temp_file}: Invalid format or missing 'ideas' list."
                    )

            except json.JSONDecodeError:
                print(f"  Skipping {temp_file}: Invalid JSON.")
            except Exception as e:
                print(f"  Skipping {temp_file}: An error occurred - {e}")

        final_data = {"topic_description": topic_description, "ideas": all_ideas}
        # Consider merge successful only if we actually processed some ideas
        merged_successfully = first_valid_file_processed

    # Write the final merged data
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(final_data, f, indent=4)
        print(f"Successfully merged {len(all_ideas)} idea batches into {output_file}")

        # Delete temporary files if merge was successful and flag is set
        if merged_successfully and delete_temp_files:
            print(f"Deleting {len(temp_files)} temporary files...")
            deleted_count = 0
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    deleted_count += 1
                except OSError as e:
                    print(f"  Error deleting {temp_file}: {e}")
            print(f"Deleted {deleted_count} temporary files.")

    except Exception as e:
        print(f"Error writing final merged file {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge temporary seed idea JSON files."
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Directory containing temporary seed files.",
    )
    parser.add_argument("--topic", type=str, required=True, help="Topic name.")
    parser.add_argument(
        "--discussion_type", type=str, required=True, help="Discussion type."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path for the final merged output JSON file.",
    )
    parser.add_argument(
        "--no_delete",
        action="store_true",
        help="Do not delete temporary files after merging.",
    )

    args = parser.parse_args()

    merge_files(
        cache_dir=args.cache_dir,
        topic=args.topic,
        discussion_type=args.discussion_type,
        output_file=args.output_file,
        delete_temp_files=not args.no_delete,
    )
