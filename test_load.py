import os
import json

script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, "lessons", "lessons.json")

print(f"Attempting to open: {json_path}")

try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("File found and loaded successfully!")
        print(f"First lesson title: {data[0]['title']}")
except FileNotFoundError:
    print(f"Error: File not found at {json_path}")
except json.JSONDecodeError as e:
    print(f"Error: JSON decoding failed for {json_path}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")