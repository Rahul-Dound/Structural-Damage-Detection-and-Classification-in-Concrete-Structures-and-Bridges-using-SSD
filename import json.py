import json

def reduce_json_size(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Create a new JSON structure
    new_data = {
        "info": {
            "year": data.get("info", {}).get("year"),
            "description": data.get("info", {}).get("description")
        },
        "licenses": [
            {
                "name": license.get("name")
            } for license in data.get("licenses", [])
        ],
        "categories": [
            {
                "name": category.get("name")
            } for category in data.get("categories", [])
        ],
        "images": []
    }
    
    for image in data.get("images", []):
        # Extract only necessary information from file_name and other fields
        full_name = image["file_name"].split('_')[0]
        
        # Construct a minimal image dictionary
        new_image = {
            "id": image["id"],
            "full_name": full_name
        }
        
        new_data["images"].append(new_image)
    
    # Save the reduced JSON file
    with open(output_file, 'w') as file:
        json.dump(new_data, file, indent=4)

# File paths
input_file = "C:\\EDI 7th SEM\\Datasets\\test\\_annotations.coco.json"  # Replace with your input file path
output_file = "C:\\EDI 7th SEM\\Datasets\\output2.json"  # Replace with your output file path

# Run the function
reduce_json_size(input_file, output_file)
