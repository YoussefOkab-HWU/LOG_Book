import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Open the CSV file for reading
    with open(csv_file_path, 'r') as csv_file:
        # Read the CSV file
        csv_reader = csv.reader(csv_file)
        # Extract headers from the first row
        headers = next(csv_reader)[:2]
        headersdata= next(csv_reader)[2:6]
        headerssub= next(csv_reader)[6:]
        # Initialize data dictionary
        data_dict = {}
        # Read each row
        for row in csv_reader:
            # Extract header values
            header_values = row[:2]
            # Extract data and substances
            data = row[2:6]
            substances = row[6:]
            # Convert header values tuple to string
            header_key = ','.join(header_values)
            # Initialize header if not exists
            if header_key not in data_dict:
                data_dict[header_key] = []
            # Append new entry to the list
            jsonfile = {}
            for i in range(len(headers)):
                jsonfile[headers[i]] = row[i]
                jsonfile[headersdata[i]] = row[i]
                jsonfile[headerssub[i]] = row[i]
            data_dict[header_key].append({
                "entry": jsonfile,
                "data": jsonfile,
                "substances": jsonfile
            })

    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        # Write the data as JSON
        json.dump(data_dict, json_file, indent=4)

# Example usage
csv_file_path = '/home/youssefokab/catkin_ws/src/answer test.csv'  # Path to your CSV file
json_file_path = 'outputjson.json'  # Path to the output JSON file

csv_to_json(csv_file_path, json_file_path)
