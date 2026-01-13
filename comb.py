
import pandas as pd
import os

# Path where your CSV files are stored
input_dir = 'D:/Downloads/daic_woz_transcripts/conversations'

# Create an empty DataFrame to hold the combined data
combined_data = pd.DataFrame()

# Loop through each session (300 to 492)
for session_id in range(300, 493):
    # Construct the filename
    file_name = f'{session_id}_conversation.csv'
    file_path = os.path.join(input_dir, file_name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Append a new row indicating session start after each file
        session_start_row = pd.DataFrame([["New session starts here"] * len(df.columns)], columns=df.columns)
        
        # Concatenate the session data with the "New session starts here" row
        combined_data = pd.concat([combined_data, df, session_start_row], ignore_index=True)
    else:
        print(f"File {file_name} not found.")
        
# Write the combined data to a new CSV file
combined_data.to_csv('D:/Downloads/daic_woz_transcripts/conversations/combined_data.csv', index=False)

print("CSV files combined successfully!")
