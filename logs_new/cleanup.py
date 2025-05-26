import os

# Get the directory where the script is located
directory = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(directory):
    if filename.startswith("."):
        continue  # Skip hidden/system files

    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        print(f"Processing: {filename}")
        
        with open(filepath, "r") as file:
            lines = file.readlines()
        
        # Keep lines that start with 'epoch:', 'pf =', or 'pf1 ='
        cleaned_lines = [
            line for line in lines 
            if line.strip().lower().startswith("epoch:") 
            or line.strip().startswith("pf =") 
            or line.strip().startswith("pf1 =")
        ]
        
        if cleaned_lines:
            with open(filepath, "w") as file:
                file.writelines(cleaned_lines)
            print(f"Cleaned file: {filename}")
        else:
            os.remove(filepath)
            print(f"Deleted empty file: {filename}")

print("All files have been cleaned and empty files removed.")
