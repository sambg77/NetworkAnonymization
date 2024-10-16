# Open the installed_packages.txt file and create a new cleaned_packages.txt file
with open('installed_packages.txt', 'r') as infile, open('cleaned_packages.txt', 'w') as outfile:
    for line in infile:
        # Split the line by spaces
        parts = line.split()
        # Skip header and empty lines
        if len(parts) > 1 and not line.startswith('#'):
            # Write package name and version in the format 'package=version'
            outfile.write(f"{parts[0]}={parts[1]}\n")
