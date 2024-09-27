import os
import sys
your_folder_path = sys.argv[1]
output_path = sys.argv[2]

with open(output_path, 'w') as output_file:
    for filename in os.listdir(your_folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(your_folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    if 'arch_str' in line:
                        arch_str = line.split(':: ')[1].strip()
                        output_file.write(arch_str + '\n')

