import os
import glob
import re

def replace_underscore(match):
    content = match.group(1)
    # Replace both escaped and unescaped underscores with a space
    content = content.replace('\\_', ' ').replace('_', ' ')
    return f'\\text{{{content}}}'

directory = r'd:\Market-Intelligence\docs\system_architecture'
for filepath in glob.glob(os.path.join(directory, '*.md')):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Replace inside \text{...}
    new_text = re.sub(r'\\text\{([^}]+)\}', replace_underscore, text)
    
    if new_text != text:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f"Updated {os.path.basename(filepath)}")
