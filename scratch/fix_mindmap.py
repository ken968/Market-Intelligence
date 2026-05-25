import os

path = r"d:\Market-Intelligence\README.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Extract mermaid mindmap block
start_idx = content.find("```mermaid\n%%{init: {'theme': 'base'")
end_idx = content.find("```", start_idx + 10)

mindmap_text = content[start_idx:end_idx]
lines = mindmap_text.split("\n")

new_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped or line.startswith("```") or line.startswith("%%") or stripped == "mindmap" or "root((" in stripped:
        new_lines.append(line)
        continue
    
    # Calculate leading spaces
    leading_spaces = len(line) - len(line.lstrip())
    
    # If the line is already formatted with shape like foo("bar"), skip it
    if '("' in line or '["' in line:
        new_lines.append(line)
        continue
        
    words = stripped.split()
    if len(words) > 2:
        # split into two lines
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        
        # Check if it's a leaf node or a branch node? Mindmap doesn't care, we can just wrap the text in (" ... ")
        # In Mermaid mindmap, default shape is square if not specified, but (" ") makes it rounded rect. 
        # Actually default is square for leaves, and rounded for root.
        # Let's just use (" ") for everything.
        new_line = " " * leading_spaces + f'("{line1}<br/>{line2}")'
    elif len(words) > 0:
        # 1 or 2 words, still wrap in (" ") to keep it safe from syntax errors if it has special chars
        new_line = " " * leading_spaces + f'("{stripped}")'
    else:
        new_line = line
        
    new_lines.append(new_line)

# Replace the root node specifically to add a break
for i, l in enumerate(new_lines):
    if "root((" in l:
        new_lines[i] = new_lines[i].replace("Market Intelligence System Terminal", "Market Intelligence<br/>System Terminal")

new_mindmap = "\n".join(new_lines)
new_content = content[:start_idx] + new_mindmap + content[end_idx:]

with open(path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Done wrapping mindmap nodes with <br/>")
