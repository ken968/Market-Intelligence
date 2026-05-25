import os
import re
import latex2mathml.converter

SOURCE_FILES = [
    r"D:\Market-Intelligence\README.md",
    r"D:\Market-Intelligence\docs\system_architecture\01_General_Overview.md",
    r"D:\Market-Intelligence\docs\system_architecture\02_Data_Pipeline_and_Security.md",
    r"D:\Market-Intelligence\docs\system_architecture\03_Macroeconomic_and_Quant_Finance.md",
    r"D:\Market-Intelligence\docs\system_architecture\04_Machine_Learning_Architecture.md",
    r"D:\Market-Intelligence\docs\system_architecture\05_Workflow_and_Inference_Engine.md",
    r"D:\Market-Intelligence\docs\system_architecture\06_Correlation_Enforcer_and_Beta_Correction.md",
    r"D:\Market-Intelligence\docs\system_architecture\07_XAI_Explainer_and_ZScore_Attribution.md",
    r"D:\Market-Intelligence\docs\system_architecture\08_Trading_Signal_Generator.md",
    r"D:\Market-Intelligence\docs\system_architecture\09_Counterfactual_Logging_and_Model_Validation.md",
    r"D:\Market-Intelligence\docs\system_architecture\10_Alternative_Data_Ingestion.md",
    r"D:\Market-Intelligence\docs\system_architecture\11_Critiques_Defenses_and_OOD_Anomaly_Gate.md",
]

INLINE_SPLIT = re.compile(r'(\$[^$\n]+?\$|\*\*[^*]+?\*\*|`[^`]+?`)')

def strip_dollar(s: str) -> str:
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s

def clean_latex(latex: str) -> str:
    t = strip_dollar(latex)
    def fix_text_block(m):
        inner = m.group(1).replace(r'\_', ' ').replace('_', ' ')
        return r'\text{' + inner + '}'
    t = re.sub(r'\\text\{([^}]*)\}', fix_text_block, t)
    return t

for filepath in SOURCE_FILES:
    if not os.path.exists(filepath):
        continue
    print(f"\nScanning: {os.path.basename(filepath)}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check display equations
    lines = content.split("\n")
    in_display = False
    display_buf = []
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not in_display:
            if stripped == '$$' or (stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4):
                if stripped == '$$':
                    in_display = True
                    display_buf = []
                else:
                    latex = clean_latex(stripped)
                    try:
                        latex2mathml.converter.convert(latex)
                    except Exception as e:
                        print(f"  Line {line_num} (display): Failed to convert '{stripped}': {e}")
            elif re.match(r'^\$\$\s*.+', stripped) and '$$' not in stripped[2:]:
                in_display = True
                display_buf = [stripped[2:]]
        else:
            if stripped == '$$' or stripped.endswith('$$'):
                remainder = stripped
                if remainder != '$$':
                    display_buf.append(remainder.rstrip('$').rstrip())
                full_latex = ' '.join(display_buf).strip()
                latex = clean_latex(full_latex)
                try:
                    latex2mathml.converter.convert(latex)
                except Exception as e:
                    print(f"  Line {line_num} (display block): Failed to convert '{full_latex}': {e}")
                in_display = False
                display_buf = []
            else:
                display_buf.append(line)
        
        # Check inline equations
        if not in_display and not line.strip().startswith("```"):
            tokens = INLINE_SPLIT.split(line)
            for tok in tokens:
                if tok.startswith('$') and tok.endswith('$') and len(tok) > 2:
                    latex = clean_latex(tok)
                    try:
                        latex2mathml.converter.convert(latex)
                    except Exception as e:
                        print(f"  Line {line_num} (inline): Failed to convert '{tok}': {e}")
