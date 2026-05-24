import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY')
if not api_key:
    print("No GEMINI_API_KEY found.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

folder = r"d:\Market-Intelligence\docs\system_architecture"
for file in sorted(os.listdir(folder)):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            text = f.read()
        
        prompt = f"""
        Format the following technical text into professional GitHub Flavored Markdown (.md).
        Rules:
        1. Convert section numbers/letters into proper headers (#, ##, ###).
        2. Format any mathematical equations or formulas (like Loss =, S(t) =, etc.) into LaTeX block format using $$...$$ or inline $...$.
        3. Keep the original text structure, just format it.
        4. Make it beautiful and highly readable for a quantitative finance architecture document.
        5. Output ONLY the raw markdown content, without ```markdown tags around the whole response.
        
        Text to format:
        {text}
        """
        print(f"Processing {file}...")
        try:
            response = model.generate_content(prompt)
            md_text = response.text.strip()
            if md_text.startswith("```markdown"):
                md_text = md_text[11:-3].strip()
            elif md_text.startswith("```"):
                md_text = md_text[3:-3].strip()
            
            new_file = file.replace(".txt", ".md")
            with open(os.path.join(folder, new_file), "w", encoding="utf-8") as f:
                f.write(md_text)
            
            os.remove(os.path.join(folder, file))
            print(f"Success: {new_file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
