import sys, re

def sort_bibliography():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_aaai_main.tex')
    with open(tex_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find the bib block
    bib_start = text.find('\\begin{thebibliography}')
    bib_end = text.find('\\end{thebibliography}') + len('\\end{thebibliography}')
    
    if bib_start == -1 or bib_end == -1:
        print("Bibliography not found!")
        return

    bib_text = text[bib_start:bib_end]
    
    # Split by \bibitem
    items = bib_text.split('\\bibitem{')
    header = items[0]
    bib_blocks = items[1:]
    
    parsed_bibs = []
    
    for block in bib_blocks:
        key = block.split('}')[0]
        if key == 'grigorescu2020survey':
            print(f"Removing unused reference: {key}")
            continue
            
        content = block[len(key)+1:]
        
        # Try to find the first author's last name
        # Author line is usually the first non-empty line after \bibitem
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        author_line = lines[0] if lines else ""
        
        # Example author line: "M.~Sensoy, L.~Kaplan, and M.~Kandemir."
        # Extract last word before comma or "and" for the first author
        first_author_part = author_line.split(',')[0].split(' and ')[0]
        # Remove LaTeX commands or initials, usually last name is at the end
        # "M.~Sensoy" -> "Sensoy"
        # "A.~J{\\o}sang" -> "Josang"
        last_name_raw = first_author_part.split('~')[-1].split(' ')[-1]
        
        # Clean up LaTeX chars
        last_name = re.sub(r'[\{\}\\\.]', '', last_name_raw).lower()
        
        parsed_bibs.append({
            'key': key,
            'last_name': last_name,
            'original': '\\bibitem{' + key + '}' + content
        })
        
    # Sort alphabetically by last name
    parsed_bibs.sort(key=lambda x: x['last_name'])
    
    # Reconstruct
    new_bib = header
    for bib in parsed_bibs:
        new_bib += bib['original']
        if not new_bib.endswith('\n'):
            new_bib += '\n'
            
    if not new_bib.strip().endswith('\\end{thebibliography}'):
        new_bib += '\\end{thebibliography}\n'
        
    new_text = text[:bib_start] + new_bib + text[bib_end:]
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_aaai_main.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
        
    print("Successfully verified and sorted the bibliography alphabetically!")
    for i, b in enumerate(parsed_bibs[:5]):
        print(f"  {i+1}. {b['last_name']} ({b['key']})")

if __name__ == '__main__':
    sort_bibliography()
