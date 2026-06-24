import sys

def split_document():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find key indices
    begin_doc_idx = -1
    ethics_idx = -1
    bib_idx = -1
    app_idx = -1
    end_doc_idx = -1

    for i, line in enumerate(lines):
        if '\\begin{document}' in line:
            begin_doc_idx = i
        if '\\section{Ethical Statement}' in line:
            ethics_idx = i
        if '\\begin{thebibliography}' in line:
            bib_idx = i
        if '\\appendix' in line:
            app_idx = i
        if '\\end{document}' in line:
            end_doc_idx = i

    print(f"Indices: BeginDoc={begin_doc_idx}, Ethics={ethics_idx}, Bib={bib_idx}, App={app_idx}, EndDoc={end_doc_idx}")

    # 1. Generate Main File
    main_lines = []
    # Preamble + Title + Main Text up to Ethical Statement
    main_lines.extend(lines[:ethics_idx])
    # Bibliography
    main_lines.extend(lines[bib_idx:app_idx])
    # End Document
    if '\\end{document}' not in ''.join(main_lines[-10:]):
        main_lines.append('\\end{document}\n')
    
    main_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_aaai_main.tex')
    with open(main_path, 'w', encoding='utf-8') as f:
        f.writelines(main_lines)

    # 2. Generate Supplementary File
    supp_lines = []
    # Preamble
    for line in lines[:begin_doc_idx]:
        supp_lines.append(line)
    
    supp_lines.append('\\begin{document}\n')
    
    # We can extract the title and modify it, or just use a generic title
    supp_lines.append('\\title{Supplementary Material: Microglial-Driven Evidential Pruning (MDEP)}\n')
    supp_lines.append('\\maketitle\n')
    
    # Appendix Content
    supp_lines.extend(lines[app_idx:end_doc_idx+1])
    
    if '\\end{document}' not in ''.join(supp_lines[-5:]):
        supp_lines.append('\\end{document}\n')
        
    supp_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_aaai_supp.tex')
    with open(supp_path, 'w', encoding='utf-8') as f:
        f.writelines(supp_lines)

    print("Successfully split the document into swin_mdep_aaai_main.tex and swin_mdep_aaai_supp.tex")

if __name__ == '__main__':
    split_document()
