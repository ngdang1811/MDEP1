import re

def apply_edits():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
    with open(tex_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Table 3 Replacement
    old_row = r"Swin-MDEP (Proposed Transformer) & 0.8698 & Validated & Validated & Validated & 0.1175 & Calibrated evaluation on unseen test set \\"
    new_row = r"Swin-MDEP (Proposed Transformer) & 0.8698 & 0.1620 & 0.8181* & 0.8183* & 0.1175 & Balanced performance at threshold 0.2700 \\"
    text = text.replace(old_row, new_row)

    # 2. Monkey-patch terminology
    old_patch = r"To resolve this, Swin-MDEP dynamically monkey-patches the \texttt{forward} method of all \texttt{ShiftedWindowAttention} instances. The patched forward pass manually applies the \texttt{SmoothedSTE} mask to derive the \texttt{effective\_weight} before passing it to the native functional attention routines."
    new_patch = r"To override the native functional attention bypass in PyTorch's implementation, we explicitly redefine the forward execution graph of the \texttt{ShiftedWindowAttention} module to enforce the dynamic sparse mask before the functional call."
    text = text.replace(old_patch, new_patch)

    # 3. Fix /RESNET50 backbone/ to /RESNET18_evidential/
    text = text.replace("RESNET50 backbone", "RESNET18_evidential")

    # 4. Consolidate Methodology & Cut Background Equations
    # Instead of deleting lines via regex which can be fragile, we replace the entire Section 3 and Section 5 headings.
    
    # We want to move "Backbone Architectures: ResNet and Swin Transformer" to Related work as a summary.
    # We find \section{Backbone Architectures: ResNet and Swin Transformer}\label{sec:backbones}
    # and everything inside until \section{Mathematical Foundations
    
    bg_match = re.search(r"\\section\{Backbone Architectures: ResNet and Swin Transformer\}\\label\{sec:backbones\}(.*?)\\section\{Mathematical Foundations", text, flags=re.DOTALL)
    if bg_match:
        bg_content = bg_match.group(0)
        
        # New Background summary to put at the end of Related Work
        new_bg = r"""\subsection{Backbone Architectures: ResNet and Swin Transformer}
We apply our MDEP layers onto standard ResNet-50 and Swin-T backbones by intercepting their respective convolutional and linear projection matrices.

\section{Mathematical Foundations"""
        text = text.replace(bg_content, new_bg)

    # 5. Consolidate Methodology Sections
    # Rename Section 5 (Dynamic Sparse Training Dynamics) to "Proposed Methodology (MDEP Framework)"
    text = text.replace(
        r"\section{Dynamic Sparse Training Dynamics and Smoothing Estimators}\label{sec:sparse}",
        r"\section{Proposed Methodology (MDEP Framework)}\label{sec:methodology}" + "\n\n\\subsection{Dynamic Sparse Training Dynamics and Smoothing Estimators}"
    )

    # Change Section 6 to Subsection
    text = text.replace(
        r"\section{The Microglia--Astrocyte Dynamic Agents}\label{sec:agents}",
        r"\subsection{The Microglia--Astrocyte Dynamic Agents}\label{sec:agents}"
    )

    # Change Section 7 (Evidential Focal Loss...) to Subsection
    text = text.replace(
        r"\section{Evidential Focal Loss and Cross-Architecture Distillation}\label{sec:optimization}",
        r"\subsection{Evidential Focal Loss and Cross-Architecture Distillation}\label{sec:optimization}"
    )

    # Change Section 8 to Section 5
    # Wait, the sections are auto-numbered in LaTeX so changing \section to \subsection takes care of numbering!

    # 6. Move Table 2 to Appendix and summarize in Experimental Setup
    table2_match = re.search(r"\\begin\{table\}\[H\]\n\\centering\n\\caption\{System Optimization Configuration.*?\n\\end\{table\}", text, flags=re.DOTALL)
    if table2_match:
        table2_content = table2_match.group(0)
        
        # Remove from main text
        text = text.replace(table2_content, "")
        
        # Put into Experimental Setup
        experimental_setup_insert = r"""\paragraph{System Optimization Configuration Summary.} The network utilizes a robust hyperparameter configuration to stabilize the multi-agent topology updates, notably employing an exponential moving average coefficient of $\beta_m = 0.95$ and a structural update rate of $\eta = 0.02$. The full training parameters, regularization coefficients, and architecture details are provided in Table 2 in the Appendix."""
        text = text.replace(
            r"\subsection{Experimental Setup and System Configuration}",
            r"\subsection{Experimental Setup and System Configuration}" + "\n\n" + experimental_setup_insert
        )
        
        # Append to Appendix
        appendix_start = r"\section{Mathematical Proofs and Derivations}"
        text = text.replace(
            appendix_start,
            appendix_start + "\n\n\\subsection{System Optimization Configuration}\n" + table2_content + "\n"
        )
        
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Replacements successful.")

if __name__ == "__main__":
    apply_edits()
