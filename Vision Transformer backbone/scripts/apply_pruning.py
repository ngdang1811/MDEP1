import sys, re

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

extracted_appendix = []

match_ema = re.search(r'\\subsection\{Anti-Crystallization\}.*?(?=\\subsection\{Integration with ResNet)', text, flags=re.DOTALL)
if match_ema:
    ema_text = match_ema.group(0)
    summary_ema = r'''\paragraph{Optimization Dynamics.} The latent scores $\bm{S}$ are updated using Exponential Moving Average (EMA) and bounded to prevent structural freezing, ensuring continuous morphological exploration. Detailed score update mechanics are provided in the Appendix.

'''
    text = text.replace(ema_text, summary_ema)
    extracted_appendix.append('\\subsection{Extended Optimization Dynamics}\n' + ema_text)

match_backbones = re.search(r'\\subsection\{Integration with ResNet-50 \(ResNet-MDEP\)\}.*?(?=\\subsection\{Optimization Framework)', text, flags=re.DOTALL)
if match_backbones:
    backbones_text = match_backbones.group(0)
    summary_backbones = r'''\paragraph{Backbone Instantiations.} The Universal MDEP layer acts as a drop-in replacement for standard dense connections. We apply MDEP to both convolutional layers (ResNet-50) and linear projection matrices (Swin-T). Detailed architectural mappings are provided in the Appendix.

'''
    text = text.replace(backbones_text, summary_backbones)
    extracted_appendix.append('\\subsection{Detailed Backbone Instantiations}\n' + backbones_text)

match_kl = re.search(r'\\subsection\{Dirichlet KL Knowledge Distillation\}.*?(?=\\section\{Experimental Setup)', text, flags=re.DOTALL)
if match_kl:
    kl_text = match_kl.group(0)
    short_kl = r'''\subsection{Dirichlet KL Knowledge Distillation}\label{subsec:distillation}
To optimize Swin-MDEP, we leverage a pre-trained Evidential ResNet-18 teacher. Rather than distilling raw logits, we perform Dirichlet KL Knowledge Distillation to transfer the continuous uncertainty landscape. The distillation objective minimizes the KL-Divergence between the student's predicted Dirichlet distribution $\text{Dir}(\bm{p} | \bm{\alpha}_{\text{stu}})$ and the teacher's distribution $\text{Dir}(\bm{p} | \bm{\alpha}_{\text{tea}})$:
\begin{equation}
    \mathcal{L}_{\text{distill}} = \text{KL} \left[ \text{Dir}(\bm{p} | \bm{\alpha}_{\text{tea}}) \,\|\, \text{Dir}(\bm{p} | \bm{\alpha}_{\text{stu}}) \right]
\end{equation}
Detailed gradient behaviors and cross-architecture alignments are discussed in the Appendix.

'''
    text = text.replace(kl_text, short_kl)
    extracted_appendix.append('\\subsection{Extended Dirichlet KL Knowledge Distillation Details}\n' + kl_text)

match_dataset = re.search(r'\\subsection\{ISIC 2024 Dataset\}.*?(?=\\subsection\{Two-Phase Training Protocol)', text, flags=re.DOTALL)
if match_dataset:
    dataset_text = match_dataset.group(0)
    extracted_appendix.append('\\subsection{Extended Dataset Description and Preprocessing}\n' + dataset_text)
    text = text.replace(dataset_text, '')

match_baselines = re.search(r'\\subsection\{Baselines\}.*?(?=\\subsection\{Evaluation Metrics)', text, flags=re.DOTALL)
if match_baselines:
    baselines_text = match_baselines.group(0)
    extracted_appendix.append('\\subsection{Detailed Baseline Configurations}\n' + baselines_text)
    text = text.replace(baselines_text, '')

match_metrics = re.search(r'\\subsection\{Evaluation Metrics\}.*?(?=\\section\{Results Analysis)', text, flags=re.DOTALL)
if match_metrics:
    metrics_text = match_metrics.group(0)
    extracted_appendix.append('\\subsection{Detailed Evaluation Metrics Formulation}\n' + metrics_text)
    text = text.replace(metrics_text, '')

short_exp = r'''\subsection{Experimental Setup}
\paragraph{Dataset and Task.} We evaluate MDEP on the clinical ISIC 2024 Challenge dataset~\cite{isic2024}, an extremely imbalanced binary classification task (malignant vs. benign skin lesions) where malignant samples represent less than 0.15\% of the data. 

\paragraph{Evaluation Metrics.} The primary official metric is the partial Area Under the ROC Curve (pAUC) over a True Positive Rate (Sensitivity) range strictly greater than 80\% ($0.8 < \text{TPR} \le 1.0$). We also report standard Macro-AUROC, Sensitivity, Specificity, and Expected Calibration Error (ECE)~\cite{naeini2015obtaining}.

\paragraph{Baselines.} We benchmark MDEP against state-of-the-art deterministic CNNs (ResNet-50, EfficientNet-B4), Vision Transformers (ViT-B/16), ensemble methods (Deep Ensembles~\cite{lakshminarayanan2017simple}, MC Dropout~\cite{gal2016dropout}), and explicit Evidential Deep Learning models (Vanilla EDL~\cite{sensoy2018evidential}, Posterior Networks~\cite{charpentier2020posterior}, and Imbalanced Open-Set EDL~\cite{pandey2023learn}).

'''
text = text.replace(r'\subsection{Two-Phase Training Protocol}', short_exp + r'\subsection{Two-Phase Training Protocol}')

# Append extracted_appendix to the end
full_appendix = '\n'.join(extracted_appendix)
text = text.replace('\\end{document}', full_appendix + '\n\\end{document}')

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Pruning executed successfully!")
