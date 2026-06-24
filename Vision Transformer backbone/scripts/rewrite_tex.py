import os

def rewrite_tex():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
    with open(tex_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Motivation in Section 1
    s1_old = r"\subsection{Evidential Deep Learning (EDL)}"
    s1_new = r"""\subsection{Evidential Deep Learning (EDL)}

\paragraph{Motivation \& Hypothesis.} The fundamental flaw of the softmax function is its sharp geometric normalization, which arbitrarily forces output logits into a rigid simplex, destroying any metric of ignorance. \textbf{Our Hypothesis} is that by modeling the parameters of a higher-order Dirichlet distribution instead of predicting point-probabilities, deep learning models can mathematically separate "lack of knowledge" (epistemic uncertainty) from "data noise" (aleatoric uncertainty).
\paragraph{Proposed Solution.} We adopt Evidential Deep Learning (EDL)~\cite{sensoy2018evidential}, which replaces softmax with a Softplus evidence activation to parameterize a Dirichlet distribution. This effectively constructs a subjective logic framework for robust uncertainty estimation."""
    text = text.replace(s1_old, s1_new)

    s1_dst_old = r"\subsection{Dynamic Sparse Training (DST)}"
    s1_dst_new = r"""\subsection{Dynamic Sparse Training (DST)}

\paragraph{Motivation \& Hypothesis.} Traditional pruning methods either require massive pre-training compute or ignore the structural flow of uncertainty. \textbf{Our Hypothesis} is that network sparsity can be optimized jointly with evidential learning by allocating computational capacity (connections) specifically to regions of the network that process highly uncertain features.
\paragraph{Proposed Solution.} We utilize Dynamic Sparse Training (DST)~\cite{evci2020rigging, mocanu2018scalable} under NVIDIA 2:4 structured sparsity. This allows the network topology to actively adapt during training without relying on post-hoc magnitude thresholds."""
    text = text.replace(s1_dst_old, s1_dst_new)

    # 2. Moving proofs to Appendix
    prop1_full = r"""\begin{proposition}[Epistemic Uncertainty Bounds]
For any valid concentration vector $\bm{\alpha}$, the value of $u_e$ always lies in the interval $(0, 1]$. The maximum value $u_e = 1.0$ is achieved if and only if $\bm{e} = \bm{0}$, i.e., $S = K$. As $S \to \infty$, $u_e \to 0$.
\end{proposition}"""
    prop1_brief = r"""\begin{proposition}[Epistemic Uncertainty Bounds]
For any valid concentration vector $\bm{\alpha}$, $u_e \in (0, 1]$. The maximum $u_e = 1.0$ is achieved iff $\bm{e} = \bm{0}$. As $S \to \infty$, $u_e \to 0$. (See Appendix for formal proof).
\end{proposition}"""
    text = text.replace(prop1_full, prop1_brief)

    prop2_full = r"""\begin{proposition}[Non-negativity of $u_a$]
For any $K > 1$, since $\alpha_c < S$ and due to the strict monotonicity of the digamma function ($\psi'(x) > 0$ for all $x > 0$), we always have $\psi(S + 1) > \psi(\alpha_c + 1)$. This guarantees that every term in the summation is positive, mathematically ensuring $u_a \ge 0$. In practical implementations, to protect against floating-point inaccuracies, $u_a \leftarrow \max(u_a, 0.0)$ should be applied.
\end{proposition}"""
    prop2_brief = r"""\begin{proposition}[Non-negativity of $u_a$]
For any $K > 1$, $u_a \ge 0$. In practical implementations, to protect against floating-point inaccuracies, $u_a \leftarrow \max(u_a, 0.0)$ is applied. (See Appendix for formal proof).
\end{proposition}"""
    text = text.replace(prop2_full, prop2_brief)

    # 3. Section 6: Glial Agents
    sec6_old = r"""\section{The Microglia--Astrocyte Dynamic Agents}\label{sec:agents}

The MDEP system introduces two adversarial but cooperative biological agents: Microglia (pruning) and Astrocyte (growing). These agents continuously monitor the network's evidential gradients and output antagonistic forces ($C_{ij}$ and $G_{ij}$), driving the evolution of the latent structural scores $S_{ij}$."""
    sec6_new = r"""\section{The Microglia--Astrocyte Dynamic Agents}\label{sec:agents}

\paragraph{Motivation \& Assumption.} Standard magnitude pruning assumes that weights with small magnitudes are unimportant. However, this assumption ignores the actual functional utility of the weight in resolving uncertainty. \textbf{Our Assumption} is that the gradients of evidential uncertainties ($u_e$ and $u_a$) with respect to network parameters provide a perfect biological analogy to synaptic pruning and neurogenesis.
\paragraph{Proposed Solution.} The MDEP system introduces two adversarial but cooperative biological agents: Microglia (pruning) and Astrocyte (growing). These agents continuously monitor the network's evidential gradients and output antagonistic forces ($C_{ij}$ and $G_{ij}$), driving the evolution of the latent structural scores $S_{ij}$.

"""
    text = text.replace(sec6_old, sec6_new)

    # 4. Section 9: Assessment / Results Overhaul
    sec9_old = r"""\section{Results Analysis, Calibration, and Risk Quantification}\label{sec:results}

MDEP's performance is analyzed across three groups of metrics: classification capacity, probability calibration, and selective prediction."""
    sec9_new = r"""\section{Results Analysis, Calibration, and Risk Quantification}\label{sec:results}

\paragraph{Assessment Goals and Objective: Selling the New Approach for EDL.} 
The primary objective of this evaluation is to demonstrate that MDEP fundamentally resolves the well-documented overconfidence and calibration crises present in standard DNNs and classical EDL systems when exposed to extreme class imbalance. By dynamically pruning noise-absorbing connections and regrowing capacity exactly where epistemic uncertainty is highest, MDEP explicitly enforces a structured and highly calibrated evidence landscape. We intend to "sell" MDEP not merely as a sparse model, but as the \textit{next-generation paradigm} for Evidential Deep Learning---proving that the integration of dynamic NVIDIA 2:4 structured sparsity with Dirichlet uncertainty is the missing key to achieving simultaneous clinical sensitivity, optimal expected calibration error (ECE), and reliable selective prediction (deferral).

MDEP's performance is analyzed across three groups of metrics: classification capacity, probability calibration, and selective prediction."""
    text = text.replace(sec9_old, sec9_new)

    # Add appendix
    appendix = r"""
\appendix
\section{Mathematical Proofs and Derivations}

\subsection{Proof of Proposition 1 (Epistemic Uncertainty Bounds)}
For any valid concentration vector $\bm{\alpha}$, the value of $u_e$ always lies in the interval $(0, 1]$. By definition, the epistemic uncertainty is formulated as $u_e = \frac{K}{S} = \frac{K}{\sum_{c=1}^K (e_c + 1)}$. Since the Softplus activation ensures that the evidence $e_c \ge 0$ for all classes $c$, the minimum possible value of the Dirichlet strength $S$ is precisely $K$ (when $e_c = 0 \, \forall c$). Therefore, the maximum value of $u_e$ is bounded at $\frac{K}{K} = 1.0$, which is achieved if and only if the network predicts absolutely zero evidence ($\bm{e} = \bm{0}$), representing total prior ignorance. Conversely, as the evidence $e_c$ grows infinitely ($S \to \infty$), the fraction $u_e \to 0$.

\subsection{Proof of Proposition 2 (Non-negativity of Aleatoric Uncertainty)}
Aleatoric uncertainty is defined by the digamma variance terms. For any $K > 1$, the Dirichlet parameters satisfy $\alpha_c < S$. Because the digamma function $\psi(x) = \frac{d}{dx} \ln \Gamma(x)$ is strictly monotonically increasing for all $x > 0$ (i.e., its derivative, the trigamma function $\psi'(x) > 0$), we mathematically guarantee that $\psi(S + 1) > \psi(\alpha_c + 1)$. This strict inequality ensures that every individual term within the summation $\sum_{c=1}^K \frac{\alpha_c}{S} [\psi(S + 1) - \psi(\alpha_c + 1)]$ is strictly positive. Consequently, the total aleatoric uncertainty $u_a$ is inherently non-negative ($u_a \ge 0$).

"""
    text = text.replace(r"\end{document}", appendix + "\n\\end{document}")

    # Write back
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, '..', 'paper', 'swin_mdep_report_en.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    rewrite_tex()
