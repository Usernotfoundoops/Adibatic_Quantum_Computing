\documentclass[11pt,a4paper]{article}
%\usepackage{fontspec}
%\setmainfont{Arial}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\geometry{margin=2cm,top=2cm}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\usepackage{titlesec}
\usepackage{siunitx} % optional, for better number formatting if needed later

\captionsetup{font=footnotesize, justification=centering}

\title{\bfseries \large \vspace{-1.4cm} Intermediary Report: Comparative Analysis of Adiabatic Error Bounds in Quantum Systems Through a Modified Landau-Zener Model} 
\date{}

\titleformat{\section}
  {\large\bfseries}   % <-- controls SIZE + weight
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\large\bfseries}   % <-- controls SIZE + weight
  {\thesubsection}{1em}{}

\begin{document}

\maketitle 
\vspace{-2cm}
\begin{center}
\vspace{-0.4cm} \href{https://github.com/Usernotfoundoops/Adibatic_Quantum_Computing/blob/master/AQC_3%20level%20jpq.ipynb}{GitHub Repository} \end{center}
\vspace{-1cm}
\section{AIM}

This project aims to provide a Python workflow for computing and comparing quantitative adiabatic error bounds for a time-dependent Hamiltonian, the sharpness of theoretical error upper bounds against the simulated error, and subsequently expand the framework to discretized adiabatic dynamics.
\vspace{-0.2cm}
\section{METHODOLOGY}

The workflow begins by defining the Hamiltonian model and simulating its time evolution using the \href{https://doi.org/10.1016/j.physrep.2025.10.001}{\textbf{QuTiP}} \textbf{Python package}. At each temporal step $t$, the code extracts instantaneous eigenstates to compute the measured adiabatic error and the theoretical upper bound defined in Section 2.3. 
\vspace{0.1cm}

To ensure mathematical rigor in the implementation of the bound, the operator norm $\Vert\cdot\Vert$ is calculated as the spectral norm (the largest singular value) via \texttt{numpy.linalg.norm(ord=2)}, specifically because QuTiP \texttt{H.norm()} defaults to the Frobenius norm. This evaluation is performed iteratively within the simulation loop at each timestep $t$, which is subsequently visualized using the \texttt{Matplotlib} package.
\vspace{-0.2cm}
\subsection{HAMILTONIAN DESIGN}

I modeled the time-dependent Hamiltonian given by
\(H(t) = H_{static} + f(t)H_{dynamic}.\) Equivalently, in the standard adiabatic parameterization,
\(H(s) = (1-f(s))H_0 + f(s)H_1\), \(s=t/T\in[0,1]\) where $T$ is the total evolution time. At $s=0$, \(H(0) = H_0 = H_{static}\). At $s=1$, the Hamiltonian $H(s)$ evolves into the target Hamiltonian \(H(1) = H_1 = H_{static}+f(1)H_{dynamic}\). Where \(H_{static}\) represents the time-independent part of the Hamiltonian containing the initial eigen energies and \(H_{dynamic}\) is turned on through the scheduling function \(f(s)\), which introduces controlled time dependence. For the three-level system studied here, the Hamiltonian (in matrix form) is:
\begin{equation}
H(s) = 
\begin{pmatrix}
-f(s) + a & b & 0 \\
b & f(s) - a & c \\
0 & c & f(s)\cdot d + e
\end{pmatrix} \text{where }a,b,c,d \in \mathbb{R} {\ge 0} 
\end{equation}

\vspace{-0.2cm}
\subsection{QuTiP IMPLEMENTATION}


The simulation models a three-level system by constructing a $3\times 3$ quantum object (\texttt{Qobj}) within the \textbf{QuTiP Python package}. The state evolution is computed via the \texttt{sesolve} solver provided by QuTiP, which works with a Hamiltonian composed of static and dynamic parts. Then the eigenvalues and eigenstates are extracted using \texttt{H.eigenstates()} method and stored in memory.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\linewidth]{Energyplot.png}
     \caption{Plots the three eigenvalues of $H(t)$ versus time for the parameters $a = 0.5, b = 0.1, c = 0.3,$ and $d = 1.$ If either off‑diagonal coupling $b$ or $c$ is set to zero, the levels cross each other. Otherwise, the coupling typically produces an avoided crossing (a nonzero minimum gap).} 
    \label{fig:eigenvalues}
\end{figure}
\vspace{-0.3cm}

\subsection{ANALYSIS OF ERROR BOUNDS}

I study the adiabatic error between the ground state and the first excited level. To quantify this, I extract the instantaneous eigenstates at all time points and compare them with the numerically solved full time-dependent Schrödinger equation. This approach aligns with the rigorous error bounds established in \href{https://arxiv.org/abs/quant-ph/0603175}{\textit {Jansen, Seiler, and Ruskai (2007)}}, which analyze the adiabatic approximation for time-dependent Hamiltonian with explicit gap dependence. The objective of this comparison is to evaluate the tightness of the bound, i.e. to see how much the error bound is overestimating the actual error. The error upper bound is given by the equation, 

\begin{equation}
    A(t) \leq \frac{m \Vert \dot{H}(0) \Vert }{g(0)^2}+ \frac{m \Vert \dot{H}(T) \Vert }{g(T)^2} +\int_0^T \left(\frac{m \Vert \ddot{H}(t)\Vert}{g(t)^2} + 7m \sqrt{m} \frac{\Vert \dot{H}(t)\Vert^2}{g(t)^3} \right) dt,
\end{equation}
The integral term is evaluated numerically using \texttt{scipy.integrate.quad()} method. Because of the time dependence of the gap, it is included inside the integrand so that the bound reflects the changing gap during evolution.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\linewidth]{Error_Bound_vs_time.png}
    \caption{Compares the simulated adiabatic error with the computed upper error bound. In interpretation, the bound is meaningful only when the gap does not close during evolution i.e., there are no level crossings.}
    \label{fig:Error_Bound_vs_Time}
\end{figure}

\vspace{-0.2cm}
\section{PROGRESS REPORT}


I implemented a Python workflow using adiabatic theorem to simulate adiabatic quantum evolution for a time-dependent Hamiltonian $H(t)$ with a linear scheduling function $f(t)$. At each time step $t$, the code calculates the instantaneous eigenstates of the Hamiltonian and quantified the adiabatic error by comparing the numerically evolved state (from the time-dependent Schrödinger equation) with the corresponding instantaneous eigenstate. The dynamics are visualized through two plots:
\begin{itemize} \vspace{-0.2cm}
    \item \hyperref[fig:eigenvalues]{Figure~\ref*{fig:eigenvalues}}: Instantaneous eigenvalue diagram \vspace{-0.2cm}
    \item \hyperref[fig:Error_Bound_vs_Time]{Figure~\ref*{fig:Error_Bound_vs_Time}}: Time-resolved plot of the adiabatic error together with the corresponding theoretical upper bound (on logarithmic scale)
\end{itemize}

\vspace{-0.2cm}
\section{TO BE STUDIED AND IMPLEMENTED}

Having established the continuous-time framework, the focus now shifts to an investigation of the error dynamics as a function of the total evolution time $T$ and the parameter $d$ associated with the third eigen level~\ref{fig:eigenvalues}. This analysis is motivated by the need to identify how the total duration of the process and the relative spacing of the energy levels influence the quality of the adiabatic transition. 
\vspace{0.1cm}

The next phase of this project replaces the continuous-time Schrödinger evolution with a discretized model defined by a sequence of unitary (walk) operators. The resulting state is then compared against instantaneous eigenstates to determine the discrete-time adiabatic error. By implementing the corresponding unitary dynamics, this approach evaluates the impact of the discretization step size on the overall fidelity and sharpness of the upper bound. 

\end{document}
