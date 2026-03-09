# Comparative Analysis of Adiabatic Error Bounds in Quantum Systems Through a Modified Landau-Zener Model


This project investigates the adiabatic theorem applied to finite dimensional quantum systems via a modified Landau-Zener model. It compares analytical error bounds against exact adiabatic errors in both continuous-time Schrödinger evolution through which we can estimate the tightness of the quantitative error bound. It also explore on how the higher eigenlevels interfere with the lower levels during adiabatic evolution which is reflected in the simulated error metrics.

Key themes explored/implemented:
  - Continuous adiabatic evolution
  - Error quantification and sharpness of analytical upper bounds
  - Influence of total evolution time T and energy level spacing on error metrics
  - Spectral geometry sensitivity via Hamiltonian slope parameters
  
Yet to be implemented:
  - Discrete adiabatic evolution frameworks

The time-dependent Hamiltonian takes the form:
  H(s) = [1 - f(s)] * H_0  +  f(s) * H_1,   s = t/T in [0, 1] with linear scheduling f(s) = s.

================================================================================
FOLDER STRUCTURE 
================================================================================
```
Usernotfoundoops/Adibatic_Quantum_Computing  [branch: Different-Implementation]
│
├── 📓 Adiabatic Theorem Implementation (Muti.T) [t fix].ipynb
│       └── Main simulation notebook — Single/Multi-T adiabatic error analysis
│
├── 📁 Test Files/
│       ├── 🐍 AQC Continuous Evolution.py
│       │       └── Continuous-time Schrödinger evolution [2-level system]
│       │
│       ├── 🐍 AQC Unitary Walk Operator.py
│       │       └── Discretized unitary walk-operator evolution [2-level system]
│       │
│       ├── 📓 AQC 3 level First Version.ipynb
│       │       └── 3-level system prototype notebook
│       │
│       └── 📓 test.ipynb
│               └── Scratch/testing notebook
│
├── 📁 Plots/
│       ├── 📁 Large Gap/
│       ├── 📁 Medium Gap/
│       └── 📁 Small Gap/
│
├── 📄 README.md
└── 📄 requirements.txt
        └── Recreate environment with: pip install -r requirements.txt
```

================================================================================
WORKFLOW - HOW TO REPRODUCE RESULTS
================================================================================

STEP 1 - Set up the environment
  - Install Python (>= 3.9 recommended) and create a virtual environment:
      python -m venv .venv
      source venv/bin/activate        # Linux/MacOS
      venv\Scripts\activate           # Windows

  - Install dependencies:
      pip install -r requirements.txt
      
  - If running on Colab:
      !pip install qutip scipy matplotlib

STEP 2 - Configure simulation parameters
  - Open Adiabatic Theorem Implementation (Muti.T) [t fix].ipynb. Execute each cell in order and set the desired values for:
      * a,b,c,d controls the Hamiltonian shape.
      * T_curr to desired Total Evolution Time (T) (In my project T_curr = 5000)

STEP 3 - Run continuous-time evolution
  - There are multiple simulations to choose from:
  - Under Single T value:
      * Varying 'd' parameter
      * Varying 'b' parameter
      * Constant a,b,c,d parameters
      
  - For Multiple instances of T execute the block under the Markdown 'Multiple T', where the simulation executes for each T values and stores it inside the list of lists. Which can be taken later for observation and analysis.
      * Adjust T1 list(default end value: 5000) and its spacing as required. (Warning: Higher the list size, longer the computational runtime. 50 is a sweet-spot used in this project.)
      * That is, for the same list size it runs loops for different Total Evolution Time (T). Since, the evolution is dependent in t/T. Which dictates the end results.
      
      * IMPORTANT: Pick one type of simulation, execute it, and visualize it. Else, it will mess up the plot. 
      * Example: I run all the cells till defining the H_static and H_dynamic cell. Then I execute "Varying 'b' parameter" cell. Here, I have already set my T_curr as 5000 in the parameters cell above. Then I go down and run the visualization cell "Variable d Parameter vs Error Metrics" designated for this particular simulation. Vice versa.
      
STEP 4 - Visualization of Plots\
  * After running a specific simulation, under 'GRAPHICAL PLOTS' markdown pick any visualization cell.
      * Certain cells cannot produce certain plots. (Single T value simulation cannot run "Total Evolution Time T vs Error Metrics" Plot, since we need Multiple T values to visualize.)
      * Cells such as "Eigen Levels E vs Time t" and "Error Metrics and Transition Probability vs Time t" works for single and multiple T simulations.
      * To get an animated result of the Error Metrics (Transition Probability is also included but commented out), run the cell under 'ANIMATION'.
      * Make sure to uncomment the ani.save line to check the animated .GIF
      * BONUS: Open Plots folder and each sub-folder has certain parameters a,b,c,d on the filename. Replace that in code to reproduce the exact results.

STEP 5 - Analytical error bounds
  - The Adiabatic error and exact errors are already setup inside each cell
  - Each simulation automatically executes the analytical upper bound A(s) (Jansen et al., 2007) which also uses scipy.integrate.quad() method for dynamic integration.
  - If different scheduling function f(s) is used then its first derivative and second derivative of H has to be calculated manually and inserted into each simulation. As long as f(s) is linear this comment can be ignored.
  - Different error bounds can be implemented by changing the "total_equation" line.
  
================================================================================
SOFTWARE AND LIBRARY DEPENDENCIES
================================================================================

Language:\
  Python = 3.12

Core Libraries:\
  numpy          >= 1.24.0    # Array operations, linear algebra\
  scipy          >= 1.10.0    # ODE solvers (solve_ivp)\
  matplotlib     >= 3.7.0     # Plotting energy spectra and error curves\
  jupyter        >= 1.0.0

Optional:
  qutip          >= 5.0.0     # Quantum system simulation and state evolution\
  sympy          >= 1.12.0    # Integration

================================================================================
CONTACT
================================================================================

  Author  : Pravin Mani Kannan\
  Email   : pravin.mani@fau.de\
  Date    : 08-03-2026

================================================================================
