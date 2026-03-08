import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

#time varying function for control
def f(t,args):
    T=args['T']
    return t/T #np.sin(np.pi * t / (2 * T))**2 #t/T

a = 0.1 #time-independent coupling constant
T = 800 #time evolution interval
args = {'T': T}
t=0
sxx = qt.sigmax()
szz = qt.sigmaz()

#defining the components of hamiltonian
H_static = a * sxx - 0.5 * szz
#H_dynamic = szz #function is not added since it's static here

H_0=H_static+(f(t,args)*szz) #hamiltonian of the ground state

H_total=[H_static,[szz,f]]
eigenvalue,eigenvector=H_0.eigenstates()
psi0=eigenvector[0]

#time-stepping
t_list=np.linspace(t,T,2000)

#Unitary_Evolution
U_list = qt.propagator(H_total, t_list, args=args)
states_from_unitary = [] #intitialise the states

for U in U_list:
    psi_t = U * psi0 #applying the unitary operator to the initial ground state
    states_from_unitary.append(psi_t)

# final = qt.sesolve(H_total, psi0, t_list, args=args)

#Visualization
errors_norm = []
errors_unitary = []
gap = []

for i, t in enumerate(t_list):
    f_val = f(t, args)
    H_t = H_static + f_val * szz
    evals, estates = H_t.eigenstates()
    psi_ideal = estates[0]
    psi_evolved = states_from_unitary[i]
    gap.append(evals[1] - evals[0])

    # Calculate Error (using Fidelity): 1 - |<psi_ideal | psi_evolved>|^2
    fidelity = np.abs(psi_ideal.overlap(psi_evolved)) ** 2
    errors_unitary.append(1 - fidelity)

    #Norm
    diff_vector = psi_ideal - psi_evolved
    vector_distance = diff_vector.norm()
    errors_norm.append(vector_distance)

# # Plotting
# f_t_values = f(t_list, args)
#
# fig,ax3=plt.subplots()
# ax3.plot(t_list,f_t_values)
# plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Time ($t$)', fontsize=12)
ax1.set_ylabel('Error Magnitude (in log)', fontsize=12)

# Plotting both error metrics
ax1.plot(t_list, errors_unitary, color='tab:red', label='1 - Fidelity (Prob)')
ax1.plot(t_list, errors_norm, color='tab:green', linestyle=':', label='Vector Distance (Norm)')

ax1.set_yscale('log') # Both metrics are best viewed on a log scale
ax1.grid(True, which="both", ls="-", alpha=0.8)

# Energy Gap on the second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel(r'Energy Gap ($\Delta E$)', color='tab:blue', fontsize=12)
ax2.plot(t_list, gap, color='tab:blue', linestyle='--', label='Energy Gap')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Final formatting
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.title(f'Comparison of Error Metrics vs Energy Gap ($T={T}$)', fontsize=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

#for f vs T
f_values = [f(t, args) for t in t_list]
fidelity_values = [1 - err for err in errors_unitary]
plt.figure()
plt.plot(t_list, f_values)
plt.title('Function f vs Timesteps')
plt.xlabel('Timesteps (T)')
plt.ylabel('f(t,T)')
plt.grid()

#for Fidelity vs Time
plt.figure()
plt.plot(t_list, fidelity_values, color='orange')
plt.title('Fidelity vs Time')
plt.xlabel('Timesteps (t)')
plt.ylabel('Fidelity')
plt.ylim(0, 1.1) # Sets the scale to see the top clearly
plt.grid()



plt.show()