import numpy as np
from numba import njit, prange

from matplotlib import pyplot as plt
import matplotlib as mpl

######################## SIMULATION PARAMETERS ########################
rate = 1.0  # rate of coupling to the resevoirs compared to the rate of the system
Nx = 7
Ny = 7
N = Nx * Ny
V = -1.0
num_iterations = 1000000
steps = 300  # Multiples of number of bonds
########################################################################

# Extract currents
bonds = []
for y in range(Ny):
    for x in range(Nx-1):
        n1 = x % Nx + y*Nx
        n2 = (x+1) % Nx + y*Nx

        bonds.append((n1, n2))
for x in range(Nx):
    for y in range(Ny-1):
        n1 = x % Nx + y*Nx
        n2 = x % Nx + (y+1)*Nx

        bonds.append((n1, n2))

bonds = np.array(bonds, dtype=np.int32)  # Convert to numpy array for numba compatibility
num_bonds = len(bonds)

@njit
def random_state():
    """Generate a random initial state."""
    state = np.array([np.random.randint(2) for _ in range(N)])
    return state


@njit
def local_energy(state, bond_index):
    """Calculate the local energy for a given bond."""
    n1 = bonds[bond_index, 0]
    n2 = bonds[bond_index, 1]
    
    E = 0.0

    for bond in [x for x in bonds if ((n1 in x) or (n2 in x))]:
        if state[bond[0]] == state[bond[1]] and state[bond[0]] == 1:
            E += V

    return E


@njit 
def acceptance_check(E_new, E_old):
    """Check if the new state is accepted based on the energy difference."""
    if E_new < E_old:
        return True
    
    if np.random.rand() < np.exp(E_old - E_new):
        return True
    
    return False


@njit
def trajectory(procid, data):
    
    state = random_state()  # Use a new random state for each trajectory

    previous_state = state.copy()

    n_list = np.zeros((steps+1, N), dtype=np.int8)
    n_list[0] = state

    current_list = np.zeros((steps+1, num_bonds), dtype=np.float64)
    current_list[0] = 0.0

    for step in range(steps*num_bonds):

        reservoir_prob = 2*rate / (num_bonds + 2*rate)  # Probability of selecting a reservoir bond

        reservoir = np.random.rand() < reservoir_prob

        if reservoir:
            if np.random.rand() < 0.5:
                state[0] = 1
            else:
                state[-1] = 0

        else:
            bond_index = np.random.randint(num_bonds)
            n1 = bonds[bond_index, 0]
            n2 = bonds[bond_index, 1]

            # Swap n1 and n2
            state[n1], state[n2] = state[n2], state[n1]


        # Accept or reject the state change
        if state[n1] == state[n2] or reservoir:
            accepted = True  # No change in state, do nothing
        else:
            E_new = local_energy(state, bond_index)
            E_old = local_energy(previous_state, bond_index)

            accepted = acceptance_check(E_new, E_old)

        if accepted:
            previous_state = state.copy()

            if not reservoir:
                current_list[step // num_bonds + 1, bond_index] += state[n2] - state[n1]  # Record current flow

        else:
            state = previous_state.copy()

        if step % N == 0:
            n_list[step // num_bonds + 1] = state.copy()

    return n_list, current_list


@njit(parallel=True)
def run_simulation():
    n_avg = np.zeros((steps+1, N))
    avg_currents = np.zeros((steps+1, num_bonds))

    for ii in prange(num_iterations):
        n_list, current_list = trajectory(ii, None)
        n_avg += n_list / num_iterations
        avg_currents += current_list / num_iterations

    return n_avg, avg_currents

        
if __name__ == "__main__":

    ######################## RUN SIMULATION #######################
    print("Running trajectories...")
    n_avg, avg_currents = run_simulation()
    print("Finished!")
    ################ PLOT THE FINAL TIME SNAPSHOT  #################

    plotting_threshold = 0.0  # Threshold for plotting currents
    marker_size = 750 * (3/Nx)**2  # Size of the markers for the density plot
    arrow_width = 0.035 * 3/Nx  # Width of the arrows in the quiver plot
    
    # single line definition of empty lists for X, Y, U, V, C
    X = []; Y = []; U = []; V = []; C = []
    for i, bond in enumerate(bonds):
        # convert back from n to x,y coordinates
        x1, y1 = bond[0] % Nx, bond[0] // Nx
        x2, y2 = bond[1] % Nx, bond[1] // Nx
    
        if np.abs(avg_currents[-1,i]) > plotting_threshold*np.abs(avg_currents[-1,:]).max():
            C.append(np.abs(avg_currents[-1,i]))
    
            if np.real(avg_currents[-1,i]) > 0:
                X.append(x1)
                Y.append(y1)
                U.append(x2-x1)
                V.append(y2-y1)
            else:
                X.append(x2)
                Y.append(y2)
                U.append(x1-x2)
                V.append(y1-y2)
    
    fig, ax = plt.subplots(figsize = (6,6))

    norm = mpl.colors.Normalize(vmin=0, vmax=0.2)
    cmap_range = mpl.cm.YlGn(np.linspace(0, 1, 500))
    cmap = mpl.colors.ListedColormap(cmap_range)
    p1 = ax.quiver(X, Y, U, V, C, cmap=cmap, norm=norm, angles='xy', scale_units='xy', scale=1, width=arrow_width)
    cb1 = plt.colorbar(p1, ax=ax, orientation='vertical', shrink=0.53, pad=0.1)
    cb1.set_ticks(np.arange(0, 0.21, 0.04))
    cb1.ax.tick_params(labelsize=13)
    cb1.set_label('Current Magnitude', labelpad=1, fontsize = 15)
    
    X = []; Y = []; C = []
    for x in range(Nx):
        for y in range(Ny):
            n = x % Nx + y*Nx
            X.append(x)
            Y.append(y)
            C.append(n_avg[-1,n])
    
    cmap_range = mpl.cm.RdBu_r(np.linspace(-0.5, 1.5, 500))
    cmap = mpl.colors.ListedColormap(cmap_range)
    p2 = ax.scatter(X, Y, c=C, cmap=cmap, s=marker_size, edgecolors= "black", vmin=0, vmax=1)
    cb2 = plt.colorbar(p2, ax=ax, orientation='vertical', shrink=0.53, pad=0.1)
    cb2.set_ticks(np.arange(0, 1.1, 0.2))
    cb2.ax.tick_params(labelsize=13)
    cb2.set_label('Density', labelpad=1, fontsize = 15)
    
    # Add automatic padding to prevent cutoff
    ax.margins(0.1 * 3/Nx)  # Add padding around the data
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    plt.savefig(f"figures/SSEP_{Nx}x{Ny}_steps{steps}_V+1_trajectories{num_iterations}.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.show()