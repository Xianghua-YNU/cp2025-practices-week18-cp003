import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
a = 5.29e-2  # Bohr radius (nm)
D_max = 1.1  # Maximum probability density
r0 = 0.25    # Convergence radius (nm)

# Probability density function for the 1s orbital
def probability_density(r):
    return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# Generate points in spherical coordinates and convert to Cartesian
def generate_spherical_points(num_points=10000):
    # Polar angle θ (0 to π), azimuthal angle φ (0 to 2π)
    theta = np.random.uniform(0, np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    # Radial distance r (0 to r0)
    r = np.random.uniform(0, r0, num_points)
    
    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Calculate probability density and filter points
    prob = probability_density(r)
    prob_ratio = prob / D_max
    mask = np.random.uniform(0, 1, num_points) < prob_ratio
    
    return x[mask], y[mask], z[mask]

# Visualize the electron cloud
def visualize_electron_cloud(x, y, z):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='blue', alpha=0.3, label='Electron Cloud Distribution')
    
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title('Hydrogen Atom Electron Cloud Simulation')
    ax.legend()
    plt.show()

# Analyze the effect of parameters (e.g., r0)
def analyze_parameter_effect():
    r0_values = [0.2, 0.25, 0.3]  # Different convergence radii
    fig, axes = plt.subplots(1, len(r0_values), figsize=(15, 5), subplot_kw={'projection': '3d'})
    
    for i, r0_val in enumerate(r0_values):
        theta = np.random.uniform(0, np.pi, 5000)
        phi = np.random.uniform(0, 2 * np.pi, 5000)
        r = np.random.uniform(0, r0_val, 5000)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        prob = (4 * r**2 / a**3) * np.exp(-2 * r / a)
        prob_ratio = prob / D_max
        mask = np.random.uniform(0, 1, 5000) < prob_ratio
        
        axes[i].scatter(x[mask], y[mask], z[mask], s=1, c='green', alpha=0.3)
        axes[i].set_title(f'r0={r0_val} nm')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('Z')
    
    plt.suptitle('Effect of Convergence Radius on Electron Cloud Distribution')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate and visualize electron cloud
    x, y, z = generate_spherical_points()
    visualize_electron_cloud(x, y, z)
    
    # Analyze parameter effects
    analyze_parameter_effect()
