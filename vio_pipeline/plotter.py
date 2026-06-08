import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_analysis():
    with open('drone_drift_data.json', 'r') as f:
        data = json.load(f)

    est = np.array([data['est_x'], data['est_y'], data['est_z']]).T
    tgt = np.array([data['target_x'], data['target_y'], data['target_z']]).T
    error = est - tgt

    fig = plt.figure(figsize=(14, 8))
    
    # 3D Trajectory Plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(est[:,0], est[:,1], est[:,2], label='Actual Path', color='b')
    ax.plot(tgt[:,0], tgt[:,1], tgt[:,2], label='Target Path', color='r', linestyle='--')
    ax.set_title("3D Flight Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    # 3D Error Vector Plot (Quiver)
    ax2 = fig.add_subplot(122, projection='3d')
    # Plotting every 20th point to keep the graph readable
    stride = 20
    ax2.quiver(est[::stride,0], est[::stride,1], est[::stride,2], 
               error[::stride,0], error[::stride,1], error[::stride,2], 
               color='r', length=0.1, label='Error Vector')
    ax2.set_title("Drift Error Analysis (Error Vectors)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_analysis()
