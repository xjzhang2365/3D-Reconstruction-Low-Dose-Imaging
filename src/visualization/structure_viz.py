"""
3D structure visualization tools.

Full implementation - standard visualization techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict
import warnings


class StructureVisualizer:
    """
    Visualize 3D atomic structures.
    
    Provides both static (matplotlib) and interactive (plotly)
    visualizations of reconstructed structures.
    
    Examples
    --------
    >>> viz = StructureVisualizer()
    >>> structure = np.load('structure.npy')
    >>> viz.plot_3d(structure, save_path='structure_3d.png')
    >>> viz.plot_interactive(structure, save_path='structure.html')
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
    
    def plot_3d(self,
               positions: np.ndarray,
               colors: Optional[np.ndarray] = None,
               color_by: str = 'z',
               title: str = '3D Atomic Structure',
               save_path: Optional[str] = None,
               show: bool = True) -> plt.Figure:
        """
        Create 3D scatter plot of structure.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
        colors : np.ndarray, optional
            Custom colors for each atom
        color_by : str
            Color atoms by: 'z' (height), 'index', or 'uniform'
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        show : bool
            Whether to display plot
            
        Returns
        -------
        fig : matplotlib.Figure
            Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine colors
        if colors is None:
            if color_by == 'z':
                colors = positions[:, 2]
                cmap = 'viridis'
            elif color_by == 'index':
                colors = np.arange(len(positions))
                cmap = 'rainbow'
            else:
                colors = 'blue'
                cmap = None
        else:
            cmap = 'viridis'
        
        # Plot
        if cmap:
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=colors, cmap=cmap, s=50, alpha=0.8,
                               edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Z Height (Å)' if color_by == 'z' else 'Value')
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=colors, s=50, alpha=0.8,
                      edgecolors='black', linewidth=0.5)
        
        # Labels
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Equal aspect ratio
        self._set_equal_aspect_3d(ax, positions)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved figure to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_interactive(self,
                        positions: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        title: str = '3D Atomic Structure (Interactive)',
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive 3D plot using Plotly.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions, shape (N, 3)
        colors : np.ndarray, optional
            Colors for atoms (z-height if None)
        title : str
            Plot title
        save_path : str, optional
            Path to save HTML file
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure
        """
        if colors is None:
            colors = positions[:, 2]  # Color by z-height
        
        fig = go.Figure(data=[go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Z Height (Å)"),
                line=dict(color='black', width=0.5)
            ),
            text=[f'Atom {i}<br>x={x:.2f}, y={y:.2f}, z={z:.2f}' 
                  for i, (x, y, z) in enumerate(positions)],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"✓ Saved interactive plot to {save_path}")
        
        return fig
    
    def plot_comparison(self,
                       structure1: np.ndarray,
                       structure2: np.ndarray,
                       labels: Tuple[str, str] = ('Structure 1', 'Structure 2'),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare two structures side-by-side.
        
        Parameters
        ----------
        structure1, structure2 : np.ndarray
            Structures to compare, shape (N, 3)
        labels : tuple of str
            Labels for each structure
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig = plt.figure(figsize=(16, 7))
        
        # Structure 1
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(structure1[:, 0], structure1[:, 1], structure1[:, 2],
                              c=structure1[:, 2], cmap='viridis', s=50, alpha=0.8,
                              edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title(labels[0], fontsize=14, fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, label='Z (Å)')
        self._set_equal_aspect_3d(ax1, structure1)
        
        # Structure 2
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(structure2[:, 0], structure2[:, 1], structure2[:, 2],
                              c=structure2[:, 2], cmap='viridis', s=50, alpha=0.8,
                              edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.set_zlabel('Z (Å)')
        ax2.set_title(labels[1], fontsize=14, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Z (Å)')
        self._set_equal_aspect_3d(ax2, structure2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")
        
        return fig
    
    def _set_equal_aspect_3d(self, ax, positions: np.ndarray):
        """Set equal aspect ratio for 3D plot"""
        # Get data limits
        x_limits = [positions[:, 0].min(), positions[:, 0].max()]
        y_limits = [positions[:, 1].min(), positions[:, 1].max()]
        z_limits = [positions[:, 2].min(), positions[:, 2].max()]
        
        # Find max range
        max_range = max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0]
        ) / 2
        
        # Set limits centered on data
        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_height_map(positions: np.ndarray,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D height map (top view colored by z).
    
    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (N, 3)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=positions[:, 2], cmap='RdYlBu_r',
                        s=100, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Z Height (Å)', fontsize=12)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_title('Height Map (Top View)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    """Test visualization"""
    
    print("Structure Visualization - Test")
    print("="*60)
    
    # Create test structure
    n_x, n_y = 10, 10
    positions = []
    
    for i in range(n_x):
        for j in range(n_y):
            x = i * 2.5
            y = j * 2.5
            z = 0.5 * np.sin(x/5) * np.cos(y/5)
            positions.append([x, y, z])
    
    positions = np.array(positions)
    
    print(f"\n1. Created test structure: {len(positions)} atoms")
    
    # Test visualizer
    print("\n2. Testing visualization...")
    viz = StructureVisualizer()
    
    fig1 = viz.plot_3d(positions, title='Test Structure', show=False)
    print("   ✓ 3D plot created")
    
    fig2 = plot_height_map(positions)
    print("   ✓ Height map created")
    
    plt.close('all')
    
    print("\n✓ Visualization module test complete!")