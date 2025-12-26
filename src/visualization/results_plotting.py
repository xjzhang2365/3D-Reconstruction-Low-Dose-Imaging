"""
Results plotting and analysis visualization.

Tools for creating publication-quality figures showing:
- Convergence plots
- Accuracy vs dose
- Comparison plots
- Analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class ResultsPlotter:
    """
    Create publication-quality plots of research results.
    
    Examples
    --------
    >>> plotter = ResultsPlotter()
    >>> plotter.plot_convergence(convergence_data)
    >>> plotter.plot_accuracy_vs_dose(dose_data)
    """
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize plotter.
        
        Parameters
        ----------
        style : str
            Plotting style: 'publication', 'presentation', or 'notebook'
        """
        self.style = style
        self._set_style()
    
    def _set_style(self):
        """Configure plotting style"""
        if self.style == 'publication':
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['savefig.dpi'] = 300
        elif self.style == 'presentation':
            plt.rcParams['font.size'] = 14
            plt.rcParams['figure.dpi'] = 100
    
    def plot_convergence(self,
                        convergence_history: List[float],
                        title: str = 'Optimization Convergence',
                        xlabel: str = 'Iteration',
                        ylabel: str = 'χ² (Energy)',
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization convergence curve.
        
        As shown in Fig 1a of the paper.
        
        Parameters
        ----------
        convergence_history : list of float
            Energy/χ² values at each iteration
        title : str
            Plot title
        xlabel, ylabel : str
            Axis labels
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        iterations = np.arange(len(convergence_history))
        
        ax.plot(iterations, convergence_history, 'o-', 
                linewidth=2, markersize=8, color='#2E86AB',
                markeredgecolor='black', markeredgewidth=0.5)
        
        # Add convergence threshold line (if applicable)
        if len(convergence_history) > 1:
            final_value = convergence_history[-1]
            ax.axhline(y=final_value, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Final: {final_value:.1f}')
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with stats
        improvement = ((convergence_history[0] - convergence_history[-1]) / 
                      convergence_history[0] * 100)
        textstr = f'Iterations: {len(convergence_history)}\n'
        textstr += f'Initial: {convergence_history[0]:.1f}\n'
        textstr += f'Final: {convergence_history[-1]:.1f}\n'
        textstr += f'Improvement: {improvement:.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved convergence plot to {save_path}")
        
        return fig
    
    def plot_accuracy_vs_dose(self,
                             dose_levels: np.ndarray,
                             accuracies: np.ndarray,
                             threshold_dose: Optional[float] = None,
                             title: str = 'Reconstruction Accuracy vs. Electron Dose',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot accuracy vs electron dose.
        
        As shown in Fig 4 of the paper.
        
        Parameters
        ----------
        dose_levels : np.ndarray
            Electron dose levels (e⁻/Å²)
        accuracies : np.ndarray
            RMSD values at each dose
        threshold_dose : float, optional
            Critical dose threshold to highlight
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Main plot
        ax.plot(dose_levels, accuracies, 'o-', 
               linewidth=2.5, markersize=10, color='#A23B72',
               markeredgecolor='black', markeredgewidth=0.5,
               label='Reconstruction RMSD')
        
        # Highlight threshold if provided
        if threshold_dose is not None:
            ax.axvline(x=threshold_dose, color='red', linestyle='--',
                      linewidth=2, alpha=0.7, 
                      label=f'Threshold: {threshold_dose:.1e} e⁻/Å²')
        
        ax.set_xlabel('Electron Dose (e⁻/Å²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11)
        
        # Add annotations for key points
        min_idx = np.argmin(accuracies)
        ax.annotate(f'Best: {accuracies[min_idx]:.2f} Å\n@ {dose_levels[min_idx]:.1e}',
                   xy=(dose_levels[min_idx], accuracies[min_idx]),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dose vs accuracy plot to {save_path}")
        
        return fig
    
    def plot_comparison_bar(self,
                           methods: List[str],
                           accuracies: List[float],
                           doses: Optional[List[str]] = None,
                           title: str = 'Method Comparison',
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Bar chart comparing different methods.
        
        Parameters
        ----------
        methods : list of str
            Method names
        accuracies : list of float
            Accuracy values
        doses : list of str, optional
            Dose requirements for each method
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(methods)))
        
        bars = ax.bar(x, accuracies, color=colors, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Highlight best method
        best_idx = np.argmin(accuracies)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            label = f'{acc:.2f} Å'
            if doses:
                label += f'\n({doses[i]})'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9,
                   fontweight='bold' if i == best_idx else 'normal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_bond_length_distribution(self,
                                     bond_lengths: np.ndarray,
                                     expected: float = 1.42,
                                     title: str = 'Bond Length Distribution',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of bond lengths.
        
        Parameters
        ----------
        bond_lengths : np.ndarray
            Array of bond length measurements
        expected : float
            Expected bond length (for graphene: 1.42 Å)
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = ax.hist(bond_lengths, bins=30, 
                                   color='skyblue', edgecolor='black',
                                   alpha=0.7, density=True)
        
        # Add expected value line
        ax.axvline(x=expected, color='red', linestyle='--',
                  linewidth=2, label=f'Expected: {expected:.2f} Å')
        
        # Add measured mean
        mean_val = np.mean(bond_lengths)
        ax.axvline(x=mean_val, color='green', linestyle='--',
                  linewidth=2, label=f'Measured: {mean_val:.2f} Å')
        
        ax.set_xlabel('Bond Length (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        std_val = np.std(bond_lengths)
        textstr = f'Mean: {mean_val:.3f} Å\n'
        textstr += f'Std: {std_val:.3f} Å\n'
        textstr += f'Deviation: {abs(mean_val - expected):.3f} Å\n'
        textstr += f'N bonds: {len(bond_lengths)}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved bond length distribution to {save_path}")
        
        return fig
    
    def plot_height_distribution(self,
                                z_heights: np.ndarray,
                                title: str = 'Height Distribution (Rippling)',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot z-height distribution showing rippling.
        
        As shown in Fig 3a of the paper.
        
        Parameters
        ----------
        z_heights : np.ndarray
            Z-height values
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(z_heights, bins=30, color='coral', 
               edgecolor='black', alpha=0.7, density=True)
        
        # Fit Gaussian for reference
        mean = np.mean(z_heights)
        std = np.std(z_heights)
        
        x = np.linspace(z_heights.min(), z_heights.max(), 100)
        gaussian = (1 / (std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x - mean) / std)**2)
        ax.plot(x, gaussian, 'r-', linewidth=2, 
               label=f'Gaussian fit (σ={std:.2f} Å)')
        
        ax.set_xlabel('Z Height (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        textstr = f'Mean: {mean:.3f} Å\n'
        textstr += f'Std: {std:.3f} Å\n'
        textstr += f'Range: {z_heights.max() - z_heights.min():.3f} Å\n'
        textstr += f'RMS: {np.sqrt(np.mean(z_heights**2)):.3f} Å'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved height distribution to {save_path}")
        
        return fig


def create_paper_figure_1(convergence_data: Dict,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Recreate Figure 1 from paper (validation on synthetic data).
    
    Parameters
    ----------
    convergence_data : dict
        Must contain 'iterations' and 'chi_squared' keys
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=(12, 10))
    
    # Subplot 1: Convergence curve
    ax1 = fig.add_subplot(221)
    ax1.plot(convergence_data['iterations'], 
            convergence_data['chi_squared'],
            'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('χ²')
    ax1.set_title('(a) SA Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Error distribution
    if 'errors' in convergence_data:
        ax2 = fig.add_subplot(222)
        ax2.hist(convergence_data['errors'], bins=30, 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Position Error (Å)')
        ax2.set_ylabel('Count')
        ax2.set_title('(b) Accuracy Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 1 to {save_path}")
    
    return fig


if __name__ == '__main__':
    """Test results plotting"""
    
    print("Results Plotting Module - Test")
    print("="*60)
    
    plotter = ResultsPlotter()
    
    # Test 1: Convergence plot
    print("\n1. Testing convergence plot...")
    convergence = [150, 135, 125, 122, 121, 120.5, 120.2]
    fig1 = plotter.plot_convergence(convergence, show=False)
    print("   ✓ Convergence plot created")
    
    # Test 2: Dose vs accuracy
    print("\n2. Testing dose vs accuracy plot...")
    doses = np.array([2.3e3, 4.6e3, 6.4e3, 9.1e3, 2.7e4])
    accuracies = np.array([1.5, 0.87, 0.54, 0.45, 0.33])
    fig2 = plotter.plot_accuracy_vs_dose(doses, accuracies, 
                                         threshold_dose=4.6e3, show=False)
    print("   ✓ Dose vs accuracy plot created")
    
    # Test 3: Method comparison
    print("\n3. Testing method comparison...")
    methods = ['Our Method', 'Standard SA', 'Exit Wave', 'Tilt Series']
    accs = [0.45, 1.2, 0.8, 0.6]
    fig3 = plotter.plot_comparison_bar(methods, accs, show=False)
    print("   ✓ Comparison plot created")
    
    # Test 4: Bond length distribution
    print("\n4. Testing bond length distribution...")
    bonds = np.random.normal(1.42, 0.05, 200)
    fig4 = plotter.plot_bond_length_distribution(bonds, show=False)
    print("   ✓ Bond length distribution created")
    
    # Test 5: Height distribution
    print("\n5. Testing height distribution...")
    heights = np.random.normal(0, 0.5, 200)
    fig5 = plotter.plot_height_distribution(heights, show=False)
    print("   ✓ Height distribution created")
    
    plt.close('all')
    
    print("\n" + "="*60)
    print("✓ Results plotting module test complete!")