"""
CO₂ Injection Rate Optimization
================================

Advanced algorithms for optimizing CO₂ injection rates in geological formations
while maintaining pressure constraints and maximizing storage efficiency.

Author: Shokhrukh Bokijonov
Email: shbokijon@gmail.com
ORCID: 0000-0002-0759-1283
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from typing import Dict, Optional
import warnings


class CO2InjectionOptimizer:
    """
    Optimizes CO₂ injection rates for geological storage applications.
    
    This class implements optimization algorithms based on:
    - Simplified analytical pressure models
    - Mass balance constraints
    - Operational rate limits
    - Safety factors for maximum pressure
    
    Example:
        >>> reservoir_props = {
        ...     'permeability': 100,
        ...     'porosity': 0.2,
        ...     'thickness': 50,
        ...     'radius': 5000,
        ...     'initial_pressure': 20e6,
        ...     'compressibility': 1e-9
        ... }
        >>> injection_constraints = {
        ...     'max_pressure': 25e6,
        ...     'max_rate': 50,
        ...     'total_mass': 1e6,
        ...     'safety_factor': 0.9
        ... }
        >>> optimizer = CO2InjectionOptimizer(reservoir_props, injection_constraints)
        >>> results = optimizer.optimize_injection_schedule(5*365.25*24*3600, 100)
    """
    
    def __init__(self, 
                 reservoir_properties: Dict[str, float],
                 injection_constraints: Dict[str, float]):
        """
        Initialize the CO₂ injection optimizer.
        
        Parameters:
            reservoir_properties: Dictionary with keys:
                - permeability (mD)
                - porosity (-)
                - thickness (m)
                - radius (m)
                - initial_pressure (Pa)
                - compressibility (1/Pa)
            injection_constraints: Dictionary with keys:
                - max_pressure (Pa)
                - max_rate (kg/s)
                - total_mass (kg)
                - safety_factor (-)
        """
        self.reservoir = reservoir_properties
        self.constraints = injection_constraints
        self.results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        required_reservoir = ['permeability', 'porosity', 'thickness', 
                            'radius', 'initial_pressure', 'compressibility']
        required_constraints = ['max_pressure', 'max_rate', 'total_mass', 'safety_factor']
        
        for param in required_reservoir:
            if param not in self.reservoir:
                raise ValueError(f"Missing reservoir parameter: {param}")
            if self.reservoir[param] <= 0:
                raise ValueError(f"{param} must be positive")
                
        for param in required_constraints:
            if param not in self.constraints:
                raise ValueError(f"Missing constraint parameter: {param}")
            if self.constraints[param] <= 0:
                raise ValueError(f"{param} must be positive")
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived reservoir parameters."""
        # Convert permeability to m²
        k_m2 = self.reservoir['permeability'] * 1e-15
        
        # Typical CO₂ properties
        mu = 5e-5  # Pa·s (viscosity)
        
        # Productivity index (simplified)
        rw = 0.1  # wellbore radius (m)
        self.PI = (2 * np.pi * k_m2 * self.reservoir['thickness']) / \
                  (mu * np.log(self.reservoir['radius'] / rw))
        
        # Storage coefficient
        self.storage_coef = (self.reservoir['porosity'] * 
                           self.reservoir['compressibility'] * 
                           self.reservoir['thickness'])
    
    def pressure_buildup(self, cumulative_volume: float) -> float:
        """
        Calculate pressure buildup from cumulative injection.
        
        Uses simplified analytical solution for pressure in bounded reservoir.
        
        Parameters:
            cumulative_volume: Total injected volume (m³)
            
        Returns:
            Pressure increase (Pa)
        """
        # Reservoir pore volume
        A = np.pi * self.reservoir['radius']**2
        pore_volume = A * self.reservoir['porosity'] * self.reservoir['thickness']
        
        # Volumetric strain
        volumetric_strain = cumulative_volume / pore_volume
        
        # Pressure increase (linear approximation for small strains)
        dp = volumetric_strain / self.reservoir['compressibility']
        
        return dp
    
    def optimize_injection_schedule(self, 
                                   time_horizon: float,
                                   n_periods: int = 50) -> Dict:
        """
        Optimize injection schedule over time horizon.
        
        Parameters:
            time_horizon: Total injection time (seconds)
            n_periods: Number of time periods
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n🔧 Starting optimization...")
        print(f"   Time periods: {n_periods}")
        print(f"   Duration: {time_horizon/(365.25*24*3600):.1f} years")
        
        # Time discretization
        times = np.linspace(0, time_horizon, n_periods + 1)
        dt = times[1] - times[0]
        
        # CO₂ density (kg/m³)
        rho_co2 = 700.0
        
        # Initial guess: constant rate to meet target
        target_rate = self.constraints['total_mass'] / time_horizon
        x0 = np.full(n_periods, min(target_rate, self.constraints['max_rate'] * 0.7))
        
        print(f"   Initial guess: {target_rate:.2f} kg/s")
        
        # Define optimization objective
        def objective(rates):
            """Minimize rate variations for smooth operation."""
            # Penalize variations
            smoothness = np.sum(np.diff(rates)**2)
            # Prefer higher rates early
            time_weight = np.sum(rates * (1.0 - np.linspace(0, 0.5, len(rates))))
            return smoothness - 0.01 * time_weight
        
        # Define constraints
        constraints = []
        
        # 1. Mass balance (equality)
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) * dt - self.constraints['total_mass']
        })
        
        # 2. Pressure constraint (inequality)
        def pressure_constraint(rates):
            """Ensure pressure stays below limit."""
            cumulative_mass = np.cumsum(rates) * dt
            cumulative_volume = cumulative_mass / rho_co2
            
            pressures = np.array([self.pressure_buildup(v) for v in cumulative_volume])
            max_pressure = np.max(pressures) + self.reservoir['initial_pressure']
            
            max_allowed = (self.constraints['max_pressure'] * 
                         self.constraints['safety_factor'])
            
            return max_allowed - max_pressure
        
        constraints.append({
            'type': 'ineq',
            'fun': pressure_constraint
        })
        
        # Bounds for rates
        bounds = [(0.0, self.constraints['max_rate']) for _ in range(n_periods)]
        
        # Optimize
        print("   Optimizing...")
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
        )
        
        if not result.success:
            print(f"   ⚠️  Warning: {result.message}")
            print("   Trying alternative method...")
            
            # Try differential evolution as backup
            result = differential_evolution(
                lambda x: objective(x) if pressure_constraint(x) >= 0 and 
                         abs(np.sum(x)*dt - self.constraints['total_mass']) < 0.01*self.constraints['total_mass']
                         else 1e10,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                atol=0,
                tol=0.01
            )
        
        # Calculate final results
        optimized_rates = result.x
        cumulative_mass = np.cumsum(optimized_rates) * dt
        cumulative_volume = cumulative_mass / rho_co2
        
        pressures = np.array([
            self.reservoir['initial_pressure'] + self.pressure_buildup(v)
            for v in cumulative_volume
        ])
        
        # Store results
        self.results = {
            'times': times[1:],
            'rates': optimized_rates,
            'pressures': pressures,
            'cumulative_mass': cumulative_mass,
            'objective_value': result.fun,
            'optimization_success': result.success,
            'max_pressure': np.max(pressures),
            'total_injected': np.sum(optimized_rates) * dt
        }
        
        # Print summary
        print(f"\n✅ Optimization complete!")
        print(f"   Success: {result.success}")
        print(f"   Total injected: {self.results['total_injected']/1000:.1f} tonnes")
        print(f"   Target: {self.constraints['total_mass']/1000:.1f} tonnes")
        print(f"   Achievement: {self.results['total_injected']/self.constraints['total_mass']*100:.1f}%")
        print(f"   Max pressure: {self.results['max_pressure']/1e6:.2f} MPa")
        print(f"   Avg rate: {np.mean(optimized_rates):.2f} kg/s")
        print(f"   Peak rate: {np.max(optimized_rates):.2f} kg/s")
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of optimization results.
        
        Parameters:
            save_path: Path to save plot (optional)
        """
        if not self.results:
            raise ValueError("No results to plot. Run optimization first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert time to years
        times_years = self.results['times'] / (365.25 * 24 * 3600)
        
        # Plot 1: Injection rates
        ax1 = axes[0, 0]
        ax1.plot(times_years, self.results['rates'], 'b-', linewidth=2.5, 
                label='Optimized Rate')
        ax1.axhline(y=self.constraints['max_rate'], color='r', linestyle='--', 
                   linewidth=1.5, label='Maximum Rate', alpha=0.7)
        ax1.fill_between(times_years, 0, self.results['rates'], alpha=0.2, color='blue')
        ax1.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Injection Rate (kg/s)', fontsize=12, fontweight='bold')
        ax1.set_title('CO₂ Injection Rate Schedule', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Pressure evolution
        ax2 = axes[0, 1]
        pressure_mpa = self.results['pressures'] / 1e6
        initial_p = self.reservoir['initial_pressure'] / 1e6
        max_p = self.constraints['max_pressure'] / 1e6
        safe_p = max_p * self.constraints['safety_factor']
        
        ax2.plot(times_years, pressure_mpa, 'g-', linewidth=2.5, label='Formation Pressure')
        ax2.axhline(y=initial_p, color='blue', linestyle=':', linewidth=2, 
                   label='Initial Pressure', alpha=0.7)
        ax2.axhline(y=safe_p, color='orange', linestyle='--', linewidth=1.5,
                   label='Safe Limit', alpha=0.7)
        ax2.axhline(y=max_p, color='r', linestyle='--', linewidth=1.5, 
                   label='Max Pressure', alpha=0.7)
        ax2.fill_between(times_years, initial_p, pressure_mpa, alpha=0.2, color='green')
        ax2.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pressure (MPa)', fontsize=12, fontweight='bold')
        ax2.set_title('Formation Pressure Evolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        
        # Plot 3: Cumulative injection
        ax3 = axes[1, 0]
        cumulative_tonnes = self.results['cumulative_mass'] / 1000
        target_tonnes = self.constraints['total_mass'] / 1000
        
        ax3.plot(times_years, cumulative_tonnes, 'purple', linewidth=3, 
                label='Cumulative Injection')
        ax3.axhline(y=target_tonnes, color='r', linestyle='--', linewidth=2, 
                   label=f'Target: {target_tonnes:.0f} tonnes', alpha=0.7)
        ax3.fill_between(times_years, 0, cumulative_tonnes, alpha=0.25, color='purple')
        ax3.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Cumulative CO₂ (tonnes)', fontsize=12, fontweight='bold')
        ax3.set_title('Cumulative CO₂ Injection', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=0)
        
        # Plot 4: Rate vs Pressure
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            self.results['rates'], 
            pressure_mpa,
            c=times_years,
            cmap='viridis',
            s=60,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        ax4.axhline(y=safe_p, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.axhline(y=max_p, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.set_xlabel('Injection Rate (kg/s)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Pressure (MPa)', fontsize=12, fontweight='bold')
        ax4.set_title('Rate vs Pressure Relationship', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Time (years)', fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str) -> None:
        """Export results to CSV file."""
        if not self.results:
            raise ValueError("No results to export. Run optimization first.")
        
        df = pd.DataFrame({
            'Time_years': self.results['times'] / (365.25 * 24 * 3600),
            'Time_days': self.results['times'] / (24 * 3600),
            'Time_seconds': self.results['times'],
            'Injection_Rate_kg_s': self.results['rates'],
            'Injection_Rate_tonnes_day': self.results['rates'] * 86.4,
            'Pressure_MPa': self.results['pressures'] / 1e6,
            'Pressure_Pa': self.results['pressures'],
            'Cumulative_Mass_tonnes': self.results['cumulative_mass'] / 1000,
            'Cumulative_Mass_kg': self.results['cumulative_mass']
        })
        
        df.to_csv(filename, index=False)
        print(f"\n📁 Results exported to: {filename}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")


def example_optimization():
    """Run example CO₂ storage optimization."""
    
    print("=" * 70)
    print("CO₂ INJECTION OPTIMIZATION - EXAMPLE CASE")
    print("=" * 70)
    print("\n🌍 Scenario: North Sea Saline Aquifer Storage Project")
    print("-" * 70)
    
    # Reservoir properties
    reservoir_props = {
        'permeability': 150,        # mD
        'porosity': 0.25,          # 25%
        'thickness': 100,          # m
        'radius': 10000,           # m
        'initial_pressure': 22e6,   # 22 MPa
        'compressibility': 8e-10    # 1/Pa
    }
    
    # Injection constraints
    injection_constraints = {
        'max_pressure': 27e6,       # 27 MPa
        'max_rate': 100,           # kg/s
        'total_mass': 2e6,         # 2000 tonnes
        'safety_factor': 0.85       # 85%
    }
    
    print("\n📋 Reservoir Properties:")
    print(f"   Permeability: {reservoir_props['permeability']} mD")
    print(f"   Porosity: {reservoir_props['porosity']:.1%}")
    print(f"   Thickness: {reservoir_props['thickness']} m")
    print(f"   Initial pressure: {reservoir_props['initial_pressure']/1e6:.1f} MPa")
    
    print("\n🎯 Injection Constraints:")
    print(f"   Target: {injection_constraints['total_mass']/1000:.0f} tonnes CO₂")
    print(f"   Max rate: {injection_constraints['max_rate']} kg/s")
    print(f"   Max pressure: {injection_constraints['max_pressure']/1e6:.1f} MPa")
    print(f"   Safety margin: {injection_constraints['safety_factor']:.0%}")
    
    # Create optimizer
    optimizer = CO2InjectionOptimizer(reservoir_props, injection_constraints)
    
    # Run optimization
    time_horizon = 3 * 365.25 * 24 * 3600  # 3 years
    results = optimizer.optimize_injection_schedule(time_horizon, n_periods=75)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Visualization
    print("\n📊 Creating visualization...")
    optimizer.plot_results('co2_optimization_results.png')
    
    # Export data
    print("\n💾 Exporting data...")
    optimizer.export_results('co2_injection_schedule.csv')
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  📊 co2_optimization_results.png")
    print("  📁 co2_injection_schedule.csv\n")


if __name__ == "__main__":
    example_optimization()
