"""
CO₂ Injection Rate Optimization
================================

Advanced algorithms for optimizing CO₂ injection rates in geological formations
while maintaining pressure constraints and maximizing storage efficiency.

This module implements physics-based optimization using:
- Darcy's Law for flow in porous media
- Pressure diffusion equations
- Storage capacity constraints
- Safety and regulatory limits

Author: Shokhrukh Bokijonov
Email: shbokijon@gmail.com
ORCID: 0000-0002-0759-1283
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings
from dataclasses import dataclass


@dataclass
class ReservoirProperties:
    """
    Container for reservoir properties.
    
    Attributes:
        permeability: Formation permeability (mD)
        porosity: Formation porosity (fraction)
        thickness: Net formation thickness (m)
        radius: Drainage radius (m)
        initial_pressure: Initial formation pressure (Pa)
        compressibility: Total compressibility (1/Pa)
        temperature: Formation temperature (K)
    """
    permeability: float
    porosity: float
    thickness: float
    radius: float
    initial_pressure: float
    compressibility: float
    temperature: float = 323.15  # 50°C default


@dataclass
class InjectionConstraints:
    """
    Container for injection constraints.
    
    Attributes:
        max_pressure: Maximum allowable pressure (Pa)
        max_rate: Maximum injection rate (kg/s)
        total_mass: Total CO₂ mass to inject (kg)
        safety_factor: Safety factor for pressure (fraction)
        min_rate: Minimum injection rate (kg/s)
    """
    max_pressure: float
    max_rate: float
    total_mass: float
    safety_factor: float
    min_rate: float = 0.0


class CO2InjectionOptimizer:
    """
    Optimizes CO₂ injection rates for geological storage applications.
    
    This class implements advanced optimization algorithms based on:
    - Darcy's Law for flow in porous media
    - Pressure diffusion equations
    - Storage capacity constraints
    - Safety and regulatory limits
    
    Example:
        >>> reservoir = ReservoirProperties(
        ...     permeability=100,
        ...     porosity=0.2,
        ...     thickness=50,
        ...     radius=5000,
        ...     initial_pressure=20e6,
        ...     compressibility=1e-9
        ... )
        >>> constraints = InjectionConstraints(
        ...     max_pressure=25e6,
        ...     max_rate=50,
        ...     total_mass=1e6,
        ...     safety_factor=0.9
        ... )
        >>> optimizer = CO2InjectionOptimizer(reservoir, constraints)
        >>> results = optimizer.optimize_injection_schedule(
        ...     time_horizon=5*365.25*24*3600,
        ...     n_periods=100
        ... )
    """
    
    # Physical constants
    GAS_CONSTANT = 8.314  # J/(mol·K)
    CO2_MOLAR_MASS = 44.01e-3  # kg/mol
    VISCOSITY_CO2 = 1e-5  # Pa·s (approximate for supercritical CO₂)
    
    def __init__(self, 
                 reservoir_properties: ReservoirProperties,
                 injection_constraints: InjectionConstraints):
        """
        Initialize the CO₂ injection optimizer.
        
        Parameters:
            reservoir_properties: ReservoirProperties instance
            injection_constraints: InjectionConstraints instance
        """
        self.reservoir = reservoir_properties
        self.constraints = injection_constraints
        self.results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
        
    def _validate_inputs(self) -> None:
        """Validate input parameters for physical consistency."""
        # Validate reservoir properties
        if self.reservoir.permeability <= 0:
            raise ValueError("Permeability must be positive")
        if not 0 < self.reservoir.porosity < 1:
            raise ValueError("Porosity must be between 0 and 1")
        if self.reservoir.thickness <= 0:
            raise ValueError("Thickness must be positive")
        if self.reservoir.radius <= 0:
            raise ValueError("Radius must be positive")
        if self.reservoir.initial_pressure <= 0:
            raise ValueError("Initial pressure must be positive")
        if self.reservoir.compressibility <= 0:
            raise ValueError("Compressibility must be positive")
            
        # Validate constraints
        if self.constraints.max_pressure <= self.reservoir.initial_pressure:
            raise ValueError("Maximum pressure must exceed initial pressure")
        if self.constraints.max_rate <= 0:
            raise ValueError("Maximum rate must be positive")
        if self.constraints.total_mass <= 0:
            raise ValueError("Total mass must be positive")
        if not 0 < self.constraints.safety_factor <= 1:
            raise ValueError("Safety factor must be between 0 and 1")
            
    def _calculate_derived_parameters(self) -> None:
        """Calculate derived reservoir parameters."""
        # Convert permeability from mD to m²
        k_m2 = self.reservoir.permeability * 1e-15
        
        # Hydraulic diffusivity (m²/s)
        self.hydraulic_diffusivity = (
            k_m2 / 
            (self.reservoir.porosity * 
             self.reservoir.compressibility * 
             self.VISCOSITY_CO2)
        )
        
        # Injectivity index (m³/(Pa·s))
        wellbore_radius = 0.1  # m (typical)
        self.injectivity_index = (
            2 * np.pi * k_m2 * self.reservoir.thickness / 
            (self.VISCOSITY_CO2 * np.log(self.reservoir.radius / wellbore_radius))
        )
        
        # Storage coefficient
        self.storage_coefficient = (
            self.reservoir.porosity * 
            self.reservoir.compressibility * 
            self.reservoir.thickness
        )
        
    def pressure_response(self, t: float, injection_rate: float) -> float:
        """
        Calculate pressure response to injection using Theis solution.
        
        This implements the analytical solution for transient pressure in 
        a confined aquifer with constant-rate injection.
        
        Parameters:
            t: Time since injection start (seconds)
            injection_rate: CO₂ injection rate (kg/s)
            
        Returns:
            Pressure increase at wellbore (Pa)
        """
        if t <= 1e-6:  # Avoid division by zero
            return 0.0
            
        # Dimensionless time
        t_d = self.hydraulic_diffusivity * t / (self.reservoir.radius**2)
        
        # Convert mass rate to volumetric rate (approximate)
        co2_density = 700  # kg/m³ (supercritical CO₂)
        volumetric_rate = injection_rate / co2_density
        
        # Theis exponential integral approximation
        if t_d < 0.01:
            # Early time approximation
            ei_approx = -0.5772 - np.log(t_d)
        else:
            # Late time approximation
            ei_approx = np.log(4 * t_d) - 0.5772
            
        # Pressure change
        pressure_increase = (
            volumetric_rate / 
            (4 * np.pi * self.injectivity_index) * 
            ei_approx
        )
        
        return max(0, pressure_increase)
    
    def optimize_injection_schedule(self, 
                                   time_horizon: float,
                                   n_periods: int = 50) -> Dict:
        """
        Optimize injection schedule over specified time horizon.
        
        This method solves a constrained optimization problem to find the
        injection rate schedule that maximizes efficiency while respecting
        pressure and rate constraints.
        
        Parameters:
            time_horizon: Total injection time (seconds)
            n_periods: Number of time periods for optimization
            
        Returns:
            Dictionary containing optimization results:
                - times: Time array (s)
                - rates: Optimized injection rates (kg/s)
                - pressures: Resulting pressures (Pa)
                - cumulative_mass: Cumulative injected mass (kg)
                - objective_value: Final objective function value
                - optimization_success: Boolean indicating convergence
                - max_pressure: Maximum pressure reached (Pa)
                - total_injected: Total mass injected (kg)
        """
        
        # Time discretization
        times = np.linspace(0, time_horizon, n_periods + 1)
        dt = times[1] - times[0]
        
        # Initial guess: constant rate
        target_rate = self.constraints.total_mass / time_horizon
        initial_rate = min(target_rate, self.constraints.max_rate * 0.8)
        x0 = np.full(n_periods, initial_rate)
        
        # Bounds: minimum to maximum rate
        bounds = [
            (self.constraints.min_rate, self.constraints.max_rate) 
            for _ in range(n_periods)
        ]
        
        # Constraints
        constraints = [
            # Mass balance constraint (equality)
            {
                'type': 'eq',
                'fun': lambda x: np.sum(x) * dt - self.constraints.total_mass
            },
            # Pressure constraints (inequality)
            {
                'type': 'ineq',
                'fun': lambda x: self._pressure_constraint(x, times, dt)
            }
        ]
        
        # Objective function: minimize injection variation (smooth operation)
        def objective(x):
            # Penalize rate variations for smooth operation
            rate_variation = np.sum(np.diff(x)**2)
            # Prefer earlier injection (time value of storage)
            time_penalty = np.sum(x * np.arange(len(x))) / len(x)
            return rate_variation + 0.01 * time_penalty
        
        # Optimize
        print("Starting optimization...")
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000, 
                'ftol': 1e-9,
                'disp': True
            }
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not fully converge: {result.message}")
        
        # Calculate results
        optimized_rates = result.x
        pressures = self._calculate_pressure_history(optimized_rates, times, dt)
        cumulative_mass = np.cumsum(optimized_rates) * dt
        
        # Store results
        self.results = {
            'times': times[1:],  # Exclude t=0
            'rates': optimized_rates,
            'pressures': pressures + self.reservoir.initial_pressure,
            'cumulative_mass': cumulative_mass,
            'objective_value': result.fun,
            'optimization_success': result.success,
            'max_pressure': np.max(pressures + self.reservoir.initial_pressure),
            'total_injected': np.sum(optimized_rates) * dt,
            'pressure_constraint_active': self._check_active_constraints(
                optimized_rates, times, dt
            )
        }
        
        print("\n✅ Optimization complete!")
        print(f"Total injected: {self.results['total_injected']/1000:.1f} tonnes")
        print(f"Max pressure: {self.results['max_pressure']/1e6:.2f} MPa")
        print(f"Success: {result.success}")
        
        return self.results
    
    def _pressure_constraint(self, rates: np.ndarray, 
                            times: np.ndarray, 
                            dt: float) -> float:
        """
        Calculate pressure constraint violation.
        
        Returns positive value if constraint satisfied, negative if violated.
        """
        pressures = self._calculate_pressure_history(rates, times, dt)
        max_allowed = (
            (self.constraints.max_pressure - self.reservoir.initial_pressure) * 
            self.constraints.safety_factor
        )
        return max_allowed - np.max(pressures)
    
    def _calculate_pressure_history(self, 
                                    rates: np.ndarray, 
                                    times: np.ndarray, 
                                    dt: float) -> np.ndarray:
        """
        Calculate pressure history using superposition principle.
        
        This method applies the superposition principle to calculate the
        pressure response from a sequence of injection rate changes.
        """
        n = len(rates)
        pressures = np.zeros(n)
        
        for i in range(n):
            t_current = times[i + 1]
            pressure_increase = 0.0
            
            # Superposition: sum contributions from all previous injections
            for j in range(i + 1):
                if rates[j] > 0:
                    time_since_injection = t_current - times[j]
                    if time_since_injection > 0:
                        # Rate contribution (differential approach)
                        if j > 0:
                            rate_change = rates[j] - rates[j-1]
                        else:
                            rate_change = rates[j]
                        
                        pressure_increase += self.pressure_response(
                            time_since_injection, 
                            rate_change
                        )
            
            pressures[i] = pressure_increase
        
        return pressures
    
    def _check_active_constraints(self, 
                                  rates: np.ndarray, 
                                  times: np.ndarray, 
                                  dt: float) -> Dict:
        """Check which constraints are active at optimum."""
        pressures = self._calculate_pressure_history(rates, times, dt)
        max_pressure = np.max(pressures)
        max_allowed = (
            (self.constraints.max_pressure - self.reservoir.initial_pressure) * 
            self.constraints.safety_factor
        )
        
        return {
            'pressure_constraint': abs(max_pressure - max_allowed) < 1e-3 * max_allowed,
            'max_rate_constraint': any(abs(rates - self.constraints.max_rate) < 0.01),
            'min_rate_constraint': any(abs(rates - self.constraints.min_rate) < 0.01)
        }
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of optimization results.
        
        Parameters:
            save_path: Path to save the plot. If None, display only.
        """
        if not self.results:
            raise ValueError("No results to plot. Run optimization first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert times to years for readability
        times_years = self.results['times'] / (365.25 * 24 * 3600)
        
        # Plot 1: Injection rates
        ax1 = axes[0, 0]
        ax1.plot(times_years, self.results['rates'], 'b-', linewidth=2, 
                label='Optimized Rate')
        ax1.axhline(y=self.constraints.max_rate, color='r', linestyle='--', 
                   linewidth=1.5, label='Maximum Rate')
        if self.constraints.min_rate > 0:
            ax1.axhline(y=self.constraints.min_rate, color='orange', 
                       linestyle='--', linewidth=1.5, label='Minimum Rate')
        ax1.set_xlabel('Time (years)', fontsize=11)
        ax1.set_ylabel('Injection Rate (kg/s)', fontsize=11)
        ax1.set_title('CO₂ Injection Rate Schedule', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pressure evolution
        ax2 = axes[0, 1]
        pressure_mpa = self.results['pressures'] / 1e6
        initial_p = self.reservoir.initial_pressure / 1e6
        max_p = self.constraints.max_pressure / 1e6
        
        ax2.plot(times_years, pressure_mpa, 'g-', linewidth=2, 
                label='Formation Pressure')
        ax2.axhline(y=initial_p, color='blue', linestyle=':', 
                   linewidth=1.5, label='Initial Pressure')
        ax2.axhline(y=max_p, color='r', linestyle='--', 
                   linewidth=1.5, label='Maximum Pressure')
        ax2.fill_between(times_years, initial_p, max_p, alpha=0.1, color='yellow')
        ax2.set_xlabel('Time (years)', fontsize=11)
        ax2.set_ylabel('Pressure (MPa)', fontsize=11)
        ax2.set_title('Formation Pressure Evolution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative injection
        ax3 = axes[1, 0]
        cumulative_tonnes = self.results['cumulative_mass'] / 1000
        target_tonnes = self.constraints.total_mass / 1000
        
        ax3.plot(times_years, cumulative_tonnes, 'purple', linewidth=2.5, 
                label='Cumulative Injection')
        ax3.axhline(y=target_tonnes, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Target: {target_tonnes:.0f} tonnes')
        ax3.fill_between(times_years, 0, cumulative_tonnes, alpha=0.2, color='purple')
        ax3.set_xlabel('Time (years)', fontsize=11)
        ax3.set_ylabel('Cumulative CO₂ (tonnes)', fontsize=11)
        ax3.set_title('Cumulative CO₂ Injection', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rate vs Pressure relationship
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            self.results['rates'], 
            pressure_mpa, 
            c=times_years, 
            cmap='viridis', 
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        ax4.axhline(y=max_p, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.set_xlabel('Injection Rate (kg/s)', fontsize=11)
        ax4.set_ylabel('Pressure (MPa)', fontsize=11)
        ax4.set_title('Rate vs Pressure Relationship', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Time (years)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, filename: str) -> None:
        """
        Export results to CSV file.
        
        Parameters:
            filename: Output CSV filename
        """
        if not self.results:
            raise ValueError("No results to export. Run optimization first.")
        
        df = pd.DataFrame({
            'Time_years': self.results['times'] / (365.25 * 24 * 3600),
            'Time_seconds': self.results['times'],
            'Time_days': self.results['times'] / (24 * 3600),
            'Injection_Rate_kg_s': self.results['rates'],
            'Injection_Rate_tonnes_day': self.results['rates'] * 86.4,
            'Pressure_Pa': self.results['pressures'],
            'Pressure_MPa': self.results['pressures'] / 1e6,
            'Pressure_increase_MPa': (self.results['pressures'] - 
                                     self.reservoir.initial_pressure) / 1e6,
            'Cumulative_Mass_kg': self.results['cumulative_mass'],
            'Cumulative_Mass_tonnes': self.results['cumulative_mass'] / 1000
        })
        
        df.to_csv(filename, index=False)
        print(f"📁 Results exported to: {filename}")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {', '.join(df.columns)}")


def example_optimization():
    """
    Run an example optimization case.
    
    This demonstrates the full workflow for a typical CO₂ storage project
    in a North Sea-type saline aquifer formation.
    """
    print("=" * 60)
    print("CO₂ INJECTION OPTIMIZATION - EXAMPLE CASE")
    print("=" * 60)
    print("\nScenario: North Sea Saline Aquifer Storage Project")
    print("-" * 60)
    
    # Define reservoir properties (typical North Sea conditions)
    reservoir = ReservoirProperties(
        permeability=150,           # mD - good permeability
        porosity=0.25,             # 25% - good porosity
        thickness=100,             # m - thick formation
        radius=10000,              # m - large drainage area
        initial_pressure=22e6,      # Pa (22 MPa) - ~2.2 km depth
        compressibility=8e-10,      # 1/Pa - typical sandstone
        temperature=333.15          # K (60°C)
    )
    
    # Define injection constraints
    constraints = InjectionConstraints(
        max_pressure=27e6,          # Pa (27 MPa) - fracture pressure
        max_rate=100,              # kg/s - well capacity
        total_mass=2e6,            # kg (2000 tonnes)
        safety_factor=0.85,         # 85% safety margin
        min_rate=5.0               # kg/s - minimum stable rate
    )
    
    print("\n📋 Reservoir Properties:")
    print(f"   Permeability: {reservoir.permeability} mD")
    print(f"   Porosity: {reservoir.porosity:.1%}")
    print(f"   Thickness: {reservoir.thickness} m")
    print(f"   Initial pressure: {reservoir.initial_pressure/1e6:.1f} MPa")
    
    print("\n🎯 Injection Constraints:")
    print(f"   Target mass: {constraints.total_mass/1000:.0f} tonnes")
    print(f"   Max rate: {constraints.max_rate} kg/s")
    print(f"   Max pressure: {constraints.max_pressure/1e6:.1f} MPa")
    print(f"   Safety factor: {constraints.safety_factor:.0%}")
    
    # Create optimizer
    optimizer = CO2InjectionOptimizer(reservoir, constraints)
    
    # Run optimization for 3 years
    time_horizon = 3 * 365.25 * 24 * 3600  # 3 years in seconds
    
    print(f"\n⏱️  Optimization period: 3 years")
    print(f"   Time steps: 75")
    print("\n🔧 Running optimization algorithm...")
    print("-" * 60)
    
    results = optimizer.optimize_injection_schedule(
        time_horizon=time_horizon,
        n_periods=75
    )
    
    # Display summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n✅ Optimization status: {'SUCCESS' if results['optimization_success'] else 'PARTIAL'}")
    print(f"\n📊 Key Metrics:")
    print(f"   Total CO₂ injected: {results['total_injected']/1000:.1f} tonnes")
    print(f"   Target achievement: {results['total_injected']/constraints.total_mass:.1%}")
    print(f"   Maximum pressure: {results['max_pressure']/1e6:.2f} MPa")
    print(f"   Pressure utilization: {(results['max_pressure']-reservoir.initial_pressure)/(constraints.max_pressure-reservoir.initial_pressure):.1%}")
    print(f"   Average rate: {np.mean(results['rates']):.2f} kg/s")
    print(f"   Peak rate: {np.max(results['rates']):.2f} kg/s")
    print(f"   Minimum rate: {np.min(results['rates']):.2f} kg/s")
    
    print(f"\n🔒 Active Constraints:")
    for constraint, is_active in results['pressure_constraint_active'].items():
        status = "✓ ACTIVE" if is_active else "○ Inactive"
        print(f"   {constraint}: {status}")
    
    # Create visualization
    print("\n📊 Generating visualization...")
    optimizer.plot_results('co2_optimization_results.png')
    
    # Export results
    print("\n💾 Exporting detailed results...")
    optimizer.export_results('co2_injection_schedule.csv')
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n📁 Output files:")
    print("   - co2_optimization_results.png (visualization)")
    print("   - co2_injection_schedule.csv (detailed data)")
    print("\n")


if __name__ == "__main__":
    # Run example optimization
    example_optimization()
