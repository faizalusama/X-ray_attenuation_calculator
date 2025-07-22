import numpy as np
import matplotlib.pyplot as plt
from xraydb import material_mu
import re


def calculate_weight_fractions(fractions, molar_masses, fraction_type):
    if fraction_type == 'mol':
        mol_fractions = {k: v / 100 for k, v in fractions.items()}
        unnormalized_weights = {k: mol_fractions[k] * molar_masses[k] for k in mol_fractions}
        total_weight = sum(unnormalized_weights.values())
        weight_fractions = {k: unnormalized_weights[k] / total_weight for k in mol_fractions}
    elif fraction_type == 'wt':
        weight_fractions = {k: v / 100 for k, v in fractions.items()}
        total_weight = sum(weight_fractions.values())
        weight_fractions = {k: v / total_weight for k, v in weight_fractions.items()}
    else:
        raise ValueError("fraction_type must be 'mol' or 'wt'")
    return weight_fractions


def format_component_name(comp):
    # Only add subscript markers
    return re.sub(r"(\\d+)", r"_{\1}", comp)


def calculate_total_mu(weight_fractions, energy, system_density):
    mu = np.zeros_like(energy)
    for comp, weight_frac in weight_fractions.items():
        mu += weight_frac * material_mu(comp, energy, density=system_density * weight_frac)
    return mu


def plot_transmission(energy, transmission, t, E, e_start, e_end, fractions):
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    for spine in ['bottom', 'left', 'right', 'top']:
        ax.spines[spine].set_linewidth(2)
    plt.plot(energy, transmission, label='Transmission')
    plt.plot(energy, 1-transmission, label='Attenuation')
    plt.axvline(x=E, color='purple', linestyle='--', label=f'Reference at {E/1000:.0f} keV')
    attenuation_at_E = 1 - np.interp(E, energy, transmission)
    plt.axhline(y=attenuation_at_E, color='red', linestyle='--', label=f"Attenuation at {E/1000:.0f} keV: {attenuation_at_E:.2f}")
    
    # Annotation
    plt.text(E + 200, attenuation_at_E + 0.02, f'Attenuation: {attenuation_at_E:.2f}', color='black', fontsize= 12, fontweight ='bold')
    
    if len(fractions) > 1:
        title = f"X-ray Absorption through {t} µm thick material (multi-component system)"
    else:
        comp_name = list(fractions.keys())[0]
        title = f"X-ray Absorption through {t} µm thick {format_component_name(comp_name)} (single-component system)"
    plt.title(title, pad=25, fontsize= 14, fontweight ='bold')

    plt.xlim(e_start, e_end)
    plt.tick_params(axis='both', which='major', labelsize=14, width=2)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel('Energy (eV)', fontsize= 14, fontweight ='bold')
    plt.ylabel('Transmitted / Attenuated fraction', fontsize= 14, fontweight ='bold')
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.show()

    
def print_one_over_e_depth(mu, energy, E):
    mu_at_target = np.interp(E, energy, mu)
    depth_mm = 10 / mu_at_target
    depth_um = depth_mm * 1e3
    print(f"1/e attenuation depth at {E/1000:.1f} keV: {depth_mm:.4f} mm ({depth_um:.1f} µm)")


def run_with_inputs(fractions, molar_masses, fraction_type, density, t, e_start, e_end, E):
    energy = np.linspace(e_start, e_end, 201)
    weight_fractions = calculate_weight_fractions(fractions, molar_masses, fraction_type)
    print("\nNormalized weight fractions:")
    for comp, frac in weight_fractions.items():
        print(f"{format_component_name(comp)}: {frac:.4f}")
    mu = calculate_total_mu(weight_fractions, energy, density)
    transmission = np.exp(-(t / 10000) * mu)
    plot_transmission(energy, transmission, t, E, e_start, e_end, fractions)
    print_one_over_e_depth(mu, energy, E)


def ordinal(n):
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def main():
    n = int(input("Enter number of components: "))
    fractions = {}
    molar_masses = {}

    for i in range(1, n + 1):
        if n == 1:
            comp = input("Enter the component name: ")
        else:
            comp = input(f"Enter {ordinal(i)} component name: ")
        percent = float(input(f"Enter percentage of {comp}: "))
        fractions[comp] = percent

        molar_mass = float(input(f"Enter molar mass of {comp} in g/mol: "))
        molar_masses[comp] = molar_mass

    fraction_type = input("Are these fractions in mol% or weight%? (type only: mol/wt): ").strip().lower()
    density = float(input("Enter system bulk density (in g/cm^3): "))
    t = float(input("Enter sample thickness (µm): "))
    e_start = float(input("Enter starting energy (eV): "))
    e_end = float(input("Enter ending energy (eV): "))
    E = float(input("Enter target energy (eV): "))

    run_with_inputs(fractions, molar_masses, fraction_type, density, t, e_start, e_end, E)


if __name__ == "__main__":
    main()