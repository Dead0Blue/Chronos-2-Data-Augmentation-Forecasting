import pandas as pd
import numpy as np
import os

def simulate_ipp(mean_psi, var_psi, tau=0.1, zeta=0.5, T=300, sub_T=30):
    """
    Simulate Interrupted Poisson Process (IPP) to generate instantaneous traffic.
    Based on Section II-C of KTH.pdf.
    
    mean_psi: Average traffic (E[Ψ]) in the 5-min (300s) window.
    var_psi: Variance of traffic (Var[Ψ]) in the 5-min window.
    tau: Transition rate OFF -> ON
    zeta: Transition rate ON -> OFF
    T: Sampling time of original data (300s)
    sub_T: Targeted instantaneous time step (30s)
    """
    # Number of sub-samples per window
    N_samples = int(T / sub_T)
    
    if mean_psi <= 0:
        return np.zeros(N_samples)

    # Calculate lambda and E(psi) based on Eq. (11) and (12)
    # Using the approximation for (tau + zeta)T >> 1
    # Eq (11): lambda = ((tau + zeta) / (tau * T)) * [ (Var(Psi) / E(Psi)^2) - (2*zeta / (tau * T * (tau + zeta))) ]
    # Wait, Eq (11) in paper text actually looks like:
    # lambda = ((tau + zeta) / (tau * T)) * [ (Var(Psi) / E(Psi)^2) - ... ]
    # Re-checking the paper's LaTeX/PDF logic for lambda:
    # lambda = ( (tau + zeta)/(tau * T) ) * [ (Var(Psi)/E(Psi)^2) - 1 ] ? No.
    
    # Let's use the simplified relationship if (tau+zeta)T >> 1:
    # E(U) = lambda * tau / (tau + zeta) * T
    # Var(U) approx E(U) * [1 + 2 * lambda * zeta / (tau + zeta)^2]
    
    # From Eq (9) and (10):
    # E(Psi) = E(U) * E(psi_single)
    # Var(Psi) = E(U) * Var(psi_single) + Var(U) * E(psi_single)^2
    
    # Assuming exponential distribution for psi_single: E(psi_single) = theta, Var(psi_single) = theta^2
    # E(Psi) = E(U) * theta
    # Var(Psi) = E(U) * theta^2 + Var(U) * theta^2 = theta^2 * (E(U) + Var(U))
    
    # If we assume tau, zeta are fixed (0.1, 0.5):
    p_on = tau / (tau + zeta)
    # E(U) = lambda * p_on * T
    # Var(U) = E(U) * (1 + 2 * lambda * zeta / (tau + zeta)**2)
    
    # Substitutions:
    # theta = E(Psi) / E(U)
    # Var(Psi) = (E(Psi)/E(U))**2 * (E(U) + Var(U))
    # Var(Psi) / E(Psi)**2 = (E(U) + Var(U)) / E(U)**2 = 1/E(U) + Var(U)/E(U)**2
    
    # Let R = Var(Psi) / E(Psi)**2 (dispersion ratio)
    # R = 1/E(U) + [E(U)*(1 + 2*lambda*zeta/(tau+zeta)**2)] / E(U)**2
    # R = 2/E(U) + 2*lambda*zeta / (E(U)*(tau+zeta)**2)
    # Since E(U) = lambda * p_on * T:
    # R = 2/(lambda * p_on * T) + 2*lambda*zeta / (lambda * p_on * T * (tau+zeta)**2)
    # R = 2/(lambda * p_on * T) + 2*zeta / (p_on * T * (tau+zeta)**2)
    
    # Solve for lambda:
    # 2/(lambda * p_on * T) = R - [2*zeta / (p_on * T * (tau+zeta)**2)]
    rhs = (var_psi / (mean_psi**2)) - (2 * zeta / (p_on * T * (tau + zeta)**2))
    
    if rhs <= 0:
        # Fallback if the variance is too low for the IPP model parameters
        lambda_val = 1.0 # arbitrary small value
    else:
        lambda_val = 2 / (p_on * T * rhs)
    
    # Now find E(psi_single) = theta
    e_u = lambda_val * p_on * T
    theta = mean_psi / e_u if e_u > 0 else 0
    
    # Generate instantaneous samples
    # For each 30s interval, we check if we are ON or OFF
    # Transition probabilities for 30s:
    # P(ON|OFF) = (tau / (tau+zeta)) * (1 - exp(-(tau+zeta)*sub_T))
    # P(OFF|ON) = (zeta / (tau+zeta)) * (1 - exp(-(tau+zeta)*sub_T))
    
    p_on_off = (tau / (tau + zeta)) * (1 - np.exp(-(tau + zeta) * sub_T))
    p_off_on = (zeta / (tau + zeta)) * (1 - np.exp(-(tau + zeta) * sub_T))
    
    state = 1 if np.random.rand() < p_on else 0 # 1 for ON, 0 for OFF
    samples = []
    
    for _ in range(N_samples):
        if state == 1:
            # Poisson arrivals during 30s
            u_sub = np.random.poisson(lambda_val * sub_T / T * (T/sub_T)) # This is just lambda_val * sub_T / T scaled?
            # Actually lambda is arrivals per T. So per sub_T it is lambda * (sub_T / T).
            u_sub = np.random.poisson(lambda_val * (sub_T / T))
            val = np.sum(np.random.exponential(theta, u_sub)) if u_sub > 0 else 0
            samples.append(val)
            # Transition?
            if np.random.rand() < p_off_on:
                state = 0
        else:
            samples.append(0)
            if np.random.rand() < p_on_off:
                state = 1
                
    # Normalize to ensure the average matches mean_psi (scaling to compensate for random noise)
    arr = np.array(samples)
    if np.sum(arr) > 0:
        arr = arr * (mean_psi * N_samples / np.sum(arr))
    else:
        # If all zeros but mean_psi > 0, just distribute mean_psi
        arr = np.full(N_samples, mean_psi)
        
    return arr

def main():
    input_path = 'c:/Users/PC/Desktop/chronos/project/histo_trafic.csv'
    output_path = 'c:/Users/PC/Desktop/chronos/project/augmented_trafic.csv'
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, sep=';', encoding='latin-1')
    
    # Clean columns
    df = df[['secteur', 'site', 'tstamp', 'trafic_mbps']].copy()
    df['trafic_mbps'] = pd.to_numeric(df['trafic_mbps'], errors='coerce').fillna(0)
    
    # We need to assume a site/sector to work with, or do all.
    # The request says "all this for me", so let's do all sectors but maybe limited time to avoid huge file.
    # Or just do all, 24k rows -> 240k rows is manageable.
    
    augmented_data = []
    
    print("Augmenting data using IPP model...")
    # Dispersion ratio k = 0.05 as per our plan
    k = 0.05
    
    # To speed up, we can process row by row
    for i, row in df.iterrows():
        mean_val = row['trafic_mbps']
        var_val = k * (mean_val**2) if mean_val > 0 else 0
        
        instantaneous = simulate_ipp(mean_val, var_val)
        
        for j, val in enumerate(instantaneous):
            augmented_data.append({
                'secteur': row['secteur'],
                'site': row['site'],
                'tstamp_orig': row['tstamp'],
                'sub_interval': j,
                'trafic_instantaneous': val
            })
            
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} rows...")
            
    aug_df = pd.DataFrame(augmented_data)
    print(f"Saving to {output_path}...")
    aug_df.to_csv(output_path, index=False)
    print("Augmentation complete.")

if __name__ == "__main__":
    main()
