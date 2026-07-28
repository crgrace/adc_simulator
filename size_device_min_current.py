#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:38:18 2026

@author: carlgrace
"""

import numpy as np

def size_device_min_current(
    gm_over_id=12.0,       # Chosen efficiency (V^-1)
    J_D_uA_per_um=15.0,    # Extracted from SPICE LUT at (gm/ID, L)
    c_gg_fF_per_um=1.2,    # Extracted gate cap density (fF/um)
    c_dd_fF_per_um=0.8,    # Extracted drain cap density (fF/um)
    Cs=1.0e-12,             # Sampling cap (F)
    Cf=1.0e-12,             # Feedback cap (F)
    Cs_next=0.5e-12,        # Load cap of next stage (F)
    t_settle=4.5e-9,        # Settling time budget (seconds)
    N_bits=10               # Resolution bits
):
    # Convert densities to SI units
    J_D = J_D_uA_per_um * 1e-6 / 1e-6       # A / m
    c_gg = c_gg_fF_per_um * 1e-15 / 1e-6    # F / m
    c_dd = c_dd_fF_per_um * 1e-15 / 1e-6    # F / m

    N_tau = (N_bits + 1.0) * np.log(2.0)
    
    # Initial guess for W (assume zero parasitics)
    beta_0 = Cf / (Cs + Cf)
    CL_0 = Cs_next + (Cf * Cs) / (Cs + Cf)
    gm_req = (2.0 * np.pi * N_tau * CL_0) / (2.0 * t_settle * beta_0)
    I_D = gm_req / gm_over_id
    W = I_D / J_D

    # Fixed-point iteration loop (converges in 3-5 steps)
    for iteration in range(10):
        # 1. Update parasitic capacitances
        Cp = c_gg * W
        C_out_par = c_dd * W

        # 2. Update feedback factor and effective load
        beta = Cf / (Cs + Cf + Cp)
        C_fb_series = (Cf * (Cs + Cp)) / (Cs + Cf + Cp)
        CL_eff = Cs_next + C_out_par + C_fb_series

        # 3. Recalculate required gm and updated current/width
        gm_req = (2.0 * np.pi * N_tau * CL_eff) / (2.0 * t_settle * beta)
        I_D = gm_req / gm_over_id
        W_new = I_D / J_D

        # Convergence check
        if np.abs(W_new - W) / W < 1e-4:
            break
        W = W_new

    return {
        "Width_um": W * 1e6,
        "Current_mA": I_D * 1e3,
        "gm_mS": gm_req * 1e3,
        "Beta": beta,
        "Iterations": iteration + 1
    }

# Example run
sizing = size_device_min_current(gm_over_id=12.0)
print(f"Optimal Width  : {sizing['Width_um']:.2f} µm")
print(f"Minimal Current: {sizing['Current_mA']:.3f} mA")
print(f"Final Beta     : {sizing['Beta']:.3f}")