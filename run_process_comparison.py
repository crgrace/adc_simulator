# run_process_comparison.py
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

from adc_simulator import ADCConfig, FullPipelinedADCSimulator


# ==============================================================================
# HELPER: SCALE CAPACITORS TO ACHIEVE EXACT TARGET ENOB
# ==============================================================================
def scale_config_for_target_enob(cfg: ADCConfig, target_enob: float, f_clk_target: float) -> ADCConfig:
    """Scales sampling capacitors and parasitic loads to hit exact target ENOB."""
    target_snr_db = 6.02 * target_enob + 1.76
    
    adc_init = FullPipelinedADCSimulator.from_config(replace(cfg, f_clk=f_clk_target))
    _, summary = adc_init.run_static_analysis(f_clk=f_clk_target, t_non_overlap=cfg.t_non_overlap)
    current_snr_db = float(summary["Thermal SNR"].split()[0])
    
    scale_factor = 10 ** ((target_snr_db - current_snr_db) / 10.0)
    
    return replace(
        cfg,
        sha_Cs=cfg.sha_Cs * scale_factor,
        sha_Cf=cfg.sha_Cf * scale_factor,
        sha_Cp=cfg.sha_Cp * scale_factor,
        sha_C_cmfb=cfg.sha_C_cmfb * scale_factor,
        Cs_profile=[cs * scale_factor for cs in cfg.Cs_profile],
        mdac_Cp=cfg.mdac_Cp * scale_factor,
        mdac_C_cmfb=cfg.mdac_C_cmfb * scale_factor
    )


# ==============================================================================
# EQUAL-ENOB PROCESS COMPARISON SWEEP
# ==============================================================================
def run_device_comparison(
    cfg_1v2_raw: ADCConfig, 
    cfg_2v5_raw: ADCConfig, 
    f_clk_target=100e6, 
    target_enob=9.5,
    L_1v2_nm=65,
    L_2v5_nm=280
):
    f_target_mhz = f_clk_target / 1e6
    target_snr_db = 6.02 * target_enob + 1.76
    total_bits = cfg_1v2_raw.num_stages + cfg_1v2_raw.quantizer_bits

    print("="*80)
    print(" 65nm COMMERCIAL CMOS IMPLEMENTATION DECISION STUDY")
    print(f" Nominal Resolution: {total_bits}-Bit  |  Target ENOB: {target_enob:.2f} Bits ({target_snr_db:.2f} dB SNR)")
    print(f" Target Sampling Rate: {f_target_mhz:.1f} MHz")
    print(" 1.2V Thin-Oxide (L=65nm) vs 2.5V Thick-Oxide (L=280nm)")
    print("="*80)

    gm_id_range = np.linspace(6.0, 20.0, 25)

    power_1v2, power_2v5 = [], []
    cap_tot_1v2, cap_tot_2v5 = [], []
    area_gate_1v2, area_gate_2v5 = [], []
    f_max_fixed_gm_1v2, f_max_fixed_gm_2v5 = [], []

    # 1. Establish baseline fixed gm and Cs at nominal gm/ID = 12.0
    cfg_1v2_base = scale_config_for_target_enob(cfg_1v2_raw, target_enob, f_clk_target)
    adc_1v2_base = FullPipelinedADCSimulator.from_config(replace(cfg_1v2_base, f_clk=f_clk_target))
    specs_1v2_sha_base = adc_1v2_base.sha.calculate_settling_and_gm(
        Cs_next_stage=adc_1v2_base.stages[0].Cs, total_adc_bits=adc_1v2_base.total_bits,
        f_clk=f_clk_target, t_non_overlap=cfg_1v2_base.t_non_overlap, Vdd=cfg_1v2_base.Vdd
    )
    gm_fixed_1v2 = specs_1v2_sha_base["gm_mS"] * 1e-3  # Fixed target gm (Siemens)

    cfg_2v5_base = scale_config_for_target_enob(cfg_2v5_raw, target_enob, f_clk_target)
    adc_2v5_base = FullPipelinedADCSimulator.from_config(replace(cfg_2v5_base, f_clk=f_clk_target))
    specs_2v5_sha_base = adc_2v5_base.sha.calculate_settling_and_gm(
        Cs_next_stage=adc_2v5_base.stages[0].Cs, total_adc_bits=adc_2v5_base.total_bits,
        f_clk=f_clk_target, t_non_overlap=cfg_2v5_base.t_non_overlap, Vdd=cfg_2v5_base.Vdd
    )
    gm_fixed_2v5 = specs_2v5_sha_base["gm_mS"] * 1e-3  # Fixed target gm (Siemens)

    # 2. SWEEP GM/ID FOR BOTH PROCESSES
    for gm_id in gm_id_range:
        # --- 1.2V THIN-OXIDE EVALUATION ---
        cp_scale_1v2 = (gm_id / cfg_1v2_raw.gm_over_id) ** 2
        cfg_1v2_temp = replace(
            cfg_1v2_raw, gm_over_id=gm_id,
            sha_Cp=cfg_1v2_raw.sha_Cp * cp_scale_1v2, mdac_Cp=cfg_1v2_raw.mdac_Cp * cp_scale_1v2
        )
        cfg_1v2_scaled = scale_config_for_target_enob(cfg_1v2_temp, target_enob, f_clk_target)
        adc_1v2 = FullPipelinedADCSimulator.from_config(replace(cfg_1v2_scaled, f_clk=f_clk_target))
        _, sum_1v2 = adc_1v2.run_static_analysis(f_clk=f_clk_target, t_non_overlap=cfg_1v2_scaled.t_non_overlap)
        
        p_1v2 = float(sum_1v2["Total OTA Power"].split()[0])
        c_1v2 = (adc_1v2.sha.Cs + adc_1v2.sha.Cf + adc_1v2.sha.C_cmfb + 
                 sum(s.Cs + s.Cf + s.C_cmfb for s in adc_1v2.stages)) * 1e12
        tot_cp_1v2 = (adc_1v2.sha.Cp + sum(s.Cp for s in adc_1v2.stages)) * 1e15
        a_gate_1v2 = tot_cp_1v2 * (L_1v2_nm / 1000.0)

        power_1v2.append(p_1v2)
        cap_tot_1v2.append(c_1v2)
        area_gate_1v2.append(a_gate_1v2)

        # Speed limit at FIXED gm & FIXED Cs (Cp scales linearly with gm/ID at fixed gm)
        cp_fixed_gm_1v2 = cfg_1v2_base.sha_Cp * (gm_id / cfg_1v2_base.gm_over_id)
        beta_fixed_gm_1v2 = cfg_1v2_base.sha_Cf / (cfg_1v2_base.sha_Cs + cfg_1v2_base.sha_Cf + cp_fixed_gm_1v2)
        c_series_1v2 = (cfg_1v2_base.sha_Cf * (cfg_1v2_base.sha_Cs + cp_fixed_gm_1v2)) / (cfg_1v2_base.sha_Cs + cfg_1v2_base.sha_Cf + cp_fixed_gm_1v2)
        cl_fixed_gm_1v2 = adc_1v2_base.stages[0].Cs + cfg_1v2_base.sha_C_out_par + cfg_1v2_base.sha_C_cmfb + c_series_1v2
        
        f_cl_1v2 = (beta_fixed_gm_1v2 * gm_fixed_1v2) / (2.0 * np.pi * cl_fixed_gm_1v2)
        tau_1v2 = 1.0 / (2.0 * np.pi * f_cl_1v2)
        t_settle_1v2 = (adc_1v2_base.total_bits + 1.0) * np.log(2.0) * tau_1v2
        f_max_fixed_gm_1v2.append(1.0 / (2.0 * (t_settle_1v2 + cfg_1v2_base.t_non_overlap)) / 1e6)

        # --- 2.5V THICK-OXIDE EVALUATION ---
        cp_scale_2v5 = (gm_id / cfg_2v5_raw.gm_over_id) ** 2
        cfg_2v5_temp = replace(
            cfg_2v5_raw, gm_over_id=gm_id,
            sha_Cp=cfg_2v5_raw.sha_Cp * cp_scale_2v5, mdac_Cp=cfg_2v5_raw.mdac_Cp * cp_scale_2v5
        )
        cfg_2v5_scaled = scale_config_for_target_enob(cfg_2v5_temp, target_enob, f_clk_target)
        adc_2v5 = FullPipelinedADCSimulator.from_config(replace(cfg_2v5_scaled, f_clk=f_clk_target))
        _, sum_2v5 = adc_2v5.run_static_analysis(f_clk=f_clk_target, t_non_overlap=cfg_2v5_scaled.t_non_overlap)

        p_2v5 = float(sum_2v5["Total OTA Power"].split()[0])
        c_2v5 = (adc_2v5.sha.Cs + adc_2v5.sha.Cf + adc_2v5.sha.C_cmfb + 
                 sum(s.Cs + s.Cf + s.C_cmfb for s in adc_2v5.stages)) * 1e12
        tot_cp_2v5 = (adc_2v5.sha.Cp + sum(s.Cp for s in adc_2v5.stages)) * 1e15
        a_gate_2v5 = tot_cp_2v5 * (L_2v5_nm / 1000.0)

        power_2v5.append(p_2v5)
        cap_tot_2v5.append(c_2v5)
        area_gate_2v5.append(a_gate_2v5)

        # Speed limit at FIXED gm & FIXED Cs (Cp scales linearly with gm/ID at fixed gm)
        cp_fixed_gm_2v5 = cfg_2v5_base.sha_Cp * (gm_id / cfg_2v5_base.gm_over_id)
        beta_fixed_gm_2v5 = cfg_2v5_base.sha_Cf / (cfg_2v5_base.sha_Cs + cfg_2v5_base.sha_Cf + cp_fixed_gm_2v5)
        c_series_2v5 = (cfg_2v5_base.sha_Cf * (cfg_2v5_base.sha_Cs + cp_fixed_gm_2v5)) / (cfg_2v5_base.sha_Cs + cfg_2v5_base.sha_Cf + cp_fixed_gm_2v5)
        cl_fixed_gm_2v5 = adc_2v5_base.stages[0].Cs + cfg_2v5_base.sha_C_out_par + cfg_2v5_base.sha_C_cmfb + c_series_2v5
        
        f_cl_2v5 = (beta_fixed_gm_2v5 * gm_fixed_2v5) / (2.0 * np.pi * cl_fixed_gm_2v5)
        tau_2v5 = 1.0 / (2.0 * np.pi * f_cl_2v5)
        t_settle_2v5 = (adc_2v5_base.total_bits + 1.0) * np.log(2.0) * tau_2v5
        f_max_fixed_gm_2v5.append(1.0 / (2.0 * (t_settle_2v5 + cfg_2v5_base.t_non_overlap)) / 1e6)

    opt_1v2_idx = np.argmin(power_1v2)
    opt_2v5_idx = np.argmin(power_2v5)

    print(f"\n--- OPTIMAL POWER SWEET SPOTS (Equal {target_enob:.2f} ENOB @ {f_target_mhz:.0f} MHz) ---")
    print(f"  1.2V Thin-Oxide  : Min Power = {power_1v2[opt_1v2_idx]:.2f} mW @ gm/ID = {gm_id_range[opt_1v2_idx]:.1f} V^-1")
    print(f"  2.5V Thick-Oxide : Min Power = {power_2v5[opt_2v5_idx]:.2f} mW @ gm/ID = {gm_id_range[opt_2v5_idx]:.1f} V^-1")

    # --------------------------------------------------------------------------
    # DASHBOARD PLOTS
    # --------------------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9.5))

    # Panel 1: Power vs gm/ID at Constant ENOB
    ax1.plot(gm_id_range, power_1v2, 'o-', color='tab:blue', lw=2, label='1.2V Thin-Oxide (65nm)')
    ax1.plot(gm_id_range, power_2v5, 's-', color='tab:green', lw=2, label='2.5V Thick-Oxide (280nm)')
    ax1.scatter([gm_id_range[opt_1v2_idx]], [power_1v2[opt_1v2_idx]], color='red', zorder=5, s=60)
    ax1.scatter([gm_id_range[opt_2v5_idx]], [power_2v5[opt_2v5_idx]], color='red', zorder=5, s=60)
    ax1.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax1.set_ylabel("Total OTA Power (mW)", fontsize=10)
    ax1.set_title(f"1. Total OTA Power (Equal {target_enob:.1f} ENOB @ {f_target_mhz:.0f} MHz)", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Panel 2: Relative Active Transistor Gate Area
    ax2.plot(gm_id_range, area_gate_1v2, 'o-', color='tab:blue', lw=2, label='1.2V Thin-Oxide (L=65nm)')
    ax2.plot(gm_id_range, area_gate_2v5, 's-', color='tab:green', lw=2, label='2.5V Thick-Oxide (L=280nm)')
    ax2.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax2.set_ylabel("Relative Gate Area (a.u.)", fontsize=10)
    ax2.set_title("2. Relative Transistor Active Gate Area (∝ Cp × L)", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    # Panel 3: Speed Ceiling at FIXED gm and Cs
    ax3.plot(gm_id_range, f_max_fixed_gm_1v2, 'o-', color='tab:blue', lw=2, label='1.2V Thin-Oxide (L=65nm)')
    ax3.plot(gm_id_range, f_max_fixed_gm_2v5, 's-', color='tab:green', lw=2, label='2.5V Thick-Oxide (L=280nm)')
    ax3.axhline(f_target_mhz, color='crimson', linestyle=':', label=f'Target Speed ({f_target_mhz:.0f} MHz)')
    ax3.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax3.set_ylabel("Max Sampling Rate f_clk_max (MHz)", fontsize=10)
    ax3.set_title("3. Speed Ceiling at Fixed gm & Cs (Self-Loading Effect)", fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()

    # Panel 4: Silicon Capacitance Area vs gm/ID at Constant ENOB
    ax4.plot(gm_id_range, cap_tot_1v2, 'o-', color='tab:blue', lw=2, label='1.2V Thin-Oxide (L=65nm)')
    ax4.plot(gm_id_range, cap_tot_2v5, 's-', color='tab:green', lw=2, label='2.5V Thick-Oxide (L=280nm)')
    ax4.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax4.set_ylabel("Total Silicon Capacitance (pF)", fontsize=10)
    ax4.set_title(f"4. Total Silicon Cap Area (Equal {target_enob:.1f} ENOB)", fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()

    plt.suptitle(
        f"65nm Commercial CMOS Decision Dashboard\n"
        f"Target Resolution: {total_bits}-Bit Nominal / Equal {target_enob:.1f}-Bit ENOB @ {f_target_mhz:.0f} MHz Sampling Rate",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # Raw 1.2V Thin-Oxide Devices (10-bit nominal)
    process_65nm_1v2_config = ADCConfig(
        num_stages=8, Vdd=1.2, Vref=0.5, Vcm_in=0.6, f_clk=100e6, t_non_overlap=0.3e-9,
        gm_over_id=12.0, gamma_transistor=1.6,
        sha_Cs=2.5e-12, sha_Cf=2.5e-12, sha_Cp=0.10e-12, sha_C_cmfb=0.20e-12, sha_A0_db=70.0,
        Cs_profile=[2.0e-12, 1.2e-12, 0.8e-12, 0.5e-12, 0.3e-12, 0.2e-12, 0.2e-12, 0.2e-12],
        mdac_Cp=0.10e-12, mdac_C_cmfb=0.15e-12, mdac_A0_db=62.0,
        sigma_cap_mismatch=0.0010, sigma_comp_offset=0.012, sigma_vref_noise=0.0005, sigma_a0_db=1.0
    )

    # Raw 2.5V Thick-Oxide Devices (10-bit nominal)
    process_65nm_2v5_config = ADCConfig(
        num_stages=8, Vdd=2.5, Vref=1.2, Vcm_in=1.25, f_clk=100e6, t_non_overlap=0.5e-9,
        gm_over_id=12.0, gamma_transistor=1.1,
        sha_Cs=0.6e-12, sha_Cf=0.6e-12, sha_Cp=0.40e-12, sha_C_cmfb=0.15e-12, sha_A0_db=75.0,
        Cs_profile=[0.5e-12, 0.3e-12, 0.2e-12, 0.15e-12, 0.1e-12, 0.1e-12, 0.1e-12, 0.1e-12],
        mdac_Cp=0.30e-12, mdac_C_cmfb=0.10e-12, mdac_A0_db=68.0,
        sigma_cap_mismatch=0.0008, sigma_comp_offset=0.015, sigma_vref_noise=0.0008, sigma_a0_db=0.8
    )

    # Run comparative sweep at equal ENOB
    run_device_comparison(
        process_65nm_1v2_config, 
        process_65nm_2v5_config, 
        f_clk_target=80e6, # 100 MHz target speed
        target_enob=9.5     # Equal 9.5 ENOB constraint
    )