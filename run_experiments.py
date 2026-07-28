# run_experiments.py
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

from adc_simulator import ADCConfig, FullPipelinedADCSimulator


# ==============================================================================
# EXPERIMENT 0: BASIC FUNCTIONALITY (SANITY CHECK)
# ==============================================================================
def run_sanity_check(cfg: ADCConfig):
    print("\n" + "="*70)
    print(" EXPERIMENT 0: BASIC FUNCTIONALITY (SANITY CHECK)")
    print("="*70)

    adc = FullPipelinedADCSimulator.from_config(cfg)

    df_breakdown, summary = adc.run_static_analysis(f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap)
    print("\n--- STAGE-BY-STAGE BREAKDOWN ---")
    print(df_breakdown.to_string(index=False))
    print("\n--- SYSTEM SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = cfg.Vcm_in + (0.48 * cfg.Vref) * np.sin(2 * np.pi * f_in * t)

    recon_sine = adc.run_transient_simulation(vin_sine, Vcm_in=cfg.Vcm_in)
    sndr, sfdr, enob, spectrum_db = adc.compute_coherent_fft_metrics(recon_sine, M_bin)

    print("\n--- DYNAMIC METRICS (Coherent FFT) ---")
    print(f"  SNDR : {sndr:.2f} dB")
    print(f"  SFDR : {sfdr:.2f} dB")
    print(f"  ENOB : {enob:.2f} bits")


# ==============================================================================
# EXPERIMENT 1: TEMPERATURE SWEEP
# ==============================================================================
def run_temperature_sweep(base_cfg: ADCConfig, temp_range=np.linspace(-40, 125, 12)):
    print("\n" + "="*70)
    print(" EXPERIMENT 1: TEMPERATURE SWEEP (-40°C to +125°C)")
    print("="*70)
    
    snr_list = []
    for temp_c in temp_range:
        cfg = replace(base_cfg, temp_celsius=temp_c, sigma_vref_noise=0.0)
        adc = FullPipelinedADCSimulator.from_config(cfg)
        _, summary = adc.run_static_analysis()
        snr_val = float(summary["Thermal SNR"].split()[0])
        snr_list.append(snr_val)
        print(f"  Temp = {temp_c:>6.1f} °C  -->  Thermal SNR = {snr_val:.2f} dB")

    plt.figure(figsize=(7, 4.5))
    plt.plot(temp_range, snr_list, 'o-', color='tab:red', lw=1.8)
    plt.title("Thermal SNR vs Operating Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Thermal SNR (dB)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 2: TRANSISTOR GAMMA SWEEP
# ==============================================================================
def run_gamma_sweep(base_cfg: ADCConfig, gamma_range=np.linspace(0.67, 2.5, 10)):
    print("\n" + "="*70)
    print(" EXPERIMENT 2: TRANSISTOR GAMMA SWEEP (0.67 to 2.5)")
    print("="*70)

    snr_list = []
    for g_val in gamma_range:
        cfg = replace(base_cfg, gamma_transistor=g_val, sigma_vref_noise=0.0)
        adc = FullPipelinedADCSimulator.from_config(cfg)
        _, summary = adc.run_static_analysis()
        snr_val = float(summary["Thermal SNR"].split()[0])
        snr_list.append(snr_val)
        print(f"  Gamma = {g_val:.2f}  -->  Thermal SNR = {snr_val:.2f} dB")

    plt.figure(figsize=(7, 4.5))
    plt.plot(gamma_range, snr_list, 's-', color='tab:purple', lw=1.8)
    plt.title("Thermal SNR vs Transistor Excess Noise Factor (γ)")
    plt.xlabel("Transistor Gamma (γ)")
    plt.ylabel("Thermal SNR (dB)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 3: VREF NOISE SWEEP
# ==============================================================================
def run_vref_noise_sweep(base_cfg: ADCConfig, max_vref_noise_mv=2.0, num_points=20):
    print("\n" + "="*70)
    print(f" EXPERIMENT 3: VREF NOISE SWEEP (0 to {max_vref_noise_mv:.2f} mV RMS)")
    print("="*70)

    vref_noise_range = np.linspace(0.0, max_vref_noise_mv / 1000.0, num_points)

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = base_cfg.Vcm_in + (0.48 * base_cfg.Vref) * np.sin(2 * np.pi * f_in * t)

    sndr_list = []
    for vnoise in vref_noise_range:
        cfg = replace(base_cfg, sigma_vref_noise=vnoise)
        adc = FullPipelinedADCSimulator.from_config(cfg)
        recon = adc.run_transient_simulation(vin_sine, Vcm_in=cfg.Vcm_in)
        sndr_val, _, _, _ = adc.compute_coherent_fft_metrics(recon, M_bin)
        sndr_list.append(sndr_val)
        print(f"  Vref Noise = {vnoise*1e3:.2f} mV RMS  -->  Dynamic SNDR = {sndr_val:.2f} dB")

    plt.figure(figsize=(7, 4.5))
    plt.plot(vref_noise_range * 1e3, sndr_list, '^-', color='tab:green', lw=1.8)
    plt.title(f"Dynamic SNDR vs Reference Voltage Noise (0 to {max_vref_noise_mv:.2f} mV)")
    plt.xlabel("Vref Noise Std Dev (mV RMS)")
    plt.ylabel("SNDR (dB)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 4: CONSTANT SNR VREF SWEEP (TOTAL CAPACITANCE)
# ==============================================================================
def run_capacitance_vs_vref_sweep(
    base_cfg: ADCConfig, 
    target_snr_db=70.0, 
    vref_pp_range=np.linspace(0.5, 4.0, 20)
):
    print("\n" + "="*70)
    print(f" EXPERIMENT 4: CONSTANT SNR ({target_snr_db} dB) VREF SWEEP")
    print("="*70)

    default_taper = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.1, 0.1]
    if len(default_taper) >= base_cfg.num_stages:
        taper_profile = default_taper[:base_cfg.num_stages]
    else:
        taper_profile = default_taper + [default_taper[-1]] * (base_cfg.num_stages - len(default_taper))

    total_caps_pF = []

    for vref_pp in vref_pp_range:
        vref_peak = vref_pp / 2.0
        Cs_base = 1.0e-12

        test_cfg = replace(
            base_cfg, Vref=vref_peak, sha_Cs=1.25*Cs_base, sha_Cf=1.25*Cs_base, sha_C_cmfb=0.15*Cs_base,
            Cs_profile=[Cs_base*tap for tap in taper_profile], mdac_C_cmfb=0.15*Cs_base
        )
        adc_test = FullPipelinedADCSimulator.from_config(test_cfg)
        _, summary_test = adc_test.run_static_analysis()
        snr_base_linear = 10 ** (float(summary_test["Thermal SNR"].split()[0]) / 10.0)

        target_snr_linear = 10 ** (target_snr_db / 10.0)
        Cs_scaled = Cs_base * (target_snr_linear / snr_base_linear)

        final_cfg = replace(
            base_cfg, Vref=vref_peak, sha_Cs=1.25*Cs_scaled, sha_Cf=1.25*Cs_scaled, sha_C_cmfb=0.15*Cs_scaled,
            Cs_profile=[Cs_scaled*tap for tap in taper_profile], mdac_C_cmfb=0.15*Cs_scaled
        )
        adc_final = FullPipelinedADCSimulator.from_config(final_cfg)

        c_sha = adc_final.sha.Cs + adc_final.sha.Cf + adc_final.sha.C_cmfb
        c_mdac = sum(s.Cs + s.Cf + s.C_cmfb for s in adc_final.stages)
        tot_pF = (c_sha + c_mdac) * 1e12
        total_caps_pF.append(tot_pF)

        print(f"  Vref_pp = {vref_pp:.2f} V  -->  Total Required Capacitance = {tot_pF:.2f} pF")

    idx_1v = np.argmin(np.abs(vref_pp_range - 1.0))
    theoretical_curve = total_caps_pF[idx_1v] * (1.0 / (vref_pp_range ** 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(vref_pp_range, total_caps_pF, 'o-', color='tab:blue', lw=2, label='Simulated')
    ax1.plot(vref_pp_range, theoretical_curve, '--', color='tab:red', lw=1.5, label=r'Theory ($\propto 1/V_{ref,pp}^2$)')
    ax1.set_title(f"Total Cap vs Vref Swing (Constant SNR = {target_snr_db} dB)")
    ax1.set_xlabel("Fully Differential Vref Peak-to-Peak (V)")
    ax1.set_ylabel("Total Capacitance (pF)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.loglog(vref_pp_range, total_caps_pF, 's-', color='tab:purple', lw=2, label='Simulated')
    ax2.loglog(vref_pp_range, theoretical_curve, '--', color='tab:red', lw=1.5, label=r'Theory ($\propto 1/V_{ref,pp}^2$)')
    ax2.set_title("Log-Log Scaling Profile")
    ax2.set_xlabel("Fully Differential Vref Peak-to-Peak (V)")
    ax2.set_ylabel("Total Capacitance (pF)")
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 5: DEDICATED STATIC DNL / INL MEASUREMENT
# ==============================================================================
def run_dnl_inl_experiment(base_cfg: ADCConfig, num_ramp_samples=300000):
    print("\n" + "="*70)
    print(" EXPERIMENT 5: STATIC DNL / INL MEASUREMENT")
    print("="*70)

    cfg = replace(base_cfg, sigma_cap_mismatch=0.001, sigma_comp_offset=0.012, sigma_a0_db=1.0)
    adc = FullPipelinedADCSimulator.from_config(cfg)

    print(f"Running linear ramp simulation with {num_ramp_samples:,} samples...")
    codes, dnl, inl = adc.run_ramp_dnl_inl(num_ramp_samples=num_ramp_samples, Vcm_in=cfg.Vcm_in)

    max_dnl = np.max(np.abs(dnl))
    max_inl = np.max(np.abs(inl))

    print(f"  Max |DNL| : {max_dnl:.3f} LSB")
    print(f"  Max |INL| : {max_inl:.3f} LSB")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax1.plot(codes, dnl, color='navy', lw=0.8, label='DNL')
    ax1.axhline(0, color='black', lw=0.8, linestyle='--')
    ax1.axhline(0.5, color='tab:red', lw=0.8, linestyle=':', label='±0.5 LSB Target')
    ax1.axhline(-0.5, color='tab:red', lw=0.8, linestyle=':')
    ax1.set_title(f"Differential Non-Linearity (DNL) — Max |DNL| = {max_dnl:.2f} LSB", fontsize=11)
    ax1.set_ylabel("DNL (LSB)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    ax2.plot(codes, inl, color='crimson', lw=1.0, label='INL')
    ax2.axhline(0, color='black', lw=0.8, linestyle='--')
    ax2.set_title(f"Integral Non-Linearity (INL) — Max |INL| = {max_inl:.2f} LSB", fontsize=11)
    ax2.set_xlabel("Digital Code")
    ax2.set_ylabel("INL (LSB)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.suptitle("Static Non-Linearity Analysis (Ramp Test)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 6: COMPREHENSIVE GM/ID TRADE-OFF ANALYSIS
# (1. Max Speed at Fixed Power | 2. Min Power at Fixed Speed | 3. Walden FOM)
# ==============================================================================
def run_gm_over_id_sweep(
    base_cfg: ADCConfig, 
    gm_id_range=np.linspace(6.0, 20.0, 25)
):
    print("\n" + "="*70)
    print(" EXPERIMENT 6: GM/ID TRADE-OFF ANALYSIS")
    print("="*70)

    # --------------------------------------------------------------------------
    # PART A: FIXED CLOCK SPEED (f_clk = 100 MHz) -> Power & Walden FOM
    # --------------------------------------------------------------------------
    power_fixed_fclk_mW = []
    fom_walden_fJ = []
    enob_list = []

    for gm_id in gm_id_range:
        # Parasitic cap scales quadratically in weak/subthreshold inversion
        cp_scale = (gm_id / base_cfg.gm_over_id) ** 2
        
        cfg = replace(
            base_cfg,
            gm_over_id=gm_id,
            sha_Cp=base_cfg.sha_Cp * cp_scale,
            mdac_Cp=base_cfg.mdac_Cp * cp_scale
        )
        adc = FullPipelinedADCSimulator.from_config(cfg)

        # Static power and thermal ENOB for target f_clk
        _, summary = adc.run_static_analysis(f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap)
        p_mW = float(summary["Total OTA Power"].split()[0])
        enob = float(summary["Thermal ENOB"].split()[0])

        # Walden FOM in fJ/conv-step: P / (2^ENOB * f_clk)
        p_watts = p_mW * 1e-3
        fom_fJ = (p_watts / ((2 ** enob) * cfg.f_clk)) * 1e15

        power_fixed_fclk_mW.append(p_mW)
        enob_list.append(enob)
        fom_walden_fJ.append(fom_fJ)

    # Find optimal Walden FOM sweet spot
    opt_idx = np.argmin(fom_walden_fJ)
    opt_gm_id = gm_id_range[opt_idx]
    opt_fom = fom_walden_fJ[opt_idx]

    # --------------------------------------------------------------------------
    # PART B: FIXED POWER (Fixed Tail Current) -> Max Sampling Frequency
    # --------------------------------------------------------------------------
    # Establish baseline tail currents at base_cfg.gm_over_id
    adc_base = FullPipelinedADCSimulator.from_config(base_cfg)
    sha_base_specs = adc_base.sha.calculate_settling_and_gm(
        Cs_next_stage=adc_base.stages[0].Cs, total_adc_bits=adc_base.total_bits,
        f_clk=base_cfg.f_clk, t_non_overlap=base_cfg.t_non_overlap, Vdd=base_cfg.Vdd
    )
    base_itail_sha = sha_base_specs["gm_mS"] * 1e-3 / adc_base.sha.gm_over_id

    mdac_base_itails = []
    for i, stage in enumerate(adc_base.stages):
        cs_next = adc_base.stages[i+1].Cs if (i + 1 < len(adc_base.stages)) else 0.0
        bits_rem = sum(s.effective_bits for s in adc_base.stages[i:]) + adc_base.quantizer_bits
        specs = stage.calculate_settling_and_gm(
            Cs_next_stage=cs_next, remaining_bits=bits_rem,
            f_clk=base_cfg.f_clk, t_non_overlap=base_cfg.t_non_overlap, Vdd=base_cfg.Vdd
        )
        mdac_base_itails.append(specs["gm_mS"] * 1e-3 / stage.gm_over_id)

    f_clk_max_MHz = []

    for gm_id in gm_id_range:
        cp_scale = (gm_id / base_cfg.gm_over_id) ** 2
        cfg = replace(
            base_cfg, gm_over_id=gm_id,
            sha_Cp=base_cfg.sha_Cp * cp_scale, mdac_Cp=base_cfg.mdac_Cp * cp_scale
        )
        adc = FullPipelinedADCSimulator.from_config(cfg)

        # Calculate SHA settling time with fixed tail current
        gm_sha = base_itail_sha * gm_id
        cl_sha = adc.sha.calculate_cl_load(adc.stages[0].Cs)
        beta_sha = adc.sha.beta
        tau_sha = cl_sha / (2.0 * np.pi * beta_sha * gm_sha)
        n_tau_sha = (adc.total_bits + 1.0) * np.log(2.0)
        t_settle_sha = n_tau_sha * tau_sha

        # Calculate MDAC settling times with fixed tail currents
        t_settle_mdacs = []
        for i, stage in enumerate(adc.stages):
            cs_next = adc.stages[i+1].Cs if (i + 1 < len(adc.stages)) else 0.0
            bits_rem = sum(s.effective_bits for s in adc.stages[i:]) + adc.quantizer_bits
            gm_stage = mdac_base_itails[i] * gm_id
            cl_stage = stage.calculate_cl_load(cs_next)
            beta_stage = stage.beta
            tau_stage = cl_stage / (2.0 * np.pi * beta_stage * gm_stage)
            n_tau_stage = (bits_rem + 1.0) * np.log(2.0)
            t_settle_mdacs.append(n_tau_stage * tau_stage)

        # Bottleneck stage defines maximum clock frequency
        t_settle_max = max(t_settle_sha, max(t_settle_mdacs))
        t_clk_min = 2.0 * (t_settle_max + cfg.t_non_overlap)
        f_max = 1.0 / t_clk_min
        f_clk_max_MHz.append(f_max / 1e6)

        print(f"  gm/ID = {gm_id:>5.1f} V^-1  |  Power = {power_fixed_fclk_mW[len(f_clk_max_MHz)-1]:>5.2f} mW  |  Max f_clk = {f_max/1e6:>6.1f} MHz  |  FOM = {fom_walden_fJ[len(f_clk_max_MHz)-1]:>6.1f} fJ/step")

    print(f"\n  ---> Walden FOM Sweet Spot: {opt_fom:.1f} fJ/conv-step at gm/ID = {opt_gm_id:.1f} V^-1")

    # --------------------------------------------------------------------------
    # PLOTTING THE THREE TRADEOFFS
    # --------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Max Speed at Fixed Power
    ax1.plot(gm_id_range, f_clk_max_MHz, 'o-', color='tab:red', lw=2)
    ax1.set_title("1. Max Clock Speed (Fixed Power)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax1.set_ylabel("Max Clock Frequency f_clk_max (MHz)", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Required Power at Fixed Speed (100 MHz)
    ax2.plot(gm_id_range, power_fixed_fclk_mW, 's-', color='tab:blue', lw=2)
    ax2.set_title("2. Required Power (Fixed 100 MHz Clock)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax2.set_ylabel("Total OTA Power (mW)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Walden Figure of Merit (Sweet Spot)
    ax3.plot(gm_id_range, fom_walden_fJ, 'd-', color='tab:green', lw=2, label='Walden FOM')
    ax3.axvline(opt_gm_id, color='crimson', linestyle='--', lw=1.2, label=f'Sweet Spot ({opt_gm_id:.1f} V^-1)')
    ax3.set_title("3. Walden FOM Sweet Spot", fontsize=11, fontweight='bold')
    ax3.set_xlabel("gm/ID Efficiency (V^-1)", fontsize=10)
    ax3.set_ylabel("Walden FOM (fJ/conv-step)", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')

    plt.suptitle("Pipelined ADC Transconductance Efficiency (gm/ID) Trade-Off Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # CONFIG 0: 1.2V Thin-Oxide Devices (65nm Process, L = 65nm)
    # High Speed, Larger Sampling Capacitors, Lower Reference Swing
    # ------------------------------------------------------------------------------
    
    main_config = ADCConfig(
        num_stages=8,               # 8 MDAC stages (10 bits resolution total)
        Vref=1.0,
        Vdd=1.2,
        gm_over_id=12.0,            # Default gm/ID efficiency (12 V^-1)
        temp_celsius=27.0,
        gamma_transistor=1.5,
        
        # SHA parameters
        sha_A0_db=75.0,
        sha_Cs=2.5e-12,
        sha_C_cmfb=0.25e-12,
        
        # MDAC parameters
        mdac_A0_db=65.0,
        mdac_C_cmfb=0.15e-12,
        Cs_profile=[2.0e-12, 1.2e-12, 0.8e-12, 0.5e-12, 0.3e-12, 0.2e-12, 0.2e-12, 0.2e-12],
        
        # Non-idealities & Mismatch Parameters
        sigma_cap_mismatch=0.001,   # 0.1% C mismatch
        sigma_comp_offset=0.012,    # 12 mV comparator offset
        sigma_vref_noise=0.0005,    # 0.5 mV Vref noise
        sigma_a0_db=1.0             # 1.0 dB OTA open-loop gain mismatch
    )
    
    # ------------------------------------------------------------------------------
    # CONFIG 1: 1.2V Thin-Oxide Devices (65nm Process, L = 65nm)
    # High Speed, Larger Sampling Capacitors, Lower Reference Swing
    # ------------------------------------------------------------------------------
    process_65nm_1v2_config = ADCConfig(
        num_stages=8,               # 8 MDAC stages (10 bits total)
        Vdd=1.2,                    # 1.2V Core Supply
        Vref=0.5,                   # 1.0V Peak-to-Peak Differential Swing
        Vcm_in=0.6,                 # Mid-supply common mode (0.6V)
        f_clk=150e6,                # High sampling rate (150 MHz target)
        t_non_overlap=0.3e-9,       # Faster clock transitions
    
        # Transistor technology specs (Thin-oxide short-channel)
        gm_over_id=12.0,
        # Higher noise factor due to hot carriers / velocity saturation
        gamma_transistor=1.6,
    
        # Capacitors (Larger Cs required to offset smaller 1.0Vpp signal swing)
        sha_Cs=2.5e-12,             # 2.5 pF required for ~70 dB SNR
        sha_Cf=2.5e-12,
        sha_Cp=0.10e-12,            # Small parasitic cap (small L = 65nm)
        sha_C_cmfb=0.20e-12,
        sha_A0_db=70.0,             # Harder to get high gain in short channel
    
        # MDAC stage caps (scaled up for thermal noise)
        Cs_profile=[2.0e-12, 1.2e-12, 0.8e-12, 0.5e-12,
                    0.3e-12, 0.2e-12, 0.2e-12, 0.2e-12],
        mdac_Cp=0.10e-12,           # Low parasitic cap
        mdac_C_cmfb=0.15e-12,
        mdac_A0_db=62.0,            # 62 dB gain
    
        # Non-idealities & Mismatch
        sigma_cap_mismatch=0.0010,  # 0.1% C mismatch
        sigma_comp_offset=0.012,    # 12 mV comparator offset
        sigma_vref_noise=0.0005,    # 0.5 mV Vref noise
        sigma_a0_db=1.0
    )
    
    # ------------------------------------------------------------------------------
    # CONFIG 2: 2.5V Thick-Oxide Devices (65nm Process, L = 280nm)
    # Lower Speed, Small Sampling Capacitors, Large Reference Swing
    # ------------------------------------------------------------------------------
    process_65nm_2v5_config = ADCConfig(
        num_stages=8,               # 8 MDAC stages (10 bits total)
        Vdd=2.5,                    # 2.5V I/O Supply
        Vref=1.2,                   # 2.4V Peak-to-Peak Differential Swing
        Vcm_in=1.25,                # Mid-supply common mode (1.25V)
        f_clk=80e6,                 # Lower sampling rate limit (80 MHz target)
        t_non_overlap=0.5e-9,       
        
        # Transistor technology specs (Thick-oxide long-channel)
        gm_over_id=12.0,            
        gamma_transistor=1.1,       # Lower noise factor (less velocity saturation)
        
        # Capacitors (Much smaller Cs thanks to 2.4Vpp swing!)
        sha_Cs=0.6e-12,             # 0.6 pF achieves same SNR as 2.5 pF at 1.2V
        sha_Cf=0.6e-12,
        sha_Cp=0.40e-12,            # Larger parasitic cap (~4x higher due to L = 280nm)
        sha_C_cmfb=0.15e-12,
        sha_A0_db=75.0,             # High open-loop gain easily achieved
        
        # MDAC stage caps (scaled down)
        Cs_profile=[0.5e-12, 0.3e-12, 0.2e-12, 0.15e-12, 0.1e-12, 0.1e-12, 0.1e-12, 0.1e-12],
        mdac_Cp=0.30e-12,           # Higher parasitic cap
        mdac_C_cmfb=0.10e-12,
        mdac_A0_db=68.0,            # 68 dB gain
        
        # Non-idealities & Mismatch
        sigma_cap_mismatch=0.0008,  # Slightly better matching at 2.5V
        sigma_comp_offset=0.015,    # 15 mV comparator offset
        sigma_vref_noise=0.0008,    # 0.8 mV Vref noise
        sigma_a0_db=0.8
    )    

    run_sanity_check(main_config)
    run_temperature_sweep(main_config)
    run_gamma_sweep(main_config)
    run_vref_noise_sweep(main_config, max_vref_noise_mv=2.0, num_points=20)
    run_capacitance_vs_vref_sweep(main_config)
    run_dnl_inl_experiment(main_config,num_ramp_samples=100*pow((main_config.num_stages+2),2))
    run_gm_over_id_sweep(main_config)