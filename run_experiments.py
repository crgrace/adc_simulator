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
    # 0.96 * Vref near full-scale differential swing (-0.35 dBFS)
    vin_sine = cfg.Vcm_in + (0.96 * cfg.Vref) * np.sin(2 * np.pi * f_in * t)

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
def run_vref_noise_sweep(base_cfg: ADCConfig, max_vref_noise_mv=3.0, num_points=20):
    print("\n" + "="*70)
    print(f" EXPERIMENT 3: VREF NOISE SWEEP (0 to {max_vref_noise_mv:.2f} mV RMS)")
    print("="*70)

    vref_noise_range = np.linspace(0.0, max_vref_noise_mv / 1000.0, num_points)

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = base_cfg.Vcm_in + (0.96 * base_cfg.Vref) * np.sin(2 * np.pi * f_in * t)

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
    vref_pp_range=np.linspace(0.8, 4.0, 20)
):
    print("\n" + "="*70)
    print(f" EXPERIMENT 4: CONSTANT SNR ({target_snr_db} dB) VREF SWEEP")
    print("="*70)

    default_taper = [1.0, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2]
    if len(default_taper) >= base_cfg.num_stages:
        taper_profile = default_taper[:base_cfg.num_stages]
    else:
        taper_profile = default_taper + [default_taper[-1]] * (base_cfg.num_stages - len(default_taper))

    total_caps_pF = []

    for vref_pp in vref_pp_range:
        vref_peak = vref_pp / 2.0
        Cs_base = 0.5e-12

        test_cfg = replace(
            base_cfg, Vref=vref_peak, sha_Cs=1.0*Cs_base, sha_Cf=1.0*Cs_base, sha_C_cmfb=0.25*Cs_base,
            Cs_profile=[Cs_base*tap for tap in taper_profile], mdac_C_cmfb=0.20*Cs_base
        )
        adc_test = FullPipelinedADCSimulator.from_config(test_cfg)
        _, summary_test = adc_test.run_static_analysis()
        snr_base_linear = 10 ** (float(summary_test["Thermal SNR"].split()[0]) / 10.0)

        target_snr_linear = 10 ** (target_snr_db / 10.0)
        Cs_scaled = Cs_base * (target_snr_linear / snr_base_linear)

        final_cfg = replace(
            base_cfg, Vref=vref_peak, sha_Cs=1.0*Cs_scaled, sha_Cf=1.0*Cs_scaled, sha_C_cmfb=0.25*Cs_scaled,
            Cs_profile=[Cs_scaled*tap for tap in taper_profile], mdac_C_cmfb=0.20*Cs_scaled
        )
        adc_final = FullPipelinedADCSimulator.from_config(final_cfg)

        c_sha = adc_final.sha.Cs + adc_final.sha.Cf + adc_final.sha.C_cmfb
        c_mdac = sum(s.Cs + s.Cf + s.C_cmfb for s in adc_final.stages)
        tot_pF = (c_sha + c_mdac) * 1e12
        total_caps_pF.append(tot_pF)

        print(f"  Vref_pp = {vref_pp:.2f} V  -->  Total Required Capacitance = {tot_pF:.2f} pF")

    idx_ref = np.argmin(np.abs(vref_pp_range - (2.0 * base_cfg.Vref)))
    theoretical_curve = total_caps_pF[idx_ref] * ((vref_pp_range[idx_ref] / vref_pp_range) ** 2)

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
def run_dnl_inl_experiment(base_cfg: ADCConfig, num_ramp_samples=None):
    print("\n" + "="*70)
    print(" EXPERIMENT 5: STATIC DNL / INL MEASUREMENT")
    print("="*70)

    cfg = replace(base_cfg, sigma_cap_mismatch=0.0008, sigma_comp_offset=0.015, sigma_a0_db=0.8)
    adc = FullPipelinedADCSimulator.from_config(cfg)

    # Automatically set sample count to 10 hits per LSB code: 10 * 2^total_bits
    if num_ramp_samples is None:
        num_ramp_samples = 10 * (2 ** adc.total_bits)

    print(f"Running linear ramp simulation with {num_ramp_samples:,} samples ({adc.total_bits}-bit resolution, 10 hits/code)...")
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
# EXPERIMENT 6: GM/ID TRADE-OFF SWEEP (POWER, SPEED, GATE AREA, THERMAL SNR)
# ==============================================================================
def run_gm_over_id_comprehensive_sweep(base_cfg: ADCConfig, gm_id_range=np.linspace(6.0, 20.0, 25), L_nm=280):
    print("\n" + "="*70)
    print(" EXPERIMENT 6: GM/ID COMPREHENSIVE DESIGN SWEEP")
    print("="*70)

    power_mW = []
    thermal_snr_dB = []
    area_gate_relative = []
    f_max_fixed_gm_MHz = []

    adc_base = FullPipelinedADCSimulator.from_config(base_cfg)
    specs_sha_base = adc_base.sha.calculate_settling_and_gm(
        Cs_next_stage=adc_base.stages[0].Cs, total_adc_bits=adc_base.total_bits,
        f_clk=base_cfg.f_clk, t_non_overlap=base_cfg.t_non_overlap, Vdd=base_cfg.Vdd
    )
    gm_fixed = specs_sha_base["gm_mS"] * 1e-3

    for gm_id in gm_id_range:
        cp_scale = (gm_id / base_cfg.gm_over_id) ** 2
        cfg = replace(
            base_cfg, gm_over_id=gm_id,
            sha_Cp=base_cfg.sha_Cp * cp_scale, mdac_Cp=base_cfg.mdac_Cp * cp_scale
        )
        adc = FullPipelinedADCSimulator.from_config(cfg)

        _, summary = adc.run_static_analysis(f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap)
        p_val = float(summary["Total OTA Power"].split()[0])
        snr_val = float(summary["Thermal SNR"].split()[0])
        power_mW.append(p_val)
        thermal_snr_dB.append(snr_val)

        tot_cp_fF = (adc.sha.Cp + sum(s.Cp for s in adc.stages)) * 1e15
        area_gate_relative.append(tot_cp_fF * (L_nm / 1000.0))

        cp_fixed_gm = base_cfg.sha_Cp * (gm_id / base_cfg.gm_over_id)
        beta_fixed_gm = base_cfg.sha_Cf / (base_cfg.sha_Cs + base_cfg.sha_Cf + cp_fixed_gm)
        c_series = (base_cfg.sha_Cf * (base_cfg.sha_Cs + cp_fixed_gm)) / (base_cfg.sha_Cs + base_cfg.sha_Cf + cp_fixed_gm)
        cl_fixed_gm = adc_base.stages[0].Cs + base_cfg.sha_C_out_par + base_cfg.sha_C_cmfb + c_series
        
        f_cl = (beta_fixed_gm * gm_fixed) / (2.0 * np.pi * cl_fixed_gm)
        tau = 1.0 / (2.0 * np.pi * f_cl)
        t_settle = (adc_base.total_bits + 1.0) * np.log(2.0) * tau
        f_max = 1.0 / (2.0 * (t_settle + base_cfg.t_non_overlap)) / 1e6
        f_max_fixed_gm_MHz.append(f_max)

        print(f"  gm/ID = {gm_id:>5.1f} V^-1  |  Power = {p_val:>5.2f} mW  |  Thermal SNR = {snr_val:>5.2f} dB  |  f_max = {f_max:>6.1f} MHz")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    ax1.plot(gm_id_range, power_mW, 'o-', color='tab:orange', lw=2)
    ax1.set_xlabel("gm/ID Efficiency (V^-1)")
    ax1.set_ylabel("Total OTA Power (mW)")
    ax1.set_title("1. Total Power Consumption", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(gm_id_range, area_gate_relative, 's-', color='tab:green', lw=2)
    ax2.set_xlabel("gm/ID Efficiency (V^-1)")
    ax2.set_ylabel("Relative Gate Area (a.u.)")
    ax2.set_title("2. Relative Transistor Active Gate Area (∝ Cp × L)", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3.plot(gm_id_range, f_max_fixed_gm_MHz, 'd-', color='tab:red', lw=2)
    ax3.axhline(base_cfg.f_clk / 1e6, color='black', linestyle=':', label=f'Target Clock ({base_cfg.f_clk/1e6:.0f} MHz)')
    ax3.set_xlabel("gm/ID Efficiency (V^-1)")
    ax3.set_ylabel("Max Sampling Rate f_clk_max (MHz)")
    ax3.set_title("3. Speed Ceiling at Fixed gm & Cs", fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()

    ax4.plot(gm_id_range, thermal_snr_dB, '^-', color='tab:blue', lw=2)
    ax4.set_xlabel("gm/ID Efficiency (V^-1)")
    ax4.set_ylabel("Thermal SNR (dB)")
    ax4.set_title("4. Thermal SNR Penalty (via Beta Degradation)", fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Pipelined ADC Transconductance Efficiency (gm/ID) Design Space", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 7: STAGE SCALING FACTOR vs. NOISE & POWER BREAKDOWN
# ==============================================================================
def run_stage_scaling_experiment(base_cfg: ADCConfig, alpha_range=np.linspace(0.3, 1.0, 15)):
    print("\n" + "="*70)
    print(" EXPERIMENT 7: STAGE SCALING FACTOR vs. NOISE & POWER BREAKDOWN")
    print("="*70)

    snr_list = []
    power_list = []

    for alpha in alpha_range:
        cs_profile = [base_cfg.Cs_profile[0] * (alpha ** i) for i in range(base_cfg.num_stages)]
        cfg = replace(base_cfg, Cs_profile=cs_profile)
        adc = FullPipelinedADCSimulator.from_config(cfg)
        _, summary = adc.run_static_analysis(f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap)
        
        snr_val = float(summary["Thermal SNR"].split()[0])
        p_val = float(summary["Total OTA Power"].split()[0])
        snr_list.append(snr_val)
        power_list.append(p_val)
        print(f"  Scaling Factor alpha = {alpha:.2f}  -->  Thermal SNR = {snr_val:.2f} dB  |  Power = {p_val:.2f} mW")

    # Evaluate exact noise and power breakdown at alpha = 0.6
    target_alpha = 0.6
    alpha_idx = np.argmin(np.abs(alpha_range - target_alpha))
    chosen_alpha = alpha_range[alpha_idx]

    cs_profile_chosen = [base_cfg.Cs_profile[0] * (chosen_alpha ** i) for i in range(base_cfg.num_stages)]
    cfg_chosen = replace(base_cfg, Cs_profile=cs_profile_chosen)
    adc_chosen = FullPipelinedADCSimulator.from_config(cfg_chosen)

    df_chosen, _ = adc_chosen.run_static_analysis(f_clk=cfg_chosen.f_clk, t_non_overlap=cfg_chosen.t_non_overlap)

    raw_noise_sq = []
    sha_noise_sq = adc_chosen.sha.input_referred_noise_sq
    raw_noise_sq.append(sha_noise_sq)

    cumulative_gain = adc_chosen.sha.actual_gain
    for i, stage in enumerate(adc_chosen.stages):
        stage_noise_sq = stage.input_referred_noise_sq / (cumulative_gain ** 2)
        raw_noise_sq.append(stage_noise_sq)
        cumulative_gain *= stage.actual_gain

    # Group stages >= 5 into "MDAC 5+" slice
    grouped_noise_sq = []
    labels = []
    for idx in range(min(5, len(raw_noise_sq))):
        grouped_noise_sq.append(raw_noise_sq[idx])
        labels.append("SHA" if idx == 0 else f"MDAC {idx}")

    if len(raw_noise_sq) > 5:
        grouped_noise_sq.append(sum(raw_noise_sq[5:]))
        labels.append("MDAC 5+")

    total_noise_sq = sum(grouped_noise_sq)
    pct_noise = [(n_sq / total_noise_sq) * 100.0 for n_sq in grouped_noise_sq]

    # CHART 1: SNR and Power vs Stage Scaling Factor
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    color1 = 'tab:blue'
    ax1.set_xlabel("Stage Scaling Factor (α)", fontsize=11)
    ax1.set_ylabel("Thermal SNR (dB)", color=color1, fontsize=11)
    ax1.plot(alpha_range, snr_list, 'o-', color=color1, lw=2, label='SNR (dB)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel("Total OTA Power (mW)", color=color2, fontsize=11)
    ax1_twin.plot(alpha_range, power_list, 's--', color=color2, lw=2, label='Power (mW)')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1.set_title("Thermal SNR & Power vs. Stage Scaling Factor (α)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # CHART 2: Standalone Pie Chart for Grouped Noise Breakdown
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    explode = [0.08] + [0.0] * (len(labels) - 1)
    colors = plt.cm.Blues(np.linspace(0.85, 0.35, len(labels)))

    wedges, texts, autotexts = ax2.pie(
        pct_noise, labels=labels, autopct='%1.1f%%', startangle=140,
        explode=explode, colors=colors, textprops=dict(color="black")
    )
    plt.setp(autotexts, size=9, weight="bold")
    ax2.set_title(f"Input-Referred Thermal Noise Contribution\n(at Stage Scaling α = {chosen_alpha:.2f})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # CHART 3: Power Allocation Bar Chart per Stage (Grouped MDAC 5+)
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
    raw_powers = [r["OTA Power (mW)"] for r in df_chosen.to_dict('records')]
    grouped_powers = raw_powers[:5] + [sum(raw_powers[5:])]

    bars = ax3.bar(labels, grouped_powers, color='tab:orange', width=0.55, edgecolor='black', linewidth=0.8)
    ax3.set_ylabel("OTA Power Consumption (mW)", fontsize=11)
    ax3.set_title(f"Stage-by-Stage Power Allocation\n(at Stage Scaling α = {chosen_alpha:.2f})", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6, axis='y')

    tot_p = sum(grouped_powers)
    for bar in bars:
        height = bar.get_height()
        pct = (height / tot_p) * 100.0
        ax3.annotate(f"{height:.2f} mW\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 8: COMPARATOR OFFSET REDUNDANCY TOLERANCE SWEEP
# ==============================================================================
def run_comparator_offset_tolerance_experiment(base_cfg: ADCConfig, max_offset_mv=350.0, num_points=20):
    print("\n" + "="*70)
    print(" EXPERIMENT 8: COMPARATOR OFFSET REDUNDANCY TOLERANCE SWEEP")
    print("="*70)

    offsets_mv = np.linspace(0.0, max_offset_mv, num_points)
    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = base_cfg.Vcm_in + (0.96 * base_cfg.Vref) * np.sin(2 * np.pi * f_in * t)

    enob_list = []
    for offset_mv in offsets_mv:
        cfg = replace(base_cfg, sigma_comp_offset=offset_mv / 1000.0)
        adc = FullPipelinedADCSimulator.from_config(cfg)
        recon = adc.run_transient_simulation(vin_sine, Vcm_in=cfg.Vcm_in)
        _, _, enob, _ = adc.compute_coherent_fft_metrics(recon, M_bin)
        enob_list.append(enob)
        print(f"  Comp Offset Std Dev = {offset_mv:>5.1f} mV  -->  Dynamic ENOB = {enob:.2f} bits")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(offsets_mv, enob_list, 's-', color='tab:red', lw=2)
    plt.axvline(base_cfg.Vref * 1e3 / 4.0, color='black', linestyle='--', label='Theoretical RSD Bound (Vref/4)')
    plt.title("Dynamic ENOB vs. Sub-ADC Comparator Offset (RSD Margin Test)", fontsize=11, fontweight='bold')
    plt.xlabel("Comparator Offset Std Dev (mV RMS)", fontsize=10)
    plt.ylabel("Dynamic ENOB (Bits)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 9: ISOLATED OTA DC OPEN-LOOP GAIN (A0) REQUIREMENT SWEEP
# ==============================================================================
def run_ota_gain_requirement_sweep(base_cfg: ADCConfig, a0_db_range=np.linspace(35, 85, 20)):
    print("\n" + "="*70)
    print(" EXPERIMENT 9: OTA DC OPEN-LOOP GAIN (A0) REQUIREMENT SWEEP")
    print("="*70)

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = base_cfg.Vcm_in + (0.96 * base_cfg.Vref) * np.sin(2 * np.pi * f_in * t)

    sndr_list = []
    for a0_db in a0_db_range:
        # Zero out reference noise and cap mismatch to isolate pure A0 gain error ceiling
        cfg = replace(
            base_cfg, 
            sha_A0_db=a0_db + 10.0, 
            mdac_A0_db=a0_db, 
            sigma_a0_db=0.0,
            sigma_vref_noise=0.0,
            sigma_cap_mismatch=0.0
        )
        adc = FullPipelinedADCSimulator.from_config(cfg)
        recon = adc.run_transient_simulation(vin_sine, Vcm_in=cfg.Vcm_in)
        sndr, _, enob, _ = adc.compute_coherent_fft_metrics(recon, M_bin)
        sndr_list.append(sndr)
        print(f"  MDAC OTA Gain A0 = {a0_db:>4.1f} dB  -->  Dynamic SNDR = {sndr:.2f} dB  |  ENOB = {enob:.2f} bits")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(a0_db_range, sndr_list, 'd-', color='tab:purple', lw=2)
    plt.axhline(61.9, color='crimson', linestyle='--', label='10-Bit Ideal (61.9 dB)')
    plt.title("Isolated Dynamic SNDR vs. MDAC OTA DC Open-Loop Gain (A0)", fontsize=11, fontweight='bold')
    plt.xlabel("MDAC OTA Open-Loop DC Gain A0 (dB)", fontsize=10)
    plt.ylabel("Dynamic SNDR (dB)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# EXPERIMENT 10: SAMPLING FREQUENCY (f_clk) SETTLING WALL SWEEP
# ==============================================================================
def run_clock_frequency_settling_sweep(base_cfg: ADCConfig, f_clk_range_mhz=np.linspace(20, 160, 25)):
    print("\n" + "="*70)
    print(" EXPERIMENT 10: SAMPLING FREQUENCY (f_clk) SETTLING WALL SWEEP")
    print("="*70)

    # Instantiate ADC once at fixed baseline design target (80 MHz)
    adc = FullPipelinedADCSimulator.from_config(base_cfg)

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = base_cfg.Vcm_in + (0.96 * base_cfg.Vref) * np.sin(2 * np.pi * f_in * t)

    enob_list = []
    for f_mhz in f_clk_range_mhz:
        f_hz = f_mhz * 1e6
        recon = adc.run_transient_simulation(
            vin_sine, Vcm_in=base_cfg.Vcm_in, 
            f_clk=f_hz, t_non_overlap=base_cfg.t_non_overlap
        )
        _, _, enob, _ = adc.compute_coherent_fft_metrics(recon, M_bin)
        enob_list.append(enob)
        print(f"  f_clk = {f_mhz:>5.1f} MHz  -->  Dynamic ENOB = {enob:.2f} bits")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(f_clk_range_mhz, enob_list, '^-', color='tab:green', lw=2)
    plt.axvline(base_cfg.f_clk / 1e6, color='black', linestyle='--', label=f'Design Target ({base_cfg.f_clk/1e6:.0f} MHz)')
    plt.title("Effective Resolution (ENOB) vs. Sampling Rate f_clk (Settling Wall)", fontsize=11, fontweight='bold')
    plt.xlabel("Sampling Frequency f_clk (MHz)", fontsize=10)
    plt.ylabel("Dynamic Resolution ENOB (Bits)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # Configuration tailored for 2.5V thick-oxide devices (L = 280nm) in 65nm CMOS
    main_config = ADCConfig(
        num_stages=8,               # 8 MDAC stages (10 bits total resolution)
        Vdd=2.5,                    # 2.5V I/O Supply Voltage
        Vref=1.2,                   # Peak reference voltage (2.4V peak-to-peak differential)
        Vcm_in=1.25,                # Mid-supply common mode
        f_clk=80e6,                 # Target clock frequency (80 MHz for 2.5V devices)
        t_non_overlap=0.5e-9,       # Clock non-overlap time
        
        # Transistor specs for 2.5V thick-oxide (L = 280nm)
        gm_over_id=12.0,            # Transconductance efficiency (V^-1)
        gamma_transistor=1.1,       # Noise factor (lower due to less velocity saturation)
        
        # Front-End SHA Parameters
        sha_Cs=0.6e-12,             # 0.6 pF (smaller Cs required thanks to 2.4Vpp swing)
        sha_Cf=0.6e-12,
        sha_Cp=0.40e-12,            # Higher parasitic cap (~4x higher than 1.2V core due to L=280nm)
        sha_C_out_par=0.15e-12,
        sha_C_cmfb=0.15e-12,
        sha_A0_db=75.0,             # Higher intrinsic gain easily achieved
        
        # MDAC Stage Parameters (scaled down due to 2.4Vpp swing)
        Cs_profile=[0.5e-12, 0.3e-12, 0.2e-12, 0.15e-12, 0.1e-12, 0.1e-12, 0.1e-12, 0.1e-12],
        mdac_Cp=0.30e-12,           # MDAC parasitic cap
        mdac_C_out_par=0.10e-12,
        mdac_C_cmfb=0.10e-12,
        mdac_A0_db=68.0,
        
        # Non-idealities & Mismatch Parameters
        sigma_cap_mismatch=0.0008,  # Better capacitor matching at 2.5V
        sigma_comp_offset=0.015,    # 15 mV comparator offset
        sigma_vref_noise=0.0008,    # 0.8 mV Vref noise
        sigma_a0_db=0.8             # Open-loop gain mismatch
    )

    # Execute all design space exploration experiments
    run_sanity_check(main_config)
    run_temperature_sweep(main_config)
    run_gamma_sweep(main_config)
    run_vref_noise_sweep(main_config, max_vref_noise_mv=3.0, num_points=20)
    run_capacitance_vs_vref_sweep(main_config)
    run_dnl_inl_experiment(main_config)
    run_gm_over_id_comprehensive_sweep(main_config, L_nm=280)
    run_stage_scaling_experiment(main_config)
    run_comparator_offset_tolerance_experiment(main_config)
    run_ota_gain_requirement_sweep(main_config)
    run_clock_frequency_settling_sweep(main_config)