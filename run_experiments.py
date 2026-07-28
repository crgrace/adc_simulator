# run_experiments.py
import numpy as np
import matplotlib.pyplot as plt

# Import simulator classes from our core engine module
from adc_simulator import ConventionalSHA, NonIdealMDACStage, FullPipelinedADCSimulator


# ==============================================================================
# HELPER: BUILD STANDARD ADC INSTANCE
# ==============================================================================
def build_adc(
    gamma=1.5, 
    temp_c=27.0, 
    sigma_vref=0.001, 
    sigma_mismatch=0.001, 
    sigma_offset=0.012
):
    """Utility function to construct a standard 9-bit Pipelined ADC."""
    sha = ConventionalSHA(
        Cs=2.5e-12, Cf=2.5e-12, Cp=0.15e-12, C_out_par=0.08e-12,
        C_cmfb=0.25e-12, A0_db=75.0, gm_over_id=12.0,
        gamma_transistor=gamma, temp_celsius=temp_c
    )
    
    # 8-stage MDAC tapered profile
    Cs_profile = [2.0e-12, 1.2e-12, 0.8e-12, 0.5e-12, 0.3e-12, 0.2e-12, 0.2e-12, 0.2e-12]
    stages = [
        NonIdealMDACStage(
            stage_num=i+1, bits=1.5, Cs=c, Cf=c, Cp=0.1e-12, C_out_par=0.05e-12,
            C_cmfb=0.15e-12, Vref=1.0, A0_db=65.0, gamma_transistor=gamma,
            gm_over_id=12.0, sigma_cap_mismatch=sigma_mismatch,
            sigma_comp_offset=sigma_offset, sigma_vref_noise=sigma_vref,
            temp_celsius=temp_c
        ) for i, c in enumerate(Cs_profile)
    ]
    
    return FullPipelinedADCSimulator(sha=sha, stages=stages, quantizer_bits=2, Vdd=1.2)


# ==============================================================================
# EXPERIMENT 0: BASIC FUNCTIONALITY (SANITY CHECK)
# ==============================================================================
def run_sanity_check():
    print("\n" + "="*70)
    print(" EXPERIMENT 0: BASIC FUNCTIONALITY (SANITY CHECK)")
    print("="*70)

    adc = build_adc(gamma=1.5, temp_c=27.0, sigma_vref=0.0005)

    df_breakdown, summary = adc.run_static_analysis(f_clk=100e6)
    print("\n--- STAGE-BY-STAGE BREAKDOWN ---")
    print(df_breakdown.to_string(index=False))
    print("\n--- SYSTEM SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = 0.5 + 0.48 * np.sin(2 * np.pi * f_in * t)

    recon_sine = adc.run_transient_simulation(vin_sine, Vcm_in=0.5)
    sndr, sfdr, enob, spectrum_db = adc.compute_coherent_fft_metrics(recon_sine, M_bin)

    print("\n--- DYNAMIC METRICS (Coherent FFT) ---")
    print(f"  SNDR : {sndr:.2f} dB")
    print(f"  SFDR : {sfdr:.2f} dB")
    print(f"  ENOB : {enob:.2f} bits")


# ==============================================================================
# EXPERIMENT 1: TEMPERATURE SWEEP
# ==============================================================================
def run_temperature_sweep(temp_range=np.linspace(-40, 125, 12)):
    print("\n" + "="*70)
    print(" EXPERIMENT 1: TEMPERATURE SWEEP (-40°C to +125°C)")
    print("="*70)
    
    snr_list = []
    for temp_c in temp_range:
        adc = build_adc(temp_c=temp_c, sigma_vref=0.0)
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
def run_gamma_sweep(gamma_range=np.linspace(0.67, 2.5, 10)):
    print("\n" + "="*70)
    print(" EXPERIMENT 2: TRANSISTOR GAMMA SWEEP (0.67 to 2.5)")
    print("="*70)

    snr_list = []
    for g_val in gamma_range:
        adc = build_adc(gamma=g_val, sigma_vref=0.0)
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
def run_vref_noise_sweep(max_vref_noise_mv=1.0, num_points=10):
    print("\n" + "="*70)
    print(f" EXPERIMENT 3: VREF NOISE SWEEP (0 to {max_vref_noise_mv:.2f} mV RMS)")
    print("="*70)

    # Convert max noise in mV to Volts and create linspace
    max_vref_noise_v = max_vref_noise_mv / 1000.0
    vref_noise_range = np.linspace(0.0, max_vref_noise_v, num_points)

    num_samples = 2048
    M_bin = 31
    t = np.arange(num_samples)
    f_in = M_bin / num_samples
    vin_sine = 0.5 + 0.48 * np.sin(2 * np.pi * f_in * t)

    sndr_list = []
    for vnoise in vref_noise_range:
        adc = build_adc(sigma_vref=vnoise)
        recon = adc.run_transient_simulation(vin_sine)
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
    target_snr_db=70.0, 
    vref_pp_range=np.linspace(0.5, 4.0, 20)
):
    print("\n" + "="*70)
    print(f" EXPERIMENT 4: CONSTANT SNR ({target_snr_db} dB) VREF SWEEP")
    print("="*70)

    taper_profile = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.1, 0.1]
    total_caps_pF = []

    for vref_pp in vref_pp_range:
        vref_peak = vref_pp / 2.0
        Cs_base = 1.0e-12

        sha_test = ConventionalSHA(Cs=1.25*Cs_base, Cf=1.25*Cs_base, C_cmfb=0.15*Cs_base)
        stages_test = [
            NonIdealMDACStage(stage_num=i+1, Cs=Cs_base*tap, Cf=Cs_base*tap, C_cmfb=0.15*Cs_base*tap, Vref=vref_peak)
            for i, tap in enumerate(taper_profile)
        ]
        adc_test = FullPipelinedADCSimulator(sha=sha_test, stages=stages_test)
        _, summary_test = adc_test.run_static_analysis()
        snr_base_linear = 10 ** (float(summary_test["Thermal SNR"].split()[0]) / 10.0)

        target_snr_linear = 10 ** (target_snr_db / 10.0)
        Cs_scaled = Cs_base * (target_snr_linear / snr_base_linear)

        sha_final = ConventionalSHA(Cs=1.25*Cs_scaled, Cf=1.25*Cs_scaled, C_cmfb=0.15*Cs_scaled)
        stages_final = [
            NonIdealMDACStage(stage_num=i+1, Cs=Cs_scaled*tap, Cf=Cs_scaled*tap, C_cmfb=0.15*Cs_scaled*tap, Vref=vref_peak)
            for i, tap in enumerate(taper_profile)
        ]

        c_sha = sha_final.Cs + sha_final.Cf + sha_final.C_cmfb
        c_mdac = sum(s.Cs + s.Cf + s.C_cmfb for s in stages_final)
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
def run_dnl_inl_experiment(num_ramp_samples=300000):
    print("\n" + "="*70)
    print(" EXPERIMENT 5: STATIC DNL / INL MEASUREMENT")
    print("="*70)

    adc = build_adc(sigma_mismatch=0.001, sigma_offset=0.012)

    print(f"Running linear ramp simulation with {num_ramp_samples:,} samples...")
    codes, dnl, inl = adc.run_ramp_dnl_inl(num_ramp_samples=num_ramp_samples)

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
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    run_sanity_check()
    run_temperature_sweep()
    run_gamma_sweep()
    run_vref_noise_sweep(max_vref_noise_mv=2.0, num_points=20)    
    run_capacitance_vs_vref_sweep()
    run_dnl_inl_experiment()