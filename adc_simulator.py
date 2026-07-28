# adc_simulator.py
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

k_B = 1.380649e-23  # Boltzmann constant (J/K)

def celsius_to_kelvin(temp_celsius):
    return temp_celsius + 273.15


# ==============================================================================
# CONFIGURATION DATA CLASS
# ==============================================================================
@dataclass
class ADCConfig:
    """Main configuration class for Pipelined ADC parameters."""
    # Pipeline Architecture
    num_stages: int = 8              # Number of MDAC stages (proxy for resolution)
    mdac_bits: float = 1.5           # Stage resolution (1.5-bit per MDAC)
    quantizer_bits: int = 2          # Back-end flash quantizer bits
    
    # Supply & References
    Vref: float = 1.0                # Peak reference voltage (Volts)
    Vdd: float = 1.2                 # Supply voltage (Volts)
    Vcm_in: float = 0.5              # Input common-mode voltage (Volts)
    
    # Clocking
    f_clk: float = 100e6             # Clock frequency (Hz)
    t_non_overlap: float = 0.4e-9    # Non-overlapping clock time (seconds)
    
    # Transistor Efficiency & Technology
    gm_over_id: float = 12.0         # Default gm/ID efficiency (V^-1)
    sha_gm_over_id: Optional[float] = None     # Optional SHA override
    mdac_gm_over_id: Optional[float] = None    # Optional global MDAC override
    gm_over_id_profile: Optional[List[float]] = None  # Optional per-stage profile
    
    temp_celsius: float = 27.0       # Temperature (°C)
    gamma_transistor: float = 1.5    # Transistor excess noise factor
    
    # Non-idealities & Mismatch
    sigma_cap_mismatch: float = 0.001   # Capacitor mismatch std dev (0.1%)
    sigma_comp_offset: float = 0.012    # Sub-ADC comp offset std dev (12 mV)
    sigma_vref_noise: float = 0.0005    # Vref noise std dev (0.5 mV RMS)
    sigma_a0_db: float = 1.0            # OTA DC open-loop gain mismatch std dev in dB
    
    # Front-End SHA Parameters
    sha_Cs: float = 2.5e-12
    sha_Cf: float = 2.5e-12
    sha_Cp: float = 0.15e-12
    sha_C_out_par: float = 0.08e-12
    sha_C_cmfb: float = 0.25e-12
    sha_A0_db: float = 75.0
    
    # MDAC Stage Default Parameters
    Cs_profile: List[float] = field(default_factory=lambda: [
        2.0e-12, 1.2e-12, 0.8e-12, 0.5e-12, 0.3e-12, 0.2e-12, 0.2e-12, 0.2e-12
    ])
    mdac_Cp: float = 0.1e-12
    mdac_C_out_par: float = 0.05e-12
    mdac_C_cmfb: float = 0.15e-12
    mdac_A0_db: float = 65.0


# ==============================================================================
# 1. FRONT-END SAMPLE-AND-HOLD
# ==============================================================================
class ConventionalSHA:
    """Conventional (Non-Flip-Around) Single-Ended to Differential S/H."""
    def __init__(
        self, Cs=2.5e-12, Cf=2.5e-12, Cp=0.15e-12, C_out_par=0.08e-12, 
        C_cmfb=0.25e-12, A0_db=75.0, sigma_a0_db=1.0, gm_over_id=12.0, 
        gamma_transistor=1.5, temp_celsius=27.0
    ):
        self.Cs = Cs
        self.Cf = Cf
        self.Cp = Cp
        self.C_out_par = C_out_par
        self.C_cmfb = C_cmfb
        
        A0_db_actual = A0_db + np.random.normal(0, sigma_a0_db)
        self.A0_db = A0_db_actual
        self.A0 = 10 ** (A0_db_actual / 20.0)
        
        self.gm_over_id = gm_over_id
        self.gamma_transistor = gamma_transistor
        self.temp_kelvin = celsius_to_kelvin(temp_celsius)
        
        # Fixed physical sizing attributes
        self.gm = None
        self.tau = None

    @property
    def beta(self):
        return self.Cf / (self.Cs + self.Cf + self.Cp)

    @property
    def nominal_gain(self):
        return self.Cs / self.Cf

    @property
    def actual_gain(self):
        loop_gain = self.A0 * self.beta
        return self.nominal_gain * (loop_gain / (1.0 + loop_gain))

    @property
    def input_referred_noise_sq(self):
        return (2.0 * k_B * self.temp_kelvin / self.Cs) * (1.0 + (self.gamma_transistor / self.beta))

    def calculate_cl_load(self, Cs_next_stage):
        C_fb_series = (self.Cf * (self.Cs + self.Cp)) / (self.Cs + self.Cf + self.Cp)
        return Cs_next_stage + self.C_out_par + self.C_cmfb + C_fb_series

    def calculate_settling_and_gm(self, Cs_next_stage, total_adc_bits, f_clk=100e6, t_non_overlap=0.4e-9, Vdd=1.2):
        t_settle = (1.0 / (2.0 * f_clk)) - t_non_overlap
        N_tau = (total_adc_bits + 1.0) * np.log(2.0)
        tau_req = t_settle / N_tau
        f_cl = 1.0 / (2.0 * np.pi * tau_req)

        GBW = f_cl / self.beta
        C_load_eff = self.calculate_cl_load(Cs_next_stage)
        gm = 2.0 * np.pi * GBW * C_load_eff

        I_tail = gm / self.gm_over_id
        P_ota = I_tail * Vdd

        return {
            "GBW_MHz": GBW / 1e6,
            "gm_mS": gm * 1e3,
            "C_load_eff_pF": C_load_eff * 1e12,
            "Power_mW": P_ota * 1e3
        }

    def process_signal(self, Vin_se, Vcm_in=0.5, t_settle=None):
        settling_factor = 1.0
        if t_settle is not None and self.tau is not None:
            if t_settle <= 0:
                settling_factor = 0.0
            else:
                settling_factor = max(0.0, 1.0 - np.exp(-t_settle / self.tau))

        v_noise = np.random.normal(0, np.sqrt(self.input_referred_noise_sq))
        Vin_diff = (Vin_se - Vcm_in) + v_noise
        return Vin_diff * self.actual_gain * settling_factor


# ==============================================================================
# 2. PIPELINED MDAC STAGE
# ==============================================================================
class NonIdealMDACStage:
    """1.5-Bit Differential MDAC Stage."""
    def __init__(
        self, stage_num, bits=1.5, Cs=1e-12, Cf=1e-12, Cp=0.1e-12, C_out_par=0.05e-12,
        C_cmfb=0.15e-12, Vref=1.0, A0_db=65.0, sigma_a0_db=1.0, gamma_transistor=1.5, gm_over_id=12.0,
        sigma_cap_mismatch=0.001, sigma_comp_offset=0.012, sigma_vref_noise=0.001, temp_celsius=27.0
    ):
        self.stage_num = stage_num
        self.bits = bits
        self.effective_bits = bits - 0.5
        self.Vref = Vref
        self.Cp = Cp
        self.C_out_par = C_out_par
        self.C_cmfb = C_cmfb
        
        A0_db_actual = A0_db + np.random.normal(0, sigma_a0_db)
        self.A0_db = A0_db_actual
        self.A0 = 10 ** (A0_db_actual / 20.0)

        self.gamma_transistor = gamma_transistor
        self.gm_over_id = gm_over_id
        self.sigma_vref_noise = sigma_vref_noise
        self.temp_kelvin = celsius_to_kelvin(temp_celsius)

        self.Cs = Cs * (1.0 + np.random.normal(0, sigma_cap_mismatch))
        self.Cf = Cf * (1.0 + np.random.normal(0, sigma_cap_mismatch))

        vth_nom = Vref / 4.0
        self.vth_hi = vth_nom + np.random.normal(0, sigma_comp_offset)
        self.vth_lo = -vth_nom + np.random.normal(0, sigma_comp_offset)

        # Fixed physical sizing attributes
        self.gm = None
        self.tau = None

    @property
    def beta(self):
        return self.Cf / (self.Cs + self.Cf + self.Cp)

    @property
    def nominal_gain(self):
        return 1.0 + (self.Cs / self.Cf)

    @property
    def actual_gain(self):
        loop_gain = self.A0 * self.beta
        return self.nominal_gain * (loop_gain / (1.0 + loop_gain))

    @property
    def input_referred_noise_sq(self):
        return (2.0 * k_B * self.temp_kelvin / self.Cs) * (1.0 + (self.gamma_transistor / self.beta))

    def calculate_cl_load(self, Cs_next_stage):
        C_fb_series = (self.Cf * (self.Cs + self.Cp)) / (self.Cs + self.Cf + self.Cp)
        return Cs_next_stage + self.C_out_par + self.C_cmfb + C_fb_series

    def calculate_settling_and_gm(self, Cs_next_stage, remaining_bits, f_clk=100e6, t_non_overlap=0.4e-9, Vdd=1.2):
        t_settle = (1.0 / (2.0 * f_clk)) - t_non_overlap
        if t_settle <= 0:
            raise ValueError("Clock frequency too high!")

        N_tau = (remaining_bits + 1.0) * np.log(2.0)
        tau_req = t_settle / N_tau
        f_cl = 1.0 / (2.0 * np.pi * tau_req)

        GBW = f_cl / self.beta
        C_load_eff = self.calculate_cl_load(Cs_next_stage)
        gm = 2.0 * np.pi * GBW * C_load_eff

        I_tail = gm / self.gm_over_id
        P_ota = I_tail * Vdd

        return {
            "GBW_MHz": GBW / 1e6,
            "gm_mS": gm * 1e3,
            "C_load_eff_pF": C_load_eff * 1e12,
            "Power_mW": P_ota * 1e3
        }

    def process_sample(self, Vin, t_settle=None):
        settling_factor = 1.0
        if t_settle is not None and self.tau is not None:
            if t_settle <= 0:
                settling_factor = 0.0
            else:
                settling_factor = max(0.0, 1.0 - np.exp(-t_settle / self.tau))

        vref_sample = self.Vref + np.random.normal(0, self.sigma_vref_noise)
        v_noise = np.random.normal(0, np.sqrt(self.input_referred_noise_sq))
        Vin_noisy = Vin + v_noise

        if Vin_noisy > self.vth_hi:
            D = 1
        elif Vin_noisy < self.vth_lo:
            D = -1
        else:
            D = 0

        loop_gain = self.A0 * self.beta
        gain_factor = loop_gain / (1.0 + loop_gain)
        
        G_stage = (1.0 + (self.Cs / self.Cf)) * gain_factor
        Vdac = D * (self.Cs / self.Cf) * vref_sample * gain_factor
        
        Vres = ((Vin_noisy * G_stage) - Vdac) * settling_factor
        return D, Vres


# ==============================================================================
# 3. MAIN PIPELINE SIMULATOR
# ==============================================================================
class FullPipelinedADCSimulator:
    """Main Pipeline ADC System."""
    def __init__(self, sha, stages, quantizer_bits=2, Vdd=1.2):
        self.sha = sha
        self.stages = stages
        self.quantizer_bits = quantizer_bits
        self.Vdd = Vdd
        self.total_bits = int(sum(s.effective_bits for s in self.stages) + self.quantizer_bits)

    @classmethod
    def from_config(cls, cfg: ADCConfig):
        """Constructs full simulator and fixes OTA sizing/time constants at cfg.f_clk."""
        sha_gm_id = cfg.sha_gm_over_id if cfg.sha_gm_over_id is not None else cfg.gm_over_id

        sha = ConventionalSHA(
            Cs=cfg.sha_Cs, Cf=cfg.sha_Cf, Cp=cfg.sha_Cp, C_out_par=cfg.sha_C_out_par,
            C_cmfb=cfg.sha_C_cmfb, A0_db=cfg.sha_A0_db, sigma_a0_db=cfg.sigma_a0_db,
            gm_over_id=sha_gm_id, gamma_transistor=cfg.gamma_transistor, 
            temp_celsius=cfg.temp_celsius
        )

        cs_profile = list(cfg.Cs_profile)
        if len(cs_profile) < cfg.num_stages:
            last_val = cs_profile[-1] if len(cs_profile) > 0 else 0.2e-12
            cs_profile.extend([last_val] * (cfg.num_stages - len(cs_profile)))
        elif len(cs_profile) > cfg.num_stages:
            cs_profile = cs_profile[:cfg.num_stages]

        stages = []
        for i in range(cfg.num_stages):
            cs_val = cs_profile[i]
            if cfg.gm_over_id_profile is not None and i < len(cfg.gm_over_id_profile):
                stage_gm_id = cfg.gm_over_id_profile[i]
            elif cfg.mdac_gm_over_id is not None:
                stage_gm_id = cfg.mdac_gm_over_id
            else:
                stage_gm_id = cfg.gm_over_id

            stages.append(
                NonIdealMDACStage(
                    stage_num=i + 1, bits=cfg.mdac_bits, Cs=cs_val, Cf=cs_val,
                    Cp=cfg.mdac_Cp, C_out_par=cfg.mdac_C_out_par, C_cmfb=cfg.mdac_C_cmfb,
                    Vref=cfg.Vref, A0_db=cfg.mdac_A0_db, sigma_a0_db=cfg.sigma_a0_db,
                    gamma_transistor=cfg.gamma_transistor, gm_over_id=stage_gm_id, 
                    sigma_cap_mismatch=cfg.sigma_cap_mismatch, sigma_comp_offset=cfg.sigma_comp_offset, 
                    sigma_vref_noise=cfg.sigma_vref_noise, temp_celsius=cfg.temp_celsius
                )
            )

        # Fix physical OTA transconductance (gm) and time constants (tau = C_L / (beta * gm))
        total_adc_bits = int(sum(s.effective_bits for s in stages) + cfg.quantizer_bits)
        sha_specs = sha.calculate_settling_and_gm(
            Cs_next_stage=stages[0].Cs, total_adc_bits=total_adc_bits,
            f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap, Vdd=cfg.Vdd
        )
        sha.gm = sha_specs["gm_mS"] * 1e-3
        cl_sha = sha.calculate_cl_load(stages[0].Cs)
        sha.tau = cl_sha / (sha.beta * sha.gm)

        for i, stage in enumerate(stages):
            cs_next = stages[i+1].Cs if (i + 1 < len(stages)) else 0.0
            bits_rem = sum(s.effective_bits for s in stages[i:]) + cfg.quantizer_bits
            specs = stage.calculate_settling_and_gm(
                Cs_next_stage=cs_next, remaining_bits=bits_rem,
                f_clk=cfg.f_clk, t_non_overlap=cfg.t_non_overlap, Vdd=cfg.Vdd
            )
            stage.gm = specs["gm_mS"] * 1e-3
            cl_stage = stage.calculate_cl_load(cs_next)
            stage.tau = cl_stage / (stage.beta * stage.gm)

        return cls(sha=sha, stages=stages, quantizer_bits=cfg.quantizer_bits, Vdd=cfg.Vdd)

    def run_static_analysis(self, f_clk=100e6, t_non_overlap=0.4e-9):
        results = []
        
        sha_specs = self.sha.calculate_settling_and_gm(
            Cs_next_stage=self.stages[0].Cs, total_adc_bits=self.total_bits, 
            f_clk=f_clk, t_non_overlap=t_non_overlap, Vdd=self.Vdd
        )
        results.append({
            "Stage": "SHA", "Cs (pF)": self.sha.Cs*1e12, "Cf (pF)": self.sha.Cf*1e12, 
            "Beta": round(self.sha.beta, 3), "A0 (dB)": round(self.sha.A0_db, 2),
            "gm/ID (V^-1)": round(self.sha.gm_over_id, 1),
            "GBW (MHz)": round(sha_specs["GBW_MHz"], 1), "gm (mS)": round(sha_specs["gm_mS"], 2), 
            "C_eff (pF)": round(sha_specs["C_load_eff_pF"], 3), "OTA Power (mW)": round(sha_specs["Power_mW"], 2)
        })

        total_noise_sq = 0.0
        cumulative_gain = self.sha.actual_gain

        for i, stage in enumerate(self.stages):
            Cs_next = self.stages[i+1].Cs if (i + 1 < len(self.stages)) else 0.0
            bits_downstream = sum(s.effective_bits for s in self.stages[i:]) + self.quantizer_bits
            
            specs = stage.calculate_settling_and_gm(
                Cs_next_stage=Cs_next, remaining_bits=bits_downstream, 
                f_clk=f_clk, t_non_overlap=t_non_overlap, Vdd=self.Vdd
            )

            stage_noise = stage.input_referred_noise_sq
            total_noise_sq += stage_noise / (cumulative_gain ** 2)

            results.append({
                "Stage": f"MDAC {i+1}", "Cs (pF)": round(stage.Cs*1e12, 3), "Cf (pF)": round(stage.Cf*1e12, 3), 
                "Beta": round(stage.beta, 3), "A0 (dB)": round(stage.A0_db, 2),
                "gm/ID (V^-1)": round(stage.gm_over_id, 1),
                "GBW (MHz)": round(specs["GBW_MHz"], 1), "gm (mS)": round(specs["gm_mS"], 2), 
                "C_eff (pF)": round(specs["C_load_eff_pF"], 3), "OTA Power (mW)": round(specs["Power_mW"], 2)
            })
            cumulative_gain *= stage.actual_gain

        Vref = self.stages[0].Vref
        P_signal = (Vref ** 2) / 2.0
        snr_db = 10 * np.log10(P_signal / total_noise_sq)

        summary = {
            "Total ADC Resolution": f"{self.total_bits} Bits ({len(self.stages)} MDAC Stages)",
            "Thermal SNR": f"{snr_db:.2f} dB",
            "Thermal ENOB": f"{(snr_db - 1.76) / 6.02:.2f} bits",
            "Total OTA Power": f"{sum(r['OTA Power (mW)'] for r in results):.2f} mW"
        }

        return pd.DataFrame(results), summary

    def run_transient_simulation(self, vin_se_array, Vcm_in=0.5, f_clk=None, t_non_overlap=0.4e-9):
        N = len(vin_se_array)
        digital_codes = np.zeros((N, len(self.stages) + 1))
        
        t_settle = (1.0 / (2.0 * f_clk)) - t_non_overlap if f_clk is not None else None
        
        v_diff = self.sha.process_signal(vin_se_array, Vcm_in=Vcm_in, t_settle=t_settle)

        for n in range(N):
            v_in = v_diff[n]
            for stage_idx, stage in enumerate(self.stages):
                D, v_res = stage.process_sample(v_in, t_settle=t_settle)
                digital_codes[n, stage_idx] = D
                v_in = v_res
            
            final_code = np.clip(np.round(v_in / (self.stages[-1].Vref / 2.0)), -1, 2)
            digital_codes[n, -1] = final_code

        reconstructed_analog = np.zeros(N)
        for n in range(N):
            val = digital_codes[n, -1] / (2**self.quantizer_bits)
            for k in reversed(range(len(self.stages))):
                val = (val + digital_codes[n, k]) / 2.0
            reconstructed_analog[n] = val

        return reconstructed_analog

    def compute_coherent_fft_metrics(self, reconstructed_signal, M_bin):
        N = len(reconstructed_signal)
        fft_spectrum = np.abs(np.fft.rfft(reconstructed_signal)) / (N / 2.0)
        fft_spectrum[0] = 0.0  
        
        signal_power = fft_spectrum[M_bin] ** 2
        
        fft_noise_spectrum = fft_spectrum.copy()
        fft_noise_spectrum[M_bin] = 0.0
        
        max_spur_val = np.max(fft_noise_spectrum[1:])
        sfdr_db = 20 * np.log10(fft_spectrum[M_bin] / max_spur_val)
        
        noise_dist_power = np.sum(fft_noise_spectrum ** 2)
        sndr_db = 10 * np.log10(signal_power / noise_dist_power)
        enob = (sndr_db - 1.76) / 6.02
        
        spectrum_db = 20 * np.log10(fft_spectrum + 1e-12)
        return sndr_db, sfdr_db, enob, spectrum_db

    def float_to_code(self, v_reconstructed):
        num_codes = 2 ** self.total_bits
        Vref = self.stages[0].Vref
        v_norm = (v_reconstructed + Vref) / (2.0 * Vref)
        codes = np.floor(v_norm * num_codes).astype(int)
        return np.clip(codes, 0, num_codes - 1)

    def run_ramp_dnl_inl(self, num_ramp_samples=300000, Vcm_in=0.5, overdrive=1.04, guard_codes=4):
        Vref = self.stages[0].Vref
        vin_ramp = np.linspace(Vcm_in - Vref * overdrive, Vcm_in + Vref * overdrive, num_ramp_samples)
        
        reconstructed = self.run_transient_simulation(vin_ramp, Vcm_in=Vcm_in)
        codes = self.float_to_code(reconstructed)

        num_codes = 2 ** self.total_bits
        H = np.bincount(codes, minlength=num_codes)

        valid_indices = np.where(H > 0)[0]
        first_code = max(guard_codes, valid_indices[0] + guard_codes)
        last_code = min(num_codes - 1 - guard_codes, valid_indices[-1] - guard_codes)

        H_valid = H[first_code : last_code + 1]
        ideal_hits_per_code = np.mean(H_valid)

        dnl = (H_valid / ideal_hits_per_code) - 1.0
        inl_raw = np.cumsum(dnl)
        linear_trend = np.linspace(inl_raw[0], inl_raw[-1], len(inl_raw))
        inl = inl_raw - linear_trend

        code_axis = np.arange(first_code, last_code + 1)
        return code_axis, dnl, inl