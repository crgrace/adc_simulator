# pipelined_adc_cal_mc.py
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Optional

# Boltzmann constant
KB = 1.380649e-23


# ==============================================================================
# 0. MONTE CARLO CONFIGURATION DATACLASS
# ==============================================================================
@dataclass
class CalADCMCConfig:
    """Configuration class for calADC Monte Carlo Statistical Yield & Thermal Noise Study."""
    # Monte Carlo Settings
    num_mc_runs: int = 50              # Number of statistical Monte Carlo runs
    master_seed: int = 2026           # Master random seed for reproducibility
    
    # Target Output Evaluation Resolution
    effective_resolution: int = 12     # Target output resolution for system evaluation (12 bits)
    
    # Pipeline Architecture
    num_stages: int = 16              # Total number of 1.5-bit MDAC stages
    num_calibrated_stages: int = 5    # Number of front-end stages to calibrate
    backend_bits: int = 2             # Final flash quantizer bits (2-bit termination after Stage N)
    
    # Supply & Timing
    Vdd: float = 2.5                  # Supply voltage (Volts)
    Vref: float = 1.2                 # Reference voltage (2.4 Vpp differential)
    Vcm_in: float = 1.25              # Input common-mode voltage (Mid-supply)
    f_clk: float = 80e6               # Clock frequency (80 MHz)
    t_non_overlap: float = 0.5e-9     # Clock non-overlap time (0.5 ns)
    temp_kelvin: float = 300.0        # Temperature in Kelvin (300 K = 27°C)
    
    # Fixed-Point Integer Register Settings
    use_fixed_point: bool = True      # True: Integer 2's Complement | False: Double-Precision Float
    datapath_bits: int = 20           # Total bitwidth B_dp for digital reconstruction datapath
    weight_bits: int = 22             # Total bitwidth B_w for calibration weight accumulators
    mu_shift: int = 6                 # LMS step size power-of-2 right shift (mu = 2^-6 = 0.015625)
    cal_cycles_per_stage: int = 2000  # Number of calibration clock cycles per stage
    
    # Physical Nominal Values
    Cs_nominal: float = 0.6e-12       # Nominal sampling capacitor (0.6 pF)
    Cf_nominal: float = 0.6e-12       # Nominal feedback capacitor (0.6 pF)
    ota_a0_db_mean: float = 72.0       # Mean OTA DC open-loop gain (dB)
    gbw_mean: float = 400e6           # Mean OTA Gain-Bandwidth product (800 MHz)
    
    # Statistical Standard Deviations (MC Variations across stages & runs)
    ota_a0_db_std: float = 2.0        # OTA DC open-loop gain std dev (2.0 dB)
    gbw_std_rel: float = 0.08         # OTA GBW relative std dev (8% variation)
    gain_error_std: float = 0.005     # 0.5% RMS relative linear stage gain error
    cap_mismatch_std: float = 0.0008  # 0.08% RMS capacitor mismatch
    comp_offset_std: float = 0.015    # Sub-ADC comparator offset std dev (15 mV)
    
    # Dynamic Thermal Noise Standard Deviations (Per Sample)
    enable_ktc_noise: bool = True     # Include k_B T / C_s switched capacitor sampling noise
    enable_thermal_noise: bool = True # Include amplifier and reference driver thermal noise
    sigma_amp_noise: float = 200e-6   # 200 uV RMS amplifier thermal noise
    sigma_vref_noise: float = 150e-6  # 150 uV RMS reference voltage driver noise

    @property
    def datapath_frac_bits(self) -> int:
        return max(1, self.datapath_bits - 2)

    @property
    def weight_frac_bits(self) -> int:
        return max(1, self.weight_bits - 2)

    @property
    def total_bits(self) -> int:
        return int(self.num_stages + self.backend_bits - 1)

    @property
    def eval_bits(self) -> int:
        return self.effective_resolution


# ==============================================================================
# 1. 1.5-BIT MDAC STAGE WITH PHYSICAL MONTE CARLO & THERMAL NOISE
# ==============================================================================
class MDACStageMC:
    """1.5-bit MDAC stage with MC process variations, incomplete settling, and kT/C noise."""
    def __init__(self, stage_id: int, cfg: CalADCMCConfig, rng: np.random.Generator):
        self.stage_id = stage_id
        self.Vref = cfg.Vref
        self.enable_ktc = cfg.enable_ktc_noise
        self.enable_thermal = cfg.enable_thermal_noise
        self.sigma_amp = cfg.sigma_amp_noise if cfg.enable_thermal_noise else 0.0
        self.sigma_vref = cfg.sigma_vref_noise if cfg.enable_thermal_noise else 0.0
        self.rng = rng

        # Stage-specific static process variations
        c_mis_s = rng.normal(0, cfg.cap_mismatch_std)
        c_mis_f = rng.normal(0, cfg.cap_mismatch_std)
        self.Cs = cfg.Cs_nominal * (1.0 + c_mis_s)
        self.Cf = cfg.Cf_nominal * (1.0 + c_mis_f)
        
        # kT/C sampling noise std dev
        if cfg.enable_ktc_noise:
            self.sigma_ktc = np.sqrt((KB * cfg.temp_kelvin) / self.Cs)
        else:
            self.sigma_ktc = 0.0

        # OTA DC Gain & GBW variations
        a0_db = rng.normal(cfg.ota_a0_db_mean, cfg.ota_a0_db_std)
        A0 = 10 ** (a0_db / 20.0)
        gbw = rng.normal(cfg.gbw_mean, cfg.gbw_mean * cfg.gbw_std_rel)
        
        # Feedback factor & finite DC gain factor
        beta = self.Cf / (self.Cs + self.Cf)
        loop_gain_factor = (A0 * beta) / (1.0 + A0 * beta)
        
        # Incomplete linear settling factor: f_settle = 1 - exp(-t_settle / tau)
        t_settle = (1.0 / (2.0 * cfg.f_clk)) - cfg.t_non_overlap
        tau = 1.0 / (2.0 * np.pi * beta * gbw) if gbw > 0 else 1e-12
        settling_factor = 1.0 - np.exp(-max(0.0, t_settle) / tau) if t_settle > 0 else 1.0

        # Stage gain and reference step height
        g_err = rng.normal(0, cfg.gain_error_std)
        self.nominal_gain = 1.0 + (self.Cs / self.Cf)
        self.actual_gain = self.nominal_gain * (1.0 + g_err) * loop_gain_factor * settling_factor
        self.Vdac_step = (self.Cs / self.Cf) * cfg.Vref * loop_gain_factor * settling_factor
        
        # Sub-ADC decision thresholds with comparator offsets
        vth_nom = cfg.Vref / 4.0
        self.vth_hi = vth_nom + rng.normal(0, cfg.comp_offset_std)
        self.vth_lo = -vth_nom + rng.normal(0, cfg.comp_offset_std)

    def process_sample(self, Vin: float, force_D: Optional[int] = None) -> Tuple[int, float]:
        """Processes input sample through MDAC with dynamic sample-by-sample thermal noise."""
        # 1. Sample kT/C noise on input
        v_ktc = self.rng.normal(0, self.sigma_ktc) if self.sigma_ktc > 0 else 0.0
        v_in_noisy = Vin + v_ktc
        
        # 2. Sub-ADC decision
        if force_D is not None:
            D = force_D
        else:
            if v_in_noisy > self.vth_hi:
                D = 1
            elif v_in_noisy < self.vth_lo:
                D = -1
            else:
                D = 0

        # 3. Dynamic reference noise & amplifier noise
        v_vref_noise = self.rng.normal(0, self.sigma_vref) if self.sigma_vref > 0 else 0.0
        v_amp_noise = self.rng.normal(0, self.sigma_amp) if self.sigma_amp > 0 else 0.0

        # 4. Analog residue calculation
        vdac_step_noisy = self.Vdac_step * (1.0 + v_vref_noise / self.Vref)
        Vres = (self.actual_gain * v_in_noisy) - (D * vdac_step_noisy) + v_amp_noise
        return D, Vres


# ==============================================================================
# 2. TERMINATING BACKEND QUANTIZER
# ==============================================================================
class BackendQuantizer:
    """Terminating N-bit flash quantizer after final MDAC stage."""
    def __init__(self, bits: int = 2, Vref: float = 1.2):
        self.bits = bits
        self.Vref = Vref
        self.num_levels = 2 ** bits

    def process_sample(self, Vin: float) -> float:
        v_norm = np.clip((Vin + self.Vref) / (2.0 * self.Vref), 0.0, 1.0)
        code = np.floor(v_norm * self.num_levels)
        code = np.clip(code, 0, self.num_levels - 1)
        return ((code + 0.5) / self.num_levels) * 2.0 - 1.0


# ==============================================================================
# 3. calADC MONTE CARLO SIMULATOR CLASS
# ==============================================================================
class calADC_MC:
    """Single-instance calADC supporting MC variations and fixed-point calibration."""
    def __init__(self, cfg: CalADCMCConfig, run_seed: int):
        self.cfg = cfg
        self.num_stages = cfg.num_stages
        self.num_calibrated_stages = min(cfg.num_calibrated_stages, cfg.num_stages)
        self.Vref = cfg.Vref
        self.rng = np.random.default_rng(run_seed)

        # Build pipeline stages with randomized process parameters
        self.stages: List[MDACStageMC] = [
            MDACStageMC(stage_id=i+1, cfg=cfg, rng=self.rng) for i in range(cfg.num_stages)
        ]
        self.backend = BackendQuantizer(bits=cfg.backend_bits, Vref=cfg.Vref)

        # Digital calibration weights
        if cfg.use_fixed_point:
            w_pos_init = int(round(0.5 * (1 << cfg.weight_frac_bits)))
            w_neg_init = int(round(-0.5 * (1 << cfg.weight_frac_bits)))
            self.weights = {i: {"pos": w_pos_init, "neg": w_neg_init} for i in range(cfg.num_stages)}
        else:
            self.weights = {i: {"pos": 0.5, "neg": -0.5} for i in range(cfg.num_stages)}

    @staticmethod
    def _align_frac(val: int, src_frac: int, dst_frac: int) -> int:
        diff = src_frac - dst_frac
        return val >> diff if diff >= 0 else val << (-diff)

    def _saturate_int(self, val: int, bits: int) -> int:
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
        return int(np.clip(val, min_val, max_val))

    def quantize_to_eval_bits(self, y_normalized: np.ndarray) -> np.ndarray:
        bits = self.cfg.eval_bits
        num_codes = 1 << bits
        v_norm = np.clip((y_normalized + 1.0) / 2.0, 0.0, 1.0 - 1e-12)
        codes = np.clip(np.floor(v_norm * num_codes), 0, num_codes - 1)
        return ((codes + 0.5) / num_codes) * 2.0 - 1.0

    def reconstruct_sample(self, digital_codes: List[int], backend_val: float, use_calibrated: bool = True) -> float:
        if self.cfg.use_fixed_point:
            Y_next = int(round(backend_val * (1 << self.cfg.datapath_frac_bits)))
            for k in reversed(range(self.num_stages)):
                D_k = digital_codes[k]
                Y_next = Y_next >> 1
                if use_calibrated and (k < self.num_calibrated_stages):
                    w_int = self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0)
                else:
                    w_int = int(round(0.5 * D_k * (1 << self.cfg.weight_frac_bits)))
                w_dp = self._align_frac(w_int, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                Y_next = self._saturate_int(Y_next + w_dp, self.cfg.datapath_bits)
            return Y_next / (1 << self.cfg.datapath_frac_bits)
        else:
            Y_next = backend_val
            for k in reversed(range(self.num_stages)):
                D_k = digital_codes[k]
                w = (self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0.0)) if (use_calibrated and k < self.num_calibrated_stages) else 0.5 * D_k
                Y_next = 0.5 * Y_next + w
            return Y_next

    def run_calibration(self):
        for stage_idx in reversed(range(self.num_calibrated_stages)):
            stage = self.stages[stage_idx]
            w_pos = self.weights[stage_idx]["pos"]
            w_neg = self.weights[stage_idx]["neg"]
            
            for _ in range(self.cfg.cal_cycles_per_stage):
                # 1. Positive boundary (+Vref/4)
                vin_pos = (self.Vref / 4.0) + self.rng.normal(0, 0.005 * self.Vref)
                _, vres_0_pos = stage.process_sample(vin_pos, force_D=0)
                c_0_pos, b_0_pos = self._sample_downstream(stage_idx + 1, vres_0_pos)
                
                _, vres_1_pos = stage.process_sample(vin_pos, force_D=1)
                c_1_pos, b_1_pos = self._sample_downstream(stage_idx + 1, vres_1_pos)
                
                if self.cfg.use_fixed_point:
                    Y_down_0_pos = self._reconstruct_downstream_int(stage_idx + 1, c_0_pos, b_0_pos)
                    Y_down_1_pos = self._reconstruct_downstream_int(stage_idx + 1, c_1_pos, b_1_pos)
                    w_pos_dp = self._align_frac(w_pos, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                    e_pos = (Y_down_0_pos >> 1) - ((Y_down_1_pos >> 1) + w_pos_dp)
                    e_pos_w = self._align_frac(e_pos, self.cfg.datapath_frac_bits, self.cfg.weight_frac_bits)
                    w_pos = self._saturate_int(w_pos + (e_pos_w >> self.cfg.mu_shift), self.cfg.weight_bits)
                else:
                    Y_down_0_pos = self._reconstruct_downstream_float(stage_idx + 1, c_0_pos, b_0_pos)
                    Y_down_1_pos = self._reconstruct_downstream_float(stage_idx + 1, c_1_pos, b_1_pos)
                    e_pos = (0.5 * Y_down_0_pos) - (0.5 * Y_down_1_pos + w_pos)
                    w_pos += (2.0 ** (-self.cfg.mu_shift)) * e_pos

                # 2. Negative boundary (-Vref/4)
                vin_neg = (-self.Vref / 4.0) + self.rng.normal(0, 0.005 * self.Vref)
                _, vres_0_neg = stage.process_sample(vin_neg, force_D=0)
                c_0_neg, b_0_neg = self._sample_downstream(stage_idx + 1, vres_0_neg)
                
                _, vres_neg_neg = stage.process_sample(vin_neg, force_D=-1)
                c_neg_neg, b_neg_neg = self._sample_downstream(stage_idx + 1, vres_neg_neg)
                
                if self.cfg.use_fixed_point:
                    Y_down_0_neg = self._reconstruct_downstream_int(stage_idx + 1, c_0_neg, b_0_neg)
                    Y_down_neg_neg = self._reconstruct_downstream_int(stage_idx + 1, c_neg_neg, b_neg_neg)
                    w_neg_dp = self._align_frac(w_neg, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                    e_neg = (Y_down_0_neg >> 1) - ((Y_down_neg_neg >> 1) + w_neg_dp)
                    e_neg_w = self._align_frac(e_neg, self.cfg.datapath_frac_bits, self.cfg.weight_frac_bits)
                    w_neg = self._saturate_int(w_neg + (e_neg_w >> self.cfg.mu_shift), self.cfg.weight_bits)
                else:
                    Y_down_0_neg = self._reconstruct_downstream_float(stage_idx + 1, c_0_neg, b_0_neg)
                    Y_down_neg_neg = self._reconstruct_downstream_float(stage_idx + 1, c_neg_neg, b_neg_neg)
                    e_neg = (0.5 * Y_down_0_neg) - (0.5 * Y_down_neg_neg + w_neg)
                    w_neg += (2.0 ** (-self.cfg.mu_shift)) * e_neg

            self.weights[stage_idx]["pos"] = w_pos
            self.weights[stage_idx]["neg"] = w_neg

    def _sample_downstream(self, start_idx: int, Vin: float) -> Tuple[List[int], float]:
        v_curr = Vin
        codes = []
        for i in range(start_idx, self.num_stages):
            D, v_res = self.stages[i].process_sample(v_curr)
            codes.append(D)
            v_curr = v_res
        backend_val = self.backend.process_sample(v_curr)
        return codes, backend_val

    def _reconstruct_downstream_int(self, start_idx: int, codes: List[int], backend_val: float) -> int:
        Y_next = int(round(backend_val * (1 << self.cfg.datapath_frac_bits)))
        code_idx = len(codes) - 1
        for k in reversed(range(start_idx, self.num_stages)):
            D_k = codes[code_idx]
            code_idx -= 1
            Y_next = Y_next >> 1
            w_int = (self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0)) if (k < self.num_calibrated_stages) else int(round(0.5 * D_k * (1 << self.cfg.weight_frac_bits)))
            w_dp = self._align_frac(w_int, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
            Y_next = self._saturate_int(Y_next + w_dp, self.cfg.datapath_bits)
        return Y_next

    def _reconstruct_downstream_float(self, start_idx: int, codes: List[int], backend_val: float) -> float:
        Y_next = backend_val
        code_idx = len(codes) - 1
        for k in reversed(range(start_idx, self.num_stages)):
            D_k = codes[code_idx]
            code_idx -= 1
            w = (self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0.0)) if (k < self.num_calibrated_stages) else 0.5 * D_k
            Y_next = 0.5 * Y_next + w
        return Y_next

    def run_transient_sine(self, num_samples=2048, M_bin=31, use_calibrated=True) -> Tuple[float, float, float]:
        t = np.arange(num_samples)
        f_in = M_bin / num_samples
        vin_sine = 0.96 * self.Vref * np.sin(2 * np.pi * f_in * t)
        
        y_raw = np.zeros(num_samples)
        for n in range(num_samples):
            codes, backend_val = self._sample_downstream(0, vin_sine[n])
            y_raw[n] = self.reconstruct_sample(codes, backend_val, use_calibrated=use_calibrated)
            
        y_recon = self.quantize_to_eval_bits(y_raw)
        fft_spec = np.abs(np.fft.rfft(y_recon)) / (num_samples / 2.0)
        fft_spec[0] = 0.0
        signal_pwr = fft_spec[M_bin] ** 2
        
        noise_spec = fft_spec.copy()
        noise_spec[M_bin] = 0.0
        
        sfdr_db = 20 * np.log10(fft_spec[M_bin] / (np.max(noise_spec[1:]) + 1e-12))
        sndr_db = 10 * np.log10(signal_pwr / (np.sum(noise_spec ** 2) + 1e-12))
        enob = (sndr_db - 1.76) / 6.02
        return sndr_db, sfdr_db, enob

    def run_ramp_dnl_inl(self, num_samples=None, use_calibrated=True) -> Tuple[float, float]:
        eval_bits = self.cfg.eval_bits
        if num_samples is None:
            num_samples = 40 * (2 ** eval_bits) # default is 40 steps per code --> 0.025 LSB resolution

        vin_ramp = np.linspace(-1.02 * self.Vref, 1.02 * self.Vref, num_samples)
        y_raw = np.zeros(num_samples)
        for n in range(num_samples):
            codes, backend_val = self._sample_downstream(0, vin_ramp[n])
            y_raw[n] = self.reconstruct_sample(codes, backend_val, use_calibrated=use_calibrated)
            
        num_codes = 2 ** eval_bits
        v_norm = np.clip((y_raw + 1.0) / 2.0, 0.0, 1.0 - 1e-12)
        codes = np.clip(np.floor(v_norm * num_codes).astype(int), 0, num_codes - 1)
        
        H = np.bincount(codes, minlength=num_codes)
        valid_idx = np.where(H > 0)[0]
        f_code, l_code = valid_idx[4], valid_idx[-5]
        
        H_valid = H[f_code:l_code+1]
        ideal_hits = np.mean(H_valid)
        
        dnl = (H_valid / ideal_hits) - 1.0
        inl_raw = np.cumsum(dnl)
        inl = inl_raw - np.linspace(inl_raw[0], inl_raw[-1], len(inl_raw))
        return float(np.max(np.abs(dnl))), float(np.max(np.abs(inl)))


# ==============================================================================
# 4. MONTE CARLO RUNNER & DECOUPLED HIGH-RESOLUTION HISTOGRAM DASHBOARD
# ==============================================================================
def run_pipelined_adc_cal_mc(cfg: CalADCMCConfig):
    print("="*78)
    print(" pipelined_adc_cal_mc: MONTE CARLO STATISTICAL YIELD & NOISE STUDY")
    print(f" Executing {cfg.num_mc_runs} Statistical Runs ({cfg.eval_bits}-Bit Target Output)")
    print(" Includes: kT/C noise, Opamp thermal noise, Vref noise, A0/GBW variations")
    print("="*78)

    results = {
        "sndr_uncal": [], "sfdr_uncal": [], "enob_uncal": [], "dnl_uncal": [], "inl_uncal": [],
        "sndr_cal": [],   "sfdr_cal": [],   "enob_cal": [],   "dnl_cal": [],   "inl_cal": []
    }

    t_start = time.time()
    for run in range(cfg.num_mc_runs):
        run_seed = cfg.master_seed + run
        adc = calADC_MC(cfg, run_seed=run_seed)
        
        # 1. Uncalibrated Performance
        sndr_u, sfdr_u, enob_u = adc.run_transient_sine(use_calibrated=False)
        dnl_u, inl_u = adc.run_ramp_dnl_inl(use_calibrated=False)
        
        # 2. Run Integer LMS Calibration
        adc.run_calibration()
        
        # 3. Calibrated Performance
        sndr_c, sfdr_c, enob_c = adc.run_transient_sine(use_calibrated=True)
        dnl_c, inl_c = adc.run_ramp_dnl_inl(use_calibrated=True)
        
        results["sndr_uncal"].append(sndr_u)
        results["sfdr_uncal"].append(sfdr_u)
        results["enob_uncal"].append(enob_u)
        results["dnl_uncal"].append(dnl_u)
        results["inl_uncal"].append(inl_u)
        
        results["sndr_cal"].append(sndr_c)
        results["sfdr_cal"].append(sfdr_c)
        results["enob_cal"].append(enob_c)
        results["dnl_cal"].append(dnl_c)
        results["inl_cal"].append(inl_c)
        
        if (run + 1) % max(1, cfg.num_mc_runs // 10) == 0 or (run + 1) == cfg.num_mc_runs:
            print(f"  Completed Run {run+1:>3d}/{cfg.num_mc_runs} | Calibrated SNDR = {sndr_c:.2f} dB (ENOB = {enob_c:.2f}) | DNL = {dnl_c:.2f} LSB")

    t_elapsed = time.time() - t_start
    print(f"\nMonte Carlo Simulation Completed in {t_elapsed:.1f} seconds.")

    # Convert to numpy arrays for statistics
    for k in results:
        results[k] = np.array(results[k])

    # Print Summary Statistics
    print("\n" + "="*78)
    print(" MONTE CARLO STATISTICAL SUMMARY TABLE")
    print("="*78)
    print(f"{'Metric':<18} | {'Uncal Mean ± Std':<22} | {'Calibrated Mean ± Std':<22} | {'Cal Min / Max':<16}")
    print("-" * 78)
    
    m_list = [
        ("SNDR (dB)", "sndr_uncal", "sndr_cal"),
        ("SFDR (dB)", "sfdr_uncal", "sfdr_cal"),
        ("ENOB (Bits)", "enob_uncal", "enob_cal"),
        ("Max |DNL| (LSB)", "dnl_uncal", "dnl_cal"),
        ("Max |INL| (LSB)", "inl_uncal", "inl_cal")
    ]
    
    for name, key_u, key_c in m_list:
        u_m, u_s = np.mean(results[key_u]), np.std(results[key_u])
        c_m, c_s = np.mean(results[key_c]), np.std(results[key_c])
        c_min, c_max = np.min(results[key_c]), np.max(results[key_c])
        print(f"{name:<18} | {u_m:>7.2f} ± {u_s:<6.2f}         | {c_m:>7.2f} ± {c_s:<6.2f}         | {c_min:>6.2f} / {c_max:<6.2f}")

    yield_enob = np.mean(results["enob_cal"] >= 11.2) * 100.0
    yield_dnl = np.mean(results["dnl_cal"] < 0.5) * 100.0
    print("-" * 78)
    print(f" Statistical Yield (ENOB ≥ 11.2 Bits) : {yield_enob:.1f}%")
    print(f" Statistical Yield (Max |DNL| < 0.5 LSB): {yield_dnl:.1f}%")

    # ==========================================================================
    # DECOUPLED HIGH-RESOLUTION HISTOGRAM DASHBOARD (4x2 SIDE-BY-SIDE)
    # ==========================================================================
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    
    # Helper to calculate clean adaptive binning range
    def get_adaptive_bins(data, num_bins=25):
        d_min, d_max = np.min(data), np.max(data)
        margin = max((d_max - d_min) * 0.08, 0.01)
        return np.linspace(d_min - margin, d_max + margin, num_bins)

    # --------------------------------------------------------------------------
    # ROW 1: DYNAMIC SNDR (UNCAL VS CAL)
    # --------------------------------------------------------------------------
    ax_u1, ax_c1 = axes[0, 0], axes[0, 1]
    
    bins_sndr_u = get_adaptive_bins(results["sndr_uncal"], 20)
    ax_u1.hist(results["sndr_uncal"], bins=bins_sndr_u, color='tab:red', alpha=0.7, edgecolor='black')
    ax_u1.set_xlabel("Uncalibrated Dynamic SNDR (dB)")
    ax_u1.set_ylabel("Count")
    ax_u1.set_title(f"1a. Uncalibrated SNDR (μ = {np.mean(results['sndr_uncal']):.1f} dB)", fontweight='bold')
    ax_u1.grid(True, linestyle=':', alpha=0.6)

    bins_sndr_c = get_adaptive_bins(results["sndr_cal"], 25)
    ax_c1.hist(results["sndr_cal"], bins=bins_sndr_c, color='tab:blue', alpha=0.7, edgecolor='black')
    ax_c1.axvline(np.mean(results["sndr_cal"]), color='navy', linestyle='--', lw=1.5, label=f'Mean = {np.mean(results["sndr_cal"]):.2f} dB')
    ax_c1.axvline(6.02 * cfg.eval_bits + 1.76, color='black', linestyle=':', label=f'Ideal {cfg.eval_bits}-Bit Floor')
    ax_c1.set_xlabel("Calibrated Dynamic SNDR (dB)")
    ax_c1.set_ylabel("Count")
    ax_c1.set_title(f"1b. Calibrated SNDR (μ = {np.mean(results['sndr_cal']):.2f} dB, σ = {np.std(results['sndr_cal']):.2f} dB)", fontweight='bold')
    ax_c1.grid(True, linestyle=':', alpha=0.6)
    ax_c1.legend(loc='upper left', fontsize=8)

    # --------------------------------------------------------------------------
    # ROW 2: EFFECTIVE RESOLUTION ENOB (UNCAL VS CAL)
    # --------------------------------------------------------------------------
    ax_u2, ax_c2 = axes[1, 0], axes[1, 1]
    
    bins_enob_u = get_adaptive_bins(results["enob_uncal"], 20)
    ax_u2.hist(results["enob_uncal"], bins=bins_enob_u, color='tab:red', alpha=0.7, edgecolor='black')
    ax_u2.set_xlabel("Uncalibrated ENOB (Bits)")
    ax_u2.set_ylabel("Count")
    ax_u2.set_title(f"2a. Uncalibrated ENOB (μ = {np.mean(results['enob_uncal']):.2f} Bits)", fontweight='bold')
    ax_u2.grid(True, linestyle=':', alpha=0.6)

    bins_enob_c = get_adaptive_bins(results["enob_cal"], 25)
    ax_c2.hist(results["enob_cal"], bins=bins_enob_c, color='tab:green', alpha=0.7, edgecolor='black')
    ax_c2.axvline(np.mean(results["enob_cal"]), color='darkgreen', linestyle='--', lw=1.5, label=f'Mean = {np.mean(results["enob_cal"]):.2f} Bits')
    ax_c2.axvline(11.2, color='crimson', linestyle=':', label='Yield Spec (11.2 Bits)')
    ax_c2.set_xlabel("Calibrated ENOB (Bits)")
    ax_c2.set_ylabel("Count")
    ax_c2.set_title(f"2b. Calibrated ENOB (μ = {np.mean(results['enob_cal']):.2f} Bits, σ = {np.std(results['enob_cal']):.2f})", fontweight='bold')
    ax_c2.grid(True, linestyle=':', alpha=0.6)
    ax_c2.legend(loc='upper left', fontsize=8)

    # --------------------------------------------------------------------------
    # ROW 3: MAXIMUM DNL (UNCAL VS CAL)
    # --------------------------------------------------------------------------
    ax_u3, ax_c3 = axes[2, 0], axes[2, 1]
    
    bins_dnl_u = get_adaptive_bins(results["dnl_uncal"], 20)
    ax_u3.hist(results["dnl_uncal"], bins=bins_dnl_u, color='tab:red', alpha=0.7, edgecolor='black')
    ax_u3.set_xlabel(f"Uncalibrated Max |DNL| ({cfg.eval_bits}-Bit LSB)")
    ax_u3.set_ylabel("Count")
    ax_u3.set_title(f"3a. Uncalibrated Max |DNL| (μ = {np.mean(results['dnl_uncal']):.2f} LSB)", fontweight='bold')
    ax_u3.grid(True, linestyle=':', alpha=0.6)

    bins_dnl_c = get_adaptive_bins(results["dnl_cal"], 25)
    ax_c3.hist(results["dnl_cal"], bins=bins_dnl_c, color='tab:purple', alpha=0.7, edgecolor='black')
    ax_c3.axvline(np.mean(results["dnl_cal"]), color='indigo', linestyle='--', lw=1.5, label=f'Mean = {np.mean(results["dnl_cal"]):.2f} LSB')
    ax_c3.axvline(0.5, color='crimson', linestyle=':', label='Target Threshold (0.5 LSB)')
    ax_c3.set_xlabel(f"Calibrated Max |DNL| ({cfg.eval_bits}-Bit LSB)")
    ax_c3.set_ylabel("Count")
    ax_c3.set_title(f"3b. Calibrated Max |DNL| (μ = {np.mean(results['dnl_cal']):.2f} LSB, σ = {np.std(results['dnl_cal']):.3f} LSB)", fontweight='bold')
    ax_c3.grid(True, linestyle=':', alpha=0.6)
    ax_c3.legend(loc='upper right', fontsize=8)

    # --------------------------------------------------------------------------
    # ROW 4: MAXIMUM INL (UNCAL VS CAL)
    # --------------------------------------------------------------------------
    ax_u4, ax_c4 = axes[3, 0], axes[3, 1]
    
    bins_inl_u = get_adaptive_bins(results["inl_uncal"], 20)
    ax_u4.hist(results["inl_uncal"], bins=bins_inl_u, color='tab:red', alpha=0.7, edgecolor='black')
    ax_u4.set_xlabel(f"Uncalibrated Max |INL| ({cfg.eval_bits}-Bit LSB)")
    ax_u4.set_ylabel("Count")
    ax_u4.set_title(f"4a. Uncalibrated Max |INL| (μ = {np.mean(results['inl_uncal']):.2f} LSB)", fontweight='bold')
    ax_u4.grid(True, linestyle=':', alpha=0.6)

    bins_inl_c = get_adaptive_bins(results["inl_cal"], 25)
    ax_c4.hist(results["inl_cal"], bins=bins_inl_c, color='indigo', alpha=0.7, edgecolor='black')
    ax_c4.axvline(np.mean(results["inl_cal"]), color='black', linestyle='--', lw=1.5, label=f'Mean = {np.mean(results["inl_cal"]):.2f} LSB')
    ax_c4.axvline(1.0, color='crimson', linestyle=':', label='Target Threshold (1.0 LSB)')
    ax_c4.set_xlabel(f"Calibrated Max |INL| ({cfg.eval_bits}-Bit LSB)")
    ax_c4.set_ylabel("Count")
    ax_c4.set_title(f"4b. Calibrated Max |INL| (μ = {np.mean(results['inl_cal']):.2f} LSB, σ = {np.std(results['inl_cal']):.3f} LSB)", fontweight='bold')
    ax_c4.grid(True, linestyle=':', alpha=0.6)
    ax_c4.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"pipelined_adc_cal_mc: High-Resolution Statistical Distributions ({cfg.num_mc_runs} Runs, {cfg.eval_bits}-Bit Output Grid)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration for 2.5V thick-oxide process Monte Carlo Yield Study
    mc_cfg = CalADCMCConfig(
        num_mc_runs=1,                # 50 statistical runs
        effective_resolution=12,       # Evaluate at 12-bit level
        num_stages=18,                 # 16 MDAC stages
        num_calibrated_stages=7,       # Calibrate stages 1-5
        cal_cycles_per_stage=4000,      # default = 2000 
        mu_shift = 8,                   # default = 8
        
        # Thermal & Reference Noise
        enable_ktc_noise=True,         # Include kT/C sampling noise
        enable_thermal_noise=True,     # Include amplifier & ref noise
        sigma_amp_noise=200e-6,        # 200 uV RMS
        sigma_vref_noise=150e-6,       # 150 uV RMS
        
        # Mismatches & Variations
        gain_error_std=0.001,          # 0.5% gain error
        cap_mismatch_std=0.001,       # 0.08% cap mismatch
        ota_a0_db_mean=74.0,           # 68 dB mean gain
        ota_a0_db_std=2.5,             # 2.5 dB gain variation
        gbw_mean=400e6,                # 800 MHz GBW
        gbw_std_rel=0.08,              # 8% GBW mismatch
        comp_offset_std=0.015          # 15 mV comparator offset
    )

    run_pipelined_adc_cal_mc(mc_cfg)