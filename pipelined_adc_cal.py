# pipelined_adc_cal.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict, Optional


# ==============================================================================
# 0. CONFIGURATION DATACLASS (UNIFIED REGISTER WIDTHS)
# ==============================================================================
@dataclass
class CalADCConfig:
    """Configuration class for 2.5V Thick-Oxide calADC with Integer Arithmetic Datapath."""
    # Pipeline Architecture
    num_stages: int = 16              # Total number of 1.5-bit MDAC stages in pipeline
    num_calibrated_stages: int = 5    # Number of front-end stages to calibrate (1 to M)
    backend_bits: int = 2             # Final flash quantizer bits (2-bit termination after Stage N)
    
    # Supply & Reference Voltages (2.5V Thick-Oxide Process)
    Vdd: float = 2.5                  # Supply voltage (Volts)
    Vref: float = 1.2                 # Reference voltage (2.4 Vpp differential)
    Vcm_in: float = 1.25              # Input common-mode voltage (Mid-supply)
    
    # Fixed-Point Integer Register Settings
    use_fixed_point: bool = True      # True: Integer 2's Complement | False: Double-Precision Float
    datapath_bits: int = 20           # Total bitwidth B_dp for digital reconstruction datapath
    weight_bits: int = 22             # Total bitwidth B_w for calibration weight accumulators
    mu_shift: int = 6                 # LMS step size power-of-2 right shift (mu = 2^-6 = 0.015625)
    
    # Calibration Cycles
    cal_cycles_per_stage: int = 2000  # Number of calibration clock cycles per stage
    
    # Non-idealities & Process Mismatches (2.5V Thick-Oxide, 0.5% Gain Error)
    gain_error_std: float = 0.005     # 0.5% RMS relative linear stage gain error
    cap_mismatch_std: float = 0.0008  # 0.08% RMS capacitor mismatch
    ota_a0_db: float = 68.0           # MDAC OTA DC open-loop gain (68 dB)
    comp_offset_std: float = 0.015    # Sub-ADC comparator offset std dev (15 mV)
    
    # Random Seed for Reproducibility
    seed: Optional[int] = 42

    @property
    def datapath_frac_bits(self) -> int:
        """Fractional bits for datapath Q2.F format (2 integer bits to hold [-2, +2) range)."""
        return max(1, self.datapath_bits - 2)

    @property
    def weight_frac_bits(self) -> int:
        """Fractional bits for weight accumulator Q2.F format."""
        return max(1, self.weight_bits - 2)

    @property
    def total_bits(self) -> int:
        """Total nominal resolution = N_stages + backend_bits - 1."""
        return int(self.num_stages + self.backend_bits - 1)


# ==============================================================================
# 1. 1.5-BIT MDAC STAGE
# ==============================================================================
class MDACStage1p5b:
    """1.5-bit MDAC stage with configurable gain error, mismatch, and finite OTA gain."""
    def __init__(
        self, 
        stage_id: int, 
        Vref: float = 1.2,
        gain_error: float = 0.0,
        cap_mismatch: float = 0.0,
        ota_a0_db: float = 68.0,
        comp_offset: float = 0.0
    ):
        self.stage_id = stage_id
        self.Vref = Vref
        
        Cs_nominal = 0.6e-12
        Cf_nominal = 0.6e-12
        self.Cs = Cs_nominal * (1.0 + cap_mismatch)
        self.Cf = Cf_nominal
        
        A0 = 10 ** (ota_a0_db / 20.0)
        beta = self.Cf / (self.Cs + self.Cf)
        loop_gain_factor = (A0 * beta) / (1.0 + A0 * beta)
        
        self.nominal_gain = 1.0 + (self.Cs / self.Cf)
        self.actual_gain = self.nominal_gain * (1.0 + gain_error) * loop_gain_factor
        self.Vdac_step = (self.Cs / self.Cf) * Vref * loop_gain_factor
        
        vth_nom = Vref / 4.0
        self.vth_hi = vth_nom + comp_offset
        self.vth_lo = -vth_nom + comp_offset

    def process_sample(self, Vin: float, force_D: Optional[int] = None) -> Tuple[int, float]:
        """Processes input voltage through 1.5b MDAC. Supports forced decisions for calibration."""
        if force_D is not None:
            D = force_D
        else:
            if Vin > self.vth_hi:
                D = 1
            elif Vin < self.vth_lo:
                D = -1
            else:
                D = 0
                
        Vres = (self.actual_gain * Vin) - (D * self.Vdac_step)
        return D, Vres


# ==============================================================================
# 2. TERMINATING BACKEND QUANTIZER
# ==============================================================================
class BackendQuantizer:
    """Terminating N-bit flash quantizer after the final MDAC stage."""
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
# 3. calADC SIMULATOR CLASS (INTEGER DATAPATH WITH SAFE SHIFT ALIGNMENT)
# ==============================================================================
class calADC:
    """Pipelined ADC featuring integer or floating-point digital LMS calibration."""
    def __init__(self, cfg: CalADCConfig):
        self.cfg = cfg
        self.num_stages = cfg.num_stages
        self.num_calibrated_stages = min(cfg.num_calibrated_stages, cfg.num_stages)
        self.Vref = cfg.Vref
        self.Vcm_in = cfg.Vcm_in
        self.total_bits = cfg.total_bits

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        
        # Build homogeneous pipeline of 1.5-bit MDAC stages
        self.stages: List[MDACStage1p5b] = []
        for i in range(cfg.num_stages):
            g_err = np.random.normal(0, cfg.gain_error_std)
            c_mis = np.random.normal(0, cfg.cap_mismatch_std)
            c_off = np.random.normal(0, cfg.comp_offset_std)
            self.stages.append(
                MDACStage1p5b(
                    stage_id=i+1, Vref=cfg.Vref, gain_error=g_err,
                    cap_mismatch=c_mis, ota_a0_db=cfg.ota_a0_db, comp_offset=c_off
                )
            )
            
        self.backend = BackendQuantizer(bits=cfg.backend_bits, Vref=cfg.Vref)

        # Digital calibration weights
        if cfg.use_fixed_point:
            w_pos_init = int(round(0.5 * (1 << cfg.weight_frac_bits)))
            w_neg_init = int(round(-0.5 * (1 << cfg.weight_frac_bits)))
            self.weights = {
                i: {"pos": w_pos_init, "neg": w_neg_init} for i in range(cfg.num_stages)
            }
        else:
            self.weights = {
                i: {"pos": 0.5, "neg": -0.5} for i in range(cfg.num_stages)
            }

    @classmethod
    def from_config(cls, cfg: CalADCConfig) -> "calADC":
        """Factory constructor using CalADCConfig."""
        return cls(cfg)

    @staticmethod
    def _align_frac(val: int, src_frac: int, dst_frac: int) -> int:
        """Safely shifts integer value from Q.src_frac format to Q.dst_frac format without negative shift errors."""
        diff = src_frac - dst_frac
        if diff >= 0:
            return val >> diff
        else:
            return val << (-diff)

    def _saturate_int(self, val: int, bits: int) -> int:
        """Helper to clamp signed integer values to N-bit 2's complement range."""
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
        return int(np.clip(val, min_val, max_val))

    # --------------------------------------------------------------------------
    # DIGITAL RECONSTRUCTION ENGINE
    # --------------------------------------------------------------------------
    def reconstruct_sample(self, digital_codes: List[int], backend_val: float, use_calibrated: bool = True) -> float:
        """Back-to-front recursive digital reconstruction (Fixed-Point or Float)."""
        if self.cfg.use_fixed_point:
            Y_next = int(round(backend_val * (1 << self.cfg.datapath_frac_bits)))
            
            for k in reversed(range(self.num_stages)):
                D_k = digital_codes[k]
                Y_next = Y_next >> 1  # Division by 2 via arithmetic right shift
                
                if use_calibrated and (k < self.num_calibrated_stages):
                    w_int = self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0)
                else:
                    ideal_float = 0.5 * D_k
                    w_int = int(round(ideal_float * (1 << self.cfg.weight_frac_bits)))
                    
                w_dp = self._align_frac(w_int, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                Y_next = self._saturate_int(Y_next + w_dp, self.cfg.datapath_bits)
                
            return Y_next / (1 << self.cfg.datapath_frac_bits)

        else:
            Y_next = backend_val
            for k in reversed(range(self.num_stages)):
                D_k = digital_codes[k]
                if use_calibrated and (k < self.num_calibrated_stages):
                    w = self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0.0)
                else:
                    w = 0.5 * D_k
                Y_next = 0.5 * Y_next + w
            return Y_next

    # --------------------------------------------------------------------------
    # LMS CALIBRATION ALGORITHM (INTEGER OR FLOAT)
    # --------------------------------------------------------------------------
    def run_calibration(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Executes back-to-front LMS weight learning using integer arithmetic."""
        history = {}
        
        for stage_idx in reversed(range(self.num_calibrated_stages)):
            stage = self.stages[stage_idx]
            w_pos_history = np.zeros(self.cfg.cal_cycles_per_stage)
            w_neg_history = np.zeros(self.cfg.cal_cycles_per_stage)
            
            w_pos = self.weights[stage_idx]["pos"]
            w_neg = self.weights[stage_idx]["neg"]
            
            for cycle in range(self.cfg.cal_cycles_per_stage):
                # --------------------------------------------------------------
                # 1. Calibrate Positive Boundary (+Vref / 4)
                # --------------------------------------------------------------
                vin_pos = (self.Vref / 4.0) + np.random.normal(0, 0.005 * self.Vref)
                
                _, vres_0_pos = stage.process_sample(vin_pos, force_D=0)
                c_0_pos, b_0_pos = self._sample_downstream(stage_idx + 1, vres_0_pos)
                
                _, vres_1_pos = stage.process_sample(vin_pos, force_D=1)
                c_1_pos, b_1_pos = self._sample_downstream(stage_idx + 1, vres_1_pos)
                
                if self.cfg.use_fixed_point:
                    Y_down_0_pos = self._reconstruct_downstream_int(stage_idx + 1, c_0_pos, b_0_pos)
                    Y_down_1_pos = self._reconstruct_downstream_int(stage_idx + 1, c_1_pos, b_1_pos)
                    
                    w_pos_dp = self._align_frac(w_pos, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                    Y_k_0 = Y_down_0_pos >> 1
                    Y_k_1 = (Y_down_1_pos >> 1) + w_pos_dp
                    
                    e_pos = Y_k_0 - Y_k_1
                    e_pos_w = self._align_frac(e_pos, self.cfg.datapath_frac_bits, self.cfg.weight_frac_bits)
                    delta_w = e_pos_w >> self.cfg.mu_shift
                    w_pos = self._saturate_int(w_pos + delta_w, self.cfg.weight_bits)
                else:
                    Y_down_0_pos = self._reconstruct_downstream_float(stage_idx + 1, c_0_pos, b_0_pos)
                    Y_down_1_pos = self._reconstruct_downstream_float(stage_idx + 1, c_1_pos, b_1_pos)
                    e_pos = (0.5 * Y_down_0_pos) - (0.5 * Y_down_1_pos + w_pos)
                    w_pos += (2.0 ** (-self.cfg.mu_shift)) * e_pos

                # --------------------------------------------------------------
                # 2. Calibrate Negative Boundary (-Vref / 4)
                # --------------------------------------------------------------
                vin_neg = (-self.Vref / 4.0) + np.random.normal(0, 0.005 * self.Vref)
                
                _, vres_0_neg = stage.process_sample(vin_neg, force_D=0)
                c_0_neg, b_0_neg = self._sample_downstream(stage_idx + 1, vres_0_neg)
                
                _, vres_neg_neg = stage.process_sample(vin_neg, force_D=-1)
                c_neg_neg, b_neg_neg = self._sample_downstream(stage_idx + 1, vres_neg_neg)
                
                if self.cfg.use_fixed_point:
                    Y_down_0_neg = self._reconstruct_downstream_int(stage_idx + 1, c_0_neg, b_0_neg)
                    Y_down_neg_neg = self._reconstruct_downstream_int(stage_idx + 1, c_neg_neg, b_neg_neg)
                    
                    w_neg_dp = self._align_frac(w_neg, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
                    Y_k_0_neg = Y_down_0_neg >> 1
                    Y_k_neg_1 = (Y_down_neg_neg >> 1) + w_neg_dp
                    
                    e_neg = Y_k_0_neg - Y_k_neg_1
                    e_neg_w = self._align_frac(e_neg, self.cfg.datapath_frac_bits, self.cfg.weight_frac_bits)
                    delta_w_neg = e_neg_w >> self.cfg.mu_shift
                    w_neg = self._saturate_int(w_neg + delta_w_neg, self.cfg.weight_bits)
                else:
                    Y_down_0_neg = self._reconstruct_downstream_float(stage_idx + 1, c_0_neg, b_0_neg)
                    Y_down_neg_neg = self._reconstruct_downstream_float(stage_idx + 1, c_neg_neg, b_neg_neg)
                    e_neg = (0.5 * Y_down_0_neg) - (0.5 * Y_down_neg_neg + w_neg)
                    w_neg += (2.0 ** (-self.cfg.mu_shift)) * e_neg

                # Log history normalized to float scale (~0.5)
                if self.cfg.use_fixed_point:
                    w_pos_history[cycle] = w_pos / (1 << self.cfg.weight_frac_bits)
                    w_neg_history[cycle] = w_neg / (1 << self.cfg.weight_frac_bits)
                else:
                    w_pos_history[cycle] = w_pos
                    w_neg_history[cycle] = w_neg
                
            self.weights[stage_idx]["pos"] = w_pos
            self.weights[stage_idx]["neg"] = w_neg
            
            history[stage_idx + 1] = {
                "w_pos": w_pos_history,
                "w_neg": w_neg_history
            }
            
        return history

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
            if k < self.num_calibrated_stages:
                w_int = self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0)
            else:
                ideal_float = 0.5 * D_k
                w_int = int(round(ideal_float * (1 << self.cfg.weight_frac_bits)))
                
            w_dp = self._align_frac(w_int, self.cfg.weight_frac_bits, self.cfg.datapath_frac_bits)
            Y_next = self._saturate_int(Y_next + w_dp, self.cfg.datapath_bits)
        return Y_next

    def _reconstruct_downstream_float(self, start_idx: int, codes: List[int], backend_val: float) -> float:
        Y_next = backend_val
        code_idx = len(codes) - 1
        for k in reversed(range(start_idx, self.num_stages)):
            D_k = codes[code_idx]
            code_idx -= 1
            if k < self.num_calibrated_stages:
                w = self.weights[k]["pos"] if D_k == 1 else (self.weights[k]["neg"] if D_k == -1 else 0.0)
            else:
                w = 0.5 * D_k
            Y_next = 0.5 * Y_next + w
        return Y_next

    # --------------------------------------------------------------------------
    # SIMULATION EVALUATION HELPERS
    # --------------------------------------------------------------------------
    def run_transient_sine(self, num_samples=2048, M_bin=31, use_calibrated=True) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        t = np.arange(num_samples)
        f_in = M_bin / num_samples
        vin_sine = 0.96 * self.Vref * np.sin(2 * np.pi * f_in * t)
        
        y_recon = np.zeros(num_samples)
        for n in range(num_samples):
            codes, backend_val = self._sample_downstream(0, vin_sine[n])
            y_recon[n] = self.reconstruct_sample(codes, backend_val, use_calibrated=use_calibrated)
            
        fft_spec = np.abs(np.fft.rfft(y_recon)) / (num_samples / 2.0)
        fft_spec[0] = 0.0
        signal_pwr = fft_spec[M_bin] ** 2
        
        noise_spec = fft_spec.copy()
        noise_spec[M_bin] = 0.0
        
        sfdr_db = 20 * np.log10(fft_spec[M_bin] / (np.max(noise_spec[1:]) + 1e-12))
        sndr_db = 10 * np.log10(signal_pwr / (np.sum(noise_spec ** 2) + 1e-12))
        enob = (sndr_db - 1.76) / 6.02
        
        return sndr_db, sfdr_db, enob, 20 * np.log10(fft_spec + 1e-12), y_recon

    def run_ramp_dnl_inl(self, num_samples=None, use_calibrated=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_samples is None:
            num_samples = 10 * (2 ** self.total_bits)

        vin_ramp = np.linspace(-1.02 * self.Vref, 1.02 * self.Vref, num_samples)
        y_recon = np.zeros(num_samples)
        for n in range(num_samples):
            codes, backend_val = self._sample_downstream(0, vin_ramp[n])
            y_recon[n] = self.reconstruct_sample(codes, backend_val, use_calibrated=use_calibrated)
            
        num_codes = 2 ** self.total_bits
        v_norm = (y_recon + 1.0) / 2.0
        codes = np.clip(np.floor(v_norm * num_codes).astype(int), 0, num_codes - 1)
        
        H = np.bincount(codes, minlength=num_codes)
        valid_idx = np.where(H > 0)[0]
        f_code, l_code = valid_idx[4], valid_idx[-5]
        
        H_valid = H[f_code:l_code+1]
        ideal_hits = np.mean(H_valid)
        
        dnl = (H_valid / ideal_hits) - 1.0
        inl_raw = np.cumsum(dnl)
        inl = inl_raw - np.linspace(inl_raw[0], inl_raw[-1], len(inl_raw))
        
        return np.arange(f_code, l_code + 1), dnl, inl


# ==============================================================================
# 4. DESIGN SPACE EXPLORATION STUDY (SYSTEM + HARDWARE SWEEPS)
# ==============================================================================
def run_design_space_exploration():
    print("="*75)
    print(" calADC DESIGN SPACE EXPLORATION STUDY (INTEGER ARITHMETIC)")
    print(" 16-Stage All-1.5b MDAC Pipeline Architecture (17-Bit Nominal)")
    print("="*75)

    # Base Integer Configuration (20-bit Datapath, 22-bit Weight Accumulators)
    base_cfg = CalADCConfig(
        num_stages=16,               # 16 total 1.5b MDAC stages
        num_calibrated_stages=5,     # Calibrate stages 1 through 5
        backend_bits=2,              # 2-bit flash backend (17 bits total nominal)
        Vdd=2.5,                     # 2.5V supply
        Vref=1.2,                    # 1.2V reference (2.4 Vpp diff)
        Vcm_in=1.25,                 # Mid-supply common mode
        
        # Fixed-Point Integer Settings
        use_fixed_point=True,
        datapath_bits=20,            # 20-bit datapath (Q2.18)
        weight_bits=22,              # 22-bit weight accumulator (Q2.20)
        mu_shift=6,                  # mu = 2^-6 (shift right by 6)
        
        # Physical Mismatches
        gain_error_std=0.005,        # 0.5% RMS stage gain error
        cap_mismatch_std=0.0008,     # 0.08% capacitor mismatch
        ota_a0_db=68.0,              # 68 dB OTA open-loop gain
        cal_cycles_per_stage=2000,   # Calibration cycles
        seed=42
    )

    # --------------------------------------------------------------------------
    # 1. BASELINE EVALUATION
    # --------------------------------------------------------------------------
    adc_base = calADC.from_config(base_cfg)
    sndr_uncal, sfdr_uncal, enob_uncal, spec_uncal, _ = adc_base.run_transient_sine(use_calibrated=False)
    code_axis_u, dnl_u, inl_u = adc_base.run_ramp_dnl_inl(use_calibrated=False)

    history = adc_base.run_calibration()
    
    sndr_cal, sfdr_cal, enob_cal, spec_cal, _ = adc_base.run_transient_sine(use_calibrated=True)
    code_axis_c, dnl_cal, inl_cal = adc_base.run_ramp_dnl_inl(use_calibrated=True)

    print(f"\n--- UNCALIBRATED METRICS (N={base_cfg.total_bits} Nominal Bits) ---")
    print(f"  Dynamic SNDR : {sndr_uncal:.2f} dB (ENOB = {enob_uncal:.2f} bits)")
    print(f"  Dynamic SFDR : {sfdr_uncal:.2f} dB")
    print(f"  Max |DNL|    : {np.max(np.abs(dnl_u)):.2f} LSB")
    print(f"  Max |INL|    : {np.max(np.abs(inl_u)):.2f} LSB")

    print(f"\n--- CALIBRATED INTEGER METRICS ({base_cfg.datapath_bits}-Bit Datapath, {base_cfg.weight_bits}-Bit Weights) ---")
    print(f"  Dynamic SNDR : {sndr_cal:.2f} dB (ENOB = {enob_cal:.2f} bits)  [+{sndr_cal - sndr_uncal:.2f} dB]")
    print(f"  Dynamic SFDR : {sfdr_cal:.2f} dB  [+{sfdr_cal - sfdr_uncal:.2f} dB]")
    print(f"  Max |DNL|    : {np.max(np.abs(dnl_cal)):.2f} LSB")
    print(f"  Max |INL|    : {np.max(np.abs(inl_cal)):.2f} LSB")

    # --------------------------------------------------------------------------
    # 2. SYSTEM SWEEPS (FIXED BITWIDTH)
    # --------------------------------------------------------------------------
    print("\nExecuting System Sweep 1: Number of Calibrated Stages Sweep...")
    cal_stage_range = list(range(0, 11))
    sndr_vs_stages = []
    for num_c in cal_stage_range:
        cfg_sweep = replace(base_cfg, num_calibrated_stages=num_c)
        adc_sweep = calADC.from_config(cfg_sweep)
        if num_c > 0:
            adc_sweep.run_calibration()
        s_val, _, _, _, _ = adc_sweep.run_transient_sine(use_calibrated=(num_c > 0))
        sndr_vs_stages.append(s_val)

    print("Executing System Sweep 2: Calibration Loop Cycles Sweep...")
    cycles_range = [200, 500, 1000, 1500, 2500, 4000]
    sndr_vs_cycles = []
    for cyc in cycles_range:
        cfg_sweep = replace(base_cfg, cal_cycles_per_stage=cyc)
        adc_sweep = calADC.from_config(cfg_sweep)
        adc_sweep.run_calibration()
        s_val, _, _, _, _ = adc_sweep.run_transient_sine(use_calibrated=True)
        sndr_vs_cycles.append(s_val)

    # --------------------------------------------------------------------------
    # 3. HARDWARE WORDLENGTH SWEEPS
    # --------------------------------------------------------------------------
    print("\nExecuting Hardware Sweep 1: Unified Register Width (B_reg) Sweep...")
    reg_bits_range = list(range(12, 25, 2))
    sndr_vs_reg_bits = []
    for b_reg in reg_bits_range:
        cfg_sweep = replace(base_cfg, datapath_bits=b_reg, weight_bits=b_reg + 2)
        adc_sweep = calADC.from_config(cfg_sweep)
        adc_sweep.run_calibration()
        s_val, _, _, _, _ = adc_sweep.run_transient_sine(use_calibrated=True)
        sndr_vs_reg_bits.append(s_val)

    print("Executing Hardware Sweep 2: LMS Step-Size Shift (mu_shift) Sweep...")
    mu_shifts_range = list(range(3, 11))
    sndr_vs_mu_shift = []
    for s_mu in mu_shifts_range:
        cfg_sweep = replace(base_cfg, mu_shift=s_mu)
        adc_sweep = calADC.from_config(cfg_sweep)
        adc_sweep.run_calibration()
        s_val, _, _, _, _ = adc_sweep.run_transient_sine(use_calibrated=True)
        sndr_vs_mu_shift.append(s_val)

    # ==========================================================================
    # FIGURE 1: SYSTEM & ALGORITHM CALIBRATION DASHBOARD
    # ==========================================================================
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9.5))

    # Panel 1: LMS Weight Convergence Traces
    for stg in sorted(history.keys()):
        ax1.plot(history[stg]["w_pos"], label=f'Stage {stg} W+')
    ax1.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Ideal RSD (0.5)')
    ax1.set_xlabel("Calibration Iterations / Cycles", fontsize=10)
    ax1.set_ylabel("Learned Weight W+ (Normalized)", fontsize=10)
    ax1.set_title(f"1. Integer LMS Weight Convergence ({base_cfg.weight_bits}-Bit Accumulators)", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 2: Dynamic Spectrum Comparison (DC Bins Omitted)
    start_bin = 3
    freqs = np.linspace(0, 0.5, len(spec_uncal))[start_bin:]
    spec_uncal_clean = spec_uncal[start_bin:]
    spec_cal_clean = spec_cal[start_bin:]

    ax2.plot(freqs, spec_uncal_clean, color='tab:red', alpha=0.7, lw=1.2, label=f'Uncalibrated ({sndr_uncal:.1f} dB)')
    ax2.plot(freqs, spec_cal_clean, color='tab:blue', lw=1.5, label=f'Calibrated ({sndr_cal:.1f} dB)')
    ax2.set_xlabel("Normalized Frequency (f / f_s)", fontsize=10)
    ax2.set_ylabel("Power Spectral Density (dBFS)", fontsize=10)
    ax2.set_title(f"2. Dynamic Spectrum ({base_cfg.datapath_bits}-Bit Datapath)", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    # Panel 3: Performance vs. Number of Calibrated Stages
    ax3.plot(cal_stage_range, sndr_vs_stages, 'o-', color='tab:purple', lw=2)
    ax3.set_xlabel("Number of Calibrated Stages (Front-to-Back)", fontsize=10)
    ax3.set_ylabel("Calibrated Dynamic SNDR (dB)", fontsize=10)
    ax3.set_title("3. Performance vs. Calibrated Stage Count", fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Panel 4: Performance vs. Calibration Cycles per Stage
    ax4.plot(cycles_range, sndr_vs_cycles, 's-', color='tab:green', lw=2)
    ax4.set_xlabel("Calibration Clock Cycles per Stage", fontsize=10)
    ax4.set_ylabel("Calibrated Dynamic SNDR (dB)", fontsize=10)
    ax4.set_title("4. Calibration Quality vs. Loop Cycles", fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)

    fig1.suptitle(f"calADC Digital Calibration System Dashboard ({base_cfg.total_bits}-Bit Nominal Pipeline)", fontsize=13, fontweight='bold')
    fig1.tight_layout()
    plt.show()

    # ==========================================================================
    # FIGURE 2: HARDWARE WORDLENGTH DASHBOARD
    # ==========================================================================
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: SNDR vs Datapath Register Wordlength (B_reg)
    ax2_1.plot(reg_bits_range, sndr_vs_reg_bits, 'o-', color='indigo', lw=2)
    ax2_1.set_xlabel("Hardware Register Bitwidth B_reg (Bits)", fontsize=10)
    ax2_1.set_ylabel("Calibrated Dynamic SNDR (dB)", fontsize=10)
    ax2_1.set_title("1. Performance vs. Hardware Register Wordlength", fontweight='bold')
    ax2_1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: SNDR vs LMS Hardware Shift (mu_shift = 2^-S)
    ax2_2.plot(mu_shifts_range, sndr_vs_mu_shift, 's-', color='tab:olive', lw=2)
    ax2_2.set_xlabel("LMS Shift Parameter S (mu = 2^-S)", fontsize=10)
    ax2_2.set_ylabel("Calibrated Dynamic SNDR (dB)", fontsize=10)
    ax2_2.set_title("2. Performance vs. Hardware Step Size (mu_shift)", fontweight='bold')
    ax2_2.grid(True, linestyle=':', alpha=0.6)

    fig2.suptitle(f"calADC Integer Hardware Wordlength Analysis ({base_cfg.total_bits}-Bit Nominal Pipeline)", fontsize=13, fontweight='bold')
    fig2.tight_layout()
    plt.show()

    # ==========================================================================
    # FIGURE 3: STATIC NON-LINEARITY ANALYSIS (SEPARATE UNCAL & CAL PLOTS)
    # ==========================================================================
    fig3, ((ax3_1, ax3_2), (ax3_3, ax3_4)) = plt.subplots(2, 2, figsize=(13, 9.5))

    # Subplot 1: Uncalibrated DNL
    ax3_1.plot(code_axis_u, dnl_u, color='tab:red', lw=0.8)
    ax3_1.axhline(0, color='black', lw=0.8, linestyle='--')
    ax3_1.set_xlabel("Digital Output Code", fontsize=10)
    ax3_1.set_ylabel("DNL (LSB)", fontsize=10)
    ax3_1.set_title(f"1. Uncalibrated DNL (Max |DNL| = {np.max(np.abs(dnl_u)):.2f} LSB)", fontweight='bold')
    ax3_1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Calibrated DNL (Rescaled to show sub-LSB details)
    ax3_2.plot(code_axis_c, dnl_cal, color='tab:blue', lw=1.0)
    ax3_2.axhline(0, color='black', lw=0.8, linestyle='--')
    ax3_2.axhline(0.5, color='crimson', linestyle=':', lw=1.0, label='±0.5 LSB Target')
    ax3_2.axhline(-0.5, color='crimson', linestyle=':', lw=1.0)
    ax3_2.set_xlabel("Digital Output Code", fontsize=10)
    ax3_2.set_ylabel("DNL (LSB)", fontsize=10)
    ax3_2.set_title(f"2. Calibrated DNL (Max |DNL| = {np.max(np.abs(dnl_cal)):.2f} LSB)", fontweight='bold')
    ax3_2.grid(True, linestyle=':', alpha=0.6)
    ax3_2.legend(loc='upper right', fontsize=8)

    # Subplot 3: Uncalibrated INL
    ax3_3.plot(code_axis_u, inl_u, color='tab:red', lw=1.0)
    ax3_3.axhline(0, color='black', lw=0.8, linestyle='--')
    ax3_3.set_xlabel("Digital Output Code", fontsize=10)
    ax3_3.set_ylabel("INL (LSB)", fontsize=10)
    ax3_3.set_title(f"3. Uncalibrated INL (Max |INL| = {np.max(np.abs(inl_u)):.2f} LSB)", fontweight='bold')
    ax3_3.grid(True, linestyle=':', alpha=0.6)

    # Subplot 4: Calibrated INL (Rescaled to show sub-LSB details)
    ax3_4.plot(code_axis_c, inl_cal, color='tab:blue', lw=1.2)
    ax3_4.axhline(0, color='black', lw=0.8, linestyle='--')
    ax3_4.set_xlabel("Digital Output Code", fontsize=10)
    ax3_4.set_ylabel("INL (LSB)", fontsize=10)
    ax3_4.set_title(f"4. Calibrated INL (Max |INL| = {np.max(np.abs(inl_cal)):.2f} LSB)", fontweight='bold')
    ax3_4.grid(True, linestyle=':', alpha=0.6)

    fig3.suptitle(f"calADC Static Linearity Study — Pre- vs. Post-Calibration ({base_cfg.total_bits}-Bit Nominal)", fontsize=13, fontweight='bold')
    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_design_space_exploration()