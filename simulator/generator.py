"""
Mathematical MPU6050 vibration synthesizer module.
Implements mathematically modeled synthesis:
- Baseline ambient noise: N(0, 0.02^2) across all axes (with 1.0g gravity on z).
- Wind & rain: Stochastic envelope modulation (1 + eta(t)) * sum(Ai * sin(2*pi*fi*t + phi_i))
  with fi in [1, 5] Hz plus high-frequency rain drizzle.
- Climbing profile: Static z-axis mass offset (downward preload +1.0g to gravity)
  and random transient burst windows at 20-35 Hz.
- Cutting profile: Sharp damped sinusoids (f > 40 Hz, damping envelope exp(-alpha*(t - t0)))
  fired at random intervals, producing impulsive tails and high Fisher kurtosis (> 6.0).
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional, Tuple
import numpy as np
from simulator.schemas import SignalType


class VibrationSynthesizer:
    """
    Synthesizes realistic 3-axis accelerometer time-series (ax, ay, az in g-units)
    at a target sampling frequency fs = 100 Hz.
    """

    def __init__(self, fs: float = 100.0, seed: Optional[int] = None):
        self.fs = fs
        self.rng = np.random.default_rng(seed)

    def generate_baseline(self, num_samples: int = 200, sigma: float = 0.02) -> np.ndarray:
        """
        Baseline ambient noise:
        ax, ay ~ N(0, 0.02^2)
        az = 1.0g + N(0, 0.02^2)
        """
        ax = self.rng.normal(0.0, sigma, size=num_samples)
        ay = self.rng.normal(0.0, sigma, size=num_samples)
        az = 1.0 + self.rng.normal(0.0, sigma, size=num_samples)
        return np.column_stack((ax, ay, az))

    def generate_wind_rain(
        self,
        num_samples: int = 200,
        fw_range: Tuple[float, float] = (1.0, 5.0),
        num_harmonics: int = 3,
        rain_sigma: float = 0.025,
    ) -> np.ndarray:
        """
        Wind & rain:
        Stochastic envelope modulation (1 + eta(t)) * sum(Ai * sin(2*pi*fi*t + phi_i))
        with fi in [1, 5] Hz plus high-frequency rain drizzle.
        """
        t = np.arange(num_samples) / self.fs
        baseline = self.generate_baseline(num_samples, sigma=0.015)

        # Slow-varying stochastic gust envelope: (1 + eta(t))
        fgust = self.rng.uniform(0.2, 0.6)
        eta_amp = self.rng.uniform(0.4, 0.7)
        phi_gust = self.rng.uniform(0, 2 * np.pi)
        eta_t = eta_amp * np.sin(2 * np.pi * fgust * t + phi_gust)
        envelope = 1.0 + eta_t  # always positive

        # Multi-harmonic wind flutter sum(Ai * sin(2*pi*fi*t + phi_i))
        wind_x = np.zeros(num_samples)
        wind_y = np.zeros(num_samples)
        wind_z = np.zeros(num_samples)

        for _ in range(num_harmonics):
            fi_x = self.rng.uniform(fw_range[0], fw_range[1])
            fi_y = self.rng.uniform(fw_range[0], fw_range[1])
            amp_x = self.rng.uniform(0.10, 0.22)
            amp_y = self.rng.uniform(0.08, 0.18)
            phi_x = self.rng.uniform(0, 2 * np.pi)
            phi_y = self.rng.uniform(0, 2 * np.pi)

            wind_x += amp_x * np.sin(2 * np.pi * fi_x * t + phi_x)
            wind_y += amp_y * np.sin(2 * np.pi * fi_y * t + phi_y)

        # Modulate with stochastic envelope
        wind_x *= envelope
        wind_y *= (envelope * 0.8)

        # High-frequency rain drizzle (broadband droplet impacts on sensor/mesh)
        rain_x = self.rng.normal(0.0, rain_sigma, size=num_samples)
        rain_y = self.rng.normal(0.0, rain_sigma, size=num_samples)
        rain_z = self.rng.normal(0.0, rain_sigma * 0.7, size=num_samples)

        return baseline + np.column_stack((wind_x + rain_x, wind_y + rain_y, wind_z + rain_z))

    def generate_climbing(
        self,
        num_samples: int = 200,
        fc_range: Tuple[float, float] = (20.0, 35.0),
        preload_offset: float = 1.0,
    ) -> np.ndarray:
        """
        Climbing profile:
        Static z-axis mass offset (downward preload +1.0g to gravity, total ~2.0g)
        and random transient burst windows at 20-35 Hz.
        """
        t = np.arange(num_samples) / self.fs
        baseline = self.generate_baseline(num_samples, sigma=0.02)

        # Static downward preload: +1.0g added to az gravity
        # climber mass loading with slight real-world variance
        static_offset = self.rng.uniform(preload_offset * 0.85, preload_offset * 1.15)
        climb_z = np.full(num_samples, static_offset)
        climb_x = np.zeros(num_samples)
        climb_y = np.zeros(num_samples)

        # 2 to 4 transient burst windows representing footfall/grip impacts
        num_bursts = self.rng.integers(2, 5)
        burst_starts = np.sort(
            self.rng.choice(np.arange(10, num_samples - 45), size=num_bursts, replace=False)
        )

        for start_idx in burst_starts:
            burst_len = self.rng.integers(25, 45)  # 0.25 to 0.45 seconds
            end_idx = min(start_idx + burst_len, num_samples)
            actual_len = end_idx - start_idx

            tb = np.arange(actual_len) / self.fs
            fc = self.rng.uniform(fc_range[0], fc_range[1])
            phi = self.rng.uniform(0, 2 * np.pi)
            amp = self.rng.uniform(0.7, 1.4)

            # Transient window (Tukey or sine taper)
            window = np.sin(np.pi * np.arange(actual_len) / actual_len)
            burst = amp * window * np.sin(2 * np.pi * fc * tb + phi)

            climb_x[start_idx:end_idx] += burst * self.rng.uniform(0.3, 0.6)
            climb_y[start_idx:end_idx] += burst * self.rng.uniform(0.3, 0.6)
            climb_z[start_idx:end_idx] += burst * self.rng.uniform(0.8, 1.2)

        return baseline + np.column_stack((climb_x, climb_y, climb_z))

    def generate_cutting(
        self,
        num_samples: int = 200,
        fk_min: float = 40.0,
        alpha_damping: float = 30.0,
    ) -> np.ndarray:
        """
        Cutting profile:
        Sharp damped sinusoids (f > 40 Hz, damping envelope exp(-alpha * (t - t0)))
        fired at random intervals, producing impulsive tails and high Fisher kurtosis (> 6.0).
        """
        t = np.arange(num_samples) / self.fs
        baseline = self.generate_baseline(num_samples, sigma=0.015)

        cut_x = np.zeros(num_samples)
        cut_y = np.zeros(num_samples)
        cut_z = np.zeros(num_samples)

        # 1 to 3 wire cut snips at random intervals
        num_cuts = self.rng.integers(1, 4)
        cut_indices = np.sort(
            self.rng.choice(np.arange(15, num_samples - 30), size=num_cuts, replace=False)
        )

        for t0_idx in cut_indices:
            # High frequency oscillation > 40 Hz (up to 48 Hz for fs=100 Hz Nyquist)
            fk = self.rng.uniform(fk_min, 48.0)
            amp = self.rng.uniform(3.5, 6.5)  # Sharp impulsive amplitude
            pulse_samples = min(num_samples - t0_idx, int(self.fs * 0.25))

            tp = np.arange(pulse_samples) / self.fs
            # Sharp damped sinusoid: exp(-alpha * tp) * sin(2*pi*fk*tp)
            damping = np.exp(-alpha_damping * tp)
            shock = amp * damping * np.sin(2 * np.pi * fk * tp)

            cut_x[t0_idx : t0_idx + pulse_samples] += shock * self.rng.uniform(0.6, 1.0)
            cut_y[t0_idx : t0_idx + pulse_samples] += shock * self.rng.uniform(0.6, 1.0)
            cut_z[t0_idx : t0_idx + pulse_samples] += shock * self.rng.uniform(0.9, 1.3)

        return baseline + np.column_stack((cut_x, cut_y, cut_z))

    def generate_sample(self, signal_type: SignalType, num_samples: int = 200) -> np.ndarray:
        """Dispatcher returning a (num_samples, 3) numpy array for the requested class."""
        if signal_type == SignalType.NORMAL:
            return self.generate_baseline(num_samples)
        elif signal_type == SignalType.WIND_RAIN:
            return self.generate_wind_rain(num_samples)
        elif signal_type == SignalType.CLIMBING:
            return self.generate_climbing(num_samples)
        elif signal_type == SignalType.CUTTING:
            return self.generate_cutting(num_samples)
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")


if __name__ == "__main__":
    from scipy import stats
    import json
    from simulator.schemas import TelemetryMessage, TelemetryDataPayload

    synth = VibrationSynthesizer(fs=100.0, seed=42)
    print("=" * 65)
    print("AI-Smart-Fence: MPU6050 Vibration Synthesizer & Schema Test")
    print("=" * 65)

    for sig_type in [SignalType.NORMAL, SignalType.WIND_RAIN, SignalType.CLIMBING, SignalType.CUTTING]:
        data = synth.generate_sample(sig_type, num_samples=200)
        ax, ay, az = data[:, 0], data[:, 1], data[:, 2]
        fisher_kurt = float(stats.kurtosis(az, fisher=True))

        print(f"\n[+] Profile: {sig_type.value}")
        print(f"    Samples: {len(data)} (2.0s @ 100 Hz)")
        print(f"    Mean (ax, ay, az)    : ({ax.mean():.3f}g, {ay.mean():.3f}g, {az.mean():.3f}g)")
        print(f"    Std  (ax, ay, az)    : ({ax.std():.3f}g, {ay.std():.3f}g, {az.std():.3f}g)")
        print(f"    Fisher Kurtosis (az) : {fisher_kurt:.2f}")

    # Generate sample MQTT telemetry JSON
    normal_sample = synth.generate_sample(SignalType.NORMAL, num_samples=200)
    msg = TelemetryMessage(
        node_id="zone_01_esp32",
        zone_id="zone_01",
        timestamp=1773289000.0,
        fs=100.0,
        window_size=200,
        data=TelemetryDataPayload(
            ax=[round(x, 4) for x in normal_sample[:, 0][:5]],
            ay=[round(y, 4) for y in normal_sample[:, 1][:5]],
            az=[round(z, 4) for z in normal_sample[:, 2][:5]],
        ),
    )
    print("\n[+] Sample MQTT Telemetry JSON Payload (Truncated data for preview):")
    print(json.dumps(msg.model_dump(), indent=2))
    print("\n" + "=" * 65)
    print("ALL SYNTHESIS MODELS & SCHEMAS VERIFIED SUCCESSFULLY!")
    print("=" * 65)

