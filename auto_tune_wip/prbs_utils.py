# Generate Pseudo Random binary sequence of +- amplitude
import numpy as np

_PRBS_TAPS = {
    2:  [2, 1],
    3:  [3, 2],
    4:  [4, 3],
    5:  [5, 3],
    6:  [6, 5],
    7:  [7, 6],
    8:  [8, 6, 5, 4],
    9:  [9, 5],
    10: [10, 7],
    11: [11, 9],
    12: [12, 11, 10, 4],
    13: [13, 12, 11, 8],
    14: [14, 13, 12, 2],
    15: [15, 14],
    16: [16, 14, 13, 11],
}

def _lfsr_bits(order, n_bits, seed):
    """Left-shift Fibonacci LFSR, feedback into LSB (matches tap table above)."""
    taps = _PRBS_TAPS[order]
    state = seed & ((1 << order) - 1)
    if state == 0:
        state = 1

    bits = np.empty(n_bits, dtype=np.uint8)

    for i in range(n_bits):
        # output MSB as the sequence bit (bool -> {0,1})
        bits[i] = (state >> (order - 1)) & 1

        # feedback is XOR of tap positions (1-indexed)
        fb = 0
        for tp in taps:
            fb ^= (state >> (tp - 1)) & 1

        # shift LEFT, insert feedback into LSB
        state = ((state << 1) & ((1 << order) - 1)) | fb

    return bits

def prbs_waveform(
    *,
    fs=1000.0,
    samples_per_bit=4,
    order=12,
    periods=3,
    amplitude=1.0,
    bias=0.0,
    seed=1,
    as_packet=False
):
    if order not in _PRBS_TAPS:
        raise ValueError(f"Unsupported order={order}. Supported: {sorted(_PRBS_TAPS.keys())}")
    if periods < 1 or int(periods) != periods:
        raise ValueError("periods must be a positive integer.")
    if samples_per_bit < 1 or int(samples_per_bit) != samples_per_bit:
        raise ValueError("samples_per_bit must be a positive integer.")
    samples_per_bit = int(samples_per_bit)

    n_bits = (1 << order) - 1
    f_bit = fs / samples_per_bit

    bits = _lfsr_bits(order, n_bits, seed)

    # Map {0,1} -> {-A,+A} and add bias
    levels = (bits.astype(np.int8) * 2 - 1).astype(np.float64) * float(amplitude) + float(bias)

    # Repeat full periods and NRZ-hold
    levels = np.tile(levels, int(periods))
    u = np.repeat(levels, samples_per_bit).astype(np.float64)

    meta = {
        "fs": float(fs),
        "samples_per_bit": int(samples_per_bit),
        "f_bit": float(f_bit),
        "order": int(order),
        "periods": int(periods),
        "n_bits": int(n_bits),
        "n_samples": int(u.size),
        "duration_s": float(u.size / fs),
        "amplitude": float(amplitude),
        "bias": float(bias),
        "seed": int(seed),
        "fmin_est": float(f_bit / n_bits),
        "fmax_est": float(f_bit / 2.0),
    }

    if as_packet:
        return {"fs": float(fs), "cmd": u}, meta
    return u, meta
