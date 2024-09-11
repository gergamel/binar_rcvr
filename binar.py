# from gfsk import GFSK
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from math import gcd, ceil
from typing import Optional, List, Generator, Literal, Any
import datetime as dt

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.signal import firwin, convolve, decimate, iirdesign, freqz, filtfilt, lfilter
from scipy.special import erf

DEF_SYMBOLRATE = float(9604.5)  # Default GMSK symbol (bit) rate
DEF_FRAME_BYTES = int(214)  # Budgetary frame length in bytes
GFSK_BT = 0.5  # Bandwidth-Time product of our GFSK gaussian filter

DEF_GS_SAMPLERATE = float(400e3)  # Default ground station IQ samplerate
DEF_GS_CENTRE_F = float(
    437.850e6
)  # Default centre frequency of Ground Station IQ captures

DEF_CH_BW = float(12000)  # Channelisation bandwidth (FIR filter cutoff is 1/2 this)
DEF_CH_SAMPLERATE = float(
    100e3
)  # Samplerate of channelised data (decimated by 4, which is ~10x oversampling)
DEF_CH_FIR_TAPS = 1024  # Number of taps in channelisation LPF

DEF_CH_SAMPLERATE = float(100e3)

# Number of samples per GFSK symbol (bit)
SYM_LEN_SAMPLES = int(ceil(DEF_GS_SAMPLERATE / DEF_SYMBOLRATE))

# Rough number of symbols (bits) per packet
FRAME_LEN_SYMS = 8 * DEF_FRAME_BYTES

# Rough number of samples per packet. We might use this for channel power estimation filter
FRAME_LEN_SAMPLES = FRAME_LEN_SYMS * SYM_LEN_SAMPLES

class FloatString():
    def __init__(self, value:Any):
        if type(value)==str:
            self._val = float(value.replace("k", "e3").replace("M", "e6").replace("G", "e9"))
        else:
            self._val = float(value)
    @property
    def value(self)->float:
        return self._val
    def __str__(self)->str:
        value = self._val
        suffix = ''
        if value>1e6:
            suffix="M"
            value/=1e6
        elif value>1e3:
            suffix="k"
            value/=1e3
        return f"{value}{suffix}"
    def __repr__(self)->str:
        return str(self)

class SampleType():
    MAP = (
        ("f32", np.float32),
        ("f64", np.float64),
        ("c64", np.complex64),
        ("c128",np.complex128)
    )
    STR_MAP = dict(MAP)
    REV_MAP = {k:v for v,k in MAP}
    def __init__(self, value:Any):
        if value in self.REV_MAP.keys():
            self._v = value
        elif type(value)==str:
            self._v = self.STR_MAP[value]
        else:
            raise ValueError(f'Unrecognised SampleType "{value}"')
    @property
    def value(self)->np.dtype:
        return self._v
    @property
    def nbytes(self)->int:
        return self._v().nbytes
    def __str__(self)->str:
        return self.REV_MAP[self._v]
    def __repr__(self)->str:
        return str(self)

SIG_FILE_RE = re.compile(r"^(\d{14})_?([\dMk\.]*)_?([cfsu\d]*)_?([\dMk\.Hz]*)")
# See this regex in action: https://pythex.org/?regex=%5E(%5Cd%7B14%7D)_%3F(%5B%5CdMk%5C.%5D*)_%3F(%5Bcfsu%5Cd%5D*)_%3F(%5B%5CdMk%5C.Hz%5D*)&test_string=20240829195734_400k%0A20240829195734_400k_c64%0A20240902114717_192k_c64_437700000%0A20240902114717_192k_c64_437.7MHz&ignorecase=0&multiline=1&dotall=0&verbose=0

@dataclass
class SignalFile:
    path: Path
    name: str
    samplerate: float
    fcentre: float
    timestamp: dt.datetime
    dtype: SampleType
    verbose: bool = False

    def __str__(self) -> str:
        return f"SignalFile({self.name}, samplerate={self.samplerate}Hz, timestamp={self.datecode}, dtype={self.dtype} fcentre={self.fcentre:.0f}Hz)"

    @property
    def dtype_size(self) -> int:
        return self.dtype.nbytes

    @property
    def datecode(self) -> str:
        return self.timestamp.strftime("%Y%m%d%H%M%S")

    @property
    def datapath(self) -> Path:
        return self.path.parent / self.name

    @property
    def length(self) -> int:
        return int(self.path.stat().st_size / self.dtype_size)

    def load(self, offset: float = -1.0, length: float = -1.0) -> npt.NDArray:
        if offset == -1.0 and length == -1.0:
            print(f"Reading {self}")
            return np.fromfile(self.path, dtype=self.dtype.value)

        # Default limited load is 0.0s to 0.5s
        sample_offset = int(0)
        sample_count = int(0.5 * self.samplerate)
        if offset > 0.0:
            sample_offset = int(offset * self.samplerate)
        if length > 0.0:
            sample_count = int(length * self.samplerate)
        if self.verbose:
            print(
                f"Reading {self}: {sample_count} samples from {sample_offset}"
            )
        return np.fromfile(
            self.path, dtype=self.dtype.value, offset=sample_offset * self.dtype.nbytes, count=sample_count
        )

    @classmethod
    def frompath(cls, path:Path, prompt_implicit:bool=False, verbose:bool=False) -> "SignalFile":
        assert path.exists()
        name_parts = SIG_FILE_RE.findall(path.stem)
        if name_parts is None:
            raise ValueError(
                "Expecting a file named {{datecode}}_{{samplerate}}.iq or {{datecode}}_{{samplerate}}_{{dtype}}.iq"
            )
        
        datecode = name_parts[0][0]

        rename:bool = False

        # Default to 192kHz samplerate if missing
        if name_parts[0][1]!="":
            samplerate = FloatString(name_parts[0][1].replace("k", "e3").replace("M", "e6")).value
        elif prompt_implicit:
            samplerate = 192e3
            val = input(f"Filename does not specify samplerate. What is the samplerate (Hz)? [{samplerate:.0f}]: ")
            if val!='':
                samplerate = FloatString(val).value
            rename = True
    
        # Default to complex64 for files that are missing a dtype
        if name_parts[0][2]!="":
            dtype = SampleType(name_parts[0][2])
        else:
            dtype = SampleType("c64")
            val = input(f"""Filename does not specify datatype. What is the datataype:
 c64 =  complex64 (float32 I, float32 Q)
c128 = complex128 (float64 I, float64 Q)
 f32 = float32 (real)
 f64 = float64 (real)
[{dtype}]: """)
            if val!='':
                dtype = SampleType(val)
            rename = True

        # Default to complex64 for files that are missing a dtype
        if name_parts[0][3]!="":
            fcentre = FloatString(name_parts[0][3]).value
        else:
            fcentre = DEF_GS_CENTRE_F  
            val = input(f"Filename does not specify centre frequency. What was the capture centre frequency (Hz)? [{fcentre:.0f}]: ")
            if val!='':
                fcentre = FloatString(val).value
            rename = True
        
        if rename:
            new_path = path.parent / f"{datecode}_{FloatString(samplerate)}_{dtype}_{FloatString(fcentre)}.iq"
            if (input(f"Rename {path.name} to {new_path.name}? [Y]: ").lower() in ("","y")):
                path.rename(new_path)
                path = new_path

        return SignalFile(
            path=path.resolve(),
            name=path.stem,
            samplerate=samplerate,
            fcentre=fcentre,
            timestamp=dt.datetime.strptime(
                datecode, "%Y%m%d%H%M%S"
            ),  # UTC or AWST????,
            dtype=dtype,
            verbose=verbose,
        )

    def write(self, data: npt.NDArray):
        print(f"- Writing {self}")
        data.astype(self.dtype.value).tofile(self.path)

    @classmethod
    def prepare(
        cls,
        parent: Path,  # Parent path to save file
        samplerate: float,  # Data samplerate
        fcentre: float,
        timestamp: dt.datetime,  # Data timestamp
        dtype: SampleType,
        suffix: Optional[str],  # Optional suffix to append to filename
        verbose: bool = False,
    ) -> "SignalFile":
        name = f"{timestamp.strftime('%Y%m%d%H%M%S')}_{FloatString(samplerate)}_{dtype}_{FloatString(fcentre)}"
        if suffix is not None:
            name += "_" + str(suffix)
        outfile = parent / f"{name}.iq"
        parent.mkdir(parents=True, exist_ok=True)
        return SignalFile(
            path=outfile,
            name=name,
            samplerate=samplerate,
            fcentre=fcentre,
            timestamp=timestamp,
            dtype=dtype,
            verbose=verbose,
        )

    @classmethod
    def find(
        cls, rootdir: str = os.getcwd(), pattern: str = "*", verbose: bool = False
    ) -> Generator["SignalFile", None, None]:
        if verbose:
            print(f"IQ file in {rootdir}:")
        for p in Path(rootdir).glob(f"{pattern}.iq"):
            result = cls.frompath(p, verbose=verbose)
            if verbose:
                print(f"- {result}")
            yield result

class GFSK:
    def __init__(
        self,
        samplerate: float = DEF_GS_SAMPLERATE,
        symbolrate: float = DEF_SYMBOLRATE,
        bt: float = 0.5,  # Bandwidth-Time product (BT)
    ):
        self.samplerate = int(samplerate)
        self.symbolrate = int(symbolrate)
        self.bt = bt
        self.N_up = int(samplerate / gcd(self.samplerate, self.symbolrate))
        self.N_down = int(symbolrate / gcd(self.samplerate, self.symbolrate))
        self.N_chan_filter_taps = 2048

    @property
    def pulse_width_symbols(self) -> int:
        return ceil(1 / float(self.bt) + 0.5)

    @property
    def pulse_width_samples(self) -> int:
        return ceil(self.N_up * self.pulse_width_symbols)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.log(2)) / (2 * np.pi * self.bt * self.N_up))

    @property
    def pulse_t(self) -> npt.NDArray[np.float32]:
        return (
            np.arange(0, self.pulse_width_symbols, 1 / self.N_up, dtype=np.float32)
            - float(self.pulse_width_symbols) / 2
        )

    @property
    def pulse_h(self) -> npt.NDArray[np.float32]:
        a = np.pi * np.sqrt(2 / np.log(2)) * self.bt
        return -0.5 * (
            erf(a * (self.pulse_t - 0.5), dtype=np.float32)
            - erf(a * (self.pulse_t + 0.5), dtype=np.float32)
        )

    def demod(
        self, iq_samples: npt.NDArray[np.complex64], clip: Optional[float] = None
    ) -> npt.NDArray[np.float32]:
        ph = np.unwrap(np.angle(iq_samples))
        freq = np.gradient(ph)
        if clip is not None:
            freq = np.clip(freq, -clip, clip)
        return freq
        # Apply gaussian filter kernel to the output, this is not exactly a matched filter, but better than nothing for now
        # return convolve(self.pulse_h, freq)

    def frombytes(self, bufin: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        N = len(bufin)
        M = 8 * N * self.N_up + self.pulse_width_samples
        dphase = np.zeros(M, dtype=np.float32)
        for n in range(0, N):
            cur_byte = bufin[n]
            # sys.stdout.write(f"bufin[{n}] {cur_byte:02x} [ ")
            for bit in range(0, 8):
                bit_val = (cur_byte >> (8 - bit - 1)) & 0x01
                nrz_val = 1.0 if bit_val == 0x01 else -1.0
                # sys.stdout.write(f"{nrz_val} ")
                n_a = int((8 * n + bit) * self.N_up)
                n_b = int(n_a + self.pulse_width_samples)
                dphase[n_a:n_b] = dphase[n_a:n_b] + nrz_val * self.pulse_h

            # sys.stdout.write(f"]\n")
        # return dphase
        if self.N_down==1:
            return dphase
        return decimate(dphase, self.N_down, ftype="fir")

def channel_power(
    iq_data: npt.NDArray[np.complex64],
    h_channel: npt.NDArray[np.float32],
    channel: float,
    samplerate: float,
    fcentre: float,
    M_down: int,
) -> npt.NDArray[np.float32]:
    f_shift_hz: float = fcentre - channel
    print(f"- Channelising {channel*1e-6:.06f} MHz...")
    w = 2 * np.pi * f_shift_hz / samplerate
    iq_filt = np.convolve(
        h_channel,
        iq_data * (np.exp(1j * w * np.arange(0, len(iq_data))).astype(np.complex64)),
    )
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    """
    TODO: Rather than just forcing an integer decimation value, should look at using
        gcd() to work out a nice upsample -> downsample combination for resample_poly()
    """
    print(f"- Decimating {channel*1e-6:.06f} MHz...")
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    return decimate(np.abs(iq_filt), M_down, n=100 * M_down, ftype="fir")

def integrate(x: npt.NDArray[np.float32], delta_t: float = 1.0) -> npt.NDArray[np.float32]:
    print(f"delta_t = {delta_t}")
    M = len(x)
    y = np.zeros(M)
    for n in range(1, M):
        y[n] = delta_t * ((x[n] + x[n - 1]) / 2) + y[n - 1]
        # y[n] = delta_t*x[n] + y[n-1]
    return y

def channel_powers(
    sf: SignalFile,
    channels=[437.700e6, 437.850e6, 437.925e6],
    channel_bandwidth=float(12000),
    fir_taps=int(1024),
    verbose: bool = False,
) -> List[SignalFile]:
    f_cutoff_hz = channel_bandwidth / 2
    # target_ch_samplerate = 10*channel_bandwidth  # Oversampling to assist with manual alignment and demod
    bytes_per_frame = 32  # Budgetary purposes only...
    syms_per_frame = 8 * bytes_per_frame
    tgt_samplerate = (
        2 * 9600 / syms_per_frame
    )  # 2 samples per frame is enough to see beacons

    # For this fast search, we filter and decimate to target about 1 sample per frame

    M_down = int(sf.samplerate / tgt_samplerate)
    while (sf.samplerate % M_down) != 0:
        M_down -= 1

    ch_samplerate = sf.samplerate / M_down

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    h_channel = firwin(fir_taps, f_cutoff_hz, fs=sf.samplerate).astype(np.float32)
    iq_data: Optional[npt.NDArray] = None
    result: List[SignalFile] = []
    for channel in channels:
        out_sf = SignalFile.prepare(
            parent=sf.datapath,
            samplerate=ch_samplerate,
            timestamp=sf.timestamp,
            dtype=SampleType("f32"),
            fcentre=channel,
            suffix=f"bw{channel_bandwidth:.0f}_n{fir_taps}_chan_power",
            verbose=verbose,
        )
        if not out_sf.path.exists():
            if iq_data is None:
                iq_data = sf.load()
            out_sf.write(
                channel_power(iq_data, h_channel, channel, sf.samplerate, sf.fcentre, M_down)
            )
        else:
            if verbose:
                print(f"- Skip channelisation (already exists): {out_sf}")
        result.append(out_sf)
    return result

def channelise(
    iq_data: npt.NDArray[np.complex64],
    samplerate: float,
    fcentre: float,
    channel: float,
    channel_bandwidth=float(12000),
    fir_taps=int(1024),
) -> npt.NDArray[np.complex64]:
    f_shift_hz: float = fcentre - channel
    f_cutoff_hz = channel_bandwidth / 2

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    h_channel = firwin(fir_taps, f_cutoff_hz, fs=samplerate)
    print(f"- Channelising {channel*1e-6:.06f} MHz...")
    w = 2 * np.pi * f_shift_hz / samplerate
    iq_filt = np.convolve(
        h_channel,
        iq_data * (np.exp(1j * w * np.arange(0, len(iq_data))).astype(np.complex64)),
    )
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    return iq_filt
    # Rather than just forcing an integer decimation value, should look at using
    # gcd() to work out a nice upsample -> downsample combination for resample_poly()
    # # print(f"- Decimating {channel*1e-6:.06f} MHz...")
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    # return decimate(iq_filt,M_down,n=100*M_down,ftype="fir")

def channel_corr(
    sf: SignalFile,
    channels=[437.700e6, 437.850e6, 437.925e6],
    channel_bandwidth=float(12000),
    fir_taps=int(1024),
    verbose: bool = False,
) -> List[SignalFile]:
    # f_cutoff_hz = channel_bandwidth / 2
    # target_ch_samplerate = 10*channel_bandwidth  # Oversampling to assist with manual alignment and demod
    tgt_samplerate = (3 * 9600)  # 6 samples per symbol

    # For this fast search, we filter and decimate to target about 1 sample per frame

    M_down = int(sf.samplerate / tgt_samplerate)
    while (sf.samplerate % M_down) != 0:
        M_down -= 1

    ch_samplerate = sf.samplerate / M_down
    # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    # h_channel = firwin(fir_taps, f_cutoff_hz, fs=sf.samplerate).astype(np.float32)

    gfsk = GFSK(samplerate=sf.samplerate)
    START_OF_FRAME = np.array(
        [0xAA, 0xAA, 0xAA, 0xAA, 0xD3, 0x91, 0xD3, 0x91], dtype=np.uint8
    )
    h_sof = np.flip(gfsk.frombytes(START_OF_FRAME))

    iq_data: Optional[npt.NDArray] = None
    result: List[SignalFile] = []
    for channel in channels:
        out_sf = SignalFile.prepare(
            parent=sf.datapath,
            samplerate=ch_samplerate,
            timestamp=sf.timestamp,
            dtype=SampleType("f32"),
            fcentre=channel,
            suffix=f"bw{channel_bandwidth:.0f}_n{fir_taps}_chan_corr",
            verbose=verbose,
        )
        if not out_sf.path.exists():
            if iq_data is None:
                iq_data = sf.load()
            chan_data = gfsk.demod(channelise(iq_data, sf.samplerate, sf.fcentre, channel, channel_bandwidth, fir_taps))
            # decimate(np.abs(iq_filt), M_down, n=100 * M_down, ftype="fir")
            out_sf.write(
                decimate(np.convolve(h_sof, chan_data), M_down, n=100 * M_down, ftype="fir")
            )
        else:
            if verbose:
                print(f"- Skip channelisation (already exists): {out_sf}")
        result.append(out_sf)
    return result

def cmd_power(plot:bool=False, verbose:bool=False):
    for sf in SignalFile.find(verbose=verbose):
        print("-"*80)
        sf_outputs = channel_powers(
            sf, [437.700e6, 437.850e6, 437.925e6], verbose=verbose
        )
        fig = plt.figure(figsize=(16, 6), layout="tight")
        for chdata in sf_outputs:
            pwr = chdata.load()
            t_ms = np.arange(0, len(pwr)) * 1000 / chdata.samplerate
            plt.plot(t_ms, 20 * np.log10(np.abs(pwr)), label=chdata.name, alpha=0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Channel Power (dB)")
        # plt.ylim([-85,-40])
        plt.grid()
        plt.legend()
        Path("figures").mkdir(exist_ok=True)
        outfile = f"figures/{sf.datecode}_channel_powers.png"
        if plot:
            plt.show()
        else:
            fig.savefig(outfile)
            print(f"Wrote: {outfile}")

def cmd_iq(rootdir:str, verbose:bool=False):
    for sf in SignalFile.find(rootdir=rootdir, verbose=verbose):
        print(sf)

def cmd_corr(plot:bool=False, verbose:bool=False):
    for sf in SignalFile.find(verbose=verbose):
        sf_outputs = channel_corr(
            sf, [437.700e6, 437.850e6, 437.925e6], verbose=verbose
        )
        fig = plt.figure(figsize=(16, 6), layout="tight")
        plt.rcParams['agg.path.chunksize'] = 1000
        plt.xlabel("Time (ms)")
        plt.ylabel("Channel Power (dB)")
        for chdata in sf_outputs:
            pwr = chdata.load()
            t_ms = np.arange(0, len(pwr)) * 1000 / chdata.samplerate
            plt.plot(t_ms, 20 * np.log10(np.abs(pwr)), label=chdata.name, alpha=0.5)

        ymax = 20*np.round(np.max(np.log10(np.abs(pwr))))
        plt.ylim([ymax-20,ymax])
        plt.grid()
        plt.legend()
        Path("figures").mkdir(exist_ok=True)
        outfile = f"figures/{sf.datecode}_channel_corr.png"
        if plot:
            plt.show()
        else:
            fig.savefig(outfile)
            print(f"Wrote: {outfile}")

def cmd_demod(
        path:Path, 
        channel:float, 
        offset:float, 
        length:float,
        symbolrate:float=DEF_SYMBOLRATE,
        symbol_offset:float=1.6,
        verbose:bool=False
):
    sf = SignalFile.frompath(path, verbose=verbose)
    gfsk = GFSK(samplerate=sf.samplerate, symbolrate=symbolrate)
    iq_data = sf.load(offset=offset, length=length)
    h_pwr = firwin(512, 100, fs=sf.samplerate)

    START_OF_FRAME = np.array(
        [0xAA, 0xAA, 0xAA, 0xAA, 0xD3, 0x91, 0xD3, 0x91], dtype=np.uint8
    )
    exp_data = gfsk.frombytes(START_OF_FRAME)
    t_exp = np.arange(0, len(exp_data)) * 1000 / sf.samplerate

    chan_data = channelise(iq_data, sf.samplerate, sf.fcentre, channel=channel)

    mag = convolve(h_pwr, np.abs(chan_data))
    mag = np.abs(mag) / np.max(mag)
    # Drop the first 256 samples as this will be the group delay of h_pwr
    mag = np.clip(10 * (mag - 0.5), 0.0, 1.0)[256:]
    t_mag = np.arange(0, len(mag)) * 1000 / sf.samplerate
    fdemod = gfsk.demod(chan_data)
    fdemod = np.multiply(fdemod, mag[: len(fdemod)])
    t_ms = np.arange(0, len(fdemod)) * 1000 / sf.samplerate

    n_pkt_start = np.argwhere(mag > 0.0)[0]

    t_mag += offset + 1024 / sf.samplerate
    t_ms += offset + 1024 / sf.samplerate
    t_exp += t_mag[n_pkt_start]

    plt.figure(figsize=(16, 6), layout="tight")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency Deviation")
    plt.plot(t_ms, fdemod, label=f"{sf.name} demod", alpha=0.5)
    plt.plot(t_exp, exp_data * fdemod.max(), label=f"{sf.name} demod", alpha=0.5)

    n_syms = [(n_pkt_start + int((n+symbol_offset)*sf.samplerate/symbolrate)) for n in range(0,DEF_FRAME_BYTES*8)]
    plt.plot(t_mag[n_syms],fdemod[n_syms],'rx', alpha=0.3)

    bits = fdemod[n_syms]>0.0
    nbytes = 0
    bitidx = 7
    curbyte = 0
    data = b''
    sys.stdout.write("data = [\n  ")
    for bitval in fdemod[n_syms]:
        curbyte = curbyte | (int(bitval[0]>0.0)<<bitidx)
        bitidx-=1
        if bitidx==-1:
            sys.stdout.write(f" {curbyte:02x}")
            
            data += bytes([curbyte])
            bitidx = 7
            curbyte = 0
            nbytes += 1
            if nbytes%32==0:
                sys.stdout.write("\n  ")
    sys.stdout.write("\n]\n")
    # plt.plot(t_mag, mag, label=f"{sf.name} mag", alpha=0.5)
    # plt.ylim([-85,-40])
    plt.grid()
    plt.legend()
    plt.show()
    # Path("figures").mkdir(exist_ok=True)
    # outfile = f"figures/{sf.datecode}_demod.png"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="binar.py", description="Binar RF IQ processing helper script"
    )
    parser.add_argument(
        "--iq-samplerate",
        type=float,
        default=DEF_GS_SAMPLERATE,
        help="IQ samplerate (Hz)",
    )
    parser.add_argument(
        "--iq-frequency",
        type=float,
        default=DEF_GS_CENTRE_F,
        help="IQ centre frequency (Hz)",
    )
    parser.add_argument(
        "--symbolrate", type=float, default=DEF_SYMBOLRATE, help="GFSK data bitrate (Hz)"
    )
    parser.add_argument(
        "--symbol_offset", type=float, default=1.6, help="Packet demod symbol offset from magnitude peak (in symbols)"
    )
    parser.add_argument(
        "--ntaps",
        type=int,
        default=DEF_CH_FIR_TAPS,
        help="Number of taps in FIR channelisation low-pass filter",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEF_CH_BW,
        help="Channel bandwidth (Hz). Channelisation LPF cutoff is exactly half this value",
    )
    parser.add_argument(
        "--verbose","-v",
        action="store_true",
        help="Print detailed information (all file reads/writes, dtypes etc...)",
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Commands")
    pwr_parser = subparsers.add_parser(
        "power",
        help="Plot channel power estimation for all .iq files found in the current working directory",
    )
    pwr_parser.add_argument(
        "--plot",
        action="store_true",
        help="Interactive plot or, if False, write PNG file",
    )

    iq_parser = subparsers.add_parser(
        "iq",
        help="List and prepare IQ files found in the current working directory for processing",
    )

    corr_parser = subparsers.add_parser(
        "corr",
        help="Plot correlation against Binar GFSK Preamble/Sync for all .iq files found in the current working directory",
    )
    corr_parser.add_argument(
        "--plot",
        action="store_true",
        help="Interactive plot or, if False, write PNG file",
    )

    demod_parser = subparsers.add_parser(
        "demod", help="Demod a time range from a channel file"
    )
    demod_parser.add_argument("srcfile", type=str, help="Source IQ file to use")
    demod_parser.add_argument(
        "channel",
        type=float,
        help="Channelisation centre frequency (Hz) (e.g. 437.700e6, 437.850e6, or 437.925e6)",
    )
    demod_parser.add_argument(
        "--offset", type=float, default=0.0, help="Offset (seconds) to plot data from"
    )
    demod_parser.add_argument(
        "--length", type=float, default=1.0, help="Length (seconds) to plot data to"
    )
    demod_parser.add_argument(
        "--expected",
        type=float,
        default=0.0,
        help="Offset to overlay expected waveform from",
    )
    pargs = parser.parse_args()
    if pargs.verbose:
        print(pargs)

    if pargs.cmd == "iq":
        cmd_iq(os.getcwd(), verbose=pargs.verbose)
        sys.exit(0)

    if pargs.cmd == "power":
        cmd_power(plot=pargs.plot, verbose=pargs.verbose)
        sys.exit(0)

    if pargs.cmd == "corr":
        cmd_corr(plot=pargs.plot, verbose=pargs.verbose)
        sys.exit(0)

    if pargs.cmd == "demod":
        cmd_demod(
            Path(pargs.srcfile), 
            pargs.channel, 
            offset=pargs.offset, 
            length=pargs.length, 
            verbose=pargs.verbose, 
            symbolrate=pargs.symbolrate,
            symbol_offset=pargs.symbol_offset
        )
        sys.exit(0)
