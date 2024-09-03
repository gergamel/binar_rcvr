# from gfsk import GFSK
import os
from pathlib import Path
from math import gcd, ceil, floor
from typing import Optional, List, Generator, Dict, Literal
import datetime as dt
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.signal import firwin, convolve, decimate
from scipy.special import erf
from dataclasses import dataclass

DEF_SYMBOLRATE    = int(9600)        # Default GMSK symbol (bit) rate
DEF_FRAME_BYTES   = int(32)          # Budgetary frame length in bytes
GFSK_BT           = 0.5              # Bandwidth-Time product of our GFSK gaussian filter

DEF_GS_SAMPLERATE = float(400e3)     # Default ground station IQ samplerate
DEF_GS_CENTRE_F   = float(437.850e6) # Default centre frequency of Ground Station IQ captures

DEF_CH_BW         = float(12000)     # Channelisation bandwidth (FIR filter cutoff is 1/2 this)
DEF_CH_SAMPLERATE = float(100e3)     # Samplerate of channelised data (decimated by 4, which is ~10x oversampling)
DEF_CH_FIR_TAPS   = 1024             # Number of taps in channelisation LPF

DEF_CH_SAMPLERATE = float(100e3)

# Number of samples per GFSK symbol (bit)
SYM_LEN_SAMPLES   = int(ceil(DEF_GS_SAMPLERATE/DEF_SYMBOLRATE))

# Rough number of symbols (bits) per packet
FRAME_LEN_SYMS    = 8*DEF_FRAME_BYTES

# Rough number of samples per packet. We might use this for channel power estimation filter
FRAME_LEN_SAMPLES = FRAME_LEN_SYMS*SYM_LEN_SAMPLES

SampleType = Literal["f32","f64","c64","c128"]

def sampletype2np(dt:SampleType)->np.dtype:
    if (dt=="f32"):  return np.float32
    if (dt=="f64"):  return np.float64
    if (dt=="c64"):  return np.complex64
    if (dt=="c128"): return np.complex128
    raise ValueError(f"Unrecognised SampleType \"{dt}\"")

def np2sampletype(dt:np.dtype)->SampleType:
    if (dt==np.float32):    return "f32"
    if (dt==np.float64):    return "f64"
    if (dt==np.complex64):  return "c64"
    if (dt==np.complex128): return "c128"
    raise ValueError(f"Unrecognised SampleType \"{dt}\"")

@dataclass
class SignalFile():
    path:Path
    name:str
    samplerate:float
    timestamp:dt.datetime
    dtype:SampleType
    verbose:bool=False
    def __str__(self)->str:
        return f"SignalFile({self.name}, samplerate={self.samplerate}, timestamp={self.datecode}, dtype={self.dtype})"
    @property
    def dtype_size(self)->int:
        return sampletype2np(self.dtype)().nbytes()
    @property
    def datecode(self)->str:
        return self.timestamp.strftime("%Y%m%d%H%M%S")
    @property
    def datapath(self)->Path:
        return self.path.parent / self.name
    @property
    def length(self)->int:
        return int(self.path.stat().st_size/self.dtype_size)
    def load(self, offset:float=-1.0, length:float=-1.0)->npt.NDArray:
        dt = sampletype2np(self.dtype)
        if offset==-1.0 and length==-1.0:
            if self.verbose: print(f"- Reading {str(self.path)} {dt}")
            return np.fromfile(self.path, dtype=dt)
        
        # Default limited load is 0.0s to 0.5s
        sample_offset = int(0)
        sample_count = int(0.5*self.samplerate)
        if offset>0.0:
            sample_offset = int(offset*self.samplerate)
        if length>0.0:
            sample_count = int(length*self.samplerate)
        if self.verbose: print(f"- Reading {str(self.path)} {dt} {sample_count} samples from {sample_offset}")
        return np.fromfile(self.path, dtype=dt, offset=sample_offset*dt().nbytes, count=sample_count)
    @classmethod
    def frompath(cls, path:Path, verbose:bool=False)->'SignalFile':
        assert(path.exists())
        name_parts = path.stem.split("_")
        if len(name_parts)<2:
            raise ValueError(f"Expecting a file named {{datecode}}_{{samplerate}}.iq or {{datecode}}_{{samplerate}}_{{dtype}}.iq")
        datecode = name_parts[0]
        samplerate_str = name_parts[1].replace("k","e3").replace("M","e6")
        dtype = "c64" # Default to complex64 for files that are missing a dtype
        if len(name_parts)>2:
            dtype = name_parts[2]
        return SignalFile(
            path=path.resolve(),
            name=path.stem,
            samplerate=float(samplerate_str),
            timestamp=dt.datetime.strptime(datecode, "%Y%m%d%H%M%S"), # UTC or AWST????,
            dtype=dtype,
            verbose=verbose
        )
    def write(self,data:npt.NDArray):
        if self.verbose:
            print(f"- Writing {str(self.path)} {self.dtype}")
        data.astype(sampletype2np(self.dtype)).tofile(self.path)
    @classmethod
    def prepare(cls,
        parent:Path,            # Parent path to save file
        samplerate:float,       # Data samplerate
        timestamp:dt.datetime,  # Data timestamp
        dtype:SampleType,
        suffix:Optional[str],    # Optional suffix to append to filename
        verbose:bool=False
    )->'SignalFile':
        name = f"{timestamp.strftime("%Y%m%d%H%M%S")}_{samplerate:.0f}_{dtype}"
        if suffix is not None:
            name += "_" + str(suffix)
        outfile = parent / f"{name}.iq"
        parent.mkdir(parents=True,exist_ok=True)
        return SignalFile(
            path=outfile,
            name=name,
            samplerate=samplerate,
            timestamp=timestamp,
            dtype=dtype,
            verbose=verbose
        )
    @classmethod
    def find(cls, rootdir:str=os.getcwd(), pattern:str="*", verbose:bool=False)->Generator['SignalFile',None,None]:
        if pargs.verbose: print(f"IQ file in {rootdir}:")
        for p in Path(rootdir).glob(f"{pattern}.iq"):
            sf = cls.frompath(p,verbose=verbose)
            if pargs.verbose: print(f"- {sf}")
            yield sf

class GFSK():
    def __init__(self,
        samplerate  : int = DEF_GS_SAMPLERATE,
        symbolrate  : int = DEF_SYMBOLRATE,
        bt          : float = 0.5           # Bandwidth-Time product (BT)
    ):
        self.samplerate         = int(samplerate)
        self.symbolrate         = int(symbolrate)
        self.bt                 = bt
        self.N_up               = int(samplerate/gcd(self.samplerate,self.symbolrate))
        self.N_down             = int(symbolrate/gcd(self.samplerate,self.symbolrate))
        self.N_chan_filter_taps = 2048
    @property
    def pulse_width_symbols(self)->int:
        return ceil(1/float(self.bt) + 0.5)
    @property
    def pulse_width_samples(self)->int:
        return ceil(self.N_up*self.pulse_width_symbols)
    @property
    def sigma(self)->float:
        return float(np.sqrt(np.log(2)) / (2*np.pi*self.bt*self.N_up))
    @property
    def pulse_t(self)->npt.NDArray[np.float32]:
        return np.arange(0,self.pulse_width_symbols,1/self.N_up,dtype=np.float32)-float(self.pulse_width_symbols)/2
    @property
    def pulse_h(self)->npt.NDArray[np.float32]:
        a = np.pi * np.sqrt(2/np.log(2))*self.bt
        return (-0.5*(erf(a*(self.pulse_t-0.5),dtype=np.float32) - erf(a*(self.pulse_t+0.5),dtype=np.float32)))
    def demod(self, iq_samples:npt.NDArray[np.complex64], clip:Optional[float]=None)->npt.NDArray[np.float32]:
        ph = np.unwrap(np.angle(iq_samples))
        freq = np.gradient(ph)
        if clip is not None:
            freq = np.clip(freq, -clip, clip)
        return freq
        # Apply gaussian filter kernel to the output, this is not exactly a matched filter, but better than nothing for now
        # return convolve(self.pulse_h, freq)
    def frombytes(self, bufin:npt.NDArray[np.uint8])->npt.NDArray[np.float32]:
        N = len(bufin)
        M = 8*N*self.N_up + self.pulse_width_samples
        dphase = np.zeros(M, dtype=np.float32)
        for n in range(0,N):
            cur_byte = bufin[n]
            # sys.stdout.write(f"bufin[{n}] {cur_byte:02x} [ ")
            for bit in range(0,8):
                bit_val = ((cur_byte>>(8-bit-1))&0x01)
                nrz_val = 1.0 if bit_val==0x01 else -1.0
                # sys.stdout.write(f"{nrz_val} ")
                n_a = int((8*n+bit)*self.N_up)
                n_b = int(n_a + self.pulse_width_samples)
                dphase[n_a:n_b] = dphase[n_a:n_b] + nrz_val*self.pulse_h
            
            # sys.stdout.write(f"]\n")
        # return dphase
        return decimate(dphase, self.N_down, ftype="fir")

def channel_power(
    iq_data:npt.NDArray[np.complex64],
    h_channel:npt.NDArray[np.float32],
    channel:float,
    samplerate:int,
    M_down:int
)->npt.NDArray[np.float32]:
    f_shift_hz:float = DEF_GS_CENTRE_F - channel
    print(f"- Channelising {channel*1e-6:.06f} MHz...")
    w = 2 * np.pi * f_shift_hz / samplerate
    iq_filt = np.convolve(h_channel, iq_data * (np.exp(1j * w * np.arange(0, len(iq_data))).astype(np.complex64)))
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    """
    TODO: Rather than just forcing an integer decimation value, should look at using
        gcd() to work out a nice upsample -> downsample combination for resample_poly()
    """
    print(f"- Decimating {channel*1e-6:.06f} MHz...")
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    return decimate(np.abs(iq_filt),M_down,n=100*M_down,ftype="fir")

def integrate(x:npt.NDArray[np.float32], dt:float=1.0)->npt.NDArray[np.float32]:
    print(f"dt = {dt}")
    M = len(x)
    y = np.zeros(M)
    for n in range(1,M):
        y[n] = dt*((x[n] + x[n-1])/2) + y[n-1]
        # y[n] = dt*x[n] + y[n-1]
    return y

def channel_powers(
    sf:SignalFile,
    channels=[437.700e6,437.850e6,437.925e6],
    channel_bandwidth=float(12000),
    fir_taps=int(1024),
    verbose:bool=False
)->List[SignalFile]:
    f_cutoff_hz = channel_bandwidth/2
    # target_ch_samplerate = 10*channel_bandwidth  # Oversampling to assist with manual alignment and demod
    bytes_per_frame   = 32    # Budgetary purposes only...
    syms_per_frame    = 8*bytes_per_frame
    tgt_samplerate    = 2*9600/syms_per_frame # 2 samples per frame is enough to see beacons

    # For this fast search, we filter and decimate to target about 1 sample per frame

    M_down = int(sf.samplerate / tgt_samplerate)
    while (sf.samplerate % M_down)!=0:
        M_down-=1

    ch_samplerate = sf.samplerate/M_down

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    h_channel = firwin(fir_taps, f_cutoff_hz, fs=sf.samplerate).astype(np.float32)
    iq_data:Optional[npt.NDArray] = None
    result:List[SignalFile] = []
    for channel in channels:
        out_sf = SignalFile.prepare(
            parent=sf.datapath,
            samplerate=ch_samplerate,
            timestamp=sf.timestamp,
            dtype="f32",
            suffix=f"{channel*1e-6:.06f}MHz_bw{channel_bandwidth:.0f}_n{fir_taps}_chan_power",
            verbose=verbose
        )
        if not out_sf.path.exists():
            if iq_data is None: iq_data = sf.load()
            out_sf.write(channel_power(iq_data,h_channel,channel,sf.samplerate,M_down))
        else:
            if verbose:
                print(f"- Skip channelisation (already exists): {out_sf}")
        result.append(out_sf)
    return result

def channelise(
    iq_data:npt.NDArray[np.complex64],
    samplerate:int,
    channel:float,
    channel_bandwidth=float(12000),
    fir_taps=int(1024),
)->npt.NDArray[np.float32]:
    f_shift_hz:float = DEF_GS_CENTRE_F - channel
    f_cutoff_hz = channel_bandwidth/2

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    h_channel = firwin(fir_taps, f_cutoff_hz, fs=sf.samplerate)
    print(f"- Channelising {channel*1e-6:.06f} MHz...")
    w = 2 * np.pi * f_shift_hz / samplerate
    iq_filt = np.convolve(h_channel, iq_data * (np.exp(1j * w * np.arange(0, len(iq_data))).astype(np.complex64)))
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    return iq_filt
    """
    TODO: Rather than just forcing an integer decimation value, should look at using
        gcd() to work out a nice upsample -> downsample combination for resample_poly()
    """
    # print(f"- Decimating {channel*1e-6:.06f} MHz...")
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html
    # return decimate(iq_filt,M_down,n=100*M_down,ftype="fir")

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='binar.py',description='Binar RF IQ processing helper script')
    parser.add_argument('--iq-samplerate', type=int,   default=DEF_GS_SAMPLERATE, help="IQ samplerate (Hz)")
    parser.add_argument("--iq-frequency",  type=int,   default=DEF_GS_CENTRE_F,   help="IQ centre frequency (Hz)")
    parser.add_argument('--symbolrate',    type=int,   default=DEF_SYMBOLRATE,    help="GFSK data bitrate (Hz)")
    parser.add_argument('--ntaps',         type=int,   default=DEF_CH_FIR_TAPS,   help="Number of taps in FIR channelisation low-pass filter")
    parser.add_argument('--bandwidth',     type=float, default=DEF_CH_BW,         help="Channel bandwidth (Hz). Channelisation LPF cutoff is exactly half this value")
    parser.add_argument('--verbose',       action="store_true",                   help="Print detailed information (all file reads/writes, dtypes etc...)")
    subparsers = parser.add_subparsers(dest="cmd", help='Commands')
    pwr_parser = subparsers.add_parser('power', help='Plot channel power estimation for all .iq files found in the current working directory')
    pwr_parser.add_argument('--plot',      action="store_true",                   help="Interactive plot or, if False, write PNG file")

    demod_parser = subparsers.add_parser('demod', help='Demod a time range from a channel file')
    demod_parser.add_argument("srcfile",    type=str,   help="Source IQ file to use")
    demod_parser.add_argument("channel",    type=float, help="Channelisation centre frequency (Hz) (e.g. 437.700e6, 437.850e6, or 437.925e6)")
    demod_parser.add_argument("--offset",   type=float, default=0.0, help="Offset (seconds) to plot data from")
    demod_parser.add_argument("--length",   type=float, default=1.0, help="Length (seconds) to plot data to")
    demod_parser.add_argument("--expected", type=float, default=0.0, help="Offset to overlay expected waveform from")
    pargs = parser.parse_args()
    if pargs.verbose:
        print(pargs)

    if pargs.cmd is None or pargs.cmd=="power":
        for sf in SignalFile.find(verbose=pargs.verbose):
            sf_outputs = channel_powers(sf,[437.700e6,437.850e6,437.925e6],verbose=pargs.verbose)
            fig = plt.figure(figsize=(16, 6), layout="tight")
            for chdata in sf_outputs:
                pwr  = chdata.load()
                t_ms = np.arange(0,len(pwr)) * 1000 / chdata.samplerate
                plt.plot(t_ms, 20*np.log10(np.abs(pwr)), label=chdata.name, alpha=0.5)
            plt.xlabel("Time (ms)")
            plt.ylabel("Channel Power (dB)")
            # plt.ylim([-85,-40])
            plt.grid()
            plt.legend()
            Path("figures").mkdir(exist_ok=True)
            outfile = f"figures/{sf.datecode}_channel_powers.png"
            if pargs.plot:
                plt.show()
            else:
                fig.savefig(outfile)
                print(f"Wrote: {outfile}")
        exit(0)


    """ Best SNR beacons found with quick search through each channel power plot...
    20240830123521
    437.850:  13800 -  14000 (envelope looks messed up... maybe too long as well, might not be a Binar beacon)
    437.700: 405350 - 405750    python binar.py --verbose demod 20240830123521_400k.iq 437.700e6 --offset 405.400 --length 0.30
    # 437.850: 405800 - 406351    python binar.py --verbose demod 20240830123521_400k.iq 437.850e6 --offset 405.800 --length 0.5
    437.700: 399600 - 400000    python binar.py --verbose demod 20240830123521_400k.iq 437.700e6 --offset 399.600 --length 0.25
    437.925: 401150 - 401500    python binar.py --verbose demod 20240830123521_400k.iq 437.925e6 --offset 401.150 --length 0.25

    20240901190108:
    437.850:   8000 -   8500 (envelope looks messed up...)
    437.700: 771600 - 772000
    437.925: 772250 - 772700
    437.850: 774333 - 774650 (envelope looks messed up...)

    20240902100932:
    437.925: 387100 - 387450
    437.700: 387380 - 387700
    Looks like clipping at, for example, 24800 - 248250

    20240902114717:
    437.700: 333500-334000
    437.925: 334750-335250
    437.700: 339100-339450
    437.700: 344450-345000
    437.925: 345250-345750
    """
    if pargs.cmd=="demod":
        sf = SignalFile.frompath(Path(pargs.srcfile), verbose=pargs.verbose)
        gfsk = GFSK(samplerate=sf.samplerate)
        iq_data = sf.load(offset=pargs.offset, length=pargs.length)
        h_pwr = firwin(512, 100, fs=sf.samplerate)

        START_OF_FRAME = np.array([0xaa ,0xaa ,0xaa ,0xaa, 0xd3 ,0x91 ,0xd3 ,0x91],dtype=np.uint8)
        exp_data = gfsk.frombytes(START_OF_FRAME)
        t_exp    = np.arange(0,len(exp_data))*1000 / sf.samplerate

        chan_data = channelise(iq_data,sf.samplerate, channel=pargs.channel)

        mag    = convolve(h_pwr, np.abs(chan_data))
        mag    = np.abs(mag)/np.max(mag)
        # Drop the first 256 samples as this will be the group delay of h_pwr
        mag    = np.clip(10*(mag-0.5), 0.0, 1.0)[256:]
        t_mag  = np.arange(0,len(mag)) * 1000 / sf.samplerate
        fdemod = gfsk.demod(chan_data)
        fdemod = np.multiply(fdemod,mag[:len(fdemod)])
        t_ms   = np.arange(0,len(fdemod)) * 1000 / sf.samplerate
        
        t_mag += pargs.offset + 1024/sf.samplerate
        t_ms  += pargs.offset + 1024/sf.samplerate
        t_exp += t_mag[np.argwhere(mag>0.0)[0]]

        fig = plt.figure(figsize=(16, 6), layout="tight")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency Deviation")
        plt.plot(t_ms, fdemod, label=f"{sf.name} demod", alpha=0.5)
        plt.plot(t_exp, exp_data*fdemod.max(), label=f"{sf.name} demod", alpha=0.5)
        # plt.plot(t_mag, mag, label=f"{sf.name} mag", alpha=0.5)
        # plt.ylim([-85,-40])
        plt.grid()
        plt.legend()
        plt.show()
        # Path("figures").mkdir(exist_ok=True)
        outfile = f"figures/{sf.datecode}_demod.png"
        # pargs.channel
        # pargs.offset
        # pargs.length
        # pargs.expected
        exit(0)