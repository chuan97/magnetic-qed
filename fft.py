from scipy.fft import fft, fftfreq

def fft_wrapper(time, timeseries, slice_=None, dicard_half=True):
    if slice_:
        time = time[slice_]
        timeseries = timeseries[slice_]
        
    ws = fftfreq(len(time), time[1] - time[0])
    amps = fft(timeseries)
    
    if dicard_half:
        return ws[:len(ws)//2], amps[:len(ws)//2]
    else:
        return ws, amps