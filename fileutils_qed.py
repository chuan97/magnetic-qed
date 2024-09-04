import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def load_csv_file_with_columns(filepath, header = 0, columns_list = []):

    dataframe = pd.read_csv(filepath, header = header)

    if len(columns_list) == 0:
        return dataframe, dataframe
    else:
        cols = [dataframe[i] for i in columns_list]
        cols_df = pd.DataFrame(np.transpose(cols), columns = columns_list)
        return dataframe, cols_df

def load_file_with_columns(filepath, header = 0, columns_list = []):

    dataframe = pd.read_table(filepath, header = header)

    if len(columns_list) == 0:
        return dataframe, dataframe
    else:
        cols = [dataframe[i] for i in columns_list]
        cols_df = pd.DataFrame(np.transpose(cols), columns = columns_list)
        return dataframe, cols_df
    
def show_transmission(time, m_data, time_ns = True, title = "$m_x$ in time", legend = "", color = 'blue', linewidth = 1):

    figure(num = title, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.ylabel("$m_y$ [A/m]")
    plt.xlabel("Time [ns]")
    plt.plot((time/1e-9 if time_ns else time), m_data, label = legend, c = color, linewidth = linewidth)

    return plt

def show_transmission_axe(ax, time, m_data, time_ns = True, title = "", legend = "", color = 'blue', linewidth = 1):

    ax.title.set_text(title)
    ax.set_ylabel("$m_y$ [A/m]")
    ax.set_xlabel("Time [ns]")
    ax.plot((time/1e-9 if time_ns else time), m_data, label = legend, c = color, linewidth = linewidth)

    return ax

# this is for mumax3 data
def plot_single_fft(time, m_data, title = "Amplitude-FFT/Frequency $m_x$", 
                    zoom_x = None, zoom_y = None, label = "", vlines = None, vlines_label = True,
                    loc = 'upper right', linestyle = 'solid', color='red', linewidth = 1, is_pandas = True):
    
    if is_pandas:
        vals = m_data.values
    else:
        vals = m_data
    fig = figure(num = title, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (Hz)")

    if is_pandas:
        t_eval2 = time.values
    else:
        t_eval2 = time
    fft_value = np.abs(fft(vals[len(vals) // 2:])) # discard first half of time evolution
    x_linspace = fftfreq(len(t_eval2) // 2, t_eval2[1] - t_eval2[0])

    ######### añadido Juan
    if vlines:
        plt.axvline(vlines[0], c='green', ls='dashed', lw = linewidth, label = 'pol. global')
        plt.axvline(vlines[1], c='green', ls='dashed', lw = linewidth)
        plt.legend(loc=loc)
    #########

    if label == "":
        plt.plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], ls = linestyle, c = color, lw = linewidth)
    else:
        plt.plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], 
                 ls = linestyle, c = color, lw = linewidth, label = label)
        plt.legend(loc=loc)

    if zoom_x != None:
        plt.xlim(zoom_x)
        
    if zoom_y != None:
        plt.ylim(zoom_y)

    return plt

# this is for mumax3 data
def plot_single_fft_axis(ax, time, m_data, 
                    zoom_x = None, zoom_y = None, label = "", vlines = None, vlines_label = True,
                    loc = 'upper right', linestyle = 'solid', color='red', linewidth = 1, is_pandas = True):
    
    if is_pandas:
        vals = m_data.values
    else:
        vals = m_data

    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")

    if is_pandas:
        t_eval2 = time.values
    else:
        t_eval2 = time
        
    fft_value = np.abs(fft(vals[len(vals) // 2:])) # discard first half of time evolution
    x_linspace = fftfreq(len(t_eval2) // 2, t_eval2[1] - t_eval2[0])

    ######### añadido Juan
    if vlines:
        ax.axvline(vlines[0], c='green', ls='dashed', lw = linewidth, label = 'pol. global')
        ax.axvline(vlines[1], c='green', ls='dashed', lw = linewidth)
        ax.legend(loc=loc)
    #########

    if label == "":
        ax.plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], ls = linestyle, c = color, lw = linewidth)
    else:
        ax.plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], 
                 ls = linestyle, c = color, lw = linewidth, label = label)
        ax.legend(loc=loc)

    if zoom_x != None:
        ax.set_xlim(zoom_x)
        
    if zoom_y != None:
        ax.set_ylim(zoom_y)

    return ax

def plot_single(ws, amp, title = "Amplitude-FFT/Frequency $m_x$", 
                    zoom_x = None, zoom_y = None, label = "", vlines = None, vlines_label = True, 
                    loc = 'upper right', linestyle = 'solid', color='red', linewidth = 1):

    vals = ws.values
    fig = figure(num = title, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (GHz)")

    ######### añadido Juan
    if vlines:
        plt.axvline(vlines[0], c='green', ls='dashed', lw = linewidth, label = 'pol. global')
        plt.axvline(vlines[1], c='green', ls='dashed', lw = linewidth)
        plt.legend(loc=loc)
    #########

    if label == "":
        plt.plot(ws[:len(ws)//2], amp[:len(ws)//2], ls=linestyle, c = color, lw = linewidth)
    else:
        plt.plot(ws[:len(ws)//2], amp[:len(ws)//2], ls=linestyle, c = color, label = label, lw = linewidth)
        plt.legend(loc = loc)

    if zoom_x != None:
        plt.xlim(zoom_x)
        
    if zoom_y != None:
        plt.ylim(zoom_y)

    return plt

def extract_plot_csv_data(filepaths, columns_list = ['ws', 'amp', 'lam']):
    
    data_ws = []
    data_amp = []
    data_lam = []
    
    for path in filepaths:

        full_data, data_colums = load_csv_file_with_columns(path, columns_list = columns_list)
   
        ws = full_data[columns_list[0]]
        amp = full_data[columns_list[1]]
        lam = full_data[columns_list[2]]
        
        data_ws.append([ws.values])
        data_amp.append([amp.values])
        data_lam.append([lam.values])
        
    return data_ws, data_amp, data_lam

def extract_plot_csv_data_evo(filepaths, columns_list = ['t', 'evolution']):
    
    data_sol = []
    data_t = []
    
    for path in filepaths:

        full_data, data_colums = load_csv_file_with_columns(path, columns_list = columns_list)
   
        t = full_data[columns_list[0]]
        sol = full_data[columns_list[1]]
        
        data_sol.append([sol.values])
        data_t.append([t.values])
        
    return data_sol, data_t

def extract_plot_data(filepaths, columns_list = ['# t (s)', 'mx ()']):
    
    data_time = []
    data_mx = []
    
    for path in filepaths:

        full_data, data_colums = load_file_with_columns(path, columns_list = columns_list)

        time = data_colums[columns_list[0]]
        mx = data_colums[columns_list[1]]
        
        data_time.append([time.values])
        data_mx.append([mx.values])
        
    return data_time, data_mx

def create_multiple_fft_plots(time, vals, title, labels, legends, zooms, 
                              rows = 1, cols = 2, stop = False, stop_at = [0,0], vlines = None, vlines_label = True, 
                              loc = 'upper right', linestyle = 'dashed', wspace=0.2, hspace=0.5, color='red', linewidth = 1):

    fig, ax = plt.subplots(nrows=rows, ncols=cols, num = title, figsize=(15, 15), facecolor='w', edgecolor='k')
    
    #fig.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle(title)
    
    count = 0
    
    for i in list(range(rows)):
        for j in list(range(cols)):
            
            if stop == True and i == stop_at[0] and j == stop_at[1]:
                break;
                
            ######### añadido Juan
            if vlines:
                v = vlines[count + j]
                
                if cols == 1:
                    ax[i].axvline(v[0], c='green', ls=linestyle, lw = linewidth, label = 'pol. global')
                    ax[i].axvline(v[1], c='green', ls=linestyle, lw = linewidth)
                else:
                    ax[i, j].axvline(v[0], c='green', ls=linestyle, lw = linewidth, label = 'pol. global')
                    ax[i, j].axvline(v[1], c='green', ls=linestyle, lw = linewidth)
            #########

            data = vals[count + j][0]
            t_eval2 = time[0][0]

            fft_value = np.abs(fft(data[len(data) // 2:])) # discard first half of time evolution
            x_linspace = fftfreq(len(t_eval2) // 2, t_eval2[1] - t_eval2[0])

            if cols == 1:
                ax[i].plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], 'r', 
                          label = legends[count + j], c = color, lw = linewidth)

                ax[i].set_xlim(zooms[count + j][0])
                ax[i].set_ylim(zooms[count + j][1])
                ax[i].legend(loc=loc)

                ax[i].set_ylabel(labels[0])
                ax[i].set_xlabel(labels[1])
            else:
                ax[i, j].plot(x_linspace[:len(fft_value)//2], fft_value[:len(fft_value)//2], 'r', 
                              label = legends[count + j], c = color, lw = linewidth)

                ax[i, j].set_xlim(zooms[count + j][0])
                ax[i, j].set_ylim(zooms[count + j][1])
                ax[i, j].legend(loc=loc)

                ax[i, j].set_ylabel(labels[0])
                ax[i, j].set_xlabel(labels[1])
        
        count += 2
        
    fig.tight_layout() 
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    return plt, fig, ax

def append_to_plot(plt, fig, ax, rows, cols, ws_arr, amps_arr, legends, line_colour, loc = 'upper right', linestyle = 'dashed', linewidth = 1, wspace=0.2, hspace=0.5):
    
    count = 0
    
    for i in list(range(rows)):
        for j in list(range(cols)): 
            #print(count + j)
            ws = ws_arr[count + j][0]
            amps = amps_arr[count + j][0]

            if cols == 1:
                ax[i].plot(ws[:len(ws)//2], amps[:len(ws)//2], 
                         ls = linestyle, c = line_colour, lw = linewidth, label = legends[count + j])
            
                ax[i].legend(loc = loc)
            else:
                ax[i,j].plot(ws[:len(ws)//2], amps[:len(ws)//2], 
                             ls = linestyle, c = line_colour, lw = linewidth, label = legends[count + j])

                ax[i,j].legend(loc = loc)

        count += cols

    #fig.tight_layout() 
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    return plt, fig, ax

def set_title(ax, rows, cols, titles):
    count = 0
    for i in list(range(rows)):
        for j in list(range(cols)): 
            title = titles[count + j]
            
            if cols == 1:
                ax[i].title.set_text(title)
            else:
                ax[i,j].title.set_text(title)

        count += cols

def set_ylim(ax, rows, cols, ylim):
    count = 0
    for i in list(range(rows)):
        for j in list(range(cols)): 
            
            if cols == 1:
                ax[i].set_ylim(ylim)
            else:
                ax[i,j].set_ylim(ylim)

        count += cols
        
def set_xlim(ax, rows, cols, xlim):
    count = 0
    for i in list(range(rows)):
        for j in list(range(cols)): 
            
            if cols == 1:
                ax[i].set_xlim(ylim)
            else:
                ax[i,j].set_xlim(ylim)

        count += cols
        
def make_float(num):
    num = num.replace(' ','').replace(',','.').replace("−", "-")
    return float(num)

def fix_zero_xaxis(ax, rows, cols):
    count = 0
    for i in list(range(rows)):
        for j in list(range(cols)): 
            
            if cols == 1:
                labels_x = [float(make_float(item.get_text())) for item in ax[i].get_xticklabels()]
                labels_x[0] = ''
                ax[i].set_xticklabels(labels_x)
            else:
                labels_x = [float(make_float(item.get_text())) for item in ax[i,j].get_xticklabels()]
                labels_x[0] = ''
                ax[i,j].set_xticklabels(labels_x)

        count += cols
        
        
def append_to_single_plot(plt, ws, amps, legend, line_colour, loc = 'upper right', linestyle = 'dashed', linewidth = 1):
    
    plt.plot(ws[:len(ws)//2], amps[:len(ws)//2], ls = linestyle, c = line_colour, lw = linewidth, label = legend)

    plt.legend(loc = loc)

    return plt

def append_to_single_plot_evo(plt, t, evo, legend, line_colour, loc = 'upper right', linestyle = 'dashed', linewidth = 1):
    
    plt.plot(t/1e-9, evo, ls = linestyle, c = line_colour, lw = linewidth, label = legend)

    plt.legend(loc = loc)

    return plt