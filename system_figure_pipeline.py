import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from matplotlib.colors import Normalize as Normalize
import matplotlib.cm
import os
from scipy.io import readsav
import re
# from brokenaxes import brokenaxes

def update_fit_files(target_folder_name, file_prefix, fitresults_folder_name = 'fitresults_bestfit', output_dir = 'data/', 
                     hpcc_path = 'jschulte@rsync.hpcc.msu.edu:/mnt/research/Exoplanet_Lab/jack/Global_Fits/'):
    '''
    target_folder_name: The name of the folder containing the fit files. E.g. 'TOI-1855'
    file_prefix: The string that precedes all of your EXOFASTv2 output files, as set by your procedures file. Must be the same as the prefix of your priors and SED files.
    fitresults_folder_name: The name of the folder containing the fit results. By default, it is 'fitresults_bestfit'.
    output_dir: The directory where you wish to move these files.
    hpcc_path: The path to your files on their current machine.
    '''

    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/' + file_prefix + '.mcmc.idl ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.detrendedmodel.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.detrendedmodel.telescope* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.prettymodel.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.residuals.telescope* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.residuals.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.prettymodelrv* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.sed.residuals.txt ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/modelfiles/' + file_prefix + '.mcmc.atmosphere.*.txt ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + f'/{fitresults_folder_name}/' + file_prefix + '.median.csv ' + output_dir)

    # collect SED file and prior file
    os.system(f'scp ' + hpcc_path + target_folder_name + '/' + file_prefix + '.priors.final ' + output_dir)
    os.system(f'scp ' + hpcc_path + target_folder_name + '/' + file_prefix + '.sed ' + output_dir)

def t_phase_folded(t, per, t0):
    t_phase_folded = (t - t0)/per - np.floor((t - t0)/per + 0.5) # centers on zero
    return t_phase_folded

def median_scinot_corrections(median, parname):
    '''
    Multiplies parameters in the EXOFASTv2 median table by the scientific notation exponent.

    median: pandas DataFrame for median table
    param: input parameter name
    '''

    scinot = median.scinot[median.parname==parname].iloc[0]

    if type(scinot) == str:
        exp_search = re.findall(r'\\times 10\^{(.*)}', scinot)
        exponent = int(exp_search[0])
    else:
        exponent = 0

    param = median.median_value[median.parname==parname].iloc[0]
    param_corrected = param * 10**exponent

    uperr = median.upper_errorbar[median.parname==parname].iloc[0]
    uperr_corrected = uperr * 10**exponent

    lowerr = median.lower_errorbar[median.parname==parname].iloc[0]
    lowerr_corrected = lowerr * 10**exponent
    return param_corrected, uperr_corrected, lowerr_corrected

def smooth(data, window_size):
    '''
    Smooths input data using a moving average with the specified window size.

    data: Input data to smooth
    window_size: The size of the smoothing window. Should be odd
    '''

    if window_size < 3:
        return data
    
    # Create a uniform filter (boxcar filter) to use for convolution
    boxcar = np.ones(window_size) / window_size
    
    # Convolve the data with the filter
    smoothed_data = np.convolve(data, boxcar, mode='same')
    
    return smoothed_data

def gen1pagefig(object_name, lcnames, rvnames, file_prefix, path = 'data/', figure_dimensions = (17, 20), transitplot_ylim = None, transitplot_spacing = None, 
                plot_atmosphere = True, MIST = False, MIST_path = None, split_pdf = False, MIST_plotlimits = None, MIST_textoffset = None, save = True, file_extension = 'pdf'):
    '''
    object_name: a string containing the planet's name. Ex: '1855' for toi-1855

    lcnames: array of strings naming the telescopes used to collect the light curves in order of the observation date. If it is
    a tess light curve, write TESS and then the exposure time in seconds. Ex: ['TESS 1800', 'SOAR']

    rvnames: array of strings with the names of the instruments that obtained the RVs in alphabetical order. Ex: ['CHIRON (fiber)', 'CHIRON (slicer)']

    file_prefix: a string containing the prefix that the EXOFASTv2 output files have. Ex: 'toi3129' or '4711'

    path: a string containing the path to the input files. by default, it is assumed that the files are in a folder labeled 'data/' within the 
    working directory.

    figure_dimensions: a tuple containing the chosen dimensions for the figure in any unit in the format (width, height). Ex: (17, 20)

    transitplot_ylim: an array containing the y-axis limits of the transit plot that can be passed if there are significant outliers in the 
    lightcurves. Should be in the format [ymin, ymax].

    transitplot_spacing: custom spacing parameter to separate the lightcurves in the transit plot

    plot_atmosphere: a boolean to determine whether or not to plot the stellar atmosphere on the SED plot

    MIST: a boolean to determine whether or not the MIST evolution plot is being plotted

    split_pdf: a boolean representing whether or not the target has a split PDF because of a bimodality. If true, there must be a directory 
    called TOI-###_splitpdf with the EXOFAST outputs

    MIST_plotlimits: a dictionary containing the x limits and y limits of the MIST plot. Ex: ([8000, 4000], [5, 2.5])

    MIST_textoffset: a float or integer representing the Teff offset that the reference age text should have, relative to the blue points

    save: a boolean for whether or not the function should save the figure in a directory named 'output'

    file_extension: the chosen file extension for the output figure. Ex: 'pdf' or 'png'
    '''

    # Determining which TESS data are being used
    TESS_exptimes = []
    for x in lcnames:
        if 'TESS' in x:
            TESS_exptimes.append(str([int(s) for s in x.split() if s.isdigit()][0]))
    TESS_exptimes = [exptime for i, exptime in enumerate(TESS_exptimes) if exptime not in TESS_exptimes[:i]] # removing duplicates

    ####################
    # LOADING IN DATA
    ####################

    # loading median file
    median_names = ['parname', 'median_value', 'upper_errorbar', 'lower_errorbar', 'scinot']
    median = pd.read_csv(f'{path}{file_prefix}.median.csv', skiprows=1, header=None, names=median_names)

    period_median = median_scinot_corrections(median, 'Period_0')[0] * u.day
    t14_median = median_scinot_corrections(median, 't14_0')[0] * u.day
    t14_median = (t14_median.to(u.hr)).value
    planetmass_median = median_scinot_corrections(median, 'mp_0')[0]
    planetradius_median = median_scinot_corrections(median, 'rp_0')[0]
    eccentricity_median = median_scinot_corrections(median, 'e_0')[0]

    # Extracting best-fit parameters from mcmc sav files

    savfile = readsav(f'{path}{file_prefix}.mcmc.idl')
    mcmcss = savfile['mcmcss']
    slope_best = mcmcss.star[0].slope[0].best[0]
    if np.isnan(slope_best):
        slope_best = 0
    period_best = mcmcss.planet[0].period[0].best[0] * u.day
    epoch_best = mcmcss.planet[0].tc[0].best[0] * u.day
    rvepoch_best = mcmcss.rvepoch[0]
    t14_best = mcmcss.planet[0].t14[0].best[0] * u.day
    t14_best = (t14_best.to(u.hr)).value
    

    # collect per-instrument gamma and jitter
    gamma = {}
    jitter = {}
    for i in range(len(rvnames)):
        gamma[i] = median_scinot_corrections(median, f'gamma_{i}')[0]
        jitter[i] = median_scinot_corrections(median, f'jitter_{i}')[0]

    # loading mcmc transit files

    model_names = ['time', 'flux']
    residual_names = ['time', 'residuals', 'error']

    pretty_TESS_1800 = pd.DataFrame(columns=model_names)
    detrended_TESS_1800 = pd.DataFrame(columns=model_names)
    residuals_TESS_1800 = pd.DataFrame(columns=residual_names)

    pretty_TESS_600 = pd.DataFrame(columns=model_names)
    detrended_TESS_600 = pd.DataFrame(columns=model_names)
    residuals_TESS_600 = pd.DataFrame(columns=residual_names)

    pretty_TESS_200 = pd.DataFrame(columns=model_names)
    detrended_TESS_200 = pd.DataFrame(columns=model_names)
    residuals_TESS_200 = pd.DataFrame(columns=residual_names)

    pretty_TESS_120 = pd.DataFrame(columns=model_names)
    detrended_TESS_120 = pd.DataFrame(columns=model_names)
    residuals_TESS_120 = pd.DataFrame(columns=residual_names)

    pretty_TESS_20 = pd.DataFrame(columns=model_names)
    detrended_TESS_20 = pd.DataFrame(columns=model_names)
    residuals_TESS_20 = pd.DataFrame(columns=residual_names)

    followup_name_index = [] # initializing follow-up lc name index to assign names at the end
    f = 0 # integer required to track the number of follow-up lightcurves

    # dictionaries to hold follow-up dataframes
    pretty_followup = {}
    detrended_followup = {}
    residuals_followup = {}

    for i in range(len(lcnames)):
        lc_index = f'{i:03d}'

        pretty_model = pd.read_csv(f'{path}{file_prefix}.mcmc.prettymodel.transit_{lc_index}.planet_00.txt', sep=r'\s+', header=None, names=model_names)
        detrended_model = pd.read_csv(f'{path}{file_prefix}.mcmc.detrendedmodel.transit_{lc_index}.planet_00.txt', sep=r'\s+', header=None, names=model_names)
        residual = pd.read_csv(f'{path}{file_prefix}.mcmc.residuals.transit_{lc_index}.txt', sep=r'\s+', header=None, names=residual_names)

        if lcnames[i] == 'TESS 1800':
            # the "if not pretty_TESS_1800.empty else None" was added to concatenate None instead of an empty df
            pretty_TESS_1800 = pd.concat([pretty_TESS_1800 if not pretty_TESS_1800.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_1800 = pd.concat([detrended_TESS_1800 if not detrended_TESS_1800.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_1800 = pd.concat([residuals_TESS_1800 if not residuals_TESS_1800.empty else None, residual], ignore_index = True)
        elif lcnames[i] == 'TESS 600':
            pretty_TESS_600 = pd.concat([pretty_TESS_600 if not pretty_TESS_600.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_600 = pd.concat([detrended_TESS_600 if not detrended_TESS_600.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_600 = pd.concat([residuals_TESS_600 if not residuals_TESS_600.empty else None, residual], ignore_index = True)
        elif lcnames[i] == 'TESS 200':
            pretty_TESS_200 = pd.concat([pretty_TESS_200 if not pretty_TESS_200.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_200 = pd.concat([detrended_TESS_200 if not detrended_TESS_200.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_200 = pd.concat([residuals_TESS_200 if not residuals_TESS_200.empty else None, residual], ignore_index = True)  
        elif lcnames[i] == 'TESS 120':
            pretty_TESS_120 = pd.concat([pretty_TESS_120 if not pretty_TESS_120.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_120 = pd.concat([detrended_TESS_120 if not detrended_TESS_120.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_120 = pd.concat([residuals_TESS_120 if not residuals_TESS_120.empty else None, residual], ignore_index = True)
        elif lcnames[i] == 'TESS 20':
            pretty_TESS_20 = pd.concat([pretty_TESS_20 if not pretty_TESS_20.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_20 = pd.concat([detrended_TESS_20 if not detrended_TESS_20.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_20 = pd.concat([residuals_TESS_20 if not residuals_TESS_20.empty else None, residual], ignore_index = True)             
        else:
            pretty_followup[f] = pretty_model
            detrended_followup[f] = detrended_model
            residuals_followup[f] = residual
            followup_name_index.append(i)
            f += 1

    # Initializing RV dictionaries
    residuals_rv = {}
    model_rv = {}
    model_names_rv = ['time', 'rv']
    pretty_rv = pd.read_csv(f'{path}{file_prefix}.mcmc.prettymodelrv.planet.00.txt', sep=r'\s+', names=model_names_rv)
    pretty_rv['rv_trend'] = pretty_rv.rv + slope_best * (pretty_rv.time - rvepoch_best) # accounting for slope
    pretty_rv['phase'] = ((pretty_rv.time - epoch_best)/period_best.value) - np.floor((pretty_rv.time - epoch_best)/period_best.value+0.5)
    pretty_rv_phasesorted = pretty_rv.sort_values('phase')

    residuals_names_rv = ['time', 'residual', 'error']
    for i in range(len(rvnames)):
        residuals_rv[i] = pd.read_csv(f'{path}{file_prefix}.mcmc.residuals.telescope_0{i}.txt', sep=r'\s+', names=residuals_names_rv)
        residuals_rv[i]['phase'] = ((residuals_rv[i].time - epoch_best)/period_best.value) - np.floor((residuals_rv[i].time - epoch_best)/period_best.value+0.5)
        model_rv[i] = pd.read_csv(f'{path}{file_prefix}.mcmc.detrendedmodel.telescope_0{i}.txt', sep=r'\s+', names=model_names_rv)

    # Loading in SED data
    sed_cols = ['filtername', 'wavelength', 'half_bandpass', 'measured_flux', 'error', 'model_flux', 'residual', 'star_index']
    sed_residuals = pd.read_csv(f'{path}{file_prefix}.mcmc.sed.residuals.txt', sep=r'\s+', skiprows=1, header=None, names = sed_cols)
    # Loading in atmosphere model
    atmosphere_cols = ['wavelength', 'flux'] # wavelength/flux are in the same units as SED. 
    atmosphere = pd.read_csv(f'{path}{file_prefix}.mcmc.atmosphere.000.txt', sep=r'\s+', header=None, names = atmosphere_cols)

    ####################
    # MANIPULATING DATA
    ####################

    # Generating lightcurves for the TESS data
    dic_TESS_1800 = {'time': detrended_TESS_1800.time, 'flux': detrended_TESS_1800.flux + residuals_TESS_1800.residuals + 1}
    lc_TESS_1800 = pd.DataFrame(dic_TESS_1800)
    dic_TESS_600 = {'time': detrended_TESS_600.time, 'flux': detrended_TESS_600.flux + residuals_TESS_600.residuals + 1}
    lc_TESS_600 = pd.DataFrame(dic_TESS_600)
    dic_TESS_200 = {'time': detrended_TESS_200.time, 'flux': detrended_TESS_200.flux + residuals_TESS_200.residuals + 1}
    lc_TESS_200 = pd.DataFrame(dic_TESS_200)
    dic_TESS_120 = {'time': detrended_TESS_120.time, 'flux': detrended_TESS_120.flux + residuals_TESS_120.residuals + 1}
    lc_TESS_120 = pd.DataFrame(dic_TESS_120)
    dic_TESS_20 = {'time': detrended_TESS_20.time, 'flux': detrended_TESS_20.flux + residuals_TESS_20.residuals + 1}
    lc_TESS_20 = pd.DataFrame(dic_TESS_20)

    # also create dicts for TESS lightcurves/models for easier access
    lc_TESS = {}
    pretty_TESS = {}
    lc_TESS['1800'] = lc_TESS_1800
    lc_TESS['600'] = lc_TESS_600
    lc_TESS['200'] = lc_TESS_200
    lc_TESS['120'] = lc_TESS_120
    lc_TESS['20'] = lc_TESS_20

    pretty_TESS['1800'] = pretty_TESS_1800
    pretty_TESS['600'] = pretty_TESS_600
    pretty_TESS['200'] = pretty_TESS_200
    pretty_TESS['120'] = pretty_TESS_120
    pretty_TESS['20'] = pretty_TESS_20

    # Generating lightcurves for the followup data
    # Generating lightcurves for the followup data using dictionaries
    lc_followup = {}
    for i in range(len(followup_name_index)):
        dic_followup = {'time': detrended_followup[i].time, 'flux': detrended_followup[i].flux + residuals_followup[i].residuals + 1}
        lc_followup[i] = pd.DataFrame(dic_followup)

    # Phase-folding the TESS data
    for exptime in TESS_exptimes:
        lc_TESS[exptime].time = t_phase_folded(lc_TESS[exptime].time, period_best, epoch_best)*period_best.to(u.hr)
        pretty_TESS[exptime].time = t_phase_folded(pretty_TESS[exptime].time, period_best, epoch_best)*period_best.to(u.hr)

    # Phase-folding the followup data
    for i in range(len(followup_name_index)):
        lc_followup[i].time = t_phase_folded(lc_followup[i].time, period_best, epoch_best)*period_best.to(u.hr)
        pretty_followup[i].time = t_phase_folded(pretty_followup[i].time, period_best, epoch_best)*period_best.to(u.hr)

    # # binning the data
    # binwidth = t14 / 10

    # nbins_TESS = int((np.max(lc_TESS.time) - np.min(lc_TESS.time))/binwidth)

    # lc_TESS_binned_time = stats.binned_statistic(lc_TESS.time, lc_TESS.time, bins = nbins_TESS)[0]
    # lc_TESS_binned_flux = stats.binned_statistic(lc_TESS.time, lc_TESS.flux, bins = nbins_TESS)[0]

    # nbins_followup = int((np.max(lc_followup.time) - np.min(lc_followup.time))/binwidth)

    # lc_followup_binned_time = stats.binned_statistic(lc_followup.time, lc_followup.time, bins = nbins_followup)[0]
    # lc_followup_binned_flux = stats.binned_statistic(lc_followup.time, lc_followup.flux, bins = nbins_followup)[0]

    # # median combining and cleaning up the TESS model
    # pretty_TESS = pretty_TESS.sort_values('time')
    # pretty_TESS_binned_time = stats.binned_statistic(pretty_TESS.time, pretty_TESS.time, bins = nbins_TESS * 2)[0]
    # pretty_TESS_binned_flux = stats.binned_statistic(pretty_TESS.time, pretty_TESS.flux, bins = nbins_TESS * 2)[0]

    # Calculating RVs
    rvs = {}
    rvs_trendsubtracted = {}
    rvs_error = {}
    for i in range(len(rvnames)):
        rvs_trendsubtracted[i] = model_rv[i].rv + residuals_rv[i].residual - slope_best * (residuals_rv[i].time - rvepoch_best)
        rvs[i] = model_rv[i].rv + residuals_rv[i].residual
        rvs_error[i] = (residuals_rv[i].error**2 + jitter[i])**(1/2)
    
    # Bandpass effective widths to represent the errors of the SED plot (from the SVO filter profile service)
    # [Gaia G, Gbp, Grp, 2MASS J, H, Ks, WISE W1, W2, W3]
    Weff = np.array([4052.97, 2157.50, 2924.44, 1624.32, 2509.40, 2618.87, 6626.42, 10422.66, 55055.23]) * u.AA
    Weff = Weff / 2 # uncertainty should be half of the effective width
    Weff = (Weff.to(u.micron)).value

    ####################
    # PLOTTING
    ####################

    # defining a list of colors to choose from when plotting. Feel free to change or add more!
    colors = ['#009B77', '#821EA6', '#34568B', '#D1AF19', '#95DEE3', '#88B04B', '#955251', '#5B5EA6', '#9B2335', '#E6AF91', '#D65076', '#422C7A', '#378011']

    # Setting up the figure
    fig, axs = plt.subplots(ncols=2, nrows=3)
    fig.set_figwidth(figure_dimensions[0])
    fig.set_figheight(figure_dimensions[1])
    fig.suptitle(f'{object_name}\n', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.95, f'$P$ = {round(period_median.value, 3)} d | $R_P$ = {planetradius_median} R$_J$ | $M_P$ = {planetmass_median} M$_J$ | $e$ = {eccentricity_median}', ha='center'\
             , fontsize=17)   

    # Making the top left plot take up two spaces
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:2, 0]:
        ax.remove()
    ax1 = fig.add_subplot(gs[0:2, 0])

    # Transits (top left)
    TESS_lc_index = 0 # keeps track of the TESS lightcurves to space each lightcurve

    # setting the spacing of the transit lightcurves
    lightcurve_spacing = 0.025 # default spacing
    if transitplot_spacing != None:
        lightcurve_spacing = transitplot_spacing

    for exptime in TESS_exptimes:
        pretty_TESS[exptime].sort_values('time', inplace=True) # sorting model

        # separates each lightcurve by the given lightcurve spacing
        ax1.plot(pretty_TESS[exptime].time, pretty_TESS[exptime].flux + 1 - lightcurve_spacing * TESS_lc_index, c = 'k', zorder=2, linewidth=3)
        ax1.scatter(lc_TESS[exptime].time, lc_TESS[exptime].flux - lightcurve_spacing * TESS_lc_index, ls = 'None', c=colors[TESS_lc_index], s = 100, alpha = 0.7, \
            edgecolors='#000000', zorder = 1, label = 'TESS ' + exptime + 's')
        # tess=ax1.scatter(lc_TESS_binned_time, lc_TESS_binned_flux, ls = 'None', c='#9F3BC2', s = 125, alpha = 1, \
                # edgecolors='#000000', zorder = 10, label='TESS')
        TESS_lc_index += 1

    for i in range(len(followup_name_index)):
        pretty_followup[i].sort_values('time', inplace=True) # sorting model

        ax1.plot(pretty_followup[i].time, pretty_followup[i].flux + 1 - lightcurve_spacing * (i + TESS_lc_index), c = 'k', zorder=2, linewidth=3)
        ax1.scatter(lc_followup[i].time, lc_followup[i].flux - lightcurve_spacing * (i + TESS_lc_index), ls = 'None', c=colors[i + len(TESS_exptimes)], s = 100, alpha = 0.7, \
            edgecolors='#000000', zorder = 1, label = lcnames[followup_name_index[i]])
        # followup=ax1.scatter(lc_followup_binned_time, lc_followup_binned_flux-lightcurve_spacing, ls = 'None', c='#7BA2F1', \
                # s = 125, alpha = 1, edgecolors='#000000', zorder = 10, label='TFOP')

    ax1.legend(loc = 4, fontsize = 15) # add legend row for EXOFASTv2 model?

    xmin = -t14_best/2 - 1 # setting x limits based on the transit duration
    xmax = t14_best/2 + 1
    ax1.set_xlim([xmin,xmax])
    ymin = 1 - lightcurve_spacing * (i + TESS_lc_index + 1) - 0.0125
    ymax = 1.0125
    if transitplot_ylim != None:
        ax1.set_ylim(transitplot_ylim)

    ax1.set_xlabel('Time Since Conjunction [Hours]', fontsize = 20)
    ax1.set_ylabel('Normalized Flux + Constant', fontsize = 20)

    ax1.tick_params(which = 'both', direction = 'inout', top=True, right=True)
    ax1.tick_params(labelsize = 20, length = 10, width=2)
    ax1.tick_params(which = 'minor', length = 7, width = 1)

    # RVs vs. Time (top right)

    axs[0, 1].remove()
    nested_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax2_upper = fig.add_subplot(nested_gs[0])
    ax2_lower = fig.add_subplot(nested_gs[1])

    # concatenate time axes
    rvtimes = []
    for i in range(len(rvnames)):
        rvtimes.append(residuals_rv[i].time)
    rvtimes = np.concatenate(rvtimes)
    rvtimes = np.sort(rvtimes) # sorting the time axis

    # # identify gaps of >200 days in the RV data
    # gaps = np.where(np.diff(rvtimes) > 200)[0]
    # xlims = []
    # for i in range(len(gaps)):
    #     if i == 0:
    #         xlims.append((rvtimes[0], rvtimes[gaps[i]]))
    #     elif i == len(gaps) - 1:
    #         xlims.append((rvtimes[gaps[i] + 1], rvtimes[-1]))
    #     else:
    #         xlims.append((rvtimes[gaps[i] + 1], rvtimes[gaps[i + 1]]))
    # print(xlims)
    # ax2_upper = brokenaxes(xlims=xlims, subplot_spec=nested_gs[0], hspace=.05)
    # ax2_lower = brokenaxes(xlims=xlims, subplot_spec=nested_gs[1], hspace=.05)

    ax2_upper.set_xticks([]) # added to remove background ticks from the removed axes
    ax2_upper.set_xlim(np.min(rvtimes - 2457000) - 5, np.max(rvtimes - 2457000) + 5)
    ax2_lower.set_xlim(np.min(rvtimes - 2457000) - 5, np.max(rvtimes - 2457000) + 5)
    ax2_lower.set_xlabel('Time [BJD$_{\mathrm{TDB}} - 2457000$]', fontsize = 20)
    ax2_upper.set_ylabel('RV [m/s]', fontsize = 20)
    ax2_lower.set_ylabel('O-C', fontsize = 16)

    ax2_upper.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2, top=True, right=True)
    ax2_lower.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2, top=True, right=True)

    ax2_upper.plot(pretty_rv.time - 2457000, pretty_rv.rv_trend, c = 'k', zorder=1, lw = 0.15, label = 'EXOFASTv2', alpha=1)

    max_rv = [] # keeping track of the max and min rv from each dataset to set plot limits
    min_rv = []
    for i in range(len(rvnames)):
        max_rv.append(np.max(rvs[i]))
        min_rv.append(np.min(rvs[i]))

        ax2_upper.errorbar(residuals_rv[i].time - 2457000, rvs[i], yerr=rvs_error[i], fmt='o', mfc=colors[-(i+1)], 
                        mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None', label=rvnames[i], zorder=10)
        ax2_lower.errorbar(residuals_rv[i].time - 2457000, residuals_rv[i].residual, yerr=rvs_error[i], 
                        fmt='o', mfc=colors[-(i+1)], mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None', zorder=10)
    
    # ax2_upper.set_ylim(np.min(min_rv) - 100, np.max(max_rv) + 100)
    ax2_lower.axhline(0, ls='--', c='grey', lw = 2)
    # ax2_upper.legend(fontsize = 15)

    # RVs vs. Phase (bottom right)

    axs[1, 1].remove()
    nested_gs = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax3_upper = fig.add_subplot(nested_gs[0])
    ax3_lower = fig.add_subplot(nested_gs[1])

    ax3_upper.set_xticks([]) # added to remove background ticks from the removed axes
    ax3_upper.set_xlim(-0.5, 0.5)
    ax3_lower.set_xlim(-0.5, 0.5)
    ax3_lower.set_xlabel('Phase + Offset', fontsize = 20)
    ax3_upper.set_ylabel('RV [m/s]', fontsize = 20)
    ax3_lower.set_ylabel('O-C', fontsize = 16)

    ax3_upper.tick_params(which = 'major', direction = 'inout', labelsize = 20, length = 10, width=2, top=True, right=True)
    ax3_lower.tick_params(which = 'major', direction = 'inout', labelsize = 20, length = 10, width=2, top=True, right=True)

    ax3_upper.plot(pretty_rv_phasesorted.phase, pretty_rv_phasesorted.rv, c = 'k', zorder=10, lw = 2, label = 'EXOFASTv2', alpha=0.7)

    for i in range(len(rvnames)):
        ax3_upper.errorbar(residuals_rv[i].phase, rvs_trendsubtracted[i], yerr=rvs_error[i], 
                           fmt='o', mfc=colors[-(i+1)], mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None', label=rvnames[i])
        ax3_lower.errorbar(residuals_rv[i].phase, residuals_rv[i].residual, yerr=rvs_error[i], fmt='o', mfc=colors[-(i+1)], mec='k',
                ecolor=colors[-(i+1)], capsize=4, ls='None')

    ax3_lower.axhline(0, ls='--', c='grey', lw = 2)
    ax3_upper.legend(fontsize = 15, loc = 1)

    # SED Plot (bottom left)

    axs[2, 0].remove()
    nested_gs = gs[2, 0].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax4_upper = fig.add_subplot(nested_gs[0])
    ax4_lower = fig.add_subplot(nested_gs[1])
    ax4_upper.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$]', fontsize = 20)
    ax4_lower.set_xlabel('Wavelength [$\mu$m]', fontsize = 20)
    ax4_lower.set_ylabel('O-C', fontsize = 16)

    ax4_upper.tick_params(which = 'major', direction = 'inout', labelsize = 20, length = 10, width=2, top=True, right=True)
    ax4_upper.tick_params(which = 'minor', top=True, right=True)
    ax4_upper.set_xticks([])
    ax4_lower.tick_params(which = 'major', direction = 'inout', labelsize = 20, length = 10, width=2, top=True, right=True)
    ax4_lower.tick_params(which = 'minor', top=True, right=True)

    ax4_upper.errorbar(sed_residuals.wavelength, sed_residuals.measured_flux, yerr=sed_residuals.error,\
                       xerr=sed_residuals.half_bandpass, fmt='.', markersize=8, mfc=colors[-1], mec=colors[-1],\
                       ecolor=colors[-1], capsize=4, ls='None', label = 'Observations') # Are x errors the width of the wavelength band or where do we get these?

    ax4_upper.scatter(sed_residuals.wavelength, sed_residuals.model_flux, marker='o', color='k', label='EXOFASTv2')

    ax4_lower.errorbar(sed_residuals.wavelength, sed_residuals.residual, yerr=sed_residuals.error,\
                       xerr=sed_residuals.half_bandpass, fmt='.', markersize=8, mfc=colors[-1], mec=colors[-1],\
                       ecolor=colors[-1], capsize=4, ls='None', label = 'Observations') # Are x errors the width of the wavelength band or where do we get these?

    ax4_lower.axhline(0, ls='--', color='grey', lw = 2)

    ax4_upper.set_yscale('log')
    ax4_upper.set_xscale('log')
    ax4_lower.set_xscale('log')

    ax4_upper.legend(fontsize = 15)

    if plot_atmosphere:
        # ax4_upper.autoscale(False)
        ax4_upper.plot(atmosphere.wavelength, smooth(atmosphere.flux, 9), color='grey', linewidth=1, zorder=0, scaley=False) # smoothed atmosphere w/ window size of 9
        xmin, xmax = ax4_upper.get_xlim()
        ax4_lower.set_xlim(xmin, xmax)

    # MIST Plot (bottom right)
    if MIST:
        ax5 = axs[2, 1]

        toinumber = object_name[4:] # assuming the object is a TOI and named TOI-XXXX

        if MIST_path == None:
            MIST_path = path

        blackline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_Black.dat', sep=r'\s+', header=None)
        blueline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_Blue.dat', sep=r'\s+', header=None)
        greenline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_Green.dat', sep=r'\s+', header=None)

        ref_ages = pd.read_csv(f'' + path + 'TOI' + toinumber + '_age.dat', sep=r'\s+', header=None)

        if split_pdf == False:
            logg, logg_E, logg_e = median_scinot_corrections(median, 'logg_0')

            teff, teff_E, teff_e = median_scinot_corrections(median, 'teff_0')
        else:
            median_names = ['parname', 'median_value', 'upper_errorbar', 'lower_errorbar', 'scinot']
            lowmass_median = pd.read_csv(f'{path}bimodalities/{file_prefix}.lowmass.csv', skiprows=1, header=None, names=median_names)
            highmass_median = pd.read_csv(f'{path}bimodalities/{file_prefix}.highmass.csv', skiprows=1, header=None, names=median_names)
            logg_low, logg_low_E, logg_low_e = median_scinot_corrections(lowmass_median, 'logg_0')

            teff_low, teff_low_E, teff_low_e = median_scinot_corrections(lowmass_median, 'teff_0')

            logg_high, logg_high_E, logg_high_e = median_scinot_corrections(highmass_median, 'logg_0')

            teff_high, teff_high_E, teff_high_e = median_scinot_corrections(highmass_median, 'teff_0')

        # default plot limits
        if split_pdf == False:
            default_xlim = [teff + 1400, teff - 500]
            default_ylim = [logg + 0.5, logg - 0.5]
        else:
            default_xlim = [teff_high + 1400, teff_high - 500]
            default_ylim = [logg_high + 0.5, logg_high - 0.5]

        # remove ref_ages that are outside the plot bounds
        i = 0
        if MIST_plotlimits != None:
            for i in range(len(ref_ages)):
                if (ref_ages[1][i] > MIST_plotlimits[0][0] or ref_ages[1][i] < MIST_plotlimits[0][1]) or (ref_ages[2][i] > MIST_plotlimits[1][0] or ref_ages[2][i] < MIST_plotlimits[1][1]):
                    ref_ages = ref_ages.drop(i)
        else:
            for i in range(len(ref_ages)):
                if (ref_ages[1][i] > default_xlim[0] or ref_ages[1][i] < default_xlim[1]) or (ref_ages[2][i] > default_ylim[0] or ref_ages[2][i] < default_ylim[1]):
                    ref_ages = ref_ages.drop(i)

        ax5.plot(blackline[0], blackline[1], 'k', linewidth=1.5) # 1 and 2 sigma contours of the current log g and teff from MIST isochrones only
        ax5.plot(blueline[0], blueline[1], 'b', linewidth=3, zorder=7) # MIST track for the best-fit stellar mass
        ax5.plot(greenline[0], greenline[1], 'g') # 1 and 2 sigma contours of the log g and teff from MIST isochrones and the global fit

        if split_pdf == False:
            ax5.errorbar(teff, logg, yerr=[[logg_e], [logg_E]], xerr=[[teff_e], [teff_E]], ecolor='r', capsize=3, linewidth=0, elinewidth=2, zorder=8)
        else:
            low = ax5.errorbar(teff_low, logg_low, yerr=[[logg_low_e], [logg_low_E]], xerr=[[teff_low_e], [teff_low_E]], ecolor='purple', capsize=3, linewidth=0, 
                               elinewidth=2, label=r'Lower $M_{\star}$ solution', zorder=8)
            high = ax5.errorbar(teff_high, logg_high, yerr=[[logg_high_e], [logg_high_E]], xerr=[[teff_high_e], [teff_high_E]], ecolor='orange', capsize=3, linewidth=0,
                                elinewidth=2, label=r'Higher $M_{\star}$ solution', zorder=8)

        ax5.scatter(ref_ages[1], ref_ages[2], color='b', s=50, zorder=9)

        if MIST_textoffset == None:
            for i in range(len(ref_ages)):
                ax5.text(ref_ages[1][i] + 120, ref_ages[2][i] + 0.005, str(ref_ages[0][i]) + ' Gyr', color='b', fontweight='bold', fontsize=10, zorder=10)
        else:
            for i in range(len(ref_ages)):
                ax5.text(ref_ages[1][i] + MIST_textoffset, ref_ages[2][i] + 0.005, str(ref_ages[0][i]) + ' Gyr', color='b', fontweight='bold', fontsize=10, zorder=10)

        if MIST_plotlimits != None:
            ax5.set_xlim(MIST_plotlimits[0])
            ax5.set_ylim(MIST_plotlimits[1])
        else:
            ax5.set_xlim(default_xlim)
            ax5.set_ylim(default_ylim)

        ax5.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2, top=True, right=True)

        ax5.set_xlabel(r'T$_{\rm eff}$ [K]', fontsize = 20)
        ax5.set_ylabel('log g$_*$ [cgs]', fontsize = 20)
        if split_pdf:
            ax5.legend(handles=[low, high], fontsize=15)

    # Add some space between subplots
    plt.tight_layout(pad=2.5, h_pad=1)

    # Saves the figure to your local directory in a subdirectory named 'output'
    if save:
        if os.path.exists('output') == False:
            os.mkdir('output')
        plt.savefig(f'output/fullpagefig_{object_name}.{file_extension}', bbox_inches='tight', facecolor='white', transparent=False)