import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as const
from matplotlib.colors import Normalize as Normalize
import matplotlib.cm
import scipy.interpolate as interpolate
from scipy import stats
from matplotlib.gridspec import GridSpec
import os
from scipy.io import readsav

def update_fit_files(target, file_prefix = '.MIST.SED.', output_dir = 'data/'):

    hpcc_path = 'jschulte@rsync.hpcc.msu.edu:/mnt/research/Exoplanet_Lab/jack/Global_Fits/'

    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.idl ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.detrendedmodel.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.model.telescope* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.prettymodel.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.residuals.telescope* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.residuals.transit* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.rv.ps.prettymodelrv* ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'mcmc.sed.residuals.txt ' + output_dir)
    os.system(f'scp ' + hpcc_path + target + '/fitresults_bestfit/' + target + file_prefix + 'median.csv ' + output_dir)

def t_phase_folded(t, per, t0):
    t_phase_folded = (t - t0)/per - np.floor((t - t0)/per + 0.5) # centers on zero
    return t_phase_folded

def gen1pagefig(object_name, lcnames, rvnames, path = 'data/', file_prefix = '.MIST.SED.', transitplot_ylim = None, transitplot_spacing = None, MIST = False, MIST_plotlimits = None,
                MIST_textoffset = None, save = True):
    '''
    object_name: a string containing the planet's name. Ex: '1855' for toi-1855

    lcnames: array of strings naming the telescopes used to collect the light curves in order of the observation date. If it is
    a tess light curve, write TESS and then the exposure time in seconds. Ex: ['TESS 1800', 'SOAR']

    rvnames: array of strings with the names of the instruments that obtained the RVs in alphabetical order. Ex: ['CHIRON (fiber)', 'CHIRON (slicer)']

    path: a string containing the path to the input files. by default, it is assumed that the files are in a folder labeled 'data/' within the working directory.

    file_prefix: a string containing the prefix that the EXOFASTv2 output files have, following the object name. Ex: '.MIST.SED.'

    transitplot_ylim: an array containing the y-axis limits of the transit plot that can be passed if there are significant outliers in the lightcurves. Should be in the format [ymin, ymax].

    transitplot_spacing: custom spacing parameter to separate the lightcurves in the transit plot

    MIST: a boolean to determine whether or not the MIST evolution plot is being plotted

    MIST_plotlimits: a dictionary containing the x limits and y limits of the MIST plot. Ex: ([8000, 4000], [5, 2.5])

    MIST_textoffset: a float or integer representing the Teff offset that the reference age text should have, relative to the blue points

    save: a boolean for whether or not the function should save the figure in a directory named 'output'
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
    median = pd.read_csv(f'{path}{object_name}{file_prefix}median.csv')

    period_median = median[' median value'][median['#parname'] == 'Period_0'].iloc[0] * u.day
    t14_median = median[' median value'][median['#parname'] == 't14_0'].iloc[0] * u.day
    t14_median = (t14_median.to(u.hr)).value
    planetmass_median = median[' median value'][median['#parname'] == 'mp_0'].iloc[0]
    planetradius_median = median[' median value'][median['#parname'] == 'rp_0'].iloc[0]
    eccentricity_median = median[' median value'][median['#parname'] == 'e_0'].iloc[0]

    # Extracting best-fit parameters from mcmc sav files

    savfile = readsav(f'{path}{object_name}{file_prefix}mcmc.idl')
    mcmcss = savfile['mcmcss']
    slope_best = mcmcss.star[0].slope[0].best[0]
    if np.isnan(slope_best) == True:
        slope_best = 0
    period_best = mcmcss.planet[0].period[0].best[0] * u.day
    epoch_best = mcmcss.planet[0].tc[0].best[0] * u.day
    rvepoch_best = mcmcss.rvepoch[0]
    t14_best = mcmcss.planet[0].t14[0].best[0] * u.day
    t14_best = (t14_best.to(u.hr)).value
    

    for i in range(len(rvnames)): # should we be using the best-fit gamma and jitter?
        gamma_varname = f'gamma_{i}'
        jitter_varname = f'jitter_{i}'

        locals()[gamma_varname] = median[' median value'][median['#parname'] == f'gamma_{i}'].iloc[0] # assigning new variable gamma_{i}
        locals()[jitter_varname] = median[' median value'][median['#parname'] == f'jitter_{i}'].iloc[0]

    # loading mcmc transit files

    model_names = ['time', 'flux']
    residual_names = ['time', 'residuals', 'error']

    pretty_TESS_1800 = pd.DataFrame(columns=model_names)
    detrended_TESS_1800 = pd.DataFrame(columns=model_names)
    residuals_TESS_1800 = pd.DataFrame(columns=residual_names)

    pretty_TESS_600 = pd.DataFrame(columns=model_names)
    detrended_TESS_600 = pd.DataFrame(columns=model_names)
    residuals_TESS_600 = pd.DataFrame(columns=residual_names)

    pretty_TESS_120 = pd.DataFrame(columns=model_names)
    detrended_TESS_120 = pd.DataFrame(columns=model_names)
    residuals_TESS_120 = pd.DataFrame(columns=residual_names)

    followup_name_index = [] # initializing follow-up lc name index to assign names at the end
    f = 0 # integer required to track the number of follow-up lightcurves

    for i in range(len(lcnames)):
        lc_index = f'{i:03d}'

        pretty_model = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.prettymodel.transit_{lc_index}.planet_00.txt', sep='\s+', header=None, names=model_names)
        detrended_model = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.detrendedmodel.transit_{lc_index}.planet_00.txt', sep='\s+', header=None, names=model_names)
        residual = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.residuals.transit_{lc_index}.txt', sep='\s+', header=None, names=residual_names)

        if lcnames[i] == 'TESS 1800':
            # the "if not pretty_TESS_1800.empty else None" was added to concatenate None instead of an empty df
            pretty_TESS_1800 = pd.concat([pretty_TESS_1800 if not pretty_TESS_1800.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_1800 = pd.concat([detrended_TESS_1800 if not detrended_TESS_1800.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_1800 = pd.concat([residuals_TESS_1800 if not residuals_TESS_1800.empty else None, residual], ignore_index = True)
        elif lcnames[i] == 'TESS 600':
            pretty_TESS_600 = pd.concat([pretty_TESS_600 if not pretty_TESS_600.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_600 = pd.concat([detrended_TESS_600 if not detrended_TESS_600.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_600 = pd.concat([residuals_TESS_600 if not residuals_TESS_600.empty else None, residual], ignore_index = True)
        elif lcnames[i] == 'TESS 120':
            pretty_TESS_120 = pd.concat([pretty_TESS_120 if not pretty_TESS_120.empty else None, pretty_model], ignore_index = True)
            detrended_TESS_120 = pd.concat([detrended_TESS_120 if not detrended_TESS_120.empty else None, detrended_model], ignore_index = True)
            residuals_TESS_120 = pd.concat([residuals_TESS_120 if not residuals_TESS_120.empty else None, residual], ignore_index = True)         
        else:
            pretty_varname = f'pretty_followup_{f}'
            detrended_varname = f'detrended_followup_{f}'
            residuals_varname = f'residuals_followup_{f}'
            locals()[pretty_varname] = pretty_model # assigning new variable pretty_followup_{f} = pretty_model
            locals()[detrended_varname] = detrended_model
            locals()[residuals_varname] = residual
            followup_name_index.append(i)
            f += 1

    for i in range(len(rvnames)):
        residuals_rv_varname = f'residuals_rv_{i}'
        model_rv_varname = f'model_rv_{i}'

        model_names_rv = ['time', 'rv']
        pretty_rv = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.rv.ps.prettymodelrv.planet.00.txt', sep='\s+', names=model_names_rv)
        pretty_rv['rv_trend'] = pretty_rv.rv + slope_best * (pretty_rv.time - rvepoch_best) # accounting for slope
        pretty_rv['phase'] = ((pretty_rv.time - epoch_best)/period_best.value) - np.floor((pretty_rv.time - epoch_best)/period_best.value+0.5)
        pretty_rv_phasesorted = pretty_rv.sort_values('phase')

        residuals_names_rv = ['time', 'residual', 'error']
        locals()[residuals_rv_varname] = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.residuals.telescope_0{i}.txt', sep='\s+', names=residuals_names_rv)
        locals()[residuals_rv_varname]['phase'] = ((locals()[residuals_rv_varname].time - epoch_best)/period_best.value)\
              - np.floor((locals()[residuals_rv_varname].time - epoch_best)/period_best.value+0.5)
        locals()[model_rv_varname] = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.model.telescope_0{i}.txt', sep='\s+', names=model_names_rv)

    # Loading in SED data
    sed_cols = ['filtername', 'wavelength', 'model_flux', 'measured_flux', 'upper_error', 'lower_error', 'residual']
    sed_residuals = pd.read_csv(f'{path}{object_name}{file_prefix}mcmc.sed.residuals.txt', delim_whitespace=True, skiprows=1, header=None, names = sed_cols)
    ####################
    # MANIPULATING DATA
    ####################

    # Generating lightcurves for the TESS data
    dic_TESS_1800 = {'time': detrended_TESS_1800.time, 'flux': detrended_TESS_1800.flux + residuals_TESS_1800.residuals + 1}
    lc_TESS_1800 = pd.DataFrame(dic_TESS_1800)
    dic_TESS_600 = {'time': detrended_TESS_600.time, 'flux': detrended_TESS_600.flux + residuals_TESS_600.residuals + 1}
    lc_TESS_600 = pd.DataFrame(dic_TESS_600)
    dic_TESS_120 = {'time': detrended_TESS_120.time, 'flux': detrended_TESS_120.flux + residuals_TESS_120.residuals + 1}
    lc_TESS_120 = pd.DataFrame(dic_TESS_120)

    # Generating lightcurves for the followup data
    for i in range(len(followup_name_index)):
        detrended_varname = f'detrended_followup_{i}'
        residuals_varname = f'residuals_followup_{i}'
        lc_varname = f'lc_followup_{i}'
        dic_followup = {'time': locals()[detrended_varname].time, 'flux': locals()[detrended_varname].flux + locals()[residuals_varname].residuals + 1}
        locals()[lc_varname] = pd.DataFrame(dic_followup)

    # Phase-folding the TESS data
    for exptime in TESS_exptimes:
        lc_varname = 'lc_TESS_' + exptime
        pretty_varname = 'pretty_TESS_' + exptime
        locals()[lc_varname].time = t_phase_folded(locals()[lc_varname].time, period_best, epoch_best)*period_best.to(u.hr)
        locals()[pretty_varname].time = t_phase_folded(locals()[pretty_varname].time, period_best, epoch_best)*period_best.to(u.hr)

    # Phase-folding the followup data
    for i in range(len(followup_name_index)):
        lc_varname = f'lc_followup_{i}'
        pretty_varname = f'pretty_followup_{i}'
        locals()[lc_varname].time = t_phase_folded(locals()[lc_varname].time, period_best, epoch_best)*period_best.to(u.hr)
        locals()[pretty_varname].time = t_phase_folded(locals()[pretty_varname].time, period_best, epoch_best)*period_best.to(u.hr)

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
    for i in range(len(rvnames)):
        rvs_varname = f'rvs_{i}'
        rvs_trendsubtracted_varname = f'rvs_trendsubtracted_{i}'
        rvs_error_varname = f'rv_error_{i}'
        model_rv_varname = f'model_rv_{i}'
        residuals_rv_varname = f'residuals_rv_{i}'

        gamma_varname = f'gamma_{i}'
        jitter_varname = f'jitter_{i}'

        locals()[rvs_trendsubtracted_varname] = locals()[model_rv_varname].rv + locals()[residuals_rv_varname].residual - locals()[gamma_varname]\
              - slope_best * (locals()[residuals_rv_varname].time - rvepoch_best) # subtracting linear slope
        locals()[rvs_varname] = locals()[model_rv_varname].rv + locals()[residuals_rv_varname].residual - locals()[gamma_varname]
        locals()[rvs_error_varname] = (locals()[residuals_rv_varname].error**2 + locals()[jitter_varname])**(1/2)
    
    # Bandpass effective widths to represent the errors of the SED plot (from the SVO filter profile service)
    # [Gaia G, Gbp, Grp, 2MASS J, H, Ks, WISE W1, W2, W3]
    Weff = np.array([4052.97, 2157.50, 2924.44, 1624.32, 2509.40, 2618.87, 6626.42, 10422.66, 55055.23]) * u.AA
    Weff = (Weff.to(u.micron)).value

    ####################
    # PLOTTING
    ####################

    # defining a list of colors to choose from when plotting. Feel free to change or add more!
    colors = ['#009B77', '#821EA6', '#34568B', '#D1AF19', '#95DEE3', '#88B04B', '#955251', '#5B5EA6', '#9B2335', '#E6AF91', '#D65076', '#422C7A', '#DD4124']

    # Setting up the figure
    fig, axs = plt.subplots(ncols=2, nrows=3)
    fig.set_figheight(20)
    fig.set_figwidth(17)
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
        pretty_varname = 'pretty_TESS_' + exptime
        lc_varname = 'lc_TESS_' + exptime
        locals()[pretty_varname].sort_values('time', inplace=True) # sorting model

        # separates each lightcurve by the given lightcurve spacing
        ax1.plot(locals()[pretty_varname].time, locals()[pretty_varname].flux + 1 - lightcurve_spacing * TESS_lc_index, c = 'k', zorder=2, linewidth=3)
        ax1.scatter(locals()[lc_varname].time, locals()[lc_varname].flux - lightcurve_spacing * TESS_lc_index, ls = 'None', c=colors[TESS_lc_index], s = 100, alpha = 0.7, \
                edgecolors='#000000', zorder = 1, label = 'TESS ' + exptime + 's')
        # tess=ax1.scatter(lc_TESS_binned_time, lc_TESS_binned_flux, ls = 'None', c='#9F3BC2', s = 125, alpha = 1, \
                # edgecolors='#000000', zorder = 10, label='TESS')
        TESS_lc_index += 1

    for i in range(len(followup_name_index)):
        pretty_varname = f'pretty_followup_{i}'
        lc_varname = f'lc_followup_{i}'

        locals()[pretty_varname].sort_values('time', inplace=True) # sorting model

        ax1.plot(locals()[pretty_varname].time, locals()[pretty_varname].flux + 1 - lightcurve_spacing * (i + TESS_lc_index), c = 'k', zorder=2, linewidth=3)
        ax1.scatter(locals()[lc_varname].time, locals()[lc_varname].flux - lightcurve_spacing * (i + TESS_lc_index), ls = 'None', c=colors[i + len(TESS_exptimes)], s = 100, alpha = 0.7, \
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
    ax1.set_ylabel('Normalized Flux', fontsize = 20)

    ax1.tick_params(which = 'both', direction = 'inout')
    ax1.tick_params(labelsize = 20, length = 10, width=2)
    ax1.tick_params(which = 'minor', length = 7, width = 1)
    ax1.tick_params(which='both')

    # RVs vs. Time (top right)

    axs[0, 1].remove()
    nested_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax2_upper = fig.add_subplot(nested_gs[0])
    ax2_lower = fig.add_subplot(nested_gs[1])

    ax2_upper.set_xlim(np.min(pretty_rv.time - 2457000),np.max(pretty_rv.time - 2457000))
    ax2_lower.set_xlim(np.min(pretty_rv.time - 2457000),np.max(pretty_rv.time - 2457000))
    ax2_lower.set_xlabel('Time [BJD$_{\mathrm{TDB}} - 2457000$]', fontsize = 20)
    ax2_upper.set_ylabel('RV [m/s]', fontsize = 20)
    ax2_lower.set_ylabel('O-C', fontsize = 16)

    ax2_upper.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2,axis='y')
    ax2_lower.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2)

    ax2_upper.plot(pretty_rv.time - 2457000, pretty_rv.rv_trend, c = 'k', zorder=10, lw = 0.5, label = 'EXOFASTv2', alpha=0.7)

    max_rv = [] # keeping track of the max and min rv from each dataset to set plot limits
    min_rv = []
    for i in range(len(rvnames)):
        residuals_rv_varname = f'residuals_rv_{i}'
        rvs_varname = f'rvs_{i}'
        rvs_error_varname = f'rv_error_{i}'

        max_rv.append(np.max(locals()[rvs_varname]))
        min_rv.append(np.min(locals()[rvs_varname]))

        ax2_upper.errorbar(locals()[residuals_rv_varname].time - 2457000, locals()[rvs_varname], yerr=locals()[rvs_error_varname], fmt='o', mfc=colors[-(i+1)], \
                        mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None', label=rvnames[i])
        ax2_lower.errorbar(locals()[residuals_rv_varname].time - 2457000, locals()[residuals_rv_varname].residual, yerr=locals()[rvs_error_varname], \
                        fmt='o', mfc=colors[-(i+1)], mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None')
    
    ax2_upper.set_ylim(np.min(min_rv) - 100, np.max(max_rv) + 100)
    ax2_lower.axhline(0, ls='--', c='grey', lw = 2)
    ax2_upper.legend(fontsize = 15)

    # RVs vs. Phase (bottom right)

    axs[1, 1].remove()
    nested_gs = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax3_upper = fig.add_subplot(nested_gs[0])
    ax3_lower = fig.add_subplot(nested_gs[1])
    ax3_upper.set_xlim(-0.5, 0.5)
    ax3_lower.set_xlim(-0.5, 0.5)
    ax3_lower.set_xlabel('Phase', fontsize = 20)
    ax3_upper.set_ylabel('RV [m/s]', fontsize = 20)
    ax3_lower.set_ylabel('O-C', fontsize = 16)

    ax3_upper.tick_params(which = 'major', labelsize = 20, length = 10, width=2, axis='y')
    ax3_lower.tick_params(which = 'major', direction = 'inout', labelsize = 20, length = 10, width=2)

    ax3_upper.plot(pretty_rv_phasesorted.phase, pretty_rv_phasesorted.rv, c = 'k', zorder=10, lw = 2, label = 'EXOFASTv2', alpha=0.7)

    for i in range(len(rvnames)):
        residuals_rv_varname = f'residuals_rv_{i}'
        rvs_varname = f'rvs_{i}'
        rvs_trendsubtracted_varname = f'rvs_trendsubtracted_{i}'
        rvs_error_varname = f'rv_error_{i}'

        ax3_upper.errorbar(locals()[residuals_rv_varname].phase, locals()[rvs_trendsubtracted_varname], yerr=locals()[rvs_error_varname], \
                           fmt='o', mfc=colors[-(i+1)], mec='k', ecolor=colors[-(i+1)], capsize=4, ls='None', label=rvnames[i])
        ax3_lower.errorbar(locals()[residuals_rv_varname].phase, locals()[residuals_rv_varname].residual, yerr=locals()[rvs_error_varname], fmt='o', mfc=colors[-(i+1)], mec='k',\
                ecolor=colors[-(i+1)], capsize=4, ls='None')

    ax3_lower.axhline(0, ls='--', c='grey', lw = 2)
    ax3_upper.legend(fontsize = 15)

    # SED Plot (bottom left)

    axs[2, 0].remove()
    nested_gs = gs[2, 0].subgridspec(2, 1, height_ratios=[1, 0.4], hspace=0)
    ax4_upper = fig.add_subplot(nested_gs[0])
    ax4_lower = fig.add_subplot(nested_gs[1])
    ax4_upper.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$]', fontsize = 20)
    ax4_lower.set_xlabel('Wavelength [$\mu$m]', fontsize = 20)
    ax4_lower.set_ylabel('O-C', fontsize = 16)

    ax4_upper.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2, axis='y')
    ax4_upper.set_xticks([])
    ax4_lower.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2)

    ax4_upper.errorbar(sed_residuals.wavelength, sed_residuals.measured_flux, yerr=[sed_residuals.lower_error, sed_residuals.upper_error],\
                       xerr=Weff, fmt='.', markersize=8, mfc='#ff3126', mec='#ff3126',\
                       ecolor='#ff3126', capsize=4, ls='None', label = 'Observations') # Are x errors the width of the wavelength band or where do we get these?

    ax4_upper.scatter(sed_residuals.wavelength, sed_residuals.model_flux, marker='o', color='k', label='EXOFASTv2')

    ax4_lower.errorbar(sed_residuals.wavelength, sed_residuals.residual, yerr=[sed_residuals.lower_error, sed_residuals.upper_error],\
                       xerr=Weff, fmt='.', markersize=8, mfc='#ff3126', mec='#ff3126',\
                       ecolor='#ff3126', capsize=4, ls='None', label = 'Observations') # Are x errors the width of the wavelength band or where do we get these?

    ax4_lower.axhline(0, ls='--', color='grey', lw = 2)
    # ax4_lower.set_ylim(-3e-11, 3e-11)
    ax4_upper.set_yscale('log')
    ax4_upper.set_xscale('log')
    ax4_lower.set_xscale('log')

    ax4_upper.legend(fontsize = 15)

    # MIST Plot (bottom right)
    if MIST == True:
        ax5 = axs[2, 1]

        toinumber = object_name[4:] # assuming the object is a TOI and named TOI-XXXX

        blackline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_EVO/TOI' + toinumber + '_Black.dat', sep='\s+', header=None)
        blueline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_EVO/TOI' + toinumber + '_Blue.dat', sep='\s+', header=None)
        greenline = pd.read_csv(f'' + path + 'TOI' + toinumber + '_EVO/TOI' + toinumber + '_Green.dat', sep='\s+', header=None)

        ref_ages = pd.read_csv(f'' + path + 'TOI' + toinumber + '_EVO/TOI' + toinumber + '_age.dat', sep='\s+', header=None)

        logg = median[' median value'][median['#parname'] == 'logg_0'].iloc[0]
        logg_E = median[' upper errorbar'][median['#parname'] == 'logg_0'].iloc[0]
        logg_e = median[' lower errorbar'][median['#parname'] == 'logg_0'].iloc[0]

        teff = median[' median value'][median['#parname'] == 'teff_0'].iloc[0]
        teff_E = median[' upper errorbar'][median['#parname'] == 'teff_0'].iloc[0]
        teff_e = median[' lower errorbar'][median['#parname'] == 'teff_0'].iloc[0]

        # default plot limits
        default_xlim = [teff + 1400, teff - 500]
        default_ylim = [logg + 0.5, logg - 0.5]

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
        ax5.plot(blueline[0], blueline[1], 'b', linewidth=3, zorder=8) # MIST track for the best-fit stellar mass
        ax5.plot(greenline[0], greenline[1], 'g') # 1 and 2 sigma contours of the log g and teff from MIST isochrones and the global fit
        ax5.errorbar(teff, logg, yerr=[[logg_e], [logg_E]], xerr=[[teff_e], [teff_E]], ecolor='r', capsize=3)
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

        ax5.tick_params(which = 'major', direction = 'inout',labelsize = 20, length = 10, width=2)

        ax5.set_xlabel(r'T$_{\rm eff}$ (K)', fontsize = 20)
        ax5.set_ylabel('log g$_*$ (cgs)', fontsize = 20)

    # Add some space between subplots
    plt.tight_layout(pad=2.5, h_pad=1)

    # Saves the figure to your local directory in a subdirectory named 'output'
    if save == True:
        if os.path.exists('output') == False:
            os.mkdir('output')
        plt.savefig(f'output/fullpagefig_{object_name}.pdf', bbox_inches='tight', facecolor='white', transparent=False)