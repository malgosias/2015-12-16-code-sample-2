'''

REQUIRES carmcmc: https://github.com/brandonckelly/carma_pack
    
Tool to test the CARMA package on multiband long-term lightcurves of
3C273 from the Soldi et al. (2008) catalog: http://isdc.unige.ch/3c273/

read_lightcurve
Read the lightcurve data from a file.

run_carma:
Model lightcurves, find best fitting PSD, and dump cPickle with
carma_sample.

get_power_spectrum:
Copy of carmcmc's plot_power_spectrum, but no plotting option.

get_fvar_spectrum
Find fractional variability *spectrum* and credibility interval for
variability timescales corresponding to the freqeuncy range (fmin, fmax)
of the best fitting CARMA(p,q) PSD.

get_fvar
Find *monochromatic* fractional variability and credibility interval for
variability timescales corresponding to the freqeuncy range (fmin, fmax)
of the best fitting CARMA(p,q) PSD.

get_xvals
Recover energies/frequencies/wavelengths from the band name

plot_fvar
Generate plots/bband-fvar-radio-xray-0_05-25yr-flag0-log.png.

'''

import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pylab as plt
import cPickle
import carmcmc as cm # carma_pack

data_dir = 'data/'

def read_lightcurve(fname):
# Read lightcurve data from <fname> file.
#
# :rtype: 5 numpy arrays, time (float, in years, e.g. 2000.15502),
#         flux (float, in Jy), flux error (float, in Jy), flag (string),
#         observatory (string)
# :param fname: path to the data file
#
# Note: lightcurve files included in data/
#
    d = np.loadtxt(fname, dtype=np.str)
    time = d[:,1].astype('float')
    y = d[:,4].astype('float')
    ysig = d[:,5].astype('float')
    flag = d[:,6]
    observatory = d[:,7]
    return time, y, ysig, flag, observatory


def run_carma(band, filt, logscale):
# Model lightcurves with CARMA(p,q) process, find best fitting PSD, and
# dump pickle with carma_sample.
#
# :rtype: None
# :param band: string with the energy band name, e.g. '2kev', '5395a',
#              corresponding to the data file names in data/
# :param filt: data filter; 'RXTE' to use only RXTE X-ray data, 'flag0'
#              to use only 'good data' (see the catalog for details).
# :param logscale: True to work on logarithm of lightcurve
#
# Note: pickle files included in data/.
#
    data_file = data_dir + 'data_' + band + '.txt'
    
    # clean data: apply filters, calculate log if required
    if (band=='fermi'):
        data = np.loadtxt(data_file)
        time = data[:,0]
        y1 = data[:,6]
        ysig1 = data[:,7]
    else:
        time0, y0, ysig0, flag, observatory = read_lightcurve(data_file)
        if (filt.lower()=='rxte'):
            idx = np.where((observatory=='RXTE') & (y0>0))
        elif (filt.lower()=='flag0'):
            idx = np.where((flag=='0') & (y0>0))
        time = time0[idx]
        y1 = y0[idx]
        ysig1 = ysig0[idx]

    if (logscale):
        y = np.log(y1)
        ysig = ysig1/y1
    else:
        y = y1
        ysig = ysig1
   
    dt = time[1:] - time[0:-1]
    noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)

    # create new CARMA process model
    carma_model = cm.CarmaModel(time, y, ysig)
    # only search over p < 7, q < p
    pmax = 7
    # use all the processes through the multiprocessing module
    njobs = -1
    MLE, pqlist, AICc_list = carma_model.choose_order(pmax, njobs=-1)
    carma_sample = carma_model.run_mcmc(50000)
    # name root for the pickle: e.g. 2kev_flag0
    name = band + '_' + filt
    #cPickle.dump(carma_sample, open(data_dir + name + '.pickle', 'wb'))
    carma_sample.add_map(MLE)
    cPickle.dump(carma_sample, open(data_dir + name + '.addmle' +
                                    '.pickle', 'wb'))

    # only use 5000 MCMC samples for speed
    psd_lo, psd_hi, psd_mid, freq = \
        carma_sample.plot_power_spectrum(percentile=95.0, nsamples=5000)
    plt.hlines( noise_level, freq[0], 1./(2. * np.median(dt)) )
    plt.title(name)

    carma_sample.assess_fit()
  
    return


def get_power_spectrum(carma_sample, percentile=68.0, nsamples=None):
# Copy of carmcmc's plot_power_spectrum, but no plotting option.
#
# Find posterior median and the credibility interval corresponding
# to percentile of the CARMA(p,q) PSD. This function returns a tuple
# containing the lower and upper PSD credibility intervals as a function
# of frequency, the median PSD as a function of frequency, and the
# frequencies.
#
# :rtype: A tuple of numpy arrays, (lower PSD, upper PSD, median PSD,
#         frequencies).
# :param percentile: The percentile of the PSD credibility interval.
# :param nsamples: The number of MCMC samples to use to estimate the
#                  credibility interval. The default is all of them. Use
#                  less samples for increased speed.
#
    sigmas = carma_sample._samples['sigma']
    ar_coefs = carma_sample._samples['ar_coefs']
    ma_coefs = carma_sample._samples['ma_coefs']
    if nsamples is None:
        # Use all of the MCMC samples
        nsamples = sigmas.shape[0]
    else:
        try:
            nsamples <= sigmas.shape[0]
        except ValueError:
            "nsamples must be less than the total no. of MCMC samples."

        nsamples0 = sigmas.shape[0]
        index = np.arange(nsamples) * (nsamples0 / nsamples)
        sigmas = sigmas[index]
        ar_coefs = ar_coefs[index]
        ma_coefs = ma_coefs[index]

    nfreq = 1000
    dt_min = carma_sample.time[1:] - \
                        carma_sample.time[0:carma_sample.time.size - 1]
    dt_min = dt_min.min()
    dt_max = carma_sample.time.max() - carma_sample.time.min()

    # Only plot frequencies corresponding to time scales a factor
    # of 2 shorter and longer than the minimum and maximum time scales
    # probed by the time series.
    freq_max = 1.0 / dt_min
    freq_min = 1.0 / dt_max

    frequencies = np.linspace(np.log(freq_min), np.log(freq_max), \
                              num=nfreq)
    frequencies = np.exp(frequencies)
    psd_credint = np.empty((nfreq, 3))

    # lower and upper intervals for credible region
    lower = (100.0 - percentile) / 2.0
    upper = 100.0 - lower

    # Compute the PSDs from the MCMC samples
    omega = 2.0 * np.pi * 1j * frequencies
    ar_poly = np.zeros((nfreq, nsamples), dtype=complex)
    ma_poly = np.zeros_like(ar_poly)
    for k in xrange(carma_sample.p):
        # Here we compute:
        # alpha(omega) = ar_coefs[0] * omega^p +
        #                ar_coefs[1] * omega^(p-1) + ... + ar_coefs[p]
        # Note that ar_coefs[0] = 1.0.
        argrid, omgrid = np.meshgrid(ar_coefs[:, k], omega)
        ar_poly += argrid * (omgrid ** (carma_sample.p - k))
    ar_poly += ar_coefs[:, carma_sample.p]
    for k in xrange(ma_coefs.shape[1]):
        # Here we compute:
        # delta(omega) = ma_coefs[0] + ma_coefs[1] * omega + ... +
        #                ma_coefs[q] * omega^q
        magrid, omgrid = np.meshgrid(ma_coefs[:, k], omega)
        ma_poly += magrid * (omgrid ** k)

    psd_samples = np.squeeze(sigmas) ** 2 * np.abs(ma_poly) ** 2 / \
                                                    np.abs(ar_poly) ** 2

    # Now compute credibility interval for power spectrum
    psd_credint[:, 0] = np.percentile(psd_samples, lower, axis=1)
    psd_credint[:, 2] = np.percentile(psd_samples, upper, axis=1)
    psd_credint[:, 1] = np.median(psd_samples, axis=1)

    return (psd_credint[:, 0], psd_credint[:, 2], psd_credint[:, 1], \
            frequencies)


def get_fvar_spectrum(fmin, fmax, band_arr, filt='flag0', percentile=68.0):
# Find fractional variability spectrum and credibility interval for
# variability timescales corresponding to the freqeuncy range
# (fmin, fmax) of the best fitting CARMA(p,q) PSD.
#
# :rtype: 4 numpy arrays; energy, fractional variability (fvar),
#         fvar lower, fvar upper
#
# :param fmin: Minimum targeted PDS frequency
# :param fmax: Maximum targeted PDS frequency
# :param band_arr: Numpy array of strings with band names,
#                  e.g. ['2kev', '5kev'] for X-rays
# :param filt: Data filter; 'RXTE' to use only RXTE X-ray data, 'flag0'
#              to use only 'good data' (see the catalog for details).
# :param percentile: The percentile of the PSD credibility interval.

    #translate wavelengths and frequencies to energies in keV
    if ('kev' in  band_arr[0]):
        x = get_xvals(band_arr, len('keV')) * u.keV
    elif ('fermi' in band_arr[0]):
        x = np.array([5.5e6])
    elif ('a' in band_arr[0]):
        # input: wavelenghts in A
        x = get_xvals(band_arr, len('a'))
        x = const.h.to("keV * s") * const.c.cgs / (x * 1e-8 *u.cm)
    elif ('um' in band_arr[0]):
        # input: wavelenghts in um = 1e-6 m = 1e-4 cm
        x = get_xvals(band_arr, len('um'))
        x = const.h.to("keV * s") * const.c.cgs / (x * 1e-4 * u.cm)
    elif ('mm' in band_arr[0]):
        # input: wavelenghts in mm = 1e-3 m = 1e-1 cm
        x = get_xvals(band_arr, len('mm'))
        x = const.h.to("keV * s") * const.c.cgs / (x * 1e-1 * u.cm)
    elif ('ghz' in band_arr[0]):
        # input: wavelenghts in ghz
        x = get_xvals(band_arr, len('ghz'))
        x = const.h.to("keV * s") * x * 1.e9 / u.s

    y = np.array([])
    ylo = np.array([])
    yhi = np.array([])

    # Compute fractional variability
    for band in band_arr:
        fvar_mid, fvar_lo, fvar_hi = \
                     get_fvar(fmin, fmax, band, filt, percentile=68.0)
        y = np.append(y, fvar_mid)
        ylo = np.append(ylo, fvar_lo)
        yhi = np.append(yhi, fvar_hi)

    return x.value, y, ylo, yhi

  
def get_xvals(band_arr, nunit):
# Recover energies/frequencies/wavelengths from the band name
#
# :rtype: Numpy array with energies/frequencies/wavelengths
# :param band_arr: Numpy array of strings with band names,
#                  e.g. ['2kev', '5kev'] for X-rays
# :param nuint: length of the substring containing the unit,
#               e.g. 3 for 'keV' (note: use re in future)
#
    x = np.array([])
    for band in band_arr:
        x = np.append(x, np.float(band[:-nunit]))
    return x
  

def get_fvar(fmin, fmax, band, filt='flag0', percentile=68.0):
# Get monochromatic fractional variability and credibility interval for
# variability timescales corresponding to the freqeuncy range
# (fmin, fmax) of the best fitting CARMA(p,q) PSD.
#
# :rtype: 3 numpy arrays; fractional variability (fvar), fvar lower,
#         fvar upper
#
# :param fmin: Minimum targeted PDS frequency
# :param fmax: Maximum targeted PDS frequency
# :param band_arr: Strings with the band name, e.g. '2kev'
# :param filt: Data filter; 'RXTE' to use only RXTE X-ray data, 'flag0'
#              to use only 'good data' (see the catalog for details).
# :param percentile: The percentile of the PSD credibility interval.

    name = band + '_' + filt
    carma_sample = cPickle.load(open(data_dir + name + '.addmle' +
                                     '.pickle', 'rb'))
    dt = carma_sample.time[1:] - carma_sample.time[:-1]
    ysig = carma_sample.ysig

    noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)
    
    psd_lo, psd_hi, psd_mid, freq = \
        get_power_spectrum(carma_sample, percentile, nsamples=5000)

    ii = np.where((freq>fmin) & (freq<fmax))
    df = freq[ii][1:] - freq[ii][:-1]
    nn = np.size(freq[ii]) - 1

    # multiply by 2 due to CARMA normalization
    fvar_mid = np.sqrt( np.sum(2 * psd_mid[ii][:-1] * df) )
    fvar_lo  = np.sqrt( np.sum(2 * psd_lo[ii][:-1] * df) )
    fvar_hi  = np.sqrt( np.sum(2 * psd_hi[ii][:-1] * df) )
    
    return fvar_mid, fvar_mid - fvar_lo, fvar_hi - fvar_mid


# --------------- Plotting ---------------

def plot_fvar():
# Generate plots/bband-fvar-radio-xray-0_05-25yr-flag0-log.png

    d = np.loadtxt(data_dir + 'fvar_soldi.txt')
    v = d[:52,0]
    fvar = d[:52,1]
    fvarerr = d[:52,2]
    v1 = d[55:64,0]
    fvar1 = d[55:64,1]
    fvarerr1 = d[55:64,2]

    xray     = ['2kev', '5kev', '10kev', '20kev', '50kev', '100kev']
    uv       = ['3000a', '2700a', '2425a', '2100a', '1950a', '1700a', \
                                                       '1525a', '1300a']
    optical  = ['5798a', '5479a', '5395a', '4466a', '4003a']
    infrared = ['3.6um', '2.2um', '1.65um', '1.25um']
    mm       = ['3.3mm', '2.0mm', '1.3mm', '1.1mm', '0.8mm']
    radio    = ['2.5ghz', '5ghz', '8ghz', '10ghz', '15ghz', '22ghz', \
                                                                '37ghz']

    timescale = 't = 0.05-25 yr' # without gamma
  
    fig = plt.figure()
  
    # Plot results of Soldi et al. (2008)
    kwargs = {'ls':'None', 'mec':'k', 'ecolor':'k'}
    plt.errorbar(np.log10(v), fvar, yerr=fvarerr, marker='o', mfc='k', \
                 label='S08', **kwargs)
    plt.errorbar(np.log10(v1), fvar1, yerr=fvarerr1, marker='^', ms=7, \
                 mfc='None', **kwargs)

    # Plot carma results (Sobolewska et al. 2014, 5th Fermi symposium)
    color = ['teal', 'indigo', 'darkorange', 'firebrick', 'orangered', \
             'saddlebrown']
    label = ['X', 'UV', 'opt', 'IR', 'mm', 'radio']

    for i, bands in enumerate([xray, uv, optical, infrared, mm, radio]):
        x, y, ylo, yhi = get_fvar_spectrum(0.04, 10., bands)
        x_in_hz = x / const.h.to('keV * s').value
        plt.errorbar(np.log10(x_in_hz), y, yerr=[ylo, yhi], marker='d',\
                     ms=8, ls='None', label=label[i], \
                     mfc=color[i], mec=color[i], ecolor=color[i])

    plt.xlabel('log10( Frequency / [Hz] )')
    plt.ylabel('Fractional variability amplitude')
    plt.title('3C 273,   ' + timescale)
    plt.legend(bbox_to_anchor=(0.34, 0.48, 0.25, 0.3), loc=3, ncol=1, \
               mode="expand", borderaxespad=0., numpoints=1)
    return


