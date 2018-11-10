import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq


def plot_spike_psd(spike_trains, lw=2, ax=None, NFFT=512,
                   sampling_frequency=1000., rate_based=False, plot_kwargs):
    if rate_based:
        import elephant as el
        kernel = el.kernels.GaussianKernel(2*pq.ms)
    pxxs = []
    for spike_train in spike_trains:
        if not rate_based:
            spt = spike_train.times.rescale('s').magnitude
            bins = np.arange(spike_train.t_start.rescale('s').magnitude,
                             spike_train.t_stop.rescale('s').magnitude,
                             1 / sampling_frequency) #time bins for spikes
            #firing rate histogram
            hist = np.histogram(spt, bins=bins)[0].astype(float)
        else:
            hist = el.statistics.instantaneous_rate(spike_train, 1/sampling_frequency,
                                                    kernel=kernel)
            hist = hist.magnitude
            hist = np.reshape(hist, hist.size)
        hist -= hist.mean()
        Pxx, freqs = plt.mlab.psd(hist, NFFT=NFFT,
                                  Fs=sampling_frequency,
                                  noverlap=NFFT*3/4)
        pxxs.append(Pxx)
    pxxs = np.reshape(np.array(pxxs), (len(spike_trains), len(Pxx))).mean(axis=0)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.plot(freqs, pxxs, **plot_kwargs)
    return ax


def plot_tfr(times, freqs, data, mother=None):
    '''
    Plots time frequency representations of analog signal with PSD estimation


    Parameters
    ----------
    times : array
    freqs : array
    power : array
    mother : wavelet
    '''
    import pycwt

    if mother is None:
        mother = pycwt.Morlet()
    sampling_period = times[1] - times[0]

    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(data, sampling_period, freqs=freqs, wavelet=mother)

    power = (numpy.abs(wave)) ** 2
    power /= scales[:, None] #rectify the power spectrum according to the suggestions proposed by Liu et al. (2007)
    fft_power = numpy.abs(fft) ** 2

    gs = gridspec.GridSpec(3, 3)
    ax_pow = plt.subplot(gs[:2, 1:3])
    ax_pow.set_xlim(*times[[0,-1]])
    ax_pow.set_ylim(*freqs[[0,-1]])

    ax_fft = plt.subplot(gs[:2, 0], sharey=ax_pow)
    ax_sig = plt.subplot(gs[2, 1:3], sharex=ax_pow)

    ax_pow.contourf(times, freqs, power, levels=100)
    ax_sig.plot(times, data)
    ax_fft.plot(fft_power, fftfreqs)


def plot_dsyn_syn(S, f, time, low=[4, 12], high=[30, 80]):
    from timefrequency import separate_syn_dsyn
    dsync_idxs, sync_idxs, L, H = separate_syn_dsyn(S, f, low=low, high=high,
                                                    return_all=True)
    rat = np.log(L)/np.log(H)
    rat_syn_mean = np.nanmean(rat[sync_idxs])
    rat_syn_std = np.nanstd(rat[sync_idxs])
    rat_dsyn_mean = np.nanmean(rat[dsync_idxs])
    rat_dsyn_std = np.nanstd(rat[dsync_idxs])
    plt.figure()
    plt.plot(time[dsync_idxs], rat[dsync_idxs], linestyle='none',
             marker='.', color='r')
    plt.plot(time[sync_idxs], rat[sync_idxs], linestyle='none',
             marker='.', color='b')

    plt.plot(time, rat_dsyn_mean*np.ones(len(rat)) + rat_dsyn_std, '--r')
    plt.plot(time, rat_dsyn_mean*np.ones(len(rat)), 'r')
    plt.plot(time, rat_syn_mean*np.ones(len(rat)) - rat_syn_std, '--b')
    plt.plot(time, rat_syn_mean*np.ones(len(rat)), 'b')
    plt.plot(time, np.mean(rat)*np.ones(len(rat)), 'k')
    plt.xlabel('Time (s)');
    plt.ylabel('log(LH power ratio)')

    d_idcs = np.where(rat < (rat_dsyn_mean + rat_dsyn_std))[0]
    s_idcs = np.where(rat > (rat_syn_mean - rat_syn_std))[0]
    plt.figure()
    plt.loglog(L[d_idcs], H[d_idcs], linestyle='none', marker='.', color='r')
    plt.loglog(L[s_idcs], H[s_idcs], linestyle='none', marker='.', color='b')
    plt.xlabel('%s power' % low)
    plt.ylabel('%s power' % high)
