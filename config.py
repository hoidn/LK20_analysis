# 2016.09.22 18:12:27 PDT
#Embedded file name: config.py
from collections import namedtuple
import numpy as np
from dataccess.default_config import *
smd = True
noplot = False
cached_only = False
exppath = 'mec/meck2016'
try:
    expname = exppath.split('/')[1]
except:
    raise ValueError('config.exppath: incorrect format')

xtc_prefix = 'e691'
logbook_ID = '1DRtTTiatTbN2z4brGJxJ1OkWRMCj3gyoPfFRN5fvl3M'
url = 'https://docs.google.com/spreadsheets/d/%s/edit#gid=0' % logbook_ID
use_logbook = True
photon_energy = 8910.0
pulse_energy = 0.0001
DetInfo = namedtuple('DetInfo', ['device_name',
 'geometry',
 'extra_masks',
 'subregion_index'])
detinfo_map = {'quad1': DetInfo('MecTargetChamber.0:Cspad.0', 
    {'y0': 187, 'phi': 0, 'r': 910.4, 'x0': 409, 'alpha': 2.44874},
#            {'phi': 0,
#           'x0': 409,
#           'y0': 187,
#           'alpha': 2.33874,
#           'r': 1041.4},
            {'../masks/quad1_mask_10.11.npy'}, 0),
 'quad2': DetInfo('MecTargetChamber.0:Cspad.0', {'phi': -1.5899543,
           'x0': 654.165,
           'y0': 574.192,
           'alpha': 0.759423,
           'r': 970.13}, {'../masks/quad2_mask_10.11.npy'}, 1),
 'allquads': DetInfo('MecTargetChamber.0:Cspad.0', {'phi': None,
              'x0': None,
              'y0': None,
              'alpha': None,
              'r': None}, {}, -1),
 'xrts1': DetInfo('MecTargetChamber.0:Cspad2x2.1', {}, {}, -1),
 'xrts2': DetInfo('MecTargetChamber.0:Cspad2x2.2', {}, {}, -1),
 'vuv': DetInfo('MecTargetChamber.0:Princeton.1', {}, {}, -1),
 'si': DetInfo('MecTargetChamber.0:Opal1000.1', {}, {}, -1)}
NonareaInfo = namedtuple('NonareaInfo', ['type', 'src'])
nonarea = {'d1': NonareaInfo('Lusi.IpmFexV1', 'MEC-TCTR-DI-01'),
 'ipm2': NonareaInfo('Lusi.IpmFexV1', 'MEC-XT2-IPM-02'),
 'ipm3': NonareaInfo('Lusi.IpmFexV1', 'MEC-XT2-IPM-03'),
 'GMD': NonareaInfo('Bld.BldDataFEEGasDetEnergyV1', 'FEEGasDetEnergy')}
powder_angles = {'Fe3O4': [27.2,
           32.1,
           33.47,
           38.9,
           48.1,
           51.3],
 'MgO': [33.47, 38.9, 56.0, 129.8],
 #'MgO': [33.47, 38.9, 56.0],
 'Graphite': [24.0,
              30.2,
              38.4,
              41.6]}

def si_is_saturated(label, start = 400, end = 800):
    from dataccess import data_access as data
    imarr = data.get_label_data(label, 'si')[0]
    spectrum = np.sum(imarr, axis=0)[start:end]
    spectrum = spectrum - np.percentile(spectrum, 1)
    saturation_metric = np.sum(np.abs(np.diff(np.diff(spectrum)))) / np.sum(spectrum)
    return saturation_metric > 0.1


def getgood():
    good = []
    for i in range(450, 850):
        try:
            if si_is_saturated(str(i)):
                good.append(i)
        except:
            pass

        print i

    return good


def get_si_peak_boundary(run):
    if run <= 480:
        return 520
    else:
        return 600


def si_spectrometer_dark(run = None, **kwargs):
    from dataccess import data_access
    bg_label = data_access.get_label_property(str(run), 'background')
    dark, _ = data_access.get_label_data(bg_label, 'si')
    return dark


def si_background_subtracted_spectrum(imarr):
    from dataccess import xes_process as spec
    imarr = imarr.T
    cencol = spec.center_col(imarr)
    return spec.bgsubtract_linear_interpolation(spec.lineout(imarr, cencol, pxwidth=30))


def si_spectrometer_probe(imarr, run = None, **kwargs):
    boundary = get_si_peak_boundary(run)
    spectrum = si_background_subtracted_spectrum(imarr)
    return np.sum(spectrum[200:boundary])


def si_spectrometer_pump(imarr, run = None, **kwargs):
    boundary = get_si_peak_boundary(run)
    spectrum = si_background_subtracted_spectrum(imarr)
    return np.sum(spectrum[boundary:900])


def si_peak_ratio5(imarr, run = None, **kwargs):
    pump_counts = si_spectrometer_pump(imarr)
    probe_counts = si_spectrometer_probe(imarr)
    return pump_counts / probe_counts


def make_si_filter(probe_min, probe_max, pump_min, pump_max, **kwargs):

    def filter_by_si_peaks(imarr):
        spectrum = np.sum(imarr, axis=1)
        spectrum = spectrum - np.percentile(spectrum, 1)
        probe_counts = np.sum(spectrum[200:520])
        pump_counts = np.sum(spectrum[520:900])
        if probe_min < probe_counts < probe_max and pump_min < pump_counts < pump_max:
            return True
        return False

    return filter_by_si_peaks


def goodratio_lower(imarr, **kwargs):
    filt = make_si_filter(0, 100000000.0, 0, 8000000.0, **kwargs)
    return filt(imarr)


def goodratio_upper(imarr, **kwargs):
    filt = make_si_filter(0, 100000000.0, 8000000.0, 1000000000.0, **kwargs)
    return filt(imarr)


def sum_si(imarr):
    baseline = np.percentile(imarr, 1)
    return np.sum(imarr - baseline)


def identity(imarr):
    return imarr


def flux(beam_energy, label = None, size = None, **kwargs):
    from dataccess import data_access as data
    size = data.get_label_property(label, 'focal_size')
    flux = beam_energy * data.get_label_property(label, 'transmission') / (np.pi * (size * 0.5 * 0.0001) ** 2)
    return flux


def get_pulse_duration(a, run = None, nevent = None, window_size = 60):
    from dataccess import xtcav
    try:
        t0 = xtcav.get_run_epoch_time(run)
    except TypeError:
        raise ValueError('No run value provided')

    t_samples = np.linspace(t0 - window_size / 2, t0 + window_size / 2)
    return np.mean(xtcav.pulse_length_from_epoch_time(t_samples))


def make_pulse_duration_filter(duration_min, duration_max, window_size = 10):

    def filterfunc(a, run = None, nevent = None):
        pulse_duration = get_pulse_duration(a, run=run, nevent=nevent, window_size=window_size)
        accepted = duration_min < pulse_duration < duration_max
        if nevent == 0:
            if accepted:
                print 'Run %04d: accepted' % run
            else:
                print 'Run %04d: rejected' % run
        return accepted

    return filterfunc


def make_pulse_duration_and_run_filter(duration_min, duration_max, run_min, run_max):

    def runfilter(a, run = None, nevent = None):
        return run_min <= run < run_max

    pulse_duration_filter = make_pulse_duration_filter(duration_min, duration_max)

    def conjunction(a, run = None, nevent = None):
        return runfilter(a, run=run, nevent=nevent) and pulse_duration_filter(a, run=run, nevent=nevent)

    return conjunction


def every_other_filter(arr, nevent = None, **kwargs):
    """
    For use in testing mode.
    """
    return bool(nevent % 20)


best_focus_size = 2.0

def sum_window(smin, smax):
    return lambda arr: smin < np.sum(arr) < smax


def beam_intensity_diagnostic(label):
    """
    Return average beam energy in a dataset, evaluated based on Si spectrometer spectrum.
    """
    gmd_scale_factor = 1.0
    si_integral_to_eV = 7.192326355322434e-11
    from dataccess import mec
    return gmd_scale_factor * si_integral_to_eV * mec.si_spectrometer_integral(label)


def xrts1_fe_fluorescence_integral(label):
    from dataccess import mec
    return mec.xrts1_fe_fluorescence_integral(label)


def onecolor_fluence(imarr, run = None, **kwargs):
    """
    Given si spectrometer array for a single shot and a run number, return the
    incident flux. If the 'focal_size' attribute from the logging spreadsheet is
    absent for this run, return 0.
    """
    from dataccess.mec import background_subtracted_spectrum, si_integral_to_eV
    from dataccess import logbook
    si_signal = np.sum(background_subtracted_spectrum(imarr))
    label = str(run)
    size = logbook.get_label_attribute(label, 'focal_size')
    return si_integral_to_eV * si_signal / (np.pi * (size * 0.5 * 0.0001) ** 2)


testing = False
playback = False
stdout_to_file = True
plotting_mode = 'notebook'
chip_level_correction = False
from dataccess.mpl_plotly import plt
