import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from collections import defaultdict

# Set MatPlotLib defaults
plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')
# plt.rc('font', family='serif', size=11, serif='Times New Roman')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('axes', labelsize=9)
plt.rc('axes', titlesize=9)
plt.rc('legend', fontsize=7)
plt.rcParams["text.usetex"] = False
plt.rcParams['text.latex.preamble']= ''

# IEEETrans double column standard
FIG_WIDTH = 252.0 / 72.27 #1pt is 1/72.27 inches
FIG_HEIGHT = FIG_WIDTH / 1.618 #golden ratio

COLOR_MAP = matplotlib.cm.get_cmap('tab20c')


def create_dirs(path):
    _path = path.split("/")[:-1]
    _path = "/".join(_path)

    os.makedirs(_path, exist_ok=True)


def plot_subsets(filename, data):
    create_dirs(filename)
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=400)

    max_labels = len(list(data.keys()))

    for i, (label, d) in enumerate(data.items()):
        d = np.array(list(d.values())) # List of tuples (x, y)
        d = np.transpose(d)

        if "worst" in label:
            color_index = i - 1
            linestyle = "--"
            plt.plot(d[0], d[1], linewidth=1, linestyle=linestyle, color=COLOR_MAP(color_index/max_labels))
        else:
            color_index = i
            linestyle = "solid"
            plt.plot(d[0], d[1], label=label, linewidth=1, linestyle=linestyle, color=COLOR_MAP(color_index/max_labels))

    plt.xlabel("Relative subset size")
    plt.ylabel("Distance to full dataset")
    plt.grid(color='black', linestyle='--', linewidth=0.5, axis='y', zorder=0, alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=400, bbox_inches="tight")


def plot_thresholds(filename, data, start_local_optimization):
    create_dirs(filename)   
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=400)
    plt.ylabel("|BAC|")
    plt.xlabel("wall clock time [s]")

    max_labels = len(list(data.keys()))

    # Label is threshold or RS
    for i, (label, plot_run_histories) in enumerate(data.items()):
        intermediate_data = defaultdict(list)

        for plot_run_history in plot_run_histories:
            plot_data = plot_run_history.get_plot_data()

            # We have to select the cost we want
            for time_step, cost in plot_data.items():
                intermediate_data[time_step].append(cost)

        X = []
        y = []
        y_std = []

        # Those are all the entries for the same x value
        best_mean = 1
        best_std = None
        for time_step, entries in intermediate_data.items():
            entries = np.array(entries)
            mean = np.mean(entries)
            std = np.std(entries)

            X.append(time_step)
            y.append(mean)
            y_std.append(std)

        y = np.array(y)
        y_std = np.array(y_std)

        plt.plot(X, y, label=label, linewidth=1, color=COLOR_MAP(i/max_labels))
        plt.fill_between(X, y-y_std, y+y_std, color=COLOR_MAP(i/max_labels), alpha=0.3)
    
    plt.axvline(x=int(start_local_optimization), color='r', label='Full dataset', linewidth=0.25, linestyle="--")

    plt.ylim(0.17, 0.20)
    #plt.xscale("log")
    plt.grid(color='black', linestyle='--', linewidth=0.5, axis='y', zorder=0, alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=400, bbox_inches="tight")
        

def plot_correlations(filename, all_correlations: list):
    """
    data: list of dicts
    """
    create_dirs(filename)

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=400, bbox_inches="tight")

    linestyles = [
        (0, ()),
        (0, (15, 5)),
        (0, (10, 10)),
        (0, (5, 15))
    ]

    x_min = 99999
    x_max = -99999
    start = 0
    cmap = matplotlib.cm.get_cmap('Spectral')
    for index, correlations in enumerate(all_correlations):
        
        if index == 0:
            ax = plt.subplot(1, len(all_correlations), index+1)
            shareax = ax
            plt.ylabel("|spearman correlation|")
        else:
            ax = plt.subplot(1, len(all_correlations), index+1)
            plt.setp(ax.get_yticklabels(), visible=False)

        # Hide first and last xticklabels
        plt.setp(ax.get_xticklabels()[0], visible=False)    
        #plt.setp(ax.get_xticklabels()[-1], visible=False)
        
        if index == 1:
            plt.xlabel("Number of configurations")

        for i, (label, y) in enumerate(correlations.items()):
            assert len(y) > start

            plt.title(f"Round {index+1}")

            if isinstance(label, int):
                label = [label]

            x = list(range(start, len(y)))
            for ind, group in enumerate(label):
                # Group decides which color is choosen
                plt.plot(x, y[start:], linestyle=linestyles[ind % 4], linewidth=1, color=cmap(group/16))

            if min(x) < x_min: x_min = min(x)
            if max(x) > x_max: x_max = max(x)
    
        plt.xlim(x_min, x_max)
        plt.ylim(0.8, 1.2)
        plt.grid(color='black', linestyle='--', linewidth=0.5, axis='y', zorder=0, alpha=0.5)


    plt.subplots_adjust(wspace = 0)

    plt.savefig(filename, dpi=400, bbox_inches="tight")


def rgb_to_hex(r:int,g:int,b:int) -> str:
    '''
    Convert a RGB tuple into a hexadecimal color code.

    :param r: Integer value of the red channel [0,255]
    :param g: Integer value of the green channel [0,255]
    :param b: Integer value of the blue channel [0,255]
    :return: String of the hexadecimal color code for the RGB tuple
    '''
    return '#%02x%02x%02x' % (r, g, b)


def tu_color(color_id:int) -> str:
    '''
    Returns hexadecimal color values for the respective colors of the corporate design of the TU Braunschweig.
    c.f. https://www.tu-braunschweig.de/Medien-DB/presse/flyer/flyer-spk-corporate-design.pdf
    :param color_id: ID of the color (https://ifn21.ifn.ing.tu-bs.de/wiki/doku.php?id=medienmaterialien:powerpoint_vorlage)
    :return: Hexadecimal color code
    '''
    tubs_color_dict = { #TODO: Remove 'platzhalter'
        0:(190,30,60),
        1:(255,205,0),
        11:(255,220,77),
        12:(255,230,127),
        13:(255,240,178),
        14:(255,245,204),
        2:(250,110,0),
        21:(252,154,77),
        22:(252,182,127),
        23:(253,211,178),
        24:(254,226,204),
        3:(176,0,70),
        31:(192,51,107),
        32:(215,127,162),
        33:(235,191,209),
        34:(243,217,227),
        4:(124,205,230),
        41:(164,220,238),
        42:(189,230,242),
        43:(215,240,247),
        44:(229,245,250),
        5:(0,128,180),
        51:(77,165,203),
        52:(140,198,221),
        53:(191,223,236),
        54:(217,246,244),
        6:(0,83,116),
        61:(64,126,151),
        62:(140,177,192),
        63:(191,212,220),
        64:(217,229,234),
        7:(8,8,8),
        71:(95,95,95),
        72:(150,150,150),
        73:(192,192,192),
        74:(221,221,221),
        8:(198,238,0),
        81:(215,243,77),
        82:(226,246,127),
        83:(238,250,178),
        84:(244,252,204),
        9:(137,164,0),
        91:(173,191,77),
        92:(196,209,127),
        93:(219,228,178),
        94:(231,237,204),
        10:(0,113,86),
        101:(77,156,137),
        102:(140,191,179),
        103:(191,219,213),
        104:(218,234,231),
        110:(204,0,153),
        111:(222,89,189),
        112:(235,153,214),
        113:(245,204,235),
        114:(250,229,245),
        120:(118,0,118),
        121:(152,64,152),
        122:(186,127,186),
        123:(214,178,214),
        124:(235,217,235),
        130:(118,0,84),
        131:(156,77,136),
        132:(193,140,178),
        133:(221,191,212),
        134:(235,217,230)
    }

    values = list(tubs_color_dict.values())
    return rgb_to_hex(*values[color_id])