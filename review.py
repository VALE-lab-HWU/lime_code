import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

#plt.style.use('dark_background')
plt.rcParams.update({
#    "figure.facecolor": "#FFF",
    "axes.facecolor": "#444"})


# https://stackoverflow.com/a/44960748/8040287
def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, 1)


# https://stackoverflow.com/a/3943023/8040287
def get_readable_color(r, g, b):
    if (r*0.299 + g*0.587 + b*0.114) > 0.60:
        return 0, 0, 0
    else:
        return 1, 1, 1


def get_readable_color_wavelength(wavelength):
    r, g, b, a = wavelength_to_rgb(wavelength)
    return *get_readable_color(r, g, b), a


DETAIL = True

CMAP = ListedColormap([wavelength_to_rgb(x) for x in range(380, 751)])
# CMAP = plt.get_cmap('turbo')

if __name__ == '__main__':
    data = {
        'lungs cancer': [
            # ['Wang et alfluorescence', [(500, '-', 570, 'BAC2'),
            #                           (610, '-', 730, 'BAC3')]],
            ['Wang et al (2020)', [(500, '-', 570, ''),
                                   (610, '-', 730, '')]],
            ['Wang et al (2021)', [(500, '-', 570, ''),
                                   (610, '-', 730, '')]],
        ],
        #  breast cancer
        'breast cancer': [
            ['Bianchetti et al', [(425, '-', 475, 'NAD(P)H'),
                                  (520, '-', 580, 'FAD')]],
            ['Unger et al', [(390, '/', 40, 'Collagen'),
                             (470, '/', 28, 'NADH'),
                             (542, '/', 28, 'FAD'),
                             (629, '/', 53, 'Porphyrin')]],
            ['Phipps et al', [(390, '/', 40, 'unknown'), (466, '/', 40, 'unknown'),
                              (542, '/', 50, 'unknown'), (629, '/', 53, 'unknown')]]
        ],
        #  oral cancer
        'oral cancer': [
            ['Marsden et al (2020)', [(390, '±', 20, 'Collagen'),
                                      (470, '±', 14, 'NADH'),
                                      (542, '±', 25, 'FAD')]],
            ['Jo et al', [(390, '±', 20, 'Collagen'), (452, '±', 22.5, 'NADH'),
                          (500, '>', 0, 'FAD')]],
            ['Duran et al', [(390, '±', 20, 'Collagen'),
                             (452, '±', 22, 'NADH'),
                             (500, '>', 0, 'FAD')]],
        ],
        #  cervical cancer
        'cervical cancer': [
            ['Ji et al', [(447, '±', 30, 'NAD(P)H')]],
            ['Sahoo et al', [(400, '>', 0, 'unknown')]],
            ['Gu et al', [(400, '-', 680, 'unknown')]]
        ],
        'skin cancer': [
            ['Cosci et al', [(440, '±', 20, 'NADH'), (514, '±', 15, 'FAD')]],
            ['Romano et al', [(390, '±', 20, 'Collagen'),
                              (452, '±', 22, 'NADH'),
                              (496, '>', 0, 'FAD')]],
            ['Vasanthakumari et al', [(390, '±', 20, 'Collagen'),
                              (452, '±', 22.5, 'NADH'),
                              (496, '>', 0, 'FAD')]]
        ],
        'brain cancer': [
            ['Butte et al', [(360, '-', 550, 'unknown')]]
        ],
        'oropharyngeal cancer': [
            ['Weyers et al', [(390, '±', 20, 'Collagen'), (470, '±', 14, 'NAD(P)H'),
                        (542, '±', 25, 'FAD'), (629, '±', 26.5, 'Porphyrin')]]
        ],
        #  cell
        'cell': [
            ['Kröger et al', [(375, '_', 385, '1'), (401, '-', 690, 'NAD(P)H')]],
            ['Neto et al', [(458, '/', 64, 'NAD(P)H'), (520, '/', 35, 'FAD')]],
            ['Walsh et al', [(440, '/', 80, 'NAD(P)H'),
                             (550, '/', 100, 'FAD')]],
            ['Qian et al', [(440, '/', 80, 'NAD(P)H'),
                            (550, '/', 100, 'FAD')]],
        ],
        #  paprathyroid
        'parathyroid': [
            ['Marsden et al (2021)', [(390, '±', 20, 'Collagen'),
                                      (470, '±', 14, 'NADH'),
                                      (542, '±', 25, 'FAD')]]
        ],
    }


    total_len = sum([len(data[i]) for i in data])
    # help from https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
    fig, ax = plt.subplots()
    count = 0
    for i, (k, v) in enumerate(data.items()):
        for j in range(len(v)):
            d = v[j][1]
            for wave in d:
                y = count*2
                if wave[1] == '-':
                    xs = (wave[0], wave[2])
                    x = (xs[0]+xs[1])/2
                    color = x
                    detail = f'({wave[0]}-{wave[2]})'
                elif wave[1] == '_':
                    xs = (wave[0], wave[2])
                    x = (xs[0]+xs[1])/2
                    color = x
                    detail = f''

                elif wave[1] == '/':
                    x = wave[0]
                    xs = (x-wave[2]/2, x+wave[2]/2)
                    color = x
                    detail = f'({wave[0]}/{wave[2]})'
                elif wave[1] == '±':
                    x = wave[0]
                    xs = (x-wave[2], x+wave[2])
                    color = x
                    detail = f'({wave[0]}±{wave[2]})'
                elif wave[1] == '>':
                    xs = (wave[0], 750)
                    x = (wave[0] + 750)/2
                    color = x
                    detail = f'(>{wave[0]})'
                else:
                    print('wtf')
                ax.imshow([range(round(xs[0]), round(xs[1]))],
                          extent=(xs[0], xs[1], y, y+1.9), vmin=380, vmax=750,
                          cmap=CMAP, aspect="auto", zorder=1)
                ax.annotate(v[j][0], (322.5, y+0.8), fontsize=14,
                            ha='center', zorder=2)
                if DETAIL:
                    # ax.scatter(x, y+1,
                    #            color=get_readable_color_wavelength(x),
                    #            marker='|', alpha=0.8, zorder=2)
                    ax.annotate(detail, (x, y+0.3), fontsize=14,
                                ha='center', zorder=2,
                                color=get_readable_color_wavelength(x))
                    ax.annotate(wave[3], (x, y+1.2), fontsize=13,
                                ha='center', weight='bold', zorder=2,
                                color=get_readable_color_wavelength(x))

            count += 1
        # color
        ax.imshow([[i]],
                  extent=(290, 355, ((count-1-j)*2), y+1.9),
                  vmin=0, vmax=len(data), aspect='auto', alpha=0.4,
                  cmap=plt.get_cmap('Paired'), zorder=0)
        ax.imshow([[i]],
                  extent=(760, 840, ((count-1-j)*2), y+1.9),
                  vmin=0, vmax=len(data), aspect='auto', alpha=0.4,
                  cmap=plt.get_cmap('Paired'), zorder=0)
        print(count, j, i)
    ax.set_xlim(290, 840)
    ax.set_ylim(0, total_len*2)
    ax.set_xlabel('nm')
    # ax.set_yticks([i*2+1 for i in range(total_len)])
    # ax.set_yticklabels([j[0] for i in data for j in data[i]])
    d_len = np.array([len(data[i]) for i in data])
    ticks = (np.cumsum(d_len) - d_len/2)*2
    # ## right axis
    # ax2 = ax.twinx()
    # ax2.set_yticks(ticks)
    # ax2.set_yticklabels(data.keys())
    # ax2.set_ylim(0, total_len*2)
    # ## right annotate
    for i, k in enumerate(data):
        ax.annotate(k, (800, ticks[i]-0.3), fontsize=16,
                    ha='center', zorder=2)
    # ## legend
    # patches = [mpatches.Patch(color=plt.get_cmap('Paired')(i/len(data)),
    #                           label=k)
    #            for i, k in enumerate(data)]
    # patches.reverse()
    # ax.legend(handles=patches)
    # ## columns
    columns = [(370, 410), (400, 490), (496, 600, 540), (602.5, 750),
               (602.5, 655.5)]
    for c in columns:
        if len(c) == 3:
            color = c[2]
        else:
            color = (c[0] + c[1])/2
        ax.imshow([[color]],
                  extent=(c[0], c[1], 0, total_len*3),
                  vmin=380, vmax=750,
                  aspect='auto', alpha=0.2,
                  cmap=CMAP, zorder=0)
    plt.show()


def testa(x):
    print(x)
    return x
