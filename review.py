import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


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
    return (R, G, B, 0.8)


DETAIL = True

CMAP = ListedColormap([wavelength_to_rgb(x) for x in range(380, 751)])


if __name__ == '__main__':
    data = [
        ['neto2022', [(458, '/', 64), (520, '/', 35)]],
        ['bianchetti2021', [(425, '-', 475), (520, '-', 580)]],
        ['walsh2021', [(440, '/', 80), (550, '/', 100)]],
        ['qian2021', [(440, '/', 80), (550, '/', 100)]],
        ['marsden2021', [(390, '±', 20), (470, '±', 14), (542, '±', 25)]],
        ['marsden2020', [(390, '±', 20), (470, '±', 14), (542, '±', 25)]],
        ['unger2020', [(390, '/', 40), (470, '/', 28), (542, '/', 28),
                       (629, '/', 53)]],
        ['jo2018', [(390, '±', 20), (452, '±', 22.5), (500, '>')]],
        ['sahoo2018', [(400, '>')]],
        ['phipps2017', [(390, '/', 40), (466, '/', 40), (542, '/', 50),
                       (629, '/', 53)]],
        ['gu2014', [(400, '-', 680)]]
    ]
    # help from https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
    fig, ax = plt.subplots()
    for i in range(len(data)):
        d = data[i][1]
        for j in d:
            y = i*3
            if j[1] == '-':
                xs = (j[0], j[2])
                x = (xs[0]+xs[1])/2
                color = x
                detail = f'({j[0]}-{j[2]})'
            elif j[1] == '/':
                x = j[0]
                xs = (x-j[2]/2, x+j[2]/2)
                color = x
                detail = f'({j[0]}/{j[2]})'
            elif j[1] == '±':
                x = j[0]
                xs = (x-j[2], x+j[2])
                color = x
                detail = f'({j[0]}±{j[2]})'
            elif j[1] == '>':
                xs = (j[0], 750)
                x = (j[0] + 750)/2
                color = x
                detail = f'(>{j[0]})'
            else:
                print('wtf')
            ax.imshow([np.array(range(round(xs[0]-380),
                                      round(xs[1]-379))) / 370],
                      extent=(xs[0], xs[1], y, y+2), vmin=0, vmax=1, cmap=CMAP,
                      aspect="auto")
            if DETAIL:
                ax.scatter(x, y+1,
                           color='black', marker='|', alpha=0.8)
                ax.annotate(detail, (x, y+0.3),
                            fontsize=10, ha='center')
    ax.set_xlim(310, 770)
    ax.set_ylim(-1, len(data)*3)
    ax.set_xlabel('nm')
    ax.set_yticks([i*3+1 for i in range(len(data))])
    ax.set_yticklabels([i[0]for i in data])
    plt.show()
