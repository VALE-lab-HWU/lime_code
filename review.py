import matplotlib.pyplot as plt


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
    return (R, G, B, 0.7)


DETAIL = True

if __name__ == '__main__':
    data = [['bianchetti', [(425, 475), (520, 580)]],
            ['phipps', [(350, 430), (426, 506), (492, 592), (576, 682)]],
            ['unger', [(350, 430), (442, 498), (514, 570), (576, 682)]],
            ['marsden', [(370, 410), (456, 484), (517, 567)]],
            ['jo', [(370, 410), (429.5, 474.5), (500,)]]]
    # help from https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
    fig, ax = plt.subplots()
    for i in range(len(data)):
        d = data[i][1]
        for j in d:
            if len(j) == 1:
                l = 750-j[0]
                x = j[0] + l/2
                y = i*3
                ax.broken_barh([(j[0], l)], (y, 2),
                               facecolors=wavelength_to_rgb(j[0]))
                if DETAIL:
                    ax.scatter(x, y+1,
                               color='black', marker='|', alpha=0.8)
                    ax.annotate(f'(>{j[0]})', (x, y+0.6),
                                fontsize=10, ha='center')
            elif len(j) == 2:
                xs = (j[0], j[1]-j[0])
                x = xs[0]+xs[1]/2
                y = i*3
                ax.broken_barh([xs], (y, 2),
                               facecolors=wavelength_to_rgb(x))
                if DETAIL:
                    ax.scatter(x, y+1,
                               color='black', marker='|', alpha=0.8)
                    ax.annotate(f'({xs[0]+xs[1]/2}/{xs[1]/2})', (x, y+0.6),
                                fontsize=10, ha='center')
            else:
                print('wtf')
    ax.set_xlim(300, 900)
    ax.set_ylim(-1, len(data)*3)
    ax.set_xlabel('nm')
    ax.set_yticks([i*3+1 for i in range(len(data))])
    ax.set_yticklabels([i[0]for i in data])
    plt.show()
