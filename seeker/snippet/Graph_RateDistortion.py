#date: 2024-06-12T17:03:02Z
#url: https://api.github.com/gists/9e8e965b906f89176ffa0e86306c91f6
#owner: https://api.github.com/users/mrperleberg


#pip3 install numpy scipy matplotlib
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def criaCurva(rate, dist, distCurva):
    if rate[-1] < rate[0]:
        assert (dist[-1] < dist[0])
        rate = np.flipud(rate)
        dist = np.flipud(dist)

    interp = scipy.interpolate.PchipInterpolator(dist, np.log10(rate))

    rates1 = 10 ** interp(distCurva)
    return rates1

def geraGrafico(curvas, labelName = ""):

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))

    colorList = [
        "tab:blue",
        "tab:orange",
        "tab:gray",
        "tab:olive",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:cyan",
    ]

    for i in range(0, len(curvas)):
        dados = curvas[i]

        videoName = dados[0]
        rate_Pontos = np.asarray(dados[1])
        dist_Pontos = np.asarray(dados[2])

        dist_Curva = np.linspace(dist_Pontos.min(), dist_Pontos.max(), num=50, endpoint=True)
        rate_Curva = criaCurva(rate_Pontos, dist_Pontos, dist_Curva)

        ax.plot(rate_Curva, dist_Curva, '-', color=colorList[i], linewidth=2, label=videoName)
        ax.plot(rate_Pontos, dist_Pontos, 'o', color=colorList[i], linewidth=2)

    # Set axis properties
    #ax.set_title(labelName, fontsize=20)
    ax.set_xlabel("bitrate (kbps)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.grid(linewidth=0.5, alpha=0.5)
    ax.legend(ncol=2, loc="lower right")
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    fig.tight_layout(pad=0.2)

    #plt.savefig("grafico"+labelName.replace(" ", "_")+".png")
    plt.show()




def main():
    valoresUHD = [
    
    #   ["video"  ,  [kbps                                        ],  [PSNR                              ]],

        ["Video_1",  [7735.8624,  2217.8888,  994.8856,   562.5904],  [46.7327, 43.9821, 42.2013, 40.1687]],
        ["Video_2",  [8802.1384,  2816.688,   1276.548,   643.632],   [46.5293, 43.9001, 41.9001, 39.7152]],
        ["Video_3",  [48340.3184, 28727.4016, 11758.8336, 4627.5376], [45.8169, 42.0028, 37.2517, 34.2707]],
        ["Video_4",  [32855.528,  7352.5696,  2511.9648,  1158.3008], [44.1954, 41.2043, 39.7012, 38.2207]],
    ]

    geraGrafico(valoresUHD, labelName="UHD Videos")

main()
