#date: 2025-05-01T17:03:55Z
#url: https://api.github.com/gists/c1b3e4e12646aaa657036b20f656b60d
#owner: https://api.github.com/users/wilsonreis

import yt_dlp

def baixar_video_youtube(url_video, diretorio_saida="."):
    """
    Baixa um vídeo do YouTube para o diretório especificado.

    Args:
        url_video (str): A URL do vídeo do YouTube.
        diretorio_saida (str, opcional): O diretório onde o vídeo será salvo.
                                        O padrão é o diretório atual.
    """
    ydl_opts = {
        'outtmpl': f'{diretorio_saida}/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url_video])
            print(f"Vídeo baixado com sucesso para: {diretorio_saida}")
        except Exception as e:
            print(f"Ocorreu um erro ao baixar o vídeo: {e}")

if __name__ == "__main__":
    url = input("Digite a URL do vídeo do YouTube que você deseja baixar: ")
    diretorio = input("Digite o diretório onde você deseja salvar o vídeo (pressione Enter para o diretório atual): ")
    if not diretorio:
        baixar_video_youtube(url)
    else:
        baixar_video_youtube(url, diretorio)

