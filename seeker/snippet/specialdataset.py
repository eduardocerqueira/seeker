#date: 2023-01-11T17:00:49Z
#url: https://api.github.com/gists/b54b31bc10574136bb8dcc458672bdf0
#owner: https://api.github.com/users/akcoguzhan

import os
import subprocess
from pydub import AudioSegment
import config
import pandas
from managers.logmanager import log_manager

class track_manager:
    @staticmethod
    def remove_broken_files():
        """
        İşlem görülemeyen bozuk dosyaları siler
        """
        for genre_dir in os.listdir(config.DATA_GROUPED_BY_GENRE):
            track_path = os.path.join(config.DATA_GROUPED_BY_GENRE, genre_dir)
            for track in os.listdir(track_path):
                track_name_len = len(str(track).split(sep="."))
                if track_name_len != 4:
                    os.remove(os.path.join(track_path, track))
                    log_manager.log("removed broken file: " + str(os.path.join(track_path, track)))

    @staticmethod
    def genre_name_normalizer(genre_name: str):
        if genre_name == "Hip-Hop":
            return "HipHop"
        if genre_name == "Soul-RnB":
            return "SoulRnB"
        if genre_name == "Old-Time / Historic":
            return "OldTimeHistoric"
        if genre_name == "Easy Listening":
            return "EasyListening"
        return genre_name

    @staticmethod
    def mp3_to_wav(source_file_path: str, dest_file_path: str, file_name: str, new_file_name: str, keep_original_file: bool):
        """
        mp3 uzantılı dosyaları wav uzantılı dosyaya çevirir

        :param source_file_path: mp3 dosyasının bulunduğu dizin
        :param dest_file_path: wav dosyasının oluşturulacağı dizin
        :param file_name: mp3 dosyasının adı
        :param new_file_name: yeni dosyanın adı
        :param keep_original_file: kaynak dosya korunsun mu?
        """
        try:
            subprocess.call(['ffmpeg', '-i', str(os.path.join(source_file_path, file_name)),
                             str(os.path.join(dest_file_path, new_file_name + '.wav'))],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_manager.log(
                "converted: " + os.path.join(source_file_path, file_name) + " to " + os.path.join(dest_file_path, new_file_name) + '.wav')

            if not keep_original_file:
                os.remove(os.path.join(source_file_path, file_name))
                log_manager.log("removed: " + os.path.join(source_file_path, file_name))
        except Exception as e:
            log_manager.log("error: {0}".format(e))
            # hata alınıyorsa ses dosyası bozulmuş olabilir, işlemlerin durmasını istemiyoruz
            pass

    @staticmethod
    def split_wav_to_parts(interval: int, source_file_path: str, file_name: str, dest_file_path: str, keep_original_file: bool):
        """
        verilen ses dosyasını n adet interval aralıklı parçaya böler
        :param interval: ses dosyasının bölünmesi istenilen parça adedi
        :param file_name: ses dosyasının adı
        :param source_file_path: ses dosyasının dizini
        :param dest_file_path: hedef dizin
        :param keep_original_file: kaynak dosya korunsun mu?
        """
        try:
            sound = AudioSegment.from_file(str(os.path.join(source_file_path, file_name)))
            i = 0
            a = 0
            split_count = int(len(sound) / interval)
            # 30000 milisaniyelik ses dosyalarının olmaması durumunda hata almamak adına eşik sınırı 27000 olarak belirlenmiştir
            if len(sound) > 26999:
                while i < len(sound) and a < split_count:
                    t1 = i
                    t2 = i + interval
                    new_audio = sound[t1:t2]
                    new_audio.export(
                        str(os.path.join(dest_file_path, file_name.rsplit(".", 1)[0] + '.' + str(a) + '.wav')),
                        format="wav")
                    log_manager.log("split " + os.path.join(source_file_path, file_name) + " to part_iter[" + str(
                        a) + "]" + " as " + str(
                        os.path.join(dest_file_path, file_name.rsplit(".", 1)[0] + '.' + str(a) + '.wav')))
                    i += interval
                    a += 1

                if not keep_original_file:
                    os.remove(os.path.join(source_file_path, file_name))
                    log_manager.log("removed: " + os.path.join(source_file_path, file_name))
        except Exception as e:
            log_manager.log("error: {0}".format(e))
            # hata alınıyorsa ses dosyası bozulmuş olabilir, işlemlerin durmasını istemiyoruz
            pass

    @staticmethod
    def preprocess():
        """
        FMA DATASINI LİBROSA İLE KULLANIMA HAZIR HALE GETİRMEK İÇİN ÖN İŞLEMLER
            * fma datasetini filtreleme
            * dataset ile birlikte gelen ses dosyalarını uygun türlere ayırma
            * ses dosyalarını daha fazla veri elde edebilmek adına parçalara ayırma
        """
        #region Track Info içerisinde bulunmayan ses dosyalarının silinmesi
        track_info = pandas.read_csv(config.FILE_TRACKS_INFO)
        true_track_id = []
        for t in track_info["track_id"].tolist():
            len_t = len(str(t))
            if len_t != 6:
                zeros_to_add = 6 - len_t
                for i in range(0, zeros_to_add):
                    t = '0' + str(t)
            true_track_id.append(str(t))

        for file in os.listdir(config.DATA_ALL):
            file_name = file.rsplit(".", 1)[0]
            if file_name not in true_track_id:
                if not os.path.exists(os.path.join(config.DATA_ALL, file)):
                    break
                os.remove(os.path.join(config.DATA_ALL, file))
                log_manager.log("removed unused file: " + str(os.path.join(config.FILE_TRACKS_INFO, file)))
        #endregion

        #region Track Info içerisinde bulunan türlere ait alt dizinlerin oluşturulması
        genre_names = track_info["track_genre_top"].unique().tolist()
        for i in range(0, len(genre_names)):
            genre_names[i] = track_manager.genre_name_normalizer(genre_names[i])

        for genre_name in genre_names:
            genre_path = os.path.join(config.DATA_GROUPED_BY_GENRE, genre_name)
            if not os.path.exists(genre_path):
                os.mkdir(genre_path)
                log_manager.log("created dir: " + genre_path)
        #endregion

        #region Ses dosyalarının türüne ait alt dizine taşınması
        genre_name_counter = dict()
        for i in genre_names:
            genre_name_counter[i] = genre_name_counter.get(i, 0)

        for track in os.listdir(config.DATA_ALL):
            if not os.path.exists(os.path.join(config.DATA_ALL, str(track))):
                break

            track_name = track.rsplit(".", 1)[0]
            track_extension = track.rsplit(".", 1)[1]
            track_name = str(int(track_name))
            track_genre = track_manager.genre_name_normalizer(str(track_info.loc[track_info['track_id'] == int(track_name), 'track_genre_top'].iloc[0]))

            genre_file_counter = str(genre_name_counter[track_genre])
            while len(genre_file_counter) < 6:
                genre_file_counter = '0' + genre_file_counter

            os.replace(src=os.path.join(config.DATA_ALL, str(track)), dst=str(os.path.join(config.DATA_GROUPED_BY_GENRE, track_genre)) + "\\" + track_genre + "." + genre_file_counter + "." + track_extension)
            log_manager.log("moved " + str(os.path.join(config.DATA_ALL, str(track))) + " to as " + str(os.path.join(config.DATA_GROUPED_BY_GENRE, track_genre)) + "\\" + track_genre + "." + genre_file_counter + "." + track_extension)
            genre_name_counter[track_genre] += 1
        #endregion

        #region Ses dosyalarının aynı dizinde wav formatına çevrilmesi ve kaynak dosyanın silinmesi
        for item in os.listdir(config.DATA_GROUPED_BY_GENRE):
            genre_dir = os.path.join(config.DATA_GROUPED_BY_GENRE, item)
            for track in os.listdir(genre_dir):
                file_ext = str(track).rsplit(".", 1)[1]
                if file_ext != "mp3":
                    break
                track_manager.mp3_to_wav(file_name=track, source_file_path=genre_dir, dest_file_path=genre_dir, new_file_name=str(track).rsplit(".", 1)[0], keep_original_file=False)
        #endregion

        #region Wav uzantılı ses dosyalarının aynı dizinde parçalara ayrılması ve kaynak dosyanın silinmesi
        try:
            for item in os.listdir(config.DATA_GROUPED_BY_GENRE):
                genre_dir = os.path.join(config.DATA_GROUPED_BY_GENRE, item)
                for track in os.listdir(genre_dir):
                    can_split = len(track.split(sep=".")) == 3
                    if can_split:
                        track_manager.split_wav_to_parts(interval=9000, source_file_path=os.path.join(config.DATA_GROUPED_BY_GENRE, genre_dir),
                                                         file_name=track, dest_file_path=os.path.join(config.DATA_GROUPED_BY_GENRE, genre_dir),
                                                         keep_original_file=False)
                    else:
                        raise StopIteration
        except StopIteration:
            pass
        #endregion
