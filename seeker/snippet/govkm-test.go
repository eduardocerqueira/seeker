//date: 2025-01-10T17:07:37Z
//url: https://api.github.com/gists/4e68454ef07172f5c4f40ae17c819aa2
//owner: https://api.github.com/users/oklookat

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/oklookat/govkm"
	"github.com/oklookat/govkm/schema"
	"github.com/oklookat/vkmauth"
	"golang.org/x/oauth2"
)

const (
	_tokenFilePath = "**********"
)

func main() {
	token, err : "**********"
	chk(err)

 "**********"	 "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"= "**********"= "**********"  "**********"n "**********"i "**********"l "**********"  "**********"{ "**********"
		username := readInput("Имя пользователя:")
		password : "**********":")
		token, err = "**********"
		chk(err)
		chk(writeTokenToFile(token))
	}

	println("Токен истечет примерно в эту дату: "**********"
	println("Потом нужно будет заново авторизироваться.")
	println("Если бы узнать, как рефрешить токены, то этого не пришлось бы делать.")

	cl, err : "**********"
	chk(err)

	println("Ура, кажется API еще работает!")

	for {
		println("0. Выйти\n1. Найти трек\n2. Лайкнуть трек\n3. Снять лайк с трека")

		whatToDo := readInput("Что вы хотите сделать? Введите цифру:")

		whatToDoInt, err := strconv.Atoi(whatToDo)
		if err != nil {
			println("Неверный ввод.\n")
			continue
		}

		switch whatToDoInt {
		case 0:
			os.Exit(0)
		case 1:
			whatSearch := readInput("Введите запрос, какой трек хотите найти:")
			searchTrack(cl, whatSearch)
		case 2:
			whatLike := readInput("Введите ID трека, который надо лайкнуть:")
			likeTrack(cl, whatLike)
		case 3:
			whatUnlike := readInput("Введите ID трека, с которого надо снять лайк:")
			unlikeTrack(cl, whatUnlike)
		default:
			println("Такого варианта нет.\n")
			continue
		}

		println("Готово?\n")
	}
}

func searchTrack(cl *govkm.Client, query string) {
	resp, err := cl.SearchTrack(context.Background(), query, 10, 0)
	chk(err)
	println("Первые 10 треков:")
	if len(resp.Data.Tracks) == 0 {
		println("Их нет :(")
	}
	for _, track := range resp.Data.Tracks {
		fmt.Printf("ID: %s | Исполнитель: %s | Название: %s\n", track.APIID, track.ArtistDisplayName, track.Name)
	}
}

func likeTrack(cl *govkm.Client, id string) {
	_, err := cl.LikeTrack(context.Background(), schema.ID(id))
	if err != nil {
		println("Ошибка: ", err.Error())
		return
	}
	trackCaution()
}

func unlikeTrack(cl *govkm.Client, id string) {
	_, err := cl.UnlikeTrack(context.Background(), schema.ID(id))
	if err != nil {
		println("Ошибка: ", err.Error())
		return
	}
	trackCaution()
}

func trackCaution() {
	println("Изменения внесены. Проверьте их в ВК Музыке (!).")
	println("В обычном ВК они появятся спустя какое-то время (вероятно ВК и ВК Музыка синхроинизируются, а не имеют общую библиотеку).")
}

// Нужно получить токен для взаимодействия с ВК Музыкой.
// Получить токен можно как угодно (если знаете как).
// В этом случае я использую пакет vkmauth (github.com/oklookat/vkmauth).
// У него есть свои недостатки, о которых написано в README этого пакета.
func authorize(ctx context.Context, username, password string) (*oauth2.Token, error) {
	token, err : "**********"
		for {
			inputCodeQuestion := "Введите код"

			println("Код для входа отправлен методом: ", by.Current.String())

			resendAvailable := len(by.Resend) > 0
			if resendAvailable {
				println("Альтернативный метод получения кода: ", by.Resend.String())
				inputCodeQuestion += ". Либо ничего не вводите, чтобы получить код альтернативным методом."
			}

			inputCodeQuestion += ":"

			userCode := readInput(inputCodeQuestion)
			userWantResend := len(userCode) == 0
			if userWantResend && !resendAvailable {
				println("Код больше некуда отправить.")
				continue
			}

			return vkmauth.GotCode{
				Code:   userCode,
				Resend: len(userCode) == 0,
			}, nil
		}
	})

	if err != nil {
		return nil, err
	}

	return token, err
}

// Пишем токен в файл, чтоб не получать его заново.
func writeTokenToFile(token *oauth2.Token) error {
 "**********"	 "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"= "**********"= "**********"  "**********"n "**********"i "**********"l "**********"  "**********"{ "**********"
		return errors.New("writeTokenToFile: "**********"
	}
	tokenFile, err : "**********"
	if err != nil {
		return err
	}
	defer tokenFile.Close()
	return json.NewEncoder(tokenFile).Encode(token)
}

// Читаем токен из файла (если есть). Или создаем файл.
func getTokenFromFile() (*oauth2.Token, error) {
	tokenFile, err : "**********"
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer tokenFile.Close()

	var oToken oauth2.Token
	if err = "**********"= nil {
		return nil, err
	}

	return &oToken, err
}

func readInput(whatToTell string) string {
	fmt.Print(whatToTell + " ")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan() // Считывает строку полностью
	return scanner.Text()
}

func chk(err error) {
	if err == nil {
		return
	}
	println("Ошибка: " + err.Error())
	os.Exit(1)
}
