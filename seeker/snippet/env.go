//date: 2024-09-02T16:44:34Z
//url: https://api.github.com/gists/17864e071fd0fbd17b33e511687112d1
//owner: https://api.github.com/users/yszkst

package util

import "os"

// example
//
//	es := EnvSetter()
//	defer es.Recover()
//	es.Setenv("key1", "v1")
//	es.Setenv("key2", "v2")
//	es.Unsetenv("key3")
//
// ...
type EnvSetter struct {
	// 環境変数のキーと値のペア
	// 値がnilの場合はRecoverでunsetする
	envvars map[string]*string
}

// 上書き前の値保存
// すでに同じkeyが存在していればそのままにする
func (es *EnvSetter) persistCurrent(key string) {
	_, alreadyHas := es.envvars[key]
	if !alreadyHas {
		cur, has := os.LookupEnv(key)
		if has {
			es.envvars[key] = &cur
		} else {
			es.envvars[key] = nil
		}
	}
}

func (es *EnvSetter) Setenv(key string, value string) error {
	es.persistCurrent(key)
	return os.Setenv(key, value)
}

func (es *EnvSetter) Unsetenv(key string) error {
	es.persistCurrent(key)
	return os.Unsetenv(key)
}

func (es *EnvSetter) Recover() {
	for key, value := range es.envvars {
		if value == nil {
			os.Unsetenv(key)
		} else {
			os.Setenv(key, *value)
		}
	}
}
