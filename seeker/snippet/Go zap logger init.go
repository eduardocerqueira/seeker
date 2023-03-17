//date: 2023-03-17T16:58:37Z
//url: https://api.github.com/gists/ad2fec241f129308aeee55a1131fb0c4
//owner: https://api.github.com/users/evlic

package inits

import (
	"/internal/config"
	logg "/internal/config/logger"
	"fmt"
	"github.com/spf13/viper"
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

/*
logger 使用规范
	1. 执行频次有限的可以使用 sugar
	2. 业务主流程上的，并发度较高的部分使用 zap.L()
	3. 日志分级
		- info	关键路径上使用要克制数量
		- err	影响关键业务继续执行的报 err
		- warn	非关键可继续执行流程的部分 err

	想要实现的日志效果：
		1. 自动压缩
		2. 分级输出
		3. 控制台输出彩色日志 & 日志文件输出普通非彩色日志
*/

func initLogger() error {
	cfg := config.Get().Log
	if cfg == nil {
		return ErrorInitFundamental
	}
	enConfig := zap.NewProductionEncoderConfig()
	enConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	enConfig.EncodeLevel = zapcore.CapitalLevelEncoder

	var cores []zapcore.Core

	switch cfg.Level {
	case zapcore.InfoLevel.String():
		cores = append(cores, buildZapCore(cfg, zapcore.InfoLevel, enConfig, zapcore.InfoLevel))
	case zapcore.WarnLevel.String():
		cores = append(cores, buildZapCore(cfg, zapcore.WarnLevel, enConfig, zapcore.WarnLevel))
	default:
		cores = append(cores, buildZapCore(cfg, zapcore.DebugLevel, enConfig, zapcore.DebugLevel))
	}

	// 控制台使用 color encoder
	if cfg.LogInConsole {
		enConfigStd := zap.NewProductionEncoderConfig()
		enConfigStd.EncodeTime = zapcore.ISO8601TimeEncoder
		enConfigStd.EncodeLevel = zapcore.CapitalColorLevelEncoder

		switch cfg.Level {
		case zapcore.InfoLevel.String():
			cores = append(cores, buildZapCoreStd(zapcore.InfoLevel, enConfigStd, zapcore.InfoLevel))
		case zapcore.WarnLevel.String():
			cores = append(cores, buildZapCoreStd(zapcore.WarnLevel, enConfigStd, zapcore.WarnLevel))
		default:
			cores = append(cores, buildZapCoreStd(zapcore.DebugLevel, enConfigStd, zapcore.DebugLevel))
		}

	}

	cores = append(cores, buildZapCore(cfg, zapcore.ErrorLevel, enConfig, zapcore.ErrorLevel))
	zapLogger := zap.New(zapcore.NewTee(cores...), zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))
	zap.ReplaceGlobals(zapLogger)

	logWelcomeInfo()
	return nil
}

// now: output config path in use
func logWelcomeInfo() {
	zap.S().Infof("welcome, app[pp/pid: %d/%d] init form %s", os.Getppid(), os.Getpid(), viper.ConfigFileUsed())
}

func buildZapCore(config *logg.Config, level zapcore.Level, enConfig zapcore.EncoderConfig, enabler zapcore.LevelEnabler) zapcore.Core {
	writer := &lumberjack.Logger{
		Filename:   fmt.Sprintf(config.Path + level.String() + logg.FileSufferFix),
		MaxSize:    config.MaxSize,
		MaxAge:     config.MaxAge,
		MaxBackups: config.MaxBackups,
		Compress:   config.Compress,
		LocalTime:  config.LocalTime,
	}

	var writeSyncer zapcore.WriteSyncer
	writeSyncer = zapcore.AddSync(writer)

	return zapcore.NewCore(zapcore.NewConsoleEncoder(enConfig), writeSyncer, enabler)
}

func buildZapCoreStd(level zapcore.Level, enConfig zapcore.EncoderConfig, enabler zapcore.LevelEnabler) zapcore.Core {
	//var writeSyncer zapcore.WriteSyncer
	//writeSyncer = zapcore.AddSync(os.Stdin)

	return zapcore.NewCore(zapcore.NewConsoleEncoder(enConfig), os.Stdout, enabler)
}
