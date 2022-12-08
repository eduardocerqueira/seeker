//date: 2022-12-08T17:08:13Z
//url: https://api.github.com/gists/fcbc8e4a2889c6230d62f6f35de6862d
//owner: https://api.github.com/users/weedge

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/apache/rocketmq-client-go/v2"
	"github.com/apache/rocketmq-client-go/v2/primitive"
	"github.com/apache/rocketmq-client-go/v2/producer"
)

type assetIncrHandler func(ctx context.Context) (incrAssetCn int)

type IAssetCallBack interface {
	getAsset(ctx context.Context) (assetDto *UserAssetDto, err error)
}
type UserAssetDto struct {
	AssetCn   int   `json:"assetCn"`
	AssetType int   `json:"assetType"`
	UserId    int64 `json:"userId"`
}

func (m *UserAssetDto) getAsset(ctx context.Context) (assetDto *UserAssetDto, err error) {
	// mock
	return &UserAssetDto{AssetCn: 10000, AssetType: m.AssetType, UserId: m.UserId}, nil
}

type ILocalTx interface {
	UserAssetChangeTx(ctx context.Context, key string, lockerKey, eventMsgKey string, cb IAssetCallBack, handle assetIncrHandler) error
	CheckEventMsg(ctx context.Context, eventMsgKey string) (bool, error)
}

type AssetTxMsgListener struct {
	localTx ILocalTx
}

type DemoLocalTx struct{}

func (m *DemoLocalTx) UserAssetChangeTx(ctx context.Context, key string, lockerKey, eventMsgKey string, cb IAssetCallBack, handle assetIncrHandler) error {
	println("this is demo local tx to run UserAssetChangeTx")
	return nil
}
func (m *DemoLocalTx) CheckEventMsg(ctx context.Context, eventMsgKey string) (bool, error) {
	println("this is demo local tx to run CheckEventMsg")
	return false, nil
}

func NewAssetTxMsgListener(ltx ILocalTx) *AssetTxMsgListener {
	return &AssetTxMsgListener{
		localTx: ltx,
	}
}

func (m *AssetTxMsgListener) ExecuteLocalTransaction(msg *primitive.Message) primitive.LocalTransactionState {
	userId, _ := strconv.ParseInt(msg.GetProperty("userId"), 10, 64)
	assetType, _ := strconv.Atoi(msg.GetProperty("assetType"))
	dto := &UserAssetDto{AssetType: assetType, UserId: userId}

	err := m.localTx.UserAssetChangeTx(context.Background(), msg.GetProperty("assetKey"),
		msg.GetProperty("lockerKey"), msg.GetProperty("eMsgKey"), dto, func(ctx context.Context) (incrAssetCn int) {
			return 500
		})
	if err != nil {
		return primitive.RollbackMessageState
	}

	return primitive.CommitMessageState
}

func (m *AssetTxMsgListener) CheckLocalTransaction(msg *primitive.MessageExt) primitive.LocalTransactionState {
	res, err := m.localTx.CheckEventMsg(context.Background(), msg.GetProperty("eMsgKey"))
	if err != nil {
		return primitive.UnknowState
	}
	if res {
		return primitive.CommitMessageState
	}

	return primitive.UnknowState
}

type EventMessageBody struct {
	EventId   string `json:"eventId"`
	OpUserId  int64  `json:"opUserId"`
	EventType string `json:"eventType"`
	MsgData   string `json:"msgData"`
}
type InteractGiftEventMsgData struct {
	RoomId     int64  `json:"roomId"`
	InteractId int64  `json:"interactId"`
	UserId     int64  `json:"userId"`
	RecUserId  int64  `json:"recUserId"`
	Record     string `json:"record"`
	RecordOp   string `json:"recordOp"`
	GiftId     int64  `json:"giftId"`
}

func main() {
	namesrvs := []string{"127.0.0.1:9876"}
	groupName := "P_GID_GIFT_ASSET_CHANGE"
	traceCfg := &primitive.TraceConfig{
		Access:    primitive.Local,
		Resolver:  primitive.NewPassthroughResolver(namesrvs),
		GroupName: groupName,
	}
	p, _ := rocketmq.NewTransactionProducer(
		NewAssetTxMsgListener(&DemoLocalTx{}),
		producer.WithNsResolver(primitive.NewPassthroughResolver(namesrvs)),
		producer.WithGroupName(groupName),
		producer.WithRetry(2),
		producer.WithTrace(traceCfg),
	)
	err := p.Start()
	if err != nil {
		fmt.Printf("start producer error: %s\n", err.Error())
		os.Exit(1)
	}

	rawMsg, _ := json.Marshal(&InteractGiftEventMsgData{
		RoomId:     0,
		InteractId: 0,
		UserId:     0,
		RecUserId:  0,
		Record:     "",
		RecordOp:   "",
		GiftId:     0,
	})
	msg := primitive.NewMessage("TOPIC_ASSET_CHANGE_EVENT", rawMsg)
	eventId := ""
	msg.WithProperties(map[string]string{"eventId": eventId, "eventType": "", "userId": "", "assetType": "", "assetKey": "", "lockerKey": "", "eMsgKey": ""})
	msg.WithKeys([]string{eventId})
	msg.WithTag("TAG_SEND_GIFT")
	res, err := p.SendMessageInTransaction(context.Background(), msg)

	if err != nil {
		fmt.Printf("send message error: %s\n", err)
	} else {
		fmt.Printf("send message success: result=%s\n", res.String())
	}

	time.Sleep(5 * time.Minute)
	err = p.Shutdown()
	if err != nil {
		fmt.Printf("shutdown producer error: %s", err.Error())
	}
}
