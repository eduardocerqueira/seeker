//date: 2023-03-31T16:43:14Z
//url: https://api.github.com/gists/2dc3c4d474ace3b6384a4e92b9f82fcd
//owner: https://api.github.com/users/hujianxiong

package watch

import (
	"encoding/json"
	"fmt"
	"github.com/go-admin-team/go-admin-core/sdk"
	v1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"log"
	"talos-admin/app/k8s/service"
	"talos-admin/common/pkg/ws"
	"time"
)


// InitWatcherInformer 创建Resource informer
func InitWatcherInformer() {
	// TODO: 需要监听集群信息变更然后重新watch events
	clusterList := []K8sCluster{}
	db := sdk.Runtime.GetDbByKey("*")

	err := db.Model(&K8sCluster{}).Find(&clusterList).Error
	if err != nil {
		log.Printf("InitWatcherPoll Get Cluster List Faild: %v", err.Error())
		return
	}
	for _, v := range clusterList {
		log.Printf("Cluster ID : %d - Cluster Name: %s - Cluster Status: %d", v.Id, v.Name, v.Status)
		if v.KubeConfig.String() != "" && v.Status == 1 {
			k8sClient, err := service.NewK8sClient(v.KubeConfig.String())
			if err != nil {
				log.Printf("Init Cluster [%s] K8sClient Faild: %s", v.Name, err.Error())
				continue
			}
			go eventInformer(v, k8sClient)
		}

	}

}

// eventInformer
func eventInformer(v K8sCluster, k8sClient *kubernetes.Clientset) {
	log.Printf("Cluster [%v] ", v.Name)	
	watchlist := cache.NewListWatchFromClient(
		k8sClient.CoreV1().RESTClient(),
		"events",
		corev1.NamespaceAll,
		fields.Everything(),
	)

	_, controller := cache.NewInformer(
		watchlist,
		&corev1.Event{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				event, ok := obj.(*corev1.Event)
				if ok {
					log.Println("AddFunc")
				}
			},
			DeleteFunc: func(obj interface{}) {
				event, ok := obj.(*corev1.Event)
				if ok {
					log.Println("DeleteFunc")
				}
				
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				event, ok := newObj.(*corev1.Event)
				if ok {
					event, ok := obj.(*corev1.Event)
					if ok {
						log.Println("UpdateFunc")
					}
				}
			},
		},
	)

	stop := make(chan struct{})
	defer close(stop)
	go controller.Run(stop)
	for {
		time.Sleep(time.Second)
	}
}