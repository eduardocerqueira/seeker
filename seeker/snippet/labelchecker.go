//date: 2023-06-07T16:54:22Z
//url: https://api.github.com/gists/f6d6caa6a157bffaa6718b414c91b3c0
//owner: https://api.github.com/users/jparrill

package main

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type labelChecker struct {
	allowedPrefixes []string
	currentKey      string
	currentValue    string
}

func (lc *labelChecker) getIdentifierLabel(pod *corev1.Pod) {
	for _, prefix := range lc.allowedPrefixes {
		if value, exists := pod.Labels[prefix]; exists {
			lc.currentKey = prefix
			lc.currentValue = value
			break
		}
	}
}

func main() {

	labels1 := map[string]string{
		"capk.cluster.x-k8s.io/kubevirt-machine-name":      "example-rscsk-bl9mj",
		"capk.cluster.x-k8s.io/kubevirt-machine-namespace": "e2e-clusters-7xb7c-example-rscsk",
		"cluster.x-k8s.io/cluster-name":                    "example-rscsk-d9qd8",
		"cluster.x-k8s.io/role":                            "worker",
		"hypershift.openshift.io/infra-id":                 "example-rscsk-d9qd8",
		"hypershift.openshift.io/nodepool-name":            "example-rscsk",
		"kubevirt.io":                                      "virt-launcher",
		"kubevirt.io/created-by":                           "9ef3e1df-75ba-486c-9a64-60fa6661ca3e",
		"kubevirt.io/vm":                                   "example-rscsk-bl9mj",
		"vm.kubevirt.io/name":                              "example-rscsk-bl9mj",
	}

	labels2 := map[string]string{
		"capk.cluster.x-k8s.io/kubevirt-machine-name":      "example-rscsk-bl9mj",
		"capk.cluster.x-k8s.io/kubevirt-machine-namespace": "e2e-clusters-7xb7c-example-rscsk",
		"cluster.x-k8s.io/cluster-name":                    "example-rscsk-d9qd8",
		"cluster.x-k8s.io/role":                            "worker",
		"hypershift.openshift.io/infra-id":                 "example-rscsk-d9qd8",
		"hypershift.openshift.io/nodepool-name":            "example-rscsk",
		"kubevirt.io":                                      "virt-launcher",
		"kubevirt.io/created-by":                           "9ef3e1df-75ba-486c-9a64-60fa6661ca3e",
		"kubevirt.io/vm":                                   "example-rscsk-bl9mj",
		"app":                                              "example-rscsk-bl9mj",
		"vm.kubevirt.io/name":                              "example-rscsk-bl9mj",
	}

	pods := []corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test1",
				Labels: labels1,
			},
		},

		{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test2",
				Labels: labels2,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test3",
				Labels: labels1,
			},
		},
	}

	lc := &labelChecker{}
	lc.allowedPrefixes = []string{"app", "name", "kubevirt.io"}

	for _, pod := range pods {
		lc.getIdentifierLabel(&pod)
		fmt.Printf("\n%s: %s\n", lc.currentKey, lc.currentValue)
	}
}
