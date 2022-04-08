//date: 2022-04-08T17:05:42Z
//url: https://api.github.com/gists/595e2a3e9ceeb53e7d8a1df0a4033aee
//owner: https://api.github.com/users/stinkyfingers


import (
	clientFake "k8s.io/client-go/kubernetes/fake"
	clientTest "k8s.io/client-go/testing"
	"k8s.io/apimachinery/pkg/watch"
	"github.com/stretchr/testify/require"
)

type watchReactor struct {
	action  clientTest.Action
	watcher watch.Interface
	err     error
}

func (w *watchReactor) Handles(action clientTest.Action) bool {
	return true
}
func (w *watchReactor) React(action clientTest.Action) (bool, watch.Interface, error) {
	return true, w.watcher, w.err
}

func TestWatchPodEventsError(t *testing.T) {
	clientset := clientFake.NewSimpleClientset()
	c := Client{
		Clientset: clientset,
	}
	watchReaction := &watchReactor{
		err: fmt.Errorf("MY BIG ERROR"),
	}
	cli.WatchReactionChain = []clientTest.WatchReactor{watchReaction}
	err := c.WatchPodEvents()
	require.EqualErr(t, err, "MY BIG ERROR")
}