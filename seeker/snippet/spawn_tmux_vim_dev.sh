#date: 2021-10-01T00:57:55Z
#url: https://api.github.com/gists/e5f13ff777d76a435d4d7486195a9a43
#owner: https://api.github.com/users/angelogladding

WORKING=~/Working
WINDOW=$(tmux new-window -c $WORKING/gaea -n PyLow -d -P)
tmux send-keys -t$WINDOW "vi pyproject.toml" ENTER
PANE=$(tmux split-window -c $WORKING/understory -d -t$WINDOW -P)
tmux send-keys -t$PANE "vi pyproject.toml" ENTER
PANE=$(tmux split-window -c $WORKING/microformats-python -d -t$WINDOW -P)
tmux send-keys -t$PANE "vi pyproject.toml" ENTER
