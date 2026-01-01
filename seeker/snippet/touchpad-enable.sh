#date: 2026-01-01T17:16:09Z
#url: https://api.github.com/gists/4890688a9898b76a1db222b749d235f8
#owner: https://api.github.com/users/lucascesar918

### Needed packages

- libinput-tools (for libinput)
- xorg-xinput (for xinput)

### List devices

Using libinput:
```bash
sudo libinput list-devices
```

Using xinput:
```bash
xinput list
```

### Listing properties

```bash
xinput list-props "Synaptics TM3096-006"
```

### Setting property

```bash
xinput set-prop "Synaptics TM3096-006" 324 0
```