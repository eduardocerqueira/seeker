#date: 2024-02-05T17:04:45Z
#url: https://api.github.com/gists/5ead1a75f4e5a154b2f42c23239385f2
#owner: https://api.github.com/users/emilstahl97

sudo ovs-ofctl show br-int
OFPT_FEATURES_REPLY (xid=0x2): dpid:00003a168a273dd7
n_tables:254, n_buffers:0
capabilities: FLOW_STATS TABLE_STATS PORT_STATS QUEUE_STATS ARP_MATCH_IP
actions: output enqueue set_vlan_vid set_vlan_pcp strip_vlan mod_dl_src mod_dl_dst mod_nw_src mod_nw_dst mod_nw_tos mod_tp_src mod_tp_dst
 2(vm1): addr:2a:71:9f:28:c6:8a
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 3(vm2): addr:2a:d6:35:3b:a6:78
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 4(vm3): addr:b6:18:bc:63:29:c6
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 5(vm4): addr:2e:8a:62:f8:61:96
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 6(vm5): addr:86:f8:45:3c:fd:b0
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 7(vm6): addr:0e:7d:3a:60:7f:9a
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 8(vm7): addr:d2:44:47:55:c0:f3
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 9(vm8): addr:c6:cc:fb:b9:9f:db
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 10(vm9): addr:2a:c6:02:88:bc:4b
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 11(vm10): addr:8a:cd:ce:eb:9b:7c
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
 12(enp2s0): addr:52:54:00:64:ee:00
     config:     0
     state:      0
     speed: 0 Mbps now, 0 Mbps max
 LOCAL(br-int): addr:3a:16:8a:27:3d:d7
     config:     PORT_DOWN
     state:      LINK_DOWN
     speed: 0 Mbps now, 0 Mbps max
OFPT_GET_CONFIG_REPLY (xid=0x4): frags=normal miss_send_len=0