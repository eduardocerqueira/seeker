#date: 2022-08-31T17:11:59Z
#url: https://api.github.com/gists/ff0a435d9d0dc304715d37b088544d1e
#owner: https://api.github.com/users/andySigler

def run(protocol):

    # NOTE: pretend I loaded labware/pipettes

    protocol.add_volume_to(trough['A1'], volume=5000, name='water')

    water_in_plate = liquid_class.Water(pipette=pipette, tip=tiprack)
    pipette.default_liquid_class = water_in_plate

    water_in_trough = liquid_class.Water(pipette=pipette, tip=tiprack)
    water_in_trough.flow_rate = 100             # modify the default Liquid-Class

    pipette.pick_up_tip()
    pipette.aspirate(200, trough['A1'], liq_cls=water_in_trough)
    pipette.dispense(200, plate['A1'])
    pipette.drop_tip()

    pipette.transfer(100, plate['A1'], plate['A2'], liq_cls=water_in_trough)
    
