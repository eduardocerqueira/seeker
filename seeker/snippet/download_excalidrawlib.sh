#date: 2022-06-15T16:56:07Z
#url: https://api.github.com/gists/b0c25870f737e0a890bd8cb0dfbebc25
#owner: https://api.github.com/users/Mearman

#!/bin/bash -e
LIBRARY_PAGE="https://libraries.excalidraw.com/?sort=downloadsTotal"

mkdir -p excalidrawlib || true
cd excalidrawlib

declare -a URLS=("
https://libraries.excalidraw.com/libraries/youritjang/software-architecture.excalidrawlib
https://libraries.excalidraw.com/libraries/youritjang/stick-figures.excalidrawlib
https://libraries.excalidraw.com/libraries/drwnio/drwnio.excalidrawlib
https://libraries.excalidraw.com/libraries/rohanp/system-design.excalidrawlib
https://libraries.excalidraw.com/libraries/cloud/cloud.excalidrawlib
https://libraries.excalidraw.com/libraries/ferminrp/awesome-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/dbssticky/data-viz.excalidrawlib
https://libraries.excalidraw.com/libraries/BjoernKW/UML-ER-library.excalidrawlib
https://libraries.excalidraw.com/libraries/spfr/lo-fi-wireframing-kit.excalidrawlib
https://libraries.excalidraw.com/libraries/michelcaradec/cloud-design-patterns.excalidrawlib
https://libraries.excalidraw.com/libraries/maeddes/technology-logos.excalidrawlib
https://libraries.excalidraw.com/libraries/arach/systems-design-components.excalidrawlib
https://libraries.excalidraw.com/libraries/aretecode/system-design-template.excalidrawlib
https://libraries.excalidraw.com/libraries/markopolo123/dev_ops.excalidrawlib
https://libraries.excalidraw.com/libraries/inwardmovement/information-architecture.excalidrawlib
https://libraries.excalidraw.com/libraries/slobodan/aws-serverless.excalidrawlib
https://libraries.excalidraw.com/libraries/aretecode/decision-flow-control.excalidrawlib
https://libraries.excalidraw.com/libraries/excacomp/web-kit.excalidrawlib
https://libraries.excalidraw.com/libraries/drwnio/storytelling.excalidrawlib
https://libraries.excalidraw.com/libraries/oehrlis/db-eng.excalidrawlib
https://libraries.excalidraw.com/libraries/ocapraro/bubbles.excalidrawlib
https://libraries.excalidraw.com/libraries/pgilfernandez/basic-shapes.excalidrawlib
https://libraries.excalidraw.com/libraries/pclainchard/it-logos.excalidrawlib
https://libraries.excalidraw.com/libraries/ferminrp/post-it.excalidrawlib
https://libraries.excalidraw.com/libraries/husainkhambaty/aws-simple-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/dwelle/network-topology-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/intradeus/algorithms-and-data-structures-arrays-matrices-trees.excalidrawlib
https://libraries.excalidraw.com/libraries/h7y/dropdowns.excalidrawlib
https://libraries.excalidraw.com/libraries/anumithaapollo12/emojis.excalidrawlib
https://libraries.excalidraw.com/libraries/kaligule/robots.excalidrawlib
https://libraries.excalidraw.com/libraries/ferminrp/gantt.excalidrawlib
https://libraries.excalidraw.com/libraries/jakubpawlina/graphs.excalidrawlib
https://libraries.excalidraw.com/libraries/simalexan/wardley-maps-symbols.excalidrawlib
https://libraries.excalidraw.com/libraries/morgemoensch/gadgets.excalidrawlib
https://libraries.excalidraw.com/libraries/niknm/systemdesignicons.excalidrawlib
https://libraries.excalidraw.com/libraries/excacomp/mobile-kit.excalidrawlib
https://libraries.excalidraw.com/libraries/corlaez/hexagonal-architecture.excalidrawlib
https://libraries.excalidraw.com/libraries/youritjang/azure-cloud-services.excalidrawlib
https://libraries.excalidraw.com/libraries/jhoughes/veeam.excalidrawlib
https://libraries.excalidraw.com/libraries/franky47/apple-devices-frames.excalidrawlib
https://libraries.excalidraw.com/libraries/kvchitrapu/data-sources.excalidrawlib
https://libraries.excalidraw.com/libraries/rkjc/schematic-symbols.excalidrawlib
https://libraries.excalidraw.com/libraries/lipis/polygons.excalidrawlib
https://libraries.excalidraw.com/libraries/xxxdeveloper/icons.excalidrawlib
https://libraries.excalidraw.com/libraries/Arqtangeles/architecture.excalidrawlib
https://libraries.excalidraw.com/libraries/shinkim/presentation-templates.excalidrawlib
https://libraries.excalidraw.com/libraries/clementbosc/gcp-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/danimaniarqsoft/scrum-board.excalidrawlib
https://libraries.excalidraw.com/libraries/ferminrp/awesome-slides.excalidrawlib
https://libraries.excalidraw.com/libraries/xxxdeveloper/system-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/esteevens/logos.excalidrawlib
https://libraries.excalidraw.com/libraries/shellerbrand/canvases.excalidrawlib
https://libraries.excalidraw.com/libraries/farisology/data-science.excalidrawlib
https://libraries.excalidraw.com/libraries/lipis/stars.excalidrawlib
https://libraries.excalidraw.com/libraries/dwelle/hearts.excalidrawlib
https://libraries.excalidraw.com/libraries/samu_x86/network-elements.excalidrawlib
https://libraries.excalidraw.com/libraries/mikhailredis/redis-grafana.excalidrawlib
https://libraries.excalidraw.com/libraries/mguidoti/google-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/braweria/customer-journey-map.excalidrawlib
https://libraries.excalidraw.com/libraries/marwinburesch/github-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/newbyca/cryptocurrencies.excalidrawlib
https://libraries.excalidraw.com/libraries/yuelfei/deep-learning.excalidrawlib
https://libraries.excalidraw.com/libraries/jdelacruz/veeam_unofficial.excalidrawlib
https://libraries.excalidraw.com/libraries/thijsdev/snowflake.excalidrawlib
https://libraries.excalidraw.com/libraries/coexist/mq.excalidrawlib
https://libraries.excalidraw.com/libraries/Vasudevatirupathinaidu/random-figure-drawings.excalidrawlib
https://libraries.excalidraw.com/libraries/selanas/it-logos.excalidrawlib
https://libraries.excalidraw.com/libraries/swissarmysam/maps.excalidrawlib
https://libraries.excalidraw.com/libraries/marwinburesch/html-input-elements.excalidrawlib
https://libraries.excalidraw.com/libraries/kinghavok/some-common-cloud-apps.excalidrawlib
https://libraries.excalidraw.com/libraries/nikordaris/team-topologies.excalidrawlib
https://libraries.excalidraw.com/libraries/claracavalcante/baby-characters.excalidrawlib
https://libraries.excalidraw.com/libraries/mguidoti/original-google-architecture-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/eho9734/2022-gantt.excalidrawlib
https://libraries.excalidraw.com/libraries/7demonsrising/azure-general.excalidrawlib
https://libraries.excalidraw.com/libraries/novakkkarel/nsx-t-vmware.excalidrawlib
https://libraries.excalidraw.com/libraries/adamkdean/comms-platform-icons.excalidrawlib
https://libraries.excalidraw.com/libraries/jjadup/mathematical-symbols.excalidrawlib
https://libraries.excalidraw.com/libraries/revolunet/raspberrypi3.excalidrawlib
https://libraries.excalidraw.com/libraries/7demonsrising/azure-network.excalidrawlib
https://libraries.excalidraw.com/libraries/shinkim/desktop-resolutions.excalidrawlib
https://libraries.excalidraw.com/libraries/rkjc/arduino-boards.excalidrawlib
https://libraries.excalidraw.com/libraries/IvanReznikov/yellow-box.excalidrawlib
https://libraries.excalidraw.com/libraries/andreandreandradecosta/3d-shapes.excalidrawlib
https://libraries.excalidraw.com/libraries/xxxdeveloper/wireframing-placeholders.excalidrawlib
https://libraries.excalidraw.com/libraries/tvoozmagnificent/coordinates.excalidrawlib
https://libraries.excalidraw.com/libraries/thebrahmnicboy/Logic-Gates.excalidrawlib
https://libraries.excalidraw.com/libraries/marcottebear/raspberrypi-zero.excalidrawlib
https://libraries.excalidraw.com/libraries/pret/chess-set.excalidrawlib
https://libraries.excalidraw.com/libraries/sudotachy/medicine.excalidrawlib
https://libraries.excalidraw.com/libraries/rochacbruno/computer-parts.excalidrawlib
https://libraries.excalidraw.com/libraries/m47812/office-items.excalidrawlib
https://libraries.excalidraw.com/libraries/sketchingdev/banners.excalidrawlib
https://libraries.excalidraw.com/libraries/sharathsanketh/biology.excalidrawlib
https://libraries.excalidraw.com/libraries/jgodoy/network-locations.excalidrawlib
https://libraries.excalidraw.com/libraries/dwelle/despair.excalidrawlib
https://libraries.excalidraw.com/libraries/7demonsrising/azure-compute.excalidrawlib
https://libraries.excalidraw.com/libraries/7demonsrising/azure-storage.excalidrawlib
https://libraries.excalidraw.com/libraries/excalidraw/valentine-s-day.excalidrawlib
https://libraries.excalidraw.com/libraries/fraoustin/bpmn.excalidrawlib
https://libraries.excalidraw.com/libraries/jasoncoelho/arduino-micro.excalidrawlib
https://libraries.excalidraw.com/libraries/mguidoti/it-tools-logos.excalidrawlib
https://libraries.excalidraw.com/libraries/tuckdiaz/internet-service-providers.excalidrawlib
https://libraries.excalidraw.com/libraries/esteevens/retail-peripherals.excalidrawlib
https://libraries.excalidraw.com/libraries/l8y/music-instruments.excalidrawlib
https://libraries.excalidraw.com/libraries/7demonsrising/azure-containers.excalidrawlib
https://libraries.excalidraw.com/libraries/dpalay/common-screen-resolutions.excalidrawlib
https://libraries.excalidraw.com/libraries/jumpingrivers/r.excalidrawlib
https://libraries.excalidraw.com/libraries/pocomane/boardgame.excalidrawlib
https://libraries.excalidraw.com/libraries/risjain/electrical-engineering.excalidrawlib
https://libraries.excalidraw.com/libraries/mppowell/circuit-components.excalidrawlib
https://libraries.excalidraw.com/libraries/jgodoy/racks-and-servers-components.excalidrawlib
https://libraries.excalidraw.com/libraries/fibreninja/fibre-network.excalidrawlib
https://libraries.excalidraw.com/libraries/zanetworker/red-hat.excalidrawlib
https://libraries.excalidraw.com/libraries/kvchitrapu/secure_shell.excalidrawlib
https://libraries.excalidraw.com/libraries/danielpza/barotrauma.excalidrawlib
https://libraries.excalidraw.com/libraries/krustvalentin/printers.excalidrawlib
")

# download all the .excalidrawlib files
for url in $URLS; do
	echo "$url"
	echo
	curl -L $url -o $(basename $url)
	echo
done
