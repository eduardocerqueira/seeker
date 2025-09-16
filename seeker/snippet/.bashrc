#date: 2025-09-16T16:51:19Z
#url: https://api.github.com/gists/bd4eaa9f66328fc2d3389c11d1eed7e6
#owner: https://api.github.com/users/hexah3dron

function dip() {
        _print_container_info() {
            local container_id
            local container_ports
            local container_ip
            local container_name
            container_id="${1}"
        
            container_ports=( $(docker port "$container_id" | grep -o "0.0.0.0:.*" | cut -f2 -d:) )
            container_name="$(docker inspect --format "{{ .Name }}" "$container_id" | sed 's/\///')"
            container_ip="$(docker inspect --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" "$container_id")"
            printf "%-13s %-40s %-20s %-80s\n" "$container_id" "$container_name" "$container_ip" "${container_ports[*]}"
        }

        local container_id
        container_id="$1"
        printf "%-13s %-40s %-20s %-80s\n" 'Container Id' 'Container Name' 'Container IP' 'Container Ports'
        if [ -z "$container_id" ]; then
            local container_id
            docker ps -a --format "{{.ID}}" | while read -r container_id ; do
                _print_container_info  "$container_id"
            done
        else
            _print_container_info  "$container_id"
        fi
}