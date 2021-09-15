#date: 2021-09-15T16:47:42Z
#url: https://api.github.com/gists/771e68d6c8b4faf0d0ece8cc7fa14f15
#owner: https://api.github.com/users/johnrichardrinehart

#!/usr/bin/env bash

fifo="tmp.fifo"

mkfifo "${fifo}"

# write the below go code to the fifo
function write_pointer_deref {
cat <<EOF > ${fifo}
     package foo
     func foo() {
         a := &struct{}{}
         _ = &*a
     }
EOF
}

# write the below go code to the fifo
function write_value {
cat <<EOF > ${fifo}
    package foo
    func foo() {
        a := &struct{}{}
        _ = a
    }
EOF
}

# -dwarf=false => no DWARF debugging symbols
# -N => don't optimize
# -S -S => print assembly listing (code and data)
# -o /dev/null => don't save the object file
write_pointer_deref &
go tool compile -dwarf=false -N -S -S -o /dev/null "${fifo}" > "pointer_deref.asm"

write_value &
go tool compile -dwarf=false -N -S -S -o /dev/null "${fifo}" > "value.asm"

printf "MD5 hash of assembly without compiler optimizations:\n%s\n%s\n\n" "$(md5sum ./pointer_deref.asm)" "$(md5sum ./value.asm)"

write_pointer_deref &
go tool compile -dwarf=false -S -S -o /dev/null "${fifo}" > "pointer_deref_unoptimized.asm"

write_value &
go tool compile -dwarf=false -S -S -o /dev/null "${fifo}" > "value_unoptimized.asm"

printf "MD5 hash of assembly with compiler optimizations: \n%s\n%s\n" "$(md5sum ./value_unoptimized.asm)" "$(md5sum ./pointer_deref_unoptimized.asm)"

# clean-up to ensure idempotency
rm "${fifo}"