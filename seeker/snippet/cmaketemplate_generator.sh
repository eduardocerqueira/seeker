#date: 2024-11-18T17:12:55Z
#url: https://api.github.com/gists/9fd6099d04ea257d8e7def018688e3a5
#owner: https://api.github.com/users/kaandesu

#!/bin/bash

cat >CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(MyProgram C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")


add_executable(my_program
  src/main.c
)

if(APPLE)
    set(HOMEBREW_DIR "/opt/homebrew/include/")
    include_directories(${HOMEBREW_DIR} ${SRC_DIR})
    target_link_libraries(my_program 
        ${CMAKE_DL_LIBS} 
    )
else()
    include_directories(${SRC_DIR})
    target_link_libraries(my_program
      ${CMAKE_DL_LIBS}
    )
endif()
EOF

echo "CMakelists.txt file created!"
