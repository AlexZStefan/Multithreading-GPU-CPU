add_library(PS_lib
    cpu_compute.cpp
    gpu_compute.cpp
)

target_include_directories(PS_lib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(PS_lib PUBLIC
    OpenGL::GL
    GLEW::GLEW
    glfw
)

add_executable(PS main.cpp)
target_link_libraries(PS PRIVATE PS_lib)
