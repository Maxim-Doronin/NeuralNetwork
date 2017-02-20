set(target "NeuralNetwork")

file(GLOB incl "../include/*.h*")
file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp")

add_library(${target} STATIC ${srcs} ${hdrs} ${incl})
target_link_libraries(${target} ${LIBRARY_DEPS})
set(LIBRARY_DEPS "${LIBRARY_DEPS};${target}" PARENT_SCOPE)

