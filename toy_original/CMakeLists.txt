macro(add_toy_chapter name)
add_dependencies(Toy_Original ${name})
add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

add_subdirectory(toy)