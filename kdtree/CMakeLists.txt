add_executable(test_kdtree test_kdtree.cc)
target_link_libraries(test_kdtree common)
install(TARGETS test_kdtree OPTIONAL)