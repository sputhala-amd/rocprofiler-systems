set(rocpd_sources
    ${CMAKE_CURRENT_LIST_DIR}/data_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_storage/database.cpp
)

set(rocpd_headers
    ${CMAKE_CURRENT_LIST_DIR}/data_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/json.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_storage/database.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_storage/queries/table_insert_query.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_storage/queries/query_builders/insert_query_builders.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${rocpd_sources} ${rocpd_headers})

target_link_libraries(
    rocprofiler-systems-core-library
    PRIVATE $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sqlite3>
)
