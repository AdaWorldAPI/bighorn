cmake_minimum_required(VERSION 3.20)

project(graphql_agi_extension)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Extension sources
set(EXTENSION_SOURCES
    src/graphql_agi_extension.cpp
    src/graphql/schema_registry.cpp
    src/graphql/query_translator.cpp
    src/graphql/resolver_engine.cpp
    src/graphql/introspection.cpp
    src/lancedb/lancedb_connector.cpp
    src/lancedb/vector_store.cpp
    src/lancedb/hybrid_search.cpp
    src/lancedb/embedding_pipeline.cpp
    src/agi/reasoning_engine.cpp
    src/agi/planning_system.cpp
    src/agi/knowledge_graph.cpp
    src/agi/autonomous_agent.cpp
    src/agi/tool_executor.cpp
    src/ladybug/query_analyzer.cpp
    src/ladybug/performance_monitor.cpp
    src/ladybug/visualization_engine.cpp
    src/ladybug/debug_tracer.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json/include
)

# Build extension shared library
add_library(graphql_agi SHARED ${EXTENSION_SOURCES})

target_link_libraries(graphql_agi PRIVATE kuzu)

# Platform-specific settings
if(APPLE)
    set_target_properties(graphql_agi PROPERTIES SUFFIX ".kuzu_extension")
elseif(WIN32)
    set_target_properties(graphql_agi PROPERTIES SUFFIX ".kuzu_extension")
else()
    set_target_properties(graphql_agi PROPERTIES SUFFIX ".kuzu_extension")
endif()

# Installation
install(TARGETS graphql_agi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
