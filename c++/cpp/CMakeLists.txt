cmake_minimum_required(VERSION 3.10)

# Project Name
project(ProjectDistributor)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the source files
set(MASTER_SOURCE master_sender.cpp)
set(WORKER_SOURCE gpu_worker.cpp)

# Create Executable for Master (Sender)
add_executable(master_sender ${MASTER_SOURCE})

# Create Executable for GPU Worker (Receiver)
add_executable(gpu_worker ${WORKER_SOURCE})

# Link Required Libraries (For Linux Networking)
if(UNIX)
    target_link_libraries(master_sender pthread)
    target_link_libraries(gpu_worker pthread)
endif()
