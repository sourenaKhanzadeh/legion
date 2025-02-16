#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <httplib.h>  // Include cpp-httplib (C++ HTTP Client)

std::vector<std::pair<std::string, int>> gpu_workers;

void zipProject(const std::string &projectPath, const std::string &zipFilePath) {
    std::string command = "zip -r " + zipFilePath + " " + projectPath;
    system(command.c_str());
}

bool sendFile(const std::string& filePath, int port, const std::string& endpoint) {
    httplib::Client cli("0.0.0.0", port);
    std::cout << "Sending file to " << filePath << " at port " << port << " and endpoint " << endpoint << std::endl;

    // Read file content
    std::string file_content;
    httplib::detail::read_file(filePath, file_content);  
    
    // Create multipart form data
    httplib::MultipartFormDataItems items{
        {"zip_file", 
         std::filesystem::path(filePath).filename().string(), 
         file_content, 
         "application/zip"}
    };
    
    auto res = cli.Post(endpoint, items);
    
    if (res && res->status == 200) {
        std::cout << "File sent successfully" << std::endl;
        return true;
    } else {
        std::cout << "Failed to send file" << std::endl;
        return false;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <ip1> <port1> <ip2> <port2> <project_path>\n";
        return 1;
    }

    std::string projectPath = "my_project";
    if (argc == 6) {
        projectPath = argv[5];
    }

    std::string zipFilePath = projectPath + ".zip";
    gpu_workers.push_back({std::string(argv[1]), std::stoi(argv[2])});
    gpu_workers.push_back({std::string(argv[3]), std::stoi(argv[4])});

    // Compress project folder
    zipProject(projectPath, zipFilePath);

    // Send ZIP to each GPU worker
    for (auto &[ip, port] : gpu_workers) {
        if (!sendFile(zipFilePath, port, "/execute_project")) {
            std::cerr << "Failed to send project to " << ip << "\n";
        }
    }

    std::cout << "Project successfully sent to all GPUs.\n";
    return 0;
}
