#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include "httplib.h"  // Include cpp-httplib (C++ HTTP Client)

std::vector<std::pair<std::string, int>> gpu_workers;

void zipProject(const std::string &projectPath, const std::string &zipFilePath) {
    std::string command = "zip -r " + zipFilePath + " " + projectPath;
    system(command.c_str());
}

bool sendFile(const std::string &serverIP, int serverPort, const std::string &filePath) {
    httplib::Client client(serverIP, serverPort);
    httplib::MultipartFormDataItems items = {
        {"zip_file", std::filesystem::path(filePath).filename().string(), httplib::detail::read_file(filePath), "application/zip"}
    };

    std::cout << "Sending project to " << serverIP << ":" << serverPort << "/execute_project ...\n";

    auto res = client.Post("/execute_project", items);

    if (res && res->status == 200) {
        std::cout << "Project sent successfully to " << serverIP << "\n";
        return true;
    } else {
        std::cerr << "Failed to send project to " << serverIP << "\n";
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
        if (!sendFile(ip, port, zipFilePath)) {
            std::cerr << "Failed to send project to " << ip << "\n";
        }
    }

    std::cout << "Project successfully sent to all GPUs.\n";
    return 0;
}
