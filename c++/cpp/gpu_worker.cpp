#include <iostream>
#include <fstream>
#include <filesystem>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/socket.h>
#include <vector>
#include <cstring>
#include <cstdlib>

#define PORT 5001  // Change this per GPU worker

void receiveFile(int clientSocket, const std::string& outputPath) {
    std::ofstream outFile(outputPath, std::ios::binary);
    char buffer[4096];
    int bytesReceived;

    while ((bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0)) > 0) {
        outFile.write(buffer, bytesReceived);
    }

    outFile.close();
    std::cout << "File received: " << outputPath << std::endl;
}

void extractAndRunProject(const std::string& zipFilePath) {
    std::filesystem::path extractPath = "gpu_project";
    
    // Ensure extract directory exists
    std::filesystem::create_directory(extractPath);

    // Extract ZIP file
    std::string command = "unzip -o " + zipFilePath + " -d " + extractPath.string();
    system(command.c_str());

    // Run main script
    std::string executeCommand = "cd " + extractPath.string() + " && python3 main.py";
    system(executeCommand.c_str());

    std::cout << "Project executed successfully.\n";
}

int main() {
    int serverSocket, clientSocket;
    struct sockaddr_in serverAddr{}, clientAddr{};
    socklen_t addrLen = sizeof(clientAddr);

    // Create socket
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Socket creation failed.\n";
        return -1;
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    // Bind socket
    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Binding failed.\n";
        return -1;
    }

    // Listen for connections
    listen(serverSocket, 3);
    std::cout << "Waiting for project files on port " << PORT << "...\n";

    while (true) {
        clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &addrLen);
        if (clientSocket < 0) {
            std::cerr << "Failed to accept connection.\n";
            continue;
        }

        std::cout << "Connected to sender!\n";
        receiveFile(clientSocket, "received_project.zip");
        close(clientSocket);

        // Extract and execute the project
        extractAndRunProject("received_project.zip");
    }

    close(serverSocket);
    return 0;
}
