#include "main.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Cannot open file: " << argv[1] << std::endl;
        return 1;
    }

    int V, E;
    input_file >> V >> E;

    // 分配主机内存：距离矩阵 V x V
    std::vector<int> h_dist(V * V, INF);

    // 初始化：自己到自己为 0
    for (int i = 0; i < V; ++i) {
        h_dist[i * V + i] = 0;
    }

    // 读边
    for (int i = 0; i < E; ++i) {
        int src, dst, w;
        input_file >> src >> dst >> w;
        h_dist[src * V + dst] = w; // 有向边
    }

    input_file.close();

    // 分配设备内存
    int *d_dist;
    hipMalloc(&d_dist, V * V * sizeof(int));
    hipMemcpy(d_dist, h_dist.data(), V * V * sizeof(int), hipMemcpyHostToDevice);

    // 调用 GPU kernel
    solve(d_dist, V);

    // 拷回结果
    hipMemcpy(h_dist.data(), d_dist, V * V * sizeof(int), hipMemcpyDeviceToHost);

    // 🔧 修复点：按行输出，每行 V 个数字，行末无多余空格
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            std::cout << h_dist[i * V + j];
            if (j < V - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl; // 每行结束后换行！
    }

    hipFree(d_dist);
    return 0;
}