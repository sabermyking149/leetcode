#ifndef _PUB_H_
#define _PUB_H_

#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

#define MAX(a, b) \
({ \
    decltype(a) _a = (a); \
    decltype(b) _b = (b); \
    (void)(&_a == &_b); \
    _a > _b ? _a : _b; \
})

constexpr unsigned short w = 1024;
constexpr int mod = 1000000007;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

template <typename T>
class Trie {
public:
    T val;
    int timestamp; // 单词时间戳
    bool IsEnd;
    vector<Trie<T> *> children;
    Trie() : IsEnd(false) {}
    Trie(T t) : val(t), IsEnd(false) {}

    // 此处其实是递归删除
    ~Trie() {
        for (auto child : children) {
            delete child;
        }
    }

    static void CreateWordTrie(Trie<char> *root, string& word);
    static void CreateWordTrie(Trie<char> *root, string& word, int timestamp);
};

class FileSystem {
public:
    FileSystem()
    {
        root = new Trie<string>("/");
    }

    vector<string> ls(string path);
    void mkdir(string path);
    void addContentToFile(string filePath, string content);
    string readContentFromFile(string filePath);
private:
    unordered_map<string, string> fileContent; // path - content
    Trie<string> *root = nullptr;
};


// 自定义哈希仿函数
template <typename T1, typename T2>
class MyHash {
public:
    size_t operator() (const pair<T1, T2>& a) const
    {
        // return reinterpret_cast<size_t>(a.first);
        return hash<T1>()(a.first) ^ hash<T2>()(a.second);
    }
};

template <typename T>
struct VectorHash {
    size_t operator()(const std::vector<T>& v) const {
        hash<T> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

class UnionFind {
private:
    vector<int> parent;  // 存储每个节点的父节点
    vector<int> rank;    // 存储每个根节点所对应的树的秩（高度）

    // 查找元素x所在集合的代表元素，并进行路径压缩
    int find(int x)
    {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 路径压缩
        }
        return parent[x];
    }

public:
    // 构造函数，初始化parent和rank数组
    UnionFind(int size) : parent(size), rank(size, 0)
    {
        for (int i = 0; i < size; ++i) {
            parent[i] = i;  // 初始时，每个节点的父节点是它自己
        }
    }

    // 合并两个元素所在的集合
    void unionSets(int x, int y)
    {
        int xRoot = find(x);
        int yRoot = find(y);

        if (xRoot == yRoot) {
            return;  // 如果x和y已经在同一个集合中，则不需要合并
        }

        // 按秩合并：将秩较小的树合并到秩较大的树下
        if (rank[xRoot] < rank[yRoot]) {
            parent[xRoot] = yRoot;
        } else if (rank[xRoot] > rank[yRoot]) {
            parent[yRoot] = xRoot;
        } else {
            parent[yRoot] = xRoot;
            rank[xRoot] += 1;  // 如果秩相等，选择一个作为根，并增加其秩
        }
    }

    // 查找元素x所在集合的代表元素
    int findSet(int x)
    {
        return find(x);
    }
};
#endif