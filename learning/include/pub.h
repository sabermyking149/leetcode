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
#endif