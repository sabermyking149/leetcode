#ifndef _PUB_H_
#define _PUB_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>

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

vector<vector<long long>> Combine(int n, int mod);

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


// 前缀树
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
    ~FileSystem()
    {
        delete root;
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
template <typename T1, typename T2, typename T3>
class MyHash {
public:
    size_t operator() (const pair<T1, T2>& a) const
    {
        // return reinterpret_cast<size_t>(a.first);
        return hash<T1>()(a.first) ^ hash<T2>()(a.second);
    }
    size_t operator() (const tuple<T1, T2, T3>& a) const
    {
        size_t seed = 0;
        hash_combine(seed, get<0>(a));
        hash_combine(seed, get<1>(a));
        hash_combine(seed, get<2>(a));
        return seed;
    }
    template <class T>
    static void hash_combine(size_t& seed, const T& v)
    {
        seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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


// 查并集
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
    // 构造函数, 初始化parent和rank数组
    UnionFind(int size) : parent(size), rank(size, 0)
    {
        for (int i = 0; i < size; ++i) {
            parent[i] = i;  // 初始时, 每个节点的父节点是它自己
        }
    }

    // 合并两个元素所在的集合
    void unionSets(int x, int y)
    {
        int xRoot = find(x);
        int yRoot = find(y);

        if (xRoot == yRoot) {
            return;  // 如果x和y已经在同一个集合中, 则不需要合并
        }

        // 按秩合并：将秩较小的树合并到秩较大的树下
        if (rank[xRoot] < rank[yRoot]) {
            parent[xRoot] = yRoot;
        } else if (rank[xRoot] > rank[yRoot]) {
            parent[yRoot] = xRoot;
        } else {
            parent[yRoot] = xRoot;
            rank[xRoot] += 1;  // 如果秩相等, 选择一个作为根, 并增加其秩
        }
    }

    // 查找元素x所在集合的代表元素
    int findSet(int x)
    {
        return find(x);
    }
};


// 线段树
class SegmentTree {
private:
    vector<int> tree;  // 线段树数组
    vector<int> lazy;  // 延迟标记数组, 用于区间更新
    int n;            // 原始数组大小

    // 构建线段树
    void build(const vector<int>& nums, int node, int start, int end) {
        if (start == end) {
            tree[node] = nums[start];
        } else {
            int mid = (end - start) / 2 + start;
            build(nums, node * 2 + 1, start, mid);    // 左子树
            build(nums, node * 2 + 2, mid + 1, end);  // 右子树
            tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];  // 求和
            // 若是查找最大值, 聚合线段树用max效率更高
            // tree[node] = max(tree[node * 2 + 1], tree[node * 2 + 2]);
        }
    }
    void push_down(int node, int start, int end) {
        if (lazy[node] != 0) {
            int mid = (start + end) / 2;
            // 更新左子树
            tree[node * 2 + 1] += lazy[node] * (mid - start + 1);
            lazy[node * 2 + 1] += lazy[node];
            // 更新右子树
            tree[node * 2 + 2] += lazy[node] * (end - mid);
            lazy[node * 2 + 2] += lazy[node];
            // 清除当前节点的标记
            lazy[node] = 0;
        }
    }
public:
    SegmentTree(const vector<int>& nums) {
        n = nums.size();
        int height = (int)ceil(log2(n));  // 树的高度
        int max_size = 2 * (int)pow(2, height) - 1;  // 线段树最大节点数
        tree.resize(max_size);
        lazy.resize(max_size);
        build(nums, 0, 0, n - 1);
    }
    // 查询区间和
    int query(int l, int r) {
        return query(0, 0, n - 1, l, r);
    }

    // 更新某个位置的值
    void update(int index, int val) {
        update(0, 0, n - 1, index, val);
    }

    // 区间更新
    void range_update(int l, int r, int val) {
        range_update(0, 0, n - 1, l, r, val);
    }

    // 查询第一个大于等于x的位置
    int find(int l, int r, int x) {
        return find(0, 0, n - 1, l, r, x);
    }
private:
    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;  // 区间不重叠
        if (l <= start && end <= r) return tree[node];  // 完全包含
        
        int mid = (start + end) / 2;
        int left_sum = query(node * 2 + 1, start, mid, l, r);
        int right_sum = query(node * 2 + 2, mid + 1, end, l, r);
        return left_sum + right_sum;
    }

    void update(int node, int start, int end, int index, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (index <= mid) {
                update(node * 2 + 1, start, mid, index, val);
            } else {
                update(node * 2 + 2, mid + 1, end, index, val);
            }
            tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];  // 更新父节点
        }
    }

    void range_update(int node, int start, int end, int l, int r, int val) {
        if (r < start || end < l) return;  // 区间不重叠
        if (l <= start && end <= r) {
            tree[node] += val * (end - start + 1);
            lazy[node] += val;
            return;
        }
        push_down(node, start, end);
        int mid = (start + end) / 2;
        range_update(node * 2 + 1, start, mid, l, r, val);
        range_update(node * 2 + 2, mid + 1, end, l, r, val);
        tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];
    }

    int find(int node, int start, int end, int L, int R, int x) {
        if (end < L || start > R) return -1;       // 区间无重叠
        if (tree[node] < x) return -1;         // 区间最大值 <x，直接剪枝
        
        if (start == end) return start;            // 找到叶子节点
        
        int mid = (start + end) / 2;
        // 先查左子树（保证最左边的解）
        int left_pos = find(node * 2 + 1, start, mid, L, R, x);
        if (left_pos != -1) return left_pos;       // 左子树有解
        
        // 左子树无解，再查右子树
        return find(node * 2 + 2, mid + 1, end, L, R, x);
    }
};


// 二进制提升法最近公共祖先节点
class BinaryLiftingLCA {
private:
    vector<vector<int>> up; // up[i][j] - i节点的2^j级祖先, up[i][0] - 父节点
    vector<int> depth;
    int LOG;
public:
    // 生成up和depth数组
    BinaryLiftingLCA(vector<vector<int>>& edges, int root)
    {
        int nodeNum = edges.size();
        LOG = log2(nodeNum) + 1;
        up = vector<vector<int>>(nodeNum + 1, vector<int>(LOG)); // 为节点编号从1开始留出空间
        depth.resize(nodeNum + 1, 0);

        function<void (int, int)> dfs = [&dfs, &edges, this](int cur, int parent) {
            int i;
            up[cur][0] = parent;
            for (i = 1; i < LOG; i++) {
                up[cur][i] = up[up[cur][i - 1]][i - 1];
            }
            for (auto& next : edges[cur]) {
                if (next != parent) {
                    depth[next] = depth[cur] + 1;
                    dfs(next, cur);
                }
            }
        };
        dfs(root, root);
    }

    const vector<int>& GetDepth() const {
        return depth;
    }

    // 节点u和v的最近公共祖先
    int lca(int u, int v)
    {
        if (depth[u] < depth[v]) {
            swap(u, v);
        }

        int i;
        int diff;

        diff = depth[u] - depth[v];
        // u, v提升到同一高度
        for (i = LOG - 1; i >= 0; i--) {
            if ((diff & (1 << i)) == (1 << i)) {
                u = up[u][i];
            }
        }
        // 在同一条链上
        if (u == v) {
            return u;
        }

        // 同时找u, v的公共祖先
        for (i = LOG - 1; i >= 0; i--) {
            if (up[u][i] != up[v][i]) {
                u = up[u][i];
                v = up[v][i];
            }
            // 此处不应该break
        }
        return up[u][0];
    }
};
#endif