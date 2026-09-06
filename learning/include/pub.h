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
long long Combine(int n, int k, int mod);
long long Combine(long long n, long long k, int mod);
int FastPow(int a, int b);
long long FastPow(long long a, long long b, int mod);

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
    vector<int> size;    // 存储每个根节点所对应的集合大小

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
    UnionFind(int size) : parent(size), rank(size, 0), size(size, 1)
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
            size[yRoot] += size[xRoot];  // 更新新根的大小
        } else if (rank[xRoot] > rank[yRoot]) {
            parent[yRoot] = xRoot;
            size[xRoot] += size[yRoot];  // 更新新根的大小
        } else {
            parent[yRoot] = xRoot;
            rank[xRoot] += 1;  // 如果秩相等, 选择一个作为根, 并增加其秩
            size[xRoot] += size[yRoot];  // 更新新根的大小
        }
    }

    // 查找元素x所在集合的代表元素
    int findSet(int x)
    {
        return find(x);
    }

    // 获取元素x所在集合的大小
    int getSize(int x)
    {
        int root = find(x);
        return size[root];
    }
};


// 线段树
class SegmentTree {
private:
    vector<long long> tree;  // 线段树数组
    vector<long long> lazy;  // 延迟标记数组, 用于区间更新
    int n;            // 原始数组大小

    // 构建线段树
    void build(const vector<long long>& nums, int node, int start, int end) {
        if (start == end) {
            tree[node] = nums[start];
        } else {
            int mid = (end - start) / 2 + start;
            build(nums, node * 2 + 1, start, mid);    // 左子树
            build(nums, node * 2 + 2, mid + 1, end);  // 右子树
            tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];  // 求和
            // 若是查找最大值, 应该使用聚合线段树, 如下
            // tree[node] = max(tree[node * 2 + 1], tree[node * 2 + 2]); // 求最大值
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
    SegmentTree(const vector<long long>& nums) {
        n = nums.size();
        int height = (int)ceil(log2(n));  // 树的高度
        int max_size = 2 * (int)pow(2, height) - 1;  // 线段树最大节点数
        tree.resize(max_size);
        lazy.resize(max_size);
        build(nums, 0, 0, n - 1);
    }
    // 查询区间和/最大值
    long long query(int l, int r) {
        return query(0, 0, n - 1, l, r);
    }

    // 下标加法
    void add(int index, long long val) {
        range_add(index, index, val);
    }

    // 区间加法
    void range_add(int l, int r, long long val) {
        range_add(0, 0, n - 1, l, r, val);
    }

    // 下面的函数需要基于最大值线段树
    // 查询第一个大于等于x的位置
    int find(int l, int r, long long x) {
        return find(0, 0, n - 1, l, r, x);
    }

    // 统计区间 [L,R] 内大于 x 的元素个数
    int countGreater(int L, int R, long long x) {
        return countGreater(0, 0, n - 1, L, R, x);
    }
private:
    long long query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;  // 区间不重叠
        if (l <= start && end <= r) return tree[node];  // 完全包含

        push_down(node, start, end);
        int mid = (start + end) / 2;
        long long left_sum = query(node * 2 + 1, start, mid, l, r);
        long long right_sum = query(node * 2 + 2, mid + 1, end, l, r);
        return left_sum + right_sum;
        // 如果是最大值线段树, query则是求区间最大值
        // return max(left_sum, right_sum);
    }

    void range_add(int node, int start, int end, int l, int r, long long val) {
        if (r < start || end < l) return;  // 区间不重叠
        if (l <= start && end <= r) {
            tree[node] += val * (end - start + 1);
            lazy[node] += val;
            return;
        }
        push_down(node, start, end);
        int mid = (start + end) / 2;
        range_add(node * 2 + 1, start, mid, l, r, val);
        range_add(node * 2 + 2, mid + 1, end, l, r, val);
        tree[node] = tree[node * 2 + 1] + tree[node * 2 + 2];
    }

    int find(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return -1;       // 区间无重叠
        if (tree[node] < x) return -1;         // 区间最大值 <x，直接剪枝

        if (start == end) return start;            // 找到叶子节点

        push_down(node, start, end);
        int mid = (start + end) / 2;
        // 先查左子树（保证最左边的解）
        int left_pos = find(node * 2 + 1, start, mid, L, R, x);
        if (left_pos != -1) return left_pos;       // 左子树有解

        // 左子树无解，再查右子树
        return find(node * 2 + 2, mid + 1, end, L, R, x);
    }

    // 统计 >= x 的个数
    int countGreaterEqual(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return 0;  // 区间不重叠
        if (tree[node] < x) return 0;         // 整个区间最大值 < x，剪枝
        
        if (start == end) {
            // 叶子节点，判断是否满足条件
            return (tree[node] >= x) ? 1 : 0;
        }
        push_down(node, start, end);
        int mid = (start + end) / 2;
        return countGreaterEqual(node * 2 + 1, start, mid, L, R, x) + 
            countGreaterEqual(node * 2 + 2, mid + 1, end, L, R, x);
    }

    // 统计 > x 的个数
    int countGreater(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return 0;
        if (tree[node] <= x) return 0;        // 整个区间最大值 <= x，剪枝
        
        if (start == end) {
            return (tree[node] > x) ? 1 : 0;
        }
        push_down(node, start, end);
        int mid = (start + end) / 2;
        return countGreater(node * 2 + 1, start, mid, L, R, x) + 
            countGreater(node * 2 + 2, mid + 1, end, L, R, x);
    }
};

class SegmentTree_Combine {
private:
    vector<long long> sumTree;   // 区间和
    vector<long long> maxTree;   // 区间最大值
    vector<long long> minTree;   // 区间最小值
    vector<long long> lazy;      // 延迟标记
    int n;

    // 构建线段树
    void build(const vector<long long>& nums, int node, int start, int end) {
        if (start == end) {
            sumTree[node] = nums[start];
            maxTree[node] = nums[start];
            minTree[node] = nums[start];
        } else {
            int mid = start + (end - start) / 2;
            build(nums, node * 2 + 1, start, mid);
            build(nums, node * 2 + 2, mid + 1, end);
            push_up(node);
        }
    }

    // 向上更新
    void push_up(int node) {
        sumTree[node] = sumTree[node * 2 + 1] + sumTree[node * 2 + 2];
        maxTree[node] = max(maxTree[node * 2 + 1], maxTree[node * 2 + 2]);
        minTree[node] = min(minTree[node * 2 + 1], minTree[node * 2 + 2]);
    }

    // 向下传递延迟标记
    void push_down(int node, int start, int end) {
        if (lazy[node] != 0 && start != end) {
            int mid = start + (end - start) / 2;
            long long val = lazy[node];
            int left = node * 2 + 1;
            int right = node * 2 + 2;

            // 更新左子树
            sumTree[left] += val * (mid - start + 1);
            maxTree[left] += val;
            minTree[left] += val;
            lazy[left] += val;

            // 更新右子树
            sumTree[right] += val * (end - mid);
            maxTree[right] += val;
            minTree[right] += val;
            lazy[right] += val;

            lazy[node] = 0;
        }
    }

public:
    // 构造函数
    SegmentTree_Combine(const vector<long long>& nums) {
        n = nums.size();
        if (n == 0) return;
        int height = (int)ceil(log2(n));
        int max_size = 2 * (int)pow(2, height) - 1;
        sumTree.resize(max_size);
        maxTree.resize(max_size);
        minTree.resize(max_size);
        lazy.resize(max_size, 0);
        build(nums, 0, 0, n - 1);
    }

    // 区间加法
    void range_add(int l, int r, long long val) {
        if (l > r) swap(l, r);
        range_add(0, 0, n - 1, l, r, val);
    }

    void add(int index, long long val) {
        range_add(index, index, val);
    }

    // 区间求和
    long long query_sum(int l, int r) {
        if (l > r) swap(l, r);
        return query_sum(0, 0, n - 1, l, r);
    }

    // 区间最大值
    long long query_max(int l, int r) {
        if (l > r) swap(l, r);
        return query_max(0, 0, n - 1, l, r);
    }

    // 区间最小值
    long long query_min(int l, int r) {
        if (l > r) swap(l, r);
        return query_min(0, 0, n - 1, l, r);
    }

    // 查找第一个 ≥ x 的位置
    int find_first_ge(int L, int R, long long x) {
        return find_first_ge(0, 0, n - 1, L, R, x);
    }

    // 统计 ≥ x 的个数
    int count_ge(int L, int R, long long x) {
        return count_ge(0, 0, n - 1, L, R, x);
    }

    // 统计 > x 的个数
    int count_gt(int L, int R, long long x) {
        return count_gt(0, 0, n - 1, L, R, x);
    }

    long long query(int index) {
        return query(0, 0, n - 1, index);
    }
private:
    // 区间加法
    void range_add(int node, int start, int end, int l, int r, long long val) {
        if (r < start || end < l) return;
        if (l <= start && end <= r) {
            sumTree[node] += val * (end - start + 1);
            maxTree[node] += val;
            minTree[node] += val;
            lazy[node] += val;
            return;
        }
        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        range_add(node * 2 + 1, start, mid, l, r, val);
        range_add(node * 2 + 2, mid + 1, end, l, r, val);
        push_up(node);
    }

    // 区间求和
    long long query_sum(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) return sumTree[node];
        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        return query_sum(node * 2 + 1, start, mid, l, r) +
               query_sum(node * 2 + 2, mid + 1, end, l, r);
    }

    // 区间最大值
    long long query_max(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return LLONG_MIN;
        if (l <= start && end <= r) return maxTree[node];
        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        return max(query_max(node * 2 + 1, start, mid, l, r),
                   query_max(node * 2 + 2, mid + 1, end, l, r));
    }

    // 单点查询
    long long query(int node, int start, int end, int index) {
        if (start == end) {
            return sumTree[node];
        }
        push_down(node, start, end);
        int mid = (start + end) / 2;
        if (index <= mid) {
            return query(node * 2 + 1, start, mid, index);
        } else {
            return query(node * 2 + 2, mid + 1, end, index);
        }
    }
    // 区间最小值
    long long query_min(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return LLONG_MAX;
        if (l <= start && end <= r) return minTree[node];
        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        return min(query_min(node * 2 + 1, start, mid, l, r),
                   query_min(node * 2 + 2, mid + 1, end, l, r));
    }

    // 查找第一个 ≥ x 的位置
    int find_first_ge(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return -1;
        if (maxTree[node] < x) return -1;
        if (start == end) return start;

        push_down(node, start, end);
        int mid = start + (end - start) / 2;

        int left_res = find_first_ge(node * 2 + 1, start, mid, L, R, x);
        if (left_res != -1) return left_res;

        return find_first_ge(node * 2 + 2, mid + 1, end, L, R, x);
    }

    // 统计 ≥ x 的个数
    int count_ge(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return 0;
        if (maxTree[node] < x) return 0;
        if (start == end) {
            return (maxTree[node] >= x) ? 1 : 0;
        }

        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        return count_ge(node * 2 + 1, start, mid, L, R, x) +
               count_ge(node * 2 + 2, mid + 1, end, L, R, x);
    }

    // 统计 > x 的个数
    int count_gt(int node, int start, int end, int L, int R, long long x) {
        if (end < L || start > R) return 0;
        if (maxTree[node] <= x) return 0;
        if (start == end) {
            return (maxTree[node] > x) ? 1 : 0;
        }

        push_down(node, start, end);
        int mid = start + (end - start) / 2;
        return count_gt(node * 2 + 1, start, mid, L, R, x) +
               count_gt(node * 2 + 2, mid + 1, end, L, R, x);
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

    // 两个节点间的距离
    int distance(int u, int v)
    {
        int c = lca(u, v);
        return depth[u] + depth[v] - depth[c] * 2;
    }

    // 节点 u -> v 路径
    vector<int> getPath(int u, int v)
    {
        int c = lca(u, v);
        vector<int> path;

        // u -> c
        int cur = u;
        while (cur != c) {
            path.emplace_back(cur);
            cur = up[cur][0];
        }
        path.emplace_back(c);

        // c -> v
        vector<int> tmp;
        cur = v;
        while (cur != c) {
            tmp.emplace_back(cur);
            cur = up[cur][0];
        }
        reverse(tmp.begin(), tmp.end());
        path.insert(path.end(), tmp.begin(), tmp.end());

        return path;
    }
};


// 树状数组
struct Fenwick {
    int n;
    vector<int> bit;
    const int INF = -1e9;

    Fenwick(int size) : n(size), bit(size + 2, INF) {}
    
    void update(int idx, int val) {
        idx++; // 转为 1-based
        while (idx <= n) {
            bit[idx] = max(bit[idx], val);
            idx += idx & -idx;
        }
    }
    // 查询 [0, idx] 的最大值
    int query(int idx) {
        if (idx < 0) {
            return INF;
        }
        idx++;
        int res = INF;
        while (idx > 0) {
            res = max(res, bit[idx]);
            idx -= idx & -idx;
        }
        return res;
    }
};


// cin, cout 不支持的__int128类型数据读取
inline __int128 read() {
    __int128 x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + (ch - '0');
        ch = getchar();
    }
    return x * f;
}

inline void write(__int128 x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x > 9) write(x / 10);
    putchar(x % 10 + '0');
}
#endif