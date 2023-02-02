#ifndef _PUB_H_
#define _PUB_H_

#include <iostream>
constexpr unsigned short w = 1024;
int mod = 1000000007;

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
    bool IsEnd;
    vector<Trie *> children;
    Trie() : IsEnd(false) {}
    Trie(T ch) : val(ch), IsEnd(false) {}
};

#endif