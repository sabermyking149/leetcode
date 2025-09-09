#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>
#include <iterator>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <list>
#include <optional>
#include <bit>
#include <climits>

#include "pub.h"

using namespace std;

int MyGcd(int a, int b)
{
    if(b == 0) {
        return a;
    }
    return gcd(b, a % b);
}


bool isVowel(unsigned char c)
{
    c = tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}


// 埃拉托斯特尼筛法
vector<int> GetPrimes(int n)
{
    int i, p;
    vector<bool> isPrime(n + 1, true);
    vector<int> primes;
    isPrime[0] = isPrime[1] = false; // 0 和 1 不是质数
 
    for (p = 2; p <= sqrt(n); p++) {
        if (isPrime[p]) {
            for (i = p * p; i <= n; i += p) {
                isPrime[i] = false;
            }
        }
    }
 
    for (p = 2; p <= n; ++p) {
        if (isPrime[p]) {
            primes.push_back(p);
        }
    }
 
    return primes;
}


bool IsPrime(int n)
{
    if (n <= 1) {
        return false;
    }
    if (n == 2 || n == 3) {
        return true;
    }
    if (n % 2 == 0 || n % 3 == 0) {
        return false;
    }

    int m = sqrt(n);
    for (int i = 3; i <= m; i += 2) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}


// 仅针对小数据的数组全排列(非去重)
vector<vector<int>> GetArrPermutation(vector<int>& source)
{
    int i;
    int n = source.size();
    vector<vector<int>> ans;

    int len = 1;
    i = n;
    while (i) {
        len *= i;
        i--;
    }
    ans.emplace_back(source);
    for (i = 1; i < len; i++) {
        next_permutation(source.begin(), source.end());
        ans.emplace_back(source);
    }
    return ans;
}

// 求组合数
long long Combine(int m, int n, map<pair<int, int>, long long>& combineData)
{
    int mod = 1e9 + 7;
    if (combineData.count(make_pair(m, n))) {
        return combineData[{m, n}];
    }
    if (n == 0 || m == n) {
        combineData[{m, n}] = 1;
        return 1;
    } else if (n == 1) {
        combineData[{m, n}] = m;
        return m;
    }
    combineData[{m, n}] = (Combine(m - 1, n - 1, combineData) + Combine(m - 1, n, combineData)) % mod;
    return combineData[{m, n}];
}

int TreeNodesNum(TreeNode *root)
{
    if (root == nullptr) {
        return 0;
    }
    return 1 + TreeNodesNum(root->left) + TreeNodesNum(root->right);
}

void initVisited(vector<int>& visited, int val)
{
    for (unsigned int i = 0; i < visited.size(); i++) {
        visited[i] = val;
    }
}
void CreateLoop(int cur, vector<int>& visited, unordered_map<int, int> relative, bool &findEnd, unordered_set<int>& loopNode)
{
    if (findEnd) {
        return;
    }
    if (loopNode.find(cur) != loopNode.end()) {
        findEnd = true;
        return;
    }
    if (visited[cur] == 1) {
        loopNode.emplace(cur);
    }
    CreateLoop(relative[cur], visited, relative, findEnd, loopNode);
}
void DFSFindLoop(unordered_map<int, unordered_set<int>>& e, int cur, int time, unordered_map<int, int>  &visitedTime,
    vector<int>& visited, bool &findLoop, int &loopSize)
{
    if (findLoop) {
        return;
    }
    if (visited[cur] == 1) {
        findLoop = true;
        loopSize = time - visitedTime[cur];
        return;
    }
    visited[cur] = 1;
    visitedTime[cur] = time;
    if (e.find(cur) != e.end()) {
        for (auto it : e[cur]) {
            if (visited[it] != 2) {
                DFSFindLoop(e, it, time + 1, visitedTime, visited, findLoop, loopSize);
            }
        }
    }
    visited[cur] = 2;
}
int longestCycle(vector<int>& edges)
{
    unsigned int i;
    unordered_map<int, unordered_set<int>> e;

    for (i = 0; i < edges.size(); i++) {
        if (edges[i] != -1) {
            e[i].emplace(edges[i]);
        }
    }
    unsigned int n = edges.size();
    vector<int> visited(n, 0);
    unordered_map<int, int> visitedTime;
    int loopSize = 0;
    bool findLoop = false;

    int ans = -1;
    for (i = 0; i < n; i++) {
        findLoop = false;
        loopSize = -1;
        DFSFindLoop(e, i, 0, visitedTime, visited, findLoop, loopSize);
        ans = max(ans, loopSize);
    }
    return ans;
}

vector<TreeNode*> BSTs;
vector<TreeNode*> CreateBSTs(int start, int end)
{
    int i;
    unsigned int j, k;
    vector<TreeNode*> trees;
    if (start > end) {
        return {nullptr};
    }
    for (i = start; i <= end; i++) {
        vector<TreeNode *> leftTree = CreateBSTs(start, i - 1);
        vector<TreeNode *> rightTree = CreateBSTs(i + 1, end);
        // cout << leftTree.size() << " " << rightTree.size() << endl;
        for (j = 0; j < leftTree.size(); j++) {
            for (k = 0; k < rightTree.size(); k++) {
                TreeNode *node = new TreeNode(i);
                node->left = leftTree[j];
                node->right = rightTree[k];
                trees.emplace_back(node);
            }
        }
    }
    return trees;
}
vector<TreeNode*> generateTrees(int n)
{
    BSTs = CreateBSTs(1, n);
    return BSTs;
}


// LC1258
vector<string> words;
unordered_map<string, int> um;
vector<string> ans;
int Find_s(string& a, int idx)
{
    if (words[idx] == a) {
        return idx;
    }
    int id = Find_s(words[idx], um[words[idx]]);
    words[idx] = words[id];
    return id;
}
void Union_s(string& a, string& b)
{
    int x, y;

    x = Find_s(a, um[a]);
    y = Find_s(b, um[b]);
    if (x != y) {
        words[y] = words[x];
    }
}
vector<string> MySplit(string& s, char separate)
{
    int i;
    int n = s.size();
    vector<string> ans;
    int cur = 0;

    for (i = 0; i < n; i++) {
        if (s[i] == separate) {
            ans.emplace_back(s.substr(cur, i - cur));
            cur = i + 1;
        }
    }
    ans.emplace_back(s.substr(cur, n - cur));
    return ans;
}
void DFSCreateSentences(vector<string>& texts, unsigned int curIdx, unordered_map<string, unordered_set<string>>& synonymsWords,
    vector<string>& words)
{
    if (curIdx == texts.size()) {
        string str;
        for (auto w : words) {
            str += w + ' ';
        }
        str[str.size() - 1] = '\0';
        ans.emplace_back(str);
        return;
    }
    if (synonymsWords.find(texts[curIdx]) == synonymsWords.end()) {
        words.push_back(texts[curIdx]);
        DFSCreateSentences(texts, curIdx + 1, synonymsWords, words);
        words.pop_back();
    } else {
        for (auto it : synonymsWords[texts[curIdx]]) {
            words.push_back(it);
            DFSCreateSentences(texts, curIdx + 1, synonymsWords, words);
            words.pop_back();
        }
    }
}
vector<string> generateSentences(vector<vector<string>>& synonyms, string text)
{
    unsigned int i;
    unordered_set<string> dict;

    for (auto s : synonyms) {
        dict.insert(s[0]);
        dict.insert(s[1]);
    }

    for (auto it : dict) {
        words.emplace_back(it);
        um[it] = words.size() - 1;
    }
    vector<string> t = words;

    for (auto s : synonyms) {
        Union_s(s[0], s[1]);
    }
    for (i = 0; i < words.size(); i++) {
        words[i] = words[Find_s(words[i], um[words[i]])];
    }
    unordered_map<string, unordered_set<string>> synonymsWords;
    for (i = 0; i < words.size(); i++) {
        synonymsWords[words[i]].insert(t[i]);
    }
    for (auto it : synonymsWords) {
        for (auto it1 : it.second) {
            if (synonymsWords.find(it1) == synonymsWords.end()) {
                synonymsWords[it1] = it.second;
            }
        }
    }
    vector<string> texts = MySplit(text, ' ');
    vector<string> strs;
    DFSCreateSentences(texts, 0, synonymsWords, strs);
    sort(ans.begin(), ans.end());
    return ans;
}

// 面试题 17.07. 婴儿名字
vector<string> trulyMostPopular(vector<string>& names, vector<string>& synonyms)
{
    unordered_set<string> dict;
    string s1, s2;
    unsigned int i;
    int idx;
    int num;
    vector<pair<string, string>> synonymsPair;
    for (auto s : synonyms) {
        idx = s.find(",");
        s1 = s.substr(1, idx - 1);
        s2 = s.substr(idx + 1, s.size() - 1 - idx - 1);
        synonymsPair.emplace_back(make_pair(s1, s2));
        dict.insert(s1);
        dict.insert(s2);
    }
    // vector<string> words;
    // 对synonyms的处理
    for (auto it : dict) {
        words.emplace_back(it);
        um[it] = words.size() - 1; // 单词坐标
    }
    // 对names的处理
    vector<pair<string, int>> namesData;
    for (auto name: names) {
        idx = name.find("(");
        s1 = name.substr(0, idx);
        num = atoi(name.substr(idx + 1).c_str()); // 多截取了一个')',atoi会忽略
        namesData.emplace_back(make_pair(s1, num));
        if (um.find(s1) == um.end()) {
            words.emplace_back(s1);
            um[s1] = words.size() - 1;
        }
    }
    vector<string> t = words;
    for (auto sp : synonymsPair) {
        Union_s(sp.first, sp.second);
    }
    for (i = 0; i < words.size(); i++) {
        words[i] = words[Find_s(words[i], um[words[i]])];
    }

    unordered_map<string, set<string>> synonymsWordsDict;
    unordered_map<string, string> root; // 查并集的根
    for (i = 0; i < words.size(); i++) {
        synonymsWordsDict[words[i]].insert(t[i]);
        root[t[i]] = words[i];
    }
    unordered_map<string, int> collvalue;
    for (auto data : namesData) {
        collvalue[*synonymsWordsDict[root[data.first]].begin()] += data.second;
    }
    vector<string> ans;
    for (auto it : collvalue) {
        ans.emplace_back(it.first + "(" + to_string(it.second) + ")");
    }
    return ans;
}

int minTotalDistance(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    vector<int> row, col;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                row.emplace_back(i);
                col.emplace_back(j);
            }
        }
    }
    sort(row.begin(), row.end());
    sort(col.begin(), col.end());
    int x, y;

    x = row[row.size() / 2];
    y = row[col.size() / 2];
    int ans = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                ans += abs(i - x) + abs(j - y);
            }
        }
    }
    return ans;
}


// LC131
bool IsPalindrome(string &s)
{
    int i, j;
    int n = s.size();

    i = 0;
    j = n - 1;
    while (i < j) {
        if (s[i] != s[j]) {
            return false;
        }
        i++;
        j--;
    }
    return true;
}
void PartPalindrome(string s, vector<string>& record, vector<vector<string>>& ans)
{
    int i;
    int n = s.size();

    if (s.size() == 0) {
        ans.emplace_back(record);
        return;
    }
    for (i = 0; i < n; i++) {
        string t = s.substr(0, i + 1);
        if (IsPalindrome(t)) {
            record.emplace_back(t);
            PartPalindrome(s.substr(i + 1), record, ans);
            record.pop_back();
        }
    }
}
vector<vector<string>> partition(string s)
{
    vector<vector<string>> ans;
    vector<string> record;
    PartPalindrome(s, record, ans);
    return ans;
}


// LC2402
class CMPMostBooked {
public:
    bool operator() (const vector<long long> &a, const vector<long long>& b)
    {
        if (a[1] != b[1]) {
            return a[1] > b[1];
        }
        return a[2] > b[2]; // 结束时间相同,返回编号小的房间
    }
};
int mostBooked(int n, vector<vector<int>>& meetings)
{
    unsigned int i;
    priority_queue<int, vector<int>, greater<int>> emptyRooms;
    priority_queue<vector<long long>, vector<vector<long long>>, CMPMostBooked> meetingTimes;
    vector<int> roomUse(n, 0);

    for (i = 0; i < static_cast<unsigned int>(n); i++) {
        emptyRooms.push(i);
    }
    sort(meetings.begin(), meetings.end());
    int emptyRoom = emptyRooms.top();
    emptyRooms.pop();
    roomUse[emptyRoom]++;

    meetingTimes.push({meetings[0][0], meetings[0][1], emptyRoom});
    for (i = 1; i < meetings.size(); i++) {
        vector<long long> t;
        while (meetingTimes.size()) {
            t = meetingTimes.top();
            if (t[1] <= meetings[i][0]) {
                emptyRoom = t[2];
                emptyRooms.push(emptyRoom);
                meetingTimes.pop();
            } else {
                break;
            }
        }
        if (meetingTimes.empty()) {
            emptyRoom = emptyRooms.top();
            emptyRooms.pop();

            roomUse[emptyRoom]++;
            meetingTimes.push({meetings[i][0], meetings[i][1], emptyRoom});
            continue;
        }
        if (emptyRooms.size() > 0) {
            emptyRoom = emptyRooms.top();
            emptyRooms.pop();

            roomUse[emptyRoom]++;
            meetingTimes.push({meetings[i][0], meetings[i][1], emptyRoom});
        } else {
            emptyRoom = t[2];
            roomUse[emptyRoom]++;
            meetingTimes.pop();
            vector<long long> newMeetings(3);
            newMeetings[0] = t[1];
            newMeetings[1] = t[1] + meetings[i][1] - meetings[i][0];
            newMeetings[2] = emptyRoom;
            meetingTimes.push(newMeetings);
        }
    }
    int maxUse = 0;
    int ans = 0;
    for (i = 0; i < static_cast<unsigned int>(n); i++) {
        if (roomUse[i] > maxUse) {
            maxUse = roomUse[i];
            ans = i;
        }
    }
    return ans;
}


// LC854
int kSimilarity(string s1, string s2)
{
    unsigned int i, j;
    unsigned int n, m;
    int step;
    unordered_set<string> visited;

    if (s1 == s2) {
        return 0;
    }
    m = s1.size();
    string ss1, ss2;
    for (i = 0; i < m; i++) {
        if (s1[i] != s2[i]) {
            ss1 += s1[i];
            ss2 += s2[i];
        }
    }

    queue<pair<string, int>> q;
    step = 0;
    m = ss1.size();
    q.push({ss1, 0});
    visited.emplace(ss1);
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            pair<string, int> p = q.front();
            string t = p.first;
            unsigned int idx = p.second;
            q.pop();
            if (t == ss2) {
                return step;
            }
            while (idx < m && t[idx] == ss2[idx]) {
                idx++;
            }
            for (j = idx; j < m; j++) {
                if (t[j] == ss2[idx] && t[j] != t[idx]) {
                    string tmp = t;
                    swap(tmp[j], tmp[idx]);
                    if (visited.find(tmp) == visited.end()) {
                        q.push({tmp, idx + 1});
                        visited.emplace(tmp);
                    }
                }
            }
        }
        step++;
    }
    return -1;
}


// LC505
vector<int> GoThrough(vector<vector<int>>& maze, vector<int>& curPos, int len, int direction, vector<vector<int>>& pathLen)
{
    int row = maze.size();
    int col = maze[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    auto t = curPos;
    int dist = 0;
    while (1) {
        t[0] += directions[direction][0];
        t[1] += directions[direction][1];
        dist++;
        if (t[0] < 0 || t[0] >= row || t[1] < 0 || t[1] >= col || maze[t[0]][t[1]] == 1) {
            t[0] -= directions[direction][0];
            t[1] -= directions[direction][1];
            dist--;
            break;
        }
    }
    if (pathLen[t[0]][t[1]] > len + dist) {
        pathLen[t[0]][t[1]] = len + dist;
        return t;
    }
    return {-1, -1};
}
int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination)
{
    int i, k;
    int n;
    int row = maze.size();
    int col = maze[0].size();
    vector<vector<int>> pathLen(row, vector<int>(col, INT_MAX));
    queue<vector<int>> q;

    q.push(start);
    pathLen[start[0]][start[1]] = 0;
    while (q.size()) {
        n = q.size();
        for (k = 0; k < n; k++) {
            auto v = q.front();
            q.pop();

            /* if (v == destination) {
                return pathLen[v[0]][v[1]];
            } */
            for (i = 0; i < 4; i++) {
                auto t = GoThrough(maze, v, pathLen[v[0]][v[1]], i, pathLen);
                if (t[0] != -1) {
                    q.push(t);
                }
            }
        }
    }
    return pathLen[destination[0]][destination[1]] == INT_MAX ? -1 : pathLen[destination[0]][destination[1]];
}


class Solution {
public:
    vector<vector<int>> ans;
    void BallsRunning(vector<string>& plate, int row, int col, int direction, int step)
    {
        int a, b;
        int r = plate.size();
        int c = plate[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        a = row;
        b = col;
        while (step) {
            row += directions[direction][0];
            col += directions[direction][1];
            if (row >= r || row < 0 || col >= c || col < 0) {
                break;
            }
            if (plate[row][col] == 'O') {
                ans.push_back({a, b});
                break;
            }
            if (plate[row][col] == 'E') {
                direction = (direction + 1) % 4;
            } else if (plate[row][col] == 'W') {
                direction = (direction + 3) % 4;
            }
            step--;
        }
    }
    vector<vector<int>> ballGame(int num, vector<string>& plate)
    {
        int i, j;
        int row = plate.size();
        int col = plate[0].size();

        for (i = 1; i < row - 1; i++) {
            if (plate[i][0] == '.') {
                BallsRunning(plate, i, 0, 0, num);
            }
            if (plate[i][col - 1] == '.') {
                BallsRunning(plate, i, col - 1, 2, num);
            }
        }
        for (j = 1; j < col - 1; j++) {
            if (plate[0][j] == '.') {
                BallsRunning(plate, 0, j, 1, num);
            }
            if (plate[row - 1][j] == '.') {
                BallsRunning(plate, row - 1, j, 3, num);
            }
        }
        return ans;
    }
};
#if 0
static bool Cmp(pair<string, int>& a, pair<string, int>& b)
{
    /*if (a.second != b.second) {
        return a.second > b.second;
    }*/
    return a.second > b.second;

}
#endif
/*
vector<string> sortPeople(vector<string>& names, vector<int>& heights)
{
    vector<pair<string, int>> vp;
    int i;
    int n = names.size();
    for (i = 0; i < n; i++) {
        vp.push_back(make_pair(names[i], heights[i]));
    }

    //sort(vp.begin(), vp.end(), Cmp);
    vector<string> ans;
    for (auto p : vp) {
        ans.emplace_back(p.second);
    }
    return ans;
}
*/

unordered_set<string> trees;
unordered_set<string> t;
vector<TreeNode *> res;

string PreorderTree(TreeNode *node)
{
    string tree;
    if (node == nullptr) {
        return "";
    }
    tree = to_string(node->val) + "(" + PreorderTree(node->left) + ")" + "(" + PreorderTree(node->right) + ")";
    if (trees.find(tree) != trees.end() && t.find(tree) == t.end()) {
        t.emplace(tree);
        res.emplace_back(node);
    }
    trees.emplace(tree);
    return tree;
}
vector<TreeNode*> lightDistribution(TreeNode* root)
{
    PreorderTree(root);
    return res;
}


vector<int> getTriggerTime(vector<vector<int>>& increase, vector<vector<int>>& requirements)
{
    int i;
    int n = increase.size();
    int size;

    vector<int> v(3, 0);
    vector<vector<int>> prefixSum;

    prefixSum.emplace_back(v);
    for (i = 0; i < n; i++) {
        size = prefixSum.size();
        prefixSum.push_back({prefixSum[size - 1][0] + increase[i][0],
            prefixSum[size - 1][1] + increase[i][1],
            prefixSum[size - 1][2] + increase[i][2]});
    }
    int left, right, mid;
    int j;
    n = requirements.size();
    vector<int> ans(n, 0);
    for (i = 0; i < n; i++) {
        for (j = 0; j < 3; j++) {
            left = 0;
            right = prefixSum.size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (prefixSum[mid][j] >= requirements[i][j]) { // right + 1;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // printf ("left = %d, right = %d\n", left, right);
            if (left >= prefixSum.size()) {
                ans[i] = -1;
                break;
            } else {
                ans[i] = max(ans[i], right + 1);
            }
        }
    }
    return ans;
}


bool Check(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}
vector<vector<int>> Flip(vector<vector<int>>& grid, int row, int col)
{
    int i;
    int m = grid.size();
    int n = grid[0].size();
    int direction[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    vector<vector<int>> ans(m, vector<int>(n, 0));
    ans = grid;

    ans[row][col] = 1 - grid[row][col];
    for (i = 0; i < 4; i++) {
        int nr = row + direction[i][0];
        int nc = col + direction[i][1];

        if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
            ans[nr][nc] = 1 - grid[nr][nc];
        }
    }
    return ans;
}
vector<int> TwoDimonsionTolinear(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    vector<int> ans;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            ans.emplace_back(grid[i][j]);
        }
    }
    return ans;
}
int minFlips(vector<vector<int>>& mat)
{
    int i, j, k;
    int m = mat.size();
    int n = mat[0].size();
    int step = 0;
    set<vector<int>> data;
    queue<vector<vector<int>>> q;

    if (Check(mat)) {
        return step;
    }

    q.push(mat);
    data.emplace(TwoDimonsionTolinear(mat));
    while (q.size()) {
        int size = q.size();
        for (k = 0; k < size; k++) {
            vector<vector<int>> t = q.front();
            q.pop();

            if (Check(t)) {
                return step;
            }
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    vector<vector<int>> vv = Flip(t, i, j);
                    vector<int> v = TwoDimonsionTolinear(vv);
                    if (data.find(v) == data.end()) {
                        data.emplace(v);
                        q.push(vv);
                    }
                }
            }
        }
        step++;
    }
    return -1;
}


int CountPlusSign(vector<vector<int>>& grid, int row, int col)
{
    if (grid[row][col] == 0) {
        return 0;
    }

    int cnt;
    int m, n, a, b, c, d;

    m = grid.size();
    n = grid[0].size();

    cnt = 1;
    a = b = row;
    c = d = col;
    while (1) {
        a--;
        b++;
        c--;
        d++;

        if (a < 0 || b >= m || c < 0 || d >= n) {
            break;
        }
        if (grid[a][col] == 0 || grid[b][col] == 0 || grid[row][c] == 0 || grid[row][d] == 0) {
            break;
        }
        cnt++;
    }
    return cnt;
}
int orderOfLargestPlusSign(int n, vector<vector<int>>& mines)
{
    int i, j;
    int ans;

    vector<vector<int>> grid(n, vector<int>(n, 1));
    for (auto v : mines) {
        grid[v[0]][v[1]] = 0;
    }
    ans = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            ans = max(ans, CountPlusSign(grid, i ,j));
        }
    }
    return ans;
}


long long maximumSubarraySum(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int t;
    long long ans;
    long long sum;
    unordered_map<int, int> numsFreq;
    map<int, unordered_set<int>, greater<int>> freqOfNums;

    ans = sum = 0;
    for (i = 0; i < k; i++) {
        sum += nums[i];
        if (numsFreq.find(nums[i]) == numsFreq.end()) {
            freqOfNums[1].emplace(nums[i]);
        } else {
            t = numsFreq[nums[i]];
            freqOfNums[t].erase(nums[i]);
            freqOfNums[t + 1].emplace(nums[i]);
        }
        numsFreq[nums[i]]++;
    }
    if (freqOfNums.begin()->first == 1) {
        ans = max(ans, sum);
    }
    for (i = 1; i <= n - k; i++) {
        t = numsFreq[nums[i - 1]];
        if (t == 1) {
            numsFreq.erase(nums[i - 1]);
            if (freqOfNums[1].size() == 1) {
                freqOfNums.erase(1);
            } else {
                freqOfNums[1].erase(nums[i - 1]);
            }
        } else {
            numsFreq[nums[i - 1]]--;
            if (freqOfNums[t].size() == 1) {
                freqOfNums.erase(t);
            } else {
                freqOfNums[t].erase(nums[i - 1]);
            }
            freqOfNums[t - 1].emplace(nums[i - 1]);
        }
        sum -= nums[i - 1];

        if (numsFreq.find(nums[i + k - 1]) == numsFreq.end()) {
            freqOfNums[1].emplace(nums[i + k - 1]);
        } else {
            t = numsFreq[nums[i + k - 1]];
            freqOfNums[t].erase(nums[i + k - 1]);
            freqOfNums[t + 1].emplace(nums[i + k - 1]);
        }
        numsFreq[nums[i + k - 1]]++;
        sum += nums[i + k - 1];

        if (freqOfNums.begin()->first == 1) {
            ans = max(ans, sum);
        }
    }
    return ans;
}


int FindIdx(string& s, vector<int>& v, int idx, string &ch) // 找到v里第一个大于等于idx的值
{
    int left, right, mid;

    left = 0;
    right = v.size() - 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        if (v[mid] >= idx) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    // printf ("s = %s, left = %d, right = %d\n", ch.c_str(), left, right);
    if (left >= v.size() || (right == -1 && v[right + 1] < idx)) {
        return -1;
    }
    return v[right + 1];
}
int numMatchingSubseq(string s, vector<string>& words)
{
    int i, j;
    int n = s.size();
    int cnt, t;
    unordered_map<char, vector<int>> strInfo;

    for (i = 0; i < n; i++) {
        strInfo[s[i]].emplace_back(i);
    }
    cnt = 0;
    for (i = 0; i < words.size(); i++) {
        t = -1;
        for (j = 0; j < words[i].size(); j++) {
            if (strInfo.find(words[i][j]) == strInfo.end()) {
                break;
            }
            t = FindIdx(s, strInfo[words[i][j]], t + 1, words[i]);
            // printf ("%s : t = %d\n", words[i].c_str(), t);
            if (t == -1) {
                break;
            }
        }
        if (j == words[i].size()) {
            cnt++;
        }
    }
    return cnt;
}


// LC802
vector<int> eventualSafeNodes(vector<vector<int>>& graph)
{
    int i, j;
    int n = graph.size();
    int m;
    set<int> safeNode;
    unordered_set<int> nodes;
    vector<int> ans;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, unordered_set<int>> reverseConnected;
    for (i = 0; i < n; i++) {
        m = graph[i].size();
        for (j = 0; j < m; j++) {
            edges[i].emplace(graph[i][j]);
            reverseConnected[graph[i][j]].emplace(i);
        }
        nodes.emplace(i);
    }

    while (1) {
        m = nodes.size();
        for (unordered_set<int>::iterator iter = nodes.begin(); iter != nodes.end();) {
            if (edges.count(*iter) == 0) {
                if (reverseConnected.count(*iter) == 0) {
                    safeNode.emplace(*iter);
                    nodes.erase(iter++);
                    continue;
                }
                safeNode.emplace(*iter);
                for (auto it : reverseConnected[*iter]) {
                    if (edges[it].size() > 1) {
                        edges[it].erase(*iter);
                    } else {
                        edges.erase(it);
                    }
                }
                reverseConnected.erase(*iter);
                nodes.erase(iter++);
            } else {
                iter++;
            }
        }
        if (nodes.size() == m) {
            break;
        }
    }
    for (auto it : safeNode) {
        ans.emplace_back(it);
    }
    return ans;
}



int expressiveWords(string s, vector<string>& words)
{
    int i, j;
    int n, cnt, ans;
    int m = s.size();
    int nums = words.size();
    bool flag = false;
    vector<pair<char, int>> ss, ww;

    cnt = 1;
    for (i = 1; i < m; i++) {
        if (s[i] == s[i - 1]) {
            cnt++;
        } else {
            ss.emplace_back(make_pair(s[i - 1], cnt));
            cnt = 1;
        }
    }
    ss.emplace_back(make_pair(s[m - 1], cnt));

    ans = 0;
    for (i = 0; i < nums; i++) {
        n = words[i].size();
        if (n > m) {
            continue;
        }

        cnt = 1;
        ww.clear();
        for (j = 1; j < n; j++) {
            if (words[i][j] == words[i][j - 1]) {
                cnt++;
            } else {
                ww.emplace_back(make_pair(words[i][j - 1], cnt));
                cnt = 1;
            }
        }
        ww.emplace_back(make_pair(words[i][n - 1], cnt));

        if (ss.size() != ww.size()) {
            continue;
        }
        flag = false;
        for (j = 0; j < ss.size(); j++) {
            if (ss[j].first != ww[j].first || ww[j].second > ss[j].second) {
                break;
            }
            if (ww[j].second != ss[j].second && ss[j].second < 3) {
                break;
            }
            if (ww[j].second + 2 <= ss[j].second) {
                flag = true;
            }
        }
        if (j != ss.size()) {
            continue;
        }
        if (flag) {
            ans++;
        }
    }
    return ans;
}


bool isPossible(int n, vector<vector<int>>& edges)
{
    unordered_map<int, int> nodeDegree;
    unordered_map<int, unordered_set<int>> ed;
    unordered_map<int, unordered_set<int>> oddDegreeEdge;
    for (auto e : edges) {
        nodeDegree[e[0]]++;
        nodeDegree[e[1]]++;
        ed[e[0]].emplace(e[1]);
        ed[e[1]].emplace(e[0]);
    }
    vector<int> oddDegreeNodes;
    for (auto it : nodeDegree) {
        if (it.second % 2 == 1) {
            oddDegreeEdge[it.first] = ed[it.first];
            oddDegreeNodes.emplace_back(it.first);
        }
    }
    int m = oddDegreeNodes.size();
    if (m != 0 && m != 2 && m != 4) {
        return false;
    }

    /* cout << "cnt = " << m << endl;
    for (auto it : oddDegreeEdge) {
        printf ("%d: ", it.first);
        for (auto it1 : it.second) {
            printf ("%d ", it1);
        }
        printf ("\n");
    }
    for (auto nn : oddDegreeNodes) cout << nn << " "; cout << endl; */

    if (m == 2) {
        for (int i = 1; i <= n; i++) {
            if (oddDegreeEdge[oddDegreeNodes[0]].count(i) == 0 &&
                oddDegreeEdge[oddDegreeNodes[1]].count(i) == 0) {
                return true;
            }
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[1]) == 1) {
            return false;
        }
    }
    if (m == 4) {
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[1]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[2]].count(oddDegreeNodes[3]) == 0) {
            return true;
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[2]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[1]].count(oddDegreeNodes[3]) == 0) {
            return true;
        }
        if (oddDegreeEdge[oddDegreeNodes[0]].count(oddDegreeNodes[3]) == 0 &&
            oddDegreeEdge[oddDegreeNodes[1]].count(oddDegreeNodes[2]) == 0) {
            return true;
        }
        return false;
    }
    return true;
}


vector<vector<int>> SplitVector(vector<int>& nums)
{
    int i;
    int a, b;
    int n = nums.size();
    vector<int> v;
    vector<vector<int>> ans;

    int cnt = 0;
    for (i = 0; i < n; i++) {
        if (nums[i] == 0) {
            cnt++;
            if (cnt == 1) {
                a = i;
            } else {
                b = i;
                if (b - a > 1) {
                    ans.emplace_back(v);
                    v.clear();
                }
                cnt = 1;
                a = i;
            }
        } else {
            v.emplace_back(nums[i] > 0 ? 1 : -1);
        }
    }
    return ans;
}
int getMaxLen(vector<int>& nums)
{
    int i;
    int m, t;
    int n = nums.size();
    vector<int> v(n + 2, 0);

    for (i = 0; i < n; i++) {
        v[i + 1] = nums[i];
    }

    int ans = 0;
    vector<int> negIdx;
    vector<vector<int>> arrays = SplitVector(v);
    for (auto arr : arrays) {
        n = arr.size();
        negIdx.clear();
        for (i = 0; i < n; i++) {
            if (arr[i] < 0) {
                negIdx.emplace_back(i);
            }
        }
        m = negIdx.size();
        if (m % 2 == 0) {
            ans = max(ans, n);
            continue;
        }
        t = max(n - (negIdx[0] + 1), negIdx[m - 1]);
        ans = max(ans, t);
    }
    return ans;
}

bool isItPossible(string word1, string word2)
{
    unordered_map<int, int> dict1, dict2;
    for (auto w : word1) {
        dict1[w]++;
    }
    for (auto w : word2) {
        dict2[w]++;
    }
    int m = dict1.size();
    int n = dict2.size();
    if (abs(m - n) > 2) {
        return false;
    }
    if (m == n) {
        for (auto it1 : dict1) {
            if (dict2.find(it1.first) != dict2.end()) {
                return true;
            } else {
                for (auto it2 : dict2) {
                    if (dict1.find(it2.first) == dict1.end()) {
                        if (it1.second == it2.second || (it1.second > 1 && it2.second > 1)) {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    if (abs(m - n) == 1) {
        if (m > n) {
            for (auto it1 : dict1) {
                if (dict2.find(it1.first) == dict2.end() && it1.second > 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end() && it2.second > 1) {
                            return true;
                        }
                    }
                }
                if (dict2.find(it1.first) == dict2.end() && it1.second == 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end() && it1.second == it2.second) {
                            return true;
                        }
                    }
                }
            }
        } else {
            for (auto it2 : dict2) {
                if (dict1.find(it2.first) == dict1.end() && it2.second > 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end() && it1.second > 1) {
                            return true;
                        }
                    }
                }
                if (dict1.find(it2.first) == dict1.end() && it2.second == 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end() && it1.second == it2.second) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
    if (abs(m - n) == 2) {
        if (m > n) {
            for (auto it1 : dict1) {
                if (dict2.find(it1.first) == dict2.end() && it1.second == 1) {
                    for (auto it2 : dict2) {
                        if (dict1.find(it2.first) != dict1.end()) {
                            return true;
                        }
                    }
                }
            }
        } else {
            for (auto it2 : dict2) {
                if (dict1.find(it2.first) == dict1.end() && it2.second == 1) {
                    for (auto it1 : dict1) {
                        if (dict2.find(it1.first) != dict2.end()) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool isItPossible_1(string word1, string word2)
{
    unordered_map<char, int> dict1, dict2, tmp1, tmp2;
    for (auto w : word1) {
        dict1[w]++;
    }
    for (auto w : word2) {
        dict2[w]++;
    }
    int m = dict1.size();
    int n = dict2.size();
    if (abs(m - n) > 2) {
        return false;
    }
    tmp1 = dict1;
    tmp2 = dict2;
    for (auto it1 : dict1) {
        if (it1.second == 1) {
            tmp1.erase(it1.first);
        } else {
            tmp1[it1.first]--;
        }
        for (auto it2 : dict2) {
            if (it2.second == 1) {
                tmp2.erase(it2.first);
            } else {
                tmp2[it2.first]--;
            }
            tmp2[it1.first]++;
            tmp1[it2.first]++;
            if (tmp1.size() == tmp2.size()) {
                return true;
            }
            // 还原
            if (tmp1[it2.first] == 1) {
                tmp1.erase(it2.first);
            } else {
                tmp1[it2.first]--;
            }
            tmp2[it2.first]++;
            if (tmp2[it1.first] == 1) {
                tmp2.erase(it1.first);
            } else {
                tmp2[it1.first]--;
            }
        }
        // 还原
        tmp1[it1.first]++;
    }
    return false;
}

bool isItPossible_2(string word1, string word2)
{
    int i, j, k;
    int a[26] = {0};
    int b[26] = {0};
    int cntA, cntB;

    for (auto w : word1) {
        a[w - 'a']++;
    }
    for (auto w : word2) {
        b[w - 'a']++;
    }
    for (i = 0; i < 26; i++) {
        if (a[i] > 0) {
            a[i]--;
            b[i]++;
            for (j = 0; j < 26; j++) {
                if (b[j] == 0) {
                    continue;
                }
                if ((i == j && b[j] == 1)) {
                    continue;
                }
                if (b[j] > 0) {
                    b[j]--;
                    a[j]++;
                }
                cntA = cntB = 0;
                for (k = 0; k < 26; k++) {
                    if (a[k] > 0) {
                        cntA++;
                    }
                    if (b[k] > 0) {
                        cntB++;
                    }
                }
                if (cntA == cntB) {
                    return true;
                }
                // 还原
                b[j]++;
                a[j]--;
            }
            // 还原
            a[i]++;
            b[i]--;
        }
    }
    return false;
}



map<string, string> dict;
map<string, set<string>> prefixDict;
map<string, int> prefixFreq;
void GenerateUniqueString(int n, int idx, int k, string t)
{
    if (idx == n) {
        string prefix = t.substr(0, t.size() - 1);
        string suffix = t.substr(1);
        dict[t] = suffix;
        prefixFreq[prefix]++;
        prefixDict[prefix].emplace(t);
        return;
    }
    int i;
    for (i = 0; i < k; i++) {
        t[idx] = i + '0';
        GenerateUniqueString(n, idx + 1, k, t);
    }
}
string crackSafe(int n, int k)
{
    int i;
    int cnt;
    string ans;
    if (n == 1) {
        for (i = 0; i < k; i++) {
            ans += i + '0';
        }
        return ans;
    }
    string t(n, ' ');
    GenerateUniqueString(n, 0, k, t);

    /*priority_queue<pair<int, string>, vector<pair<int, string>>> pq;
    for (auto pf : prefixFreq) {
        pq.push(make_pair(pf.second, pf.first));
    }*/
    ans = dict.begin()->first;
    string curString = ans;
    string curPrefix = ans.substr(0, n - 1);
    string nextPrefix = dict.begin()->second;
    prefixFreq[curPrefix]--;
    prefixDict[curPrefix].erase(curString);

    string tmp;
    while (1) {

        cnt = 0;
        for (auto it : prefixDict[nextPrefix]) {
            nextPrefix = dict[it];
            if (prefixFreq[nextPrefix] > cnt) {
                cnt = prefixFreq[nextPrefix];
                curString = it;
            }
        }
        curPrefix = curString.substr(0, curString.size() - 1);
        ans += curString[curString.size() - 1];
        nextPrefix = dict[curString];

        if (prefixFreq[curPrefix] == 1) {
            prefixFreq.erase(curPrefix);
        } else {
            prefixFreq[curPrefix]--;
        }

        if (prefixDict[curPrefix].size() == 1) {
            prefixDict.erase(curPrefix);
        } else {
            prefixDict[curPrefix].erase(curString);
        }
        if (prefixDict.size() == 0) {
            break;
        }
    }
    return ans;
}


void DeleteTree(TreeNode* &node)
{
    if (node == nullptr) {
        return;
    }
    DeleteTree(node->left);
    DeleteTree(node->right);
    delete node;
    node = nullptr;
}
void DFSCheckTreeSum(TreeNode* &node, int curSum, int limit, bool& flag)
{
    if (node == nullptr) {
        return;
    }
    curSum += node->val;
    if (node->left == nullptr && node->right == nullptr) {
        if (curSum >= limit) {
            flag = false;
        }
        return;
    }
    DFSCheckTreeSum(node->left, curSum, limit, flag);
    DFSCheckTreeSum(node->right, curSum, limit, flag);
}
void DFSSCanTree(TreeNode* &node, TreeNode *parent, char parentBranch, int curSum, int limit)
{
    bool flag = true;
    if (node == nullptr) {
        return;
    }
    DFSCheckTreeSum(node, curSum, limit, flag);
    if (flag) {
        DeleteTree(node);
        flag = true;
        if (parent != nullptr) {
            if (parentBranch == 'L') {
                parent->left = nullptr;
            } else {
                parent->right = nullptr;
            }
        } else {

        }
    }
    if (node != nullptr) {
        DFSSCanTree(node->left, node, 'L', curSum + node->val, limit);
        DFSSCanTree(node->right, node, 'R', curSum + node->val, limit);
    }
}
TreeNode* sufficientSubset(TreeNode* root, int limit)
{
    DFSSCanTree(root, nullptr, 'L', 0, limit);
    return root;
}


int waysToMakeFair(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<int> oddSum, evenSum;

    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        if (i % 2 == 0) {
            if (evenSum.empty()) {
                evenSum.emplace_back(nums[i]);
            } else {
                evenSum.emplace_back(evenSum[evenSum.size() - 1] + nums[i]);
            }
        } else {
            if (oddSum.empty()) {
                oddSum.emplace_back(nums[i]);
            } else {
                oddSum.emplace_back(oddSum[oddSum.size() - 1] + nums[i]);
            }
        }
    }
    int ans = 0;
    int odds = 0;
    int evens = 0;
    int p = oddSum.size();
    int q = evenSum.size();
    for (i = 0; i < n; i++) {
       if (i % 2 == 1) {
            if (i / 2 > 0) {
                odds = oddSum[i / 2 - 1] + evenSum[q - 1] - evenSum[i / 2];
            } else {
                odds = evenSum[q - 1] - evenSum[i / 2];
            }
            evens = evenSum[i / 2] + oddSum[p - 1] - oddSum[i / 2];
        } else {
            if (i == 0) {
                odds = evenSum[q - 1] - evenSum[0];
                evens = oddSum[p - 1];
            } else {
                odds = oddSum[i / 2 - 1] + evenSum[q - 1] - evenSum[i / 2];
                evens = evenSum[i / 2 - 1] + oddSum[p - 1] - oddSum[i / 2 - 1];
            }
        }
        if (odds == evens) {
            ans++;
        }
    }
    return ans;
}

static bool MaxScoreCmp(const pair<int, int>& a, const pair<int, int>& b)
{
    if (a.first == b.first) {
        return a.second > b.second;
    }
    return a.first > b.first;
}
long long maxScore(vector<int>& nums1, vector<int>& nums2, int k)
{
    int i;
    int n = nums1.size();
    int t;
    long long ans, sum;
    priority_queue<int, vector<int>, greater<>> pq;
    vector<pair<int, int>> vp;

    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(nums2[i], i));
    }
    sort(vp.begin(), vp.end(), MaxScoreCmp);
    sum = 0;
    for (i = 0; i < k; i++) {
        t = nums1[vp[i].second];
        sum += t;
        pq.push(t);
    }
    ans = sum * vp[k - 1].first;
    for (i = k; i < n; i++) {
        t = nums1[vp[i].second];
        if (pq.top() < t) {
            sum += t - pq.top();
            pq.pop();
            pq.push(t);
            ans = max(ans, sum * vp[i].first);
        }
    }
    return ans;
}


int FastPow(int a, int b)
{
    int ans = 1, base = a;
    while (b != 0) {
        if ((b & 1) != 0) {
            ans = (static_cast<long long>(ans) * base) % mod;
        }
        base = (static_cast<long long>(base) * base) % mod;
        b >>= 1;
    }
    return ans;
}
long long FastPow(int a, int b, int mod)
{
    long long ans = 1;
    int base = a;
    while (b != 0) {
        if ((b & 1) != 0) {
            ans = (static_cast<long long>(ans) * base) % mod;
        }
        base = (static_cast<long long>(base) * base) % mod;
        b >>= 1;
    }
    return ans;
}


long long putMarbles(vector<int>& weights, int k)
{
    int i;
    int n = weights.size();
    long long ans;
    vector<int> val(n - 1);

    for (i = 1; i < n; i++) {
        val[i - 1] = weights[i - 1] + weights[i];
    }
    sort(val.begin(), val.end());

    ans = 0;
    for (i = 0; i < k - 1; i++) {
        ans += val[n - 2 - i] - val[i];
    }
    return ans;
}


ListNode* mergeInBetween(ListNode* list1, int a, int b, ListNode* list2)
{
    int cnt;
    ListNode *prev = nullptr;
    ListNode *cur = list2;
    while (cur->next) {
        cur = cur->next;
    }
    ListNode *list2Tail = cur;

    cur = list1;
    cnt = 0;
    while (cur) {
        if (cnt == a) {
            prev->next = list2;
        }
        if (cnt == b) {
            list2Tail->next = cur->next;
            break;
        }
        cnt++;
        prev = cur;
        cur = cur->next;
    }
    return list1;
}


static bool longestStrChainCMP(const string& a, const string& b)
{
    if (a.size() != b.size()) {
        return a.size() < b.size();
    }
    return a < b;
}
bool IsParent(string& a, string& b) // a - cur  b - parent
{
    int m = a.size();
    int n = b.size();
    int i, j;

    if (n + 1 != m) {
        return false;
    }
    i = j = 0;
    while (j < n && i < m) {
        if (a[i] == b[j]) {
            i++;
            j++;
        } else {
            i++;
        }
    }
    if (j == n) {
        return true;
    }
    return false;
} 
int longestStrChain(vector<string>& words)
{
    int i, j;
    int n = words.size();
    vector<int> dp(n, 1);
    sort(words.begin(), words.end(), longestStrChainCMP);
    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (IsParent(words[i], words[j])) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    int ans = 0;
    for (auto d : dp) {
        ans = max(ans, d); 
    }
    return ans;
}


unordered_map<TreeNode *, unordered_set<TreeNode *>> edges;
vector<TreeNode *> leaves;
void DFSCreateGrid(TreeNode * node, TreeNode *parent)
{
    if (node == nullptr) {
        return;
    }
    if (node->left == nullptr && node->right == nullptr) {
        leaves.emplace_back(node);
    }
    if (parent != nullptr) {
        edges[node].emplace(parent);
        edges[parent].emplace(node);
    }
    DFSCreateGrid(node->left, node);
    DFSCreateGrid(node->right, node);
}
int CheckCanReach(TreeNode *a, int distance)
{
    int i, n;
    int step, ans;
    queue<TreeNode *> q;
    unordered_set<TreeNode *> visited;

    q.push(a);
    step = 0;
    ans = 0;
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            TreeNode *t = q.front();
            q.pop();
            if (t != a && t->left == nullptr && t->right == nullptr) {
                ans++;
            }
            visited.emplace(t);
            for (auto it : edges[t]) {
                if (visited.count(it) == 1) {
                    continue;
                }
                q.push(it);
            }
        }
        step++;
        if (step > distance) {
            break;
        }
    }
    return ans;
}
int countPairs(TreeNode* root, int distance)
{
    int ans;
    DFSCreateGrid(root, nullptr);

    int num = leaves.size();
    if (num < 2) {
        return 0;
    }
    int i;
    ans = 0;
    for (i = 0; i < num; i++) {
        ans += CheckCanReach(leaves[i], distance);
    }
    return ans / 2;
}


void DSFAdjustRoute(int cur, int parent, unordered_map<int, vector<int>>& edges, set<pair<int, int>>& realRoutePair, 
    unordered_set<int>& visited, int& ans)
{
    visited.emplace(cur);
    if (parent != -1) {
        if (realRoutePair.count(make_pair(cur, parent)) == 0) {
            ans++;
        }
    }
    for (auto it : edges[cur]) {
        if (visited.find(it) != visited.end()) {
            continue;
        }
        DSFAdjustRoute(it, cur, edges, realRoutePair, visited, ans);
    }
}
int minReorder(int n, vector<vector<int>>& connections)
{
    int ans;
    unordered_map<int, vector<int>> edges;
    set<pair<int, int>> realRoutePair;
    unordered_set<int> visited;
    for (auto conn : connections) {
        edges[conn[0]].emplace_back(conn[1]);
        edges[conn[1]].emplace_back(conn[0]);
        realRoutePair.emplace(make_pair(conn[0], conn[1]));
    }

    ans = 0;
    DSFAdjustRoute(0, -1, edges, realRoutePair, visited, ans);
    return ans;
}


string findContestMatch(int n)
{
    int i, j;
    string ans;
    vector<string> v, t;

    for (i = 1; i <= n; i++) {
        v.emplace_back(to_string(i));
    }
    while (n != 1) {
        i = 0;
        j = v.size() - 1;
        t.clear();
        while (i < j) {
            t.emplace_back("(" + v[i] + "," + v[j] + ")");
            i++;
            j--;
        }
        v = t;
        n /= 2;
    }
    ans = v[0];
    return ans;
}

template<>
void Trie<char>::CreateWordTrie(Trie<char> *root, string& word)
{
    int i, j;
    int n, m;

    n = word.size();
    for (i = 0; i < n; i++) {
        if (root->children.size() == 0) {
            Trie<char> *node = new Trie(word[i]);
            root->children.emplace_back(node);
            root = node;
        } else {
            m = root->children.size();
            for (j = 0; j < m; j++) {
                if (root->children[j]->val == word[i]) {
                    root = root->children[j];
                    break;
                }
            }
            if (j == m) {
                Trie<char> *node = new Trie(word[i]);
                root->children.emplace_back(node);
                root = node;
            }
        }
    }
    root->IsEnd = true;
}
template<>
void Trie<char>::CreateWordTrie(Trie<char> *root, string& word, int timestamp)
{
    int i, j;
    int n, m;

    n = word.size();
    for (i = 0; i < n; i++) {
        if (root->children.size() == 0) {
            Trie<char> *node = new Trie(word[i]);
            root->children.emplace_back(node);
            root = node;
            root->timestamp = timestamp;
        } else {
            m = root->children.size();
            for (j = 0; j < m; j++) {
                if (root->children[j]->val == word[i]) {
                    root = root->children[j];
                    break;
                }
            }
            if (j == m) {
                Trie<char> *node = new Trie(word[i]);
                root->children.emplace_back(node);
                root = node;
                root->timestamp = timestamp;
            }
        }
    }
    root->IsEnd = true;
    // root->timestamp = timestamp;
}
void DFSScanWordTrie(Trie<char> *root, string& t, int point, unordered_map<string, int>& wordPrefixPoint)
{
    int i;
    t.push_back(root->val);
    if (root->IsEnd) {
        point++;
        wordPrefixPoint[t.substr(1)] = point; // 去掉字符串首位的'/'
    }
    for (i = 0; i < root->children.size(); i++) {
        DFSScanWordTrie(root->children[i], t, point, wordPrefixPoint);
        t.pop_back();
    }
}
string longestWord(vector<string>& words)
{
    unsigned int i;
    int m = words.size();
    Trie<char> *root = new Trie('/');

    for (i = 0; i < m; i++) {
        Trie<char>::CreateWordTrie(root, words[i]);
    }

    unordered_map<string, int> wordPrefixPoint;
    string t;
    DFSScanWordTrie(root, t, 0, wordPrefixPoint);

    string ans = "";
    for (auto it : wordPrefixPoint) {
        // cout << it.first << " " << it.second << endl;
        if (it.first.size() == it.second) {
            if (it.second > ans.size()) {
                ans = it.first;
            } else if (it.second == ans.size()) {
                ans = min(ans, it.first);
            }
        }
    }
    delete(root);
    return ans;
}


// LC1746
int maxSumAfterOperation(vector<int>& nums)
{
    int ans;
    int n = nums.size();
    int i;
    int a, b;
    vector<vector<int>> dp(n, vector<int>(2, 0));
    // 最大子数组的变种 dp[i][0] -- 以nums[i]结尾的最大子数组和,不使用平方; dp[i][1] -- 以nums[i]结尾的最大子数组和,使用平方
    dp[0][0] = nums[0];
    dp[0][1] = nums[0] * nums[0];
    ans = max(dp[0][0], dp[0][1]);
    for (i = 1; i < n; i++) {
        dp[i][0] = dp[i - 1][0] > 0 ? dp[i - 1][0] + nums[i] : nums[i];
        a = dp[i - 1][0] > 0 ? dp[i - 1][0] + nums[i] * nums[i] : nums[i] * nums[i];
        b = dp[i - 1][1] > 0 ? dp[i - 1][1] + nums[i] : nums[i];
        dp[i][1] = max(a, b);
        ans = max(ans, max(dp[i][0], dp[i][1]));
    }
    return ans;
}


bool canBeValid(string s, string locked)
{
    int i;
    int n = s.size();
    stack<int> lockedIdxStack;
    stack<int> freeIdxStack;

    if (n % 2 == 1 || (s[0] == ')' && locked[0] == '1') || (s[n - 1] == '(' && locked[n - 1] == '1')) {
        return false;
    }

    for (i = 0; i < n; i++) {
        if (locked[i] == '0') {
            freeIdxStack.push(i);
            continue;
        }
        if (s[i] == '(') {
            lockedIdxStack.push(i);
        } else {
            if (lockedIdxStack.empty()) {
                if (freeIdxStack.empty()) {
                    return false;
                } else {
                    freeIdxStack.pop();
                }
            } else {
                lockedIdxStack.pop();
            }
        }
    }
    while (lockedIdxStack.size()) {
        if (freeIdxStack.empty() || lockedIdxStack.top() > freeIdxStack.top()) {
            return false;
        }
        lockedIdxStack.pop();
        freeIdxStack.pop();
    }
    return lockedIdxStack.empty();
}


void CheckBucket(vector<int>& tasks, int sessionTime, vector<int> &bucket, int idx, bool& isOk)
{
    unsigned int i;
    int n = bucket.size();

    if (isOk) {
        return;
    }
    if (idx == tasks.size()) {
        isOk = true;
        return;
    }
    for (i = 0; i < min(n, idx + 1); i++) {
        if (bucket[i] + tasks[idx] <= sessionTime) {
            bucket[i] += tasks[idx];
            CheckBucket(tasks, sessionTime, bucket, idx + 1, isOk);
            bucket[i] -= tasks[idx];
        }
    }
}
bool TrySessions(vector<int>& tasks, int sessionTime, int bucketSize)
{
    bool isOk = false;
    vector<int> bucket(bucketSize, 0);
    CheckBucket(tasks, sessionTime, bucket, 0, isOk);
    return isOk;
}
int minSessions(vector<int>& tasks, int sessionTime)
{
    int bucketSize;
    int n = tasks.size();
    int ans;
    sort (tasks.rbegin(), tasks.rend());
    ans = n;
    for (bucketSize = 1; bucketSize <= n; bucketSize++) {
        if (TrySessions(tasks, sessionTime, bucketSize)) {
            ans = bucketSize;
            break;
        }
    }
    return ans;
}


void DFSTrySum(vector<int>& nums, int curSum, int curMatchSubArr, int val, int k, vector<bool>& visited, bool& matchAll, int idx)
{
    unsigned int i;
    if (matchAll == true) {
        return;
    }
    for (i = idx; i < nums.size(); i++) {
        if (visited[i] == false) {
            if (curSum - curMatchSubArr * val + nums[i] == val) {
                if (curMatchSubArr + 1 == k) {
                    matchAll = true;
                    return;
                } else {
                    visited[i] = true;
                    DFSTrySum(nums, curSum + nums[i], curMatchSubArr + 1, val, k, visited, matchAll, 0);
                    visited[i] = false;
                }
            } else if (curSum - curMatchSubArr * val + nums[i] < val) {
                visited[i] = true;
                DFSTrySum(nums, curSum + nums[i], curMatchSubArr, val, k, visited, matchAll, i + 1);
                visited[i] = false;
            }
        }
    }
}
bool makesquare(vector<int>& matchsticks)
{
    int sum = 0;
    int n = matchsticks.size();

    if (n < 4) {
        return false;
    }
    for (auto m : matchsticks) {
        sum += m;
    }
    if (sum % 4 != 0) {
        return false;
    }
    int edgeLen = sum / 4;
    sort(matchsticks.rbegin(), matchsticks.rend());
    if (matchsticks[0] > edgeLen) {
        return false;
    }
    vector<int> v;
    bool f = false;
    vector<bool> visited(n, false);
    DFSTrySum(matchsticks, 0, 0, edgeLen, 4, visited, f, 0);
    return f;
}


// 从数组取k个互不相邻元素和最小值
// dp[i][j] 从前i个物品选j个物品的最小和
// dp[i][j] = min(dp[i - 1][j], dp[i - 2][j - 1] + nums[i])
int minGetSum(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();

    if (n == 1) {
        return nums[0];
    } else if (n == 2) {
        return min(nums[0], nums[1]);
    }
    vector<vector<long long>> dp(n, vector<long long>(k + 1, 0x3f3f3f3f));
    dp[0][0] = 0;
    dp[0][1] = nums[0];
    dp[1][1] = min(nums[0], nums[1]);
    for (i = 2; i < n; i++) {
        for (j = 1; j <= k; j++) {
            // dp[i][j] = min(dp[i - 1][j], dp[i - 2][j - 1] + nums[i]);
            if (dp[i - 2][j - 1] + nums[i] < dp[i - 1][j]) {
                dp[i][j] = dp[i - 2][j - 1] + nums[i];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[n - 1][k];
}


long long minCost(vector<int>& basket1, vector<int>& basket2)
{
    int i;
    int k;
    int n = basket1.size();
    long long ans;
    map<int, int> m1, m2;
    for (i = 0; i < n; i++) {
        m1[basket1[i]]++;
        m2[basket2[i]]++;
    }
    vector<vector<int>> v1, v2;
    for (auto it : m1) {
        if (m2.find(it.first) == m2.end()) {
            if (it.second % 2 != 0) {
                return -1;
            }
            v1.push_back({it.first, it.second / 2});
        } else {
            if (abs(it.second - m2[it.first]) % 2 != 0) {
                return -1;
            }
            if (it.second > m2[it.first]) {
                v1.push_back({it.first, (it.second - m2[it.first]) / 2});
            } else if (it.second < m2[it.first]) {
                v2.push_back({it.first, (m2[it.first] - it.second) / 2});
            }
        }
    }
    for (auto it : m2) {
        if (m1.find(it.first) == m1.end()) {
            if (it.second % 2 != 0) {
                return -1;
            }
            v2.push_back({it.first, it.second / 2});
        }
    }
    /* for (auto v : v1) cout << v[0] << ":" << v[1] << endl;
    cout << endl;
    for (auto v : v2) cout << v[0] << ":" << v[1] << endl;
    cout << endl;
    */
    ans = 0;
    k = min(m1.begin()->first, m2.begin()->first) * 2;
    vector<int> vv;
    for (auto v : v1) {
        for (i = 0; i < v[1]; i++) {
            vv.emplace_back(v[0]);
        }
    }
    for (auto v : v2) {
        for (i = 0; i < v[1]; i++) {
            vv.emplace_back(v[0]);
        }
    }

    sort(vv.begin(), vv.end());
    n = vv.size();
    for (i = 0; i < n / 2; i++) {
        ans += min(k, vv[i]);
    }
    return ans;
}


int maxCount(vector<int>& banned, int n, long long maxSum)
{
    int i;
    int banSize;
    int start, end;
    int cnt;
    int left, right, mid;
    long long t;
    vector<int> ban;

    sort(banned.begin(), banned.end());
    ban.emplace_back(0);
    ban.insert(ban.end(), banned.begin(), banned.end());
    ban.emplace_back(n + 1);
    banSize = ban.size();

    cnt = 0;
    for (i = 1; i < banSize; i++) {
        start = ban[i - 1] + 1;
        end = ban[i] - 1;
        if (start > end) {
            continue;
        }
        t = static_cast<long long>(start + end) * (end - start + 1) / 2;
        maxSum -= t;
        if (maxSum == 0) {
            cnt += end - start + 1;
            break;
        } else if (maxSum > 0) {
            cnt += end - start + 1;
            continue;
        }
        left = start;
        right = end;
        maxSum += t; // 还原
        while (left <= right) { // 所求为right
            mid = (right - left) / 2 + left;
            t = static_cast<long long>(start + mid) * (mid - start + 1) / 2;
            if (t <= maxSum) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        // cout << left << " " << right << endl;
        cnt += right - start + 1;
        break;
    }
    return cnt;
}


// LC769
bool CheckContinuousArr(int lastArrVal, vector<int>& arr)
{
    unsigned int i;

    sort(arr.begin(), arr.end());
    if (lastArrVal + 1 != arr[0]) {
        return false;
    }
    i = 1;
    for (; i < arr.size(); i++) {
        if (arr[i - 1] + 1 != arr[i]) {
            break;
        }
    }
    if (i != arr.size()) {
        return false;
    }
    return true;
} 
void TrySplit(vector<int>& arr, int left, vector<int>& record, int cnt, int& ans, bool& find)
{
    int i;
    int n = arr.size();
    if (find) {
        return;
    }
    if (left == n) {
        find = true;
        ans = cnt;
        return;
    }
    for (i = left; i < n; i++) {
        if (record.size() == 0) {
            record.assign(arr.begin() + left, arr.begin() + i + 1);
            if (CheckContinuousArr(-1, record) == false) {
                record.clear();
                continue;
            }
            TrySplit(arr, i + 1, record, cnt + 1, ans, find);
        } else {
            vector<int> v;
            v.assign(arr.begin() + left, arr.begin() + i + 1);
            if (CheckContinuousArr(record[record.size() - 1], v) == false) {
                v.clear();
                continue;
            }
            record.insert(record.end(), v.begin(), v.end());
            TrySplit(arr, i + 1, record, cnt + 1, ans, find);
        }
    }
}
int maxChunksToSorted(vector<int>& arr)
{
    int ans = 0;
    bool find = false;
    vector<int> record;
    TrySplit(arr, 0, record, 0, ans, find);
    return ans;
}


int CheckCanMakeWord(unordered_map<char, int>& alphabet, unordered_map<char, int>& word, vector<int>& score)
{
    int val;
    val = 0;
    for (auto it : word) {
        if (alphabet[it.first] < it.second) {
            return 0;
        }
        val += score[it.first - 'a'] * it.second;
    }
    for (auto it : word) {
        alphabet[it.first] -= it.second;
    }
    return val;
}
void DFSChooseWord(vector<unordered_map<char, int>>& wordsAlphabet, int idx, 
                unordered_map<char, int>& alphabet, vector<int>& score, int curScore, int& ans)
{
    int i;
    int n = wordsAlphabet.size();
    int val;
    if (idx == n) {
        ans = max(curScore, ans);
        return;
    }
    unordered_map<char, int> um;
    um = wordsAlphabet[idx];
    val = CheckCanMakeWord(alphabet, um, score);
    for (i = idx; i < n; i++) {
        DFSChooseWord(wordsAlphabet, i + 1, alphabet, score, curScore + val, ans);
    }
    // 回溯
    if (val != 0) {
        for (auto it : um) {
            alphabet[it.first] += it.second;
        }
    }
}
int maxScoreWords(vector<string>& words, vector<char>& letters, vector<int>& score)
{
    int i;
    int n = words.size();
    int ans;
    bool flag;
    unordered_map<char, int> alphabet;
    for (auto l : letters) {
        alphabet[l]++;
    }
    vector<unordered_map<char, int>> wordsAlphabet;
    unordered_map<char, int> um;
    for (i = 0; i < n; i++) {
        um.clear();
        flag = true;
        for (auto w : words[i]) {
            if (alphabet.find(w) == alphabet.end()) { // 剔除无法拼写单词
                flag = false;
                break;
            }
            um[w]++;
            if (alphabet[w] < um[w]) { // 剔除无法拼写单词
                flag = false;
                break;
            }
        }
        if (flag) {
            wordsAlphabet.emplace_back(um);
        }
    }
    ans = 0;
    n = wordsAlphabet.size();
    for (i = 0; i < n; i++) {
        DFSChooseWord(wordsAlphabet, i, alphabet, score, 0, ans);
    }
    return ans;
}


vector<string> removeSubfolders(vector<string>& folder)
{
    int i, j;
    int n = folder.size();

    sort (folder.begin(), folder.end());
    vector<string> ans;
    ans.emplace_back(folder[0]);
    for (i = 0; i < n;) {
        for (j = i + 1; j < n; j++) {
            if (strncmp((folder[i] + "/").c_str(), (folder[j] + "/").c_str(), folder[i].size() + 1) != 0) {
                ans.emplace_back(folder[j]);
                break;
            }
        }
        i = j;
        if (j == n) {
            break;
        }
    }
    return ans;
}


// LC1682
int longestPalindromeSubseq(string s)
{
    int n = s.size();
    vector<vector<vector<int>>> record(n, vector<vector<int>>(n, vector<int>(27, -1)));
    function<int (int, int, char)> dfs = [&dfs, &s, &record](int left, int right, char outer) {
        if (left + 1 > right) {
            return 0;
        }
        if (record[left][right][outer - 'a'] != -1) {
            return record[left][right][outer - 'a'];
        }

        int ans = 0;
        if (s[left] == s[right]) {
            if (s[left] != outer) {
                ans = dfs(left + 1, right - 1, s[left]) + 2;
            } else {
                ans = dfs(left + 1, right - 1, s[left]);
            }
        } else {
            ans = max(dfs(left + 1, right, outer), dfs(left, right - 1, outer));
        }
        return record[left][right][outer - 'a'] = ans;
        return ans;
    };
    int len = dfs(0, n - 1, '{'); // 此处用'{'是因为它是'z'的下一个ascii码
    return len;
}


int maxEqualFreq(vector<int>& nums)
{
    map<int, unordered_set<int>> freqMap;
    unordered_map<int, int> numFreq;
    int cnt, ans;

    cnt = 1;
    for (auto n : nums) {
        numFreq[n]++;
        if (numFreq[n] == 1) {
            freqMap[1].emplace(n);
        } else {
            if (freqMap[numFreq[n] - 1].size() == 1) {
                freqMap.erase(numFreq[n] - 1);
            } else {
                freqMap[numFreq[n] - 1].erase(n);
            }
            freqMap[numFreq[n]].emplace(n);
        }
        if (freqMap.size() == 1 && (freqMap.begin()->second.size() == 1 || freqMap.count(1) == 1)) { // 2 2 2 或 1 2 3
            ans = cnt;
        }
        if (freqMap.size() == 2) {
            auto it2 = freqMap.begin();
            auto it1 = it2++;
            if ((it1->first + 1 == it2->first && it2->second.size() == 1) || 
                (it1->first == 1 && it1->second.size() == 1)) { // 2 2 3 3 4 4 4 或  1 3 3 3 4 4 4
                ans = cnt;
            }
        }
        cnt++;
    }
    return ans;
}


// LC1820 匈牙利算法
bool DFSInviteLady(vector<vector<int>>& grid, unordered_map<int, int>& girlLikeBoy, int boy, int girlsNum, vector<bool>& visited)
{
    int i;
    for (i = 0; i < girlsNum; i++) {
        if (visited[i] == false && grid[boy][i] == 1) {
            visited[i] = true;
            if (girlLikeBoy.count(i) == 0 || DFSInviteLady(grid, girlLikeBoy, girlLikeBoy[i], girlsNum, visited)) {
                girlLikeBoy[i] = boy;
                return true;
            }
        }
    }
    return false;
}
int maximumInvitations(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int res = 0;
    vector<bool> visited(n, false);
    unordered_map<int, int> girlLikeBoy;

    for (i = 0; i < m; i++) {
        visited.assign(n, false);
        if (DFSInviteLady(grid, girlLikeBoy, i, n, visited)) {
            res++;
        }
    }
    return res;
}


vector<int> deckRevealedIncreasing(vector<int>& deck)
{
    int i;
    int t;
    int n = deck.size();
    vector<int> ans;
    deque<int> dq;

    sort (deck.rbegin(), deck.rend());
    dq.push_back(deck[0]);
    for (i = 1; i < n; i++) {
        t = dq.front();
        dq.pop_front();
        dq.push_back(t);
        dq.push_back(deck[i]);
    }
    while (!dq.empty()) {
        ans.emplace_back(dq.back());
        dq.pop_back();
    }
    return ans;
}


int longestWPI(vector<int>& hours)
{
    int i, j;
    int n = hours.size();
    int ans;
    vector<int> v;
    vector<vector<int>> vv;
    for (auto h : hours) {
        if (h > 8) {
            v.emplace_back(1);
        } else {
            v.emplace_back(-1);
        }
    }
    vector<int> prefixSum(n, 0);
    prefixSum[0] = v[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + v[i];
    }

    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(prefixSum[i], i));
    }
    sort (vp.begin(), vp.end());
    ans = 0;
    v.clear();
    v.emplace_back(vp[0].second);
    for (i = 1; i < n; i++) {
        if (vp[i].first == vp[i - 1].first) {
            v.emplace_back(vp[i].second);
        } else {
            vv.emplace_back(v);
            v.clear();
            v.emplace_back(vp[i].second);
        }
    }
    vv.emplace_back(v);
    // cout << vv.size() << endl;
    n = vv.size();
    for (i = 0; i < n - 1; i++) {
        if (prefixSum[vv[i][vv[i].size() - 1]] > 0) {
            ans = max(ans, vv[i][vv[i].size() - 1] + 1);
        }
        for (j = i + 1; j < n; j++) {
            if (prefixSum[vv[j][vv[j].size() - 1]] > 0) {
                ans = max(ans, vv[j][vv[j].size() - 1] + 1);
            } else {
                ans = max(vv[j][vv[j].size() - 1] - vv[i][0], ans);
            }
        }
    }
    return ans;
}


int maxWidthRamp(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int ans, minIdx;
    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(nums[i], i));
    }
    sort (vp.begin(), vp.end());
    ans = 0;
    minIdx = vp[0].second;
    for (i = 1; i < n; i++) {
        if (vp[i].second > minIdx) {
            ans = max(ans, vp[i].second - minIdx);
        }
        minIdx = min(minIdx, vp[i].second);
    }
    return ans;
}


// LC1186
// dp[i][j]  以i结尾是否已删除一个数字的最大子数组和 j = 0 未删 j = 1 已删
int maximumSum(vector<int>& arr)
{
    int i;
    int n = arr.size();
    int a, b, ans;
    vector<vector<int>> dp(n, vector<int>(2, 0));

    dp[0][0] = arr[0];
    ans = arr[0];
    for (i = 1; i < n; i++) {
        dp[i][0] = (dp[i - 1][0] < 0 ? arr[i] : dp[i - 1][0] + arr[i]);
        a = dp[i - 1][0]; // arr[i] 删除
        b = (dp[i - 1][1] < 0 ? arr[i] : dp[i - 1][1] + arr[i]); // arr[i] 不删除
        dp[i][1] = max(a, b);
        ans = max(ans, max(dp[i][0], dp[i][1]));
    }
    return ans;
}


// LC1011
int shipWithinDays(vector<int>& weights, int days)
{
    int i;
    int n = weights.size();
    int sum = 0;
    int cnt;
    int left, right, mid;

    left = weights[0];
    for (auto w : weights) {
        sum += w;
        left = max(left, w);
    }
    right = sum;
    while (left <= right) { // 所求为left
        mid = (right - left) / 2 + left;
        cnt = 0;
        sum = 0;
        for (i = 0; i < n ; i++) {
            sum += weights[i];
            if (sum > mid) {
                sum = weights[i];
                cnt++;
            }
        }
        cnt++;
        if (cnt > days) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


int func(long long n)
{
    int t;
    int mod = 1000000007;
    long long ans = 0;

    while (n > 1) {
        t = ceil(log10(n) / log10(2));
        ans = (ans + (n - static_cast<long long>(pow(2, t - 1))) * t) % mod;
        n = pow(2, t - 1);
    }
    return ans;
}


// LC2571
int minOperations(int n)
{
    int i, j;
    int step;
    int k, t;
    vector<int> nums = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    unordered_set<int> visited;
    queue<int> q;

    step = 0;
    q.push(n);
    visited.emplace(n);
    while (!q.empty()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            t = q.front();
            // cout << t << " ";
            // visited.emplace(t);
            q.pop();
            for (j = 0; j < nums.size(); j++) {
                if (t == nums[j]) {
                    return step + 1;
                }
                if (t - nums[j] > 0 && visited.count(t - nums[j]) == 0) {
                    q.push(t - nums[j]);
                    visited.emplace(t - nums[j]);
                }
                if (t + nums[j] < 131072 && visited.count(t + nums[j]) == 0) {
                    q.push(t + nums[j]);
                    visited.emplace(t + nums[j]);
                }
            }
        }
        step++;
    }
    return -1;
}


// LC898
int subarrayBitwiseORs(vector<int>& arr)
{
    int i, j;
    int n = arr.size();
    unordered_set<int> s;

    for (i = 0; i < n; i++) {
        s.emplace(arr[i]);
        for (j = i - 1; j >= 0; j--) {
            if ((arr[i] | arr[j]) == arr[j]) {
                break;
            }
            arr[j] |= arr[i];
            s.emplace(arr[j]);
        }
    }
    return s.size();
}


// LC444
bool sequenceReconstruction(vector<int>& nums, vector<vector<int>>& sequences)
{
    int i;
    unordered_map<int, unordered_set<int>> edges;
    for (auto seq : sequences) {
        for (i = 1; i < seq.size(); i++) {
            edges[seq[i - 1]].emplace(seq[i]);
        }
    }
    for (i = 1; i < nums.size(); i++) {
        if (edges[nums[i - 1]].count(nums[i]) == 0) {
            return false;
        }
    }
    return true;
}


// LC2552
// 0 <= i < j < k < l < n
// nums[i] < nums[k] < nums[j] < nums[l]
/*long long countQuadruplets(vector<int>& nums)
{
    long long ans;
    // long long left, right;
    unsigned int n = nums.size();
    vector<vector<int>> left(n, vector<int>(n, 0)), right(n, vector<int>(n, 0));
    // vector<int> left(n, 0), right(n, 0);
    int j, k;

    j = 1;
    k = n - 2;
    // left = right = 0;
    if (nums[0] < nums[k]) {
        left[1][k] = 1;
    }
    if (nums[n - 1] > nums[j]) {
        right[n - 1][j] = 1;
    }
    for ()
    ans = 0;
    for (j = 1; j < n; j++) {
        for (k = n - 2; k >= 0; k--) {
            if (j >= k) {
                break;
            }
            if (nums[k] >= nums[j]) {
                continue;
            }

            ans += static_cast<long long>(left[j][k]) * right[k][j];
        }
    }
    return ans;
}*/


// LC1131
int maxAbsValExpr(vector<int>& arr1, vector<int>& arr2)
{
    int i;
    int n = arr1.size();
    int minVal, ans;

    ans = 0;
    // arr1[i] + arr2[i] + i
    minVal = arr1[0] + arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, arr1[i] + arr2[i] + i - minVal);
        minVal = min(minVal, arr1[i] + arr2[i] + i);
    }
    // -arr1[i] + arr2[i] + i;
    minVal = -arr1[0] + arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, -arr1[i] + arr2[i] + i - minVal);
        minVal = min(minVal, -arr1[i] + arr2[i] + i);
    }
    // arr1[i] - arr2[i] + i;
    minVal = arr1[0] - arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, arr1[i] - arr2[i] + i - minVal);
        minVal = min(minVal, arr1[i] - arr2[i] + i);
    }
    // -arr1[i] - arr2[i] + i;
    minVal = -arr1[0] - arr2[0] + 0;
    for (i = 1; i < n; i++) {
        ans = max(ans, -arr1[i] - arr2[i] + i - minVal);
        minVal = min(minVal, -arr1[i] - arr2[i] + i);
    }
    return ans;
}


// LC310
int curLongestRouteLen = 0;
vector<int> longestRoute;
void DFSSearchLongestRourte(unordered_map<int, unordered_set<int>>& edge, int cur, vector<bool>& visited, vector<int>& route, bool sign, bool &quit)
{
    if (quit) {
        return;
    }
    visited[cur] = true;
    route.push_back(cur);
    if (sign == false) {
        if (curLongestRouteLen < route.size()) {
            curLongestRouteLen = route.size();
        }
    } else {
        if (route.size() == curLongestRouteLen) {
            longestRoute = route;
            quit = true;
            return;
        }
    }
    for (auto it : edge[cur]) {
        if (visited[it] == false) {
            DFSSearchLongestRourte(edge, it, visited, route, sign, quit);
        }
    }
    route.pop_back();
}
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges)
{
    int i;
    int node, num;
    int lastNode;
    vector<int> ans;
    vector<bool> visited(n, false);
    unordered_map<int, unordered_set<int>> edge;
    if (n == 1) {
        return {0};
    }
    for (auto e : edges) {
        edge[e[0]].emplace(e[1]);
        edge[e[1]].emplace(e[0]);
    }

    int dist;
    queue<int> q;

    dist = 0;
    q.push(0);
    while (q.size()) {
        num = q.size();
        for (i = 0; i < n; i++) {
            node = q.front();
            q.pop();
            visited[node] = true;
            for (auto it : edge[node]) {
                if (visited[it] == false) {
                    q.push(it);
                    lastNode = it;
                }
            }
        }
        dist++;
    }
    // cout << lastNode;
    // 从lastNode找最长路径
    visited.assign(n, false);
    vector<int> route;
    bool sign = false;
    bool quit = false;
    DFSSearchLongestRourte(edge, lastNode, visited, route, sign, quit);
    sign = true;
    visited.assign(n, false);
    DFSSearchLongestRourte(edge, lastNode, visited, route, sign, quit);
    num = longestRoute.size();
    if (num % 2 == 1) {
        ans = {longestRoute[num / 2]};
    } else {
        ans = {longestRoute[num / 2 - 1], longestRoute[num / 2]};
    }
    return ans;
}


// LC210
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites)
{
    int i;
    int n, t;
    vector<int> ans;
    vector<int> degree(numCourses, 0);
    unordered_map<int, unordered_set<int>> edge;
    for (auto p : prerequisites) {
        degree[p[0]]++;
        edge[p[1]].emplace(p[0]);
    }
    queue<int> q;
    for (i = 0; i < numCourses; i++) {
        if (degree[i] == 0) {
            q.push(i);
        }
    }
    if (q.empty()) {
        return {};
    }
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            ans.emplace_back(t);
            if (edge.count(t) == 0) {
                continue;
            }
            for (auto it : edge[t]) {
                degree[it]--;
                if (degree[it] == 0) {
                    q.push(it);
                }
            }
        }
    }
    if (ans.size() != numCourses) {
        return {};
    }
    return ans;
}


// LC2576
int maxNumOfMarkedIndices(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int ans, t;
    int left, right, mid;
    vector<bool> visited(n, false);

    sort(nums.begin(), nums.end());
    ans = 0;
    left = n / 2;
    for (i = 0; i < n / 2; i++) {
        right = n - 1;
        t = nums[i] * 2;
        while (left <= right) { // 所求为left
            mid = (right - left) / 2 + left;
            if (nums[mid] >= t) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (left >= n) {
            break;
        }
        if (visited[left] != false) {
            for (j = left; j < n; j++) {
                if (visited[j] == false) {
                    break;
                }
            }
            if (j == n + 1) {
                break;
            } else {
                left = j;
            }
        }
        visited[left] = true;
        // cout << nums[i] << " " << nums[left] << " " << left << endl;
        ans += 2;

        left++;
        if (left >= n) {
            break;
        }
    }
    return ans;
}


// 面试题 05.02
string printBin(double num)
{
    string ans = "0.";
    double carry = 0.5;
    while (carry > 0.0078125) {
        if (num >= carry) {
            ans += '1';
            num -= carry;
            if (fabs(num) < 10e-7) {
                return ans;
            }
        } else {
            ans += '0';
        }
        carry /= 2;
    }
    return "ERROR";
}


// 面试题 08.02
void DFSFindPath(vector<vector<int>>& obstacleGrid, int row, int col, vector<vector<int>>& route, vector<vector<int>>& ans, bool& find)
{
    if (find) {
        return;
    }
    int i;
    int m, n;
    int nrow, ncol;
    int directions[2][2] = {{0, 1}, {1, 0}};

    m = obstacleGrid.size();
    n = obstacleGrid[0].size();
    route.push_back({row, col});
    if (row == m - 1 && col == n - 1) {
        find = true;
        ans = route;
        return;
    }
    for (i = 0; i < 2; i++) {
        nrow = row + directions[i][0];
        ncol = col + directions[i][1];

        if (nrow >= 0 && nrow < m && ncol >= 0 && ncol < n && obstacleGrid[nrow][ncol] == 0) {
            obstacleGrid[nrow][ncol] = 1;
            DFSFindPath(obstacleGrid, nrow, ncol, route, ans, find);
            // obstacleGrid[nrow][ncol] = 0; 此处回溯会超时
        }
    }
    route.pop_back();
}
vector<vector<int>> pathWithObstacles(vector<vector<int>>& obstacleGrid)
{
    vector<vector<int>> ans;
    vector<vector<int>> route;
    vector<vector<int>> grid;
    bool find = false;
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();

    if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) {
        return ans;
    }
    grid = obstacleGrid;
    DFSFindPath(grid, 0, 0, route, ans, find);
    return ans;
}


// LC982
int countTriplets(vector<int>& nums)
{
    int i, j;
    int ans;
    int n;
    unordered_map<int, int> andCnt;

    n = nums.size();
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            andCnt[(nums[i] & nums[j])]++;
        }
    }
    ans = 0;
    for (auto it : andCnt) {
        for (auto n : nums) {
            if ((n & it.first) == 0) {
                ans += it.second;
            }
        }
    }
    return ans;
}


class LC2581 {
public:
    int ans = 0;
    map<long long, int> condition;
    map<long long, int> curUsedCondition;
    void DFS(unordered_map<int, unordered_set<int>>& gridedges, size_t cur, size_t parent, int& curk)
    {
        // cout << curk << endl;
        long long t = parent << 32 | cur;
        if (condition.count(t) == 1) {
            // printf ("[%d %d]\n", parent, cur);
            curk += condition[t];
            curUsedCondition[t] = condition[t];
        }
        for (auto e : gridedges[cur]) {
            if (e != parent) {
                DFS(gridedges, e, cur, curk);
            }
        }
    }
    void ChangeRoot(unordered_map<int, unordered_set<int>>& gridedges, size_t cur, size_t parent, int curk, int k)
    {
        long long t = cur << 32 | parent;
        if (condition.count(t) == 1 && curUsedCondition.count(t) == 0) {
            curk += condition[t];
            curUsedCondition[t] += condition[t];
        }
        t = parent << 32 | cur;
        if (condition.count(t) == 1 && curUsedCondition.count(t) == 1) {
            curk -= condition[t];
            curUsedCondition.erase(t);
        }
        if (curk >= k) {
            ans++;
        }
        for (auto it : gridedges[cur]) {
            if (it != parent) {
                ChangeRoot(gridedges, it, cur, curk, k);
            }
        }
    }
    int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k)
    {
        for (auto g : guesses) {
            condition[(long long)g[0] << 32 | g[1]]++;
        }
        unordered_map<int, unordered_set<int>> gridedges;
        for (auto e : edges) {
            gridedges[e[0]].emplace(e[1]);
            gridedges[e[1]].emplace(e[0]);
        }
        int i;
        int n = edges.size() + 1;
        int curk;

        curk = 0;
        DFS(gridedges, 0, -1, curk);
        if (curk >= k) {
            ans++;
        }
        for (auto it : gridedges[0]) {
            ChangeRoot(gridedges, it, 0, curk, k);
        }
        return ans;
    }
};


// LC1599
int minOperationsMaxProfit(vector<int>& customers, int boardingCost, int runningCost)
{
    int ans;
    int curProfit, t;
    int curTime, curCustomers;

    curTime = 0;
    curCustomers = 0;
    ans = -1;
    t = INT_MIN;
    curProfit = 0;
    while (1) {
        if (curTime < customers.size()) {
            curCustomers += customers[curTime];
        }
        if (curCustomers >= 4) {
            curProfit += 4 * boardingCost - runningCost;
            curCustomers -= 4;
        } else {
            curProfit += curCustomers * boardingCost - runningCost;
            curCustomers = 0;
        }
        if (curProfit > t) {
            t = curProfit;
            ans = curTime + 1;
        }
        curTime++;
        if (curTime >= customers.size() && curCustomers == 0) {
            break;
        }
    }
    return curProfit <= 0 ? -1 : ans;
}


// 面试题 16.09. 运算
class Operations {
public:
    Operations() {

    }
    long negative(long n)
    {
        if (n == 0) {
            return 0;
        }
        bool f = false;
        long base = -1;
        if (n < 0) {
            f = true;
            base = 1;
        }
        long i = 0;
        while (1) {
            if (f) { 
                if (i + base + n > 0) {
                    base = 1;
                } else if (i + base + n == 0) {
                    i += base;
                    break;
                } else {
                    i += base;
                    base += base;
                }
            } else {
                if (i + base + n < 0) {
                    base = -1;
                } else if (i + base + n == 0) {
                    i += base;
                    break;
                } else {
                    i += base;
                    base += base;
                }
            }
        }
        return i;
    }
    int minus(int a, int b)
    {
        return a + negative(b);
    }
    
    int multiply(int a, int b)
    {
        if (a == 0 || b == 0) {
            return 0;
        }
        bool f = false;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
            f  = true;
        }
        long aa, bb;
        if (a < 0) {
            aa = negative(a);
        } else {
            aa = a;
        }
        if (b < 0) {
            bb = negative(b);
        } else {
            bb = b;
        }

        long ans = 0;
        long base = 1;
        long i = 0;
        long t = aa;
        while (1) {
            if (i + base > bb) {
                base = 1;
                t = aa;
            } else if (i + base == bb) {
                i += base;
                ans += t;
                break;
            } else {
                ans += t;
                i += base;
                base += base;
                t += t;
            }
        }
        if (f) {
            return negative(ans);
        }
        return ans;
    }
    long my_multiply(long a, long b)
    {
        if (a == 0 || b == 0) {
            return 0;
        }
        long ans = 0;
        long base = 1;
        long i = 0;
        long t = a;
        while (1) {
            if (i + base > b) {
                base = 1;
                t = a;
            } else if (i + base == b) {
                i += base;
                ans += t;
                break;
            } else {
                ans += t;
                i += base;
                base += base;
                t += t;
            }
        }
        return ans;
    }
    long GetHalfOfNum(long n)
    {
        long i;
        long base;

        base = 1;
        i = 0;
        while (1) {
            if (i + base + i + base > n) {
                base = 1;
            } else if (i + base + i + base == n || i + base + i + base + 1 == n) {
                i += base;
                break;
            } else {
                i += base;
                base += base;
            }
        }
        return i;
    }
    int divide(int a, int b)
    {
        if (a == 0) {
            return 0;
        } else if (b == 1) {
            return a;
        } else if (b == -1) {
            return negative(a);
        }
        bool f = false;
        if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
            f  = true;
        }
        long aa, bb;
        if (a < 0) {
            aa = negative(a);
        } else {
            aa = a;
        }
        if (b < 0) {
            bb = negative(b);
        } else {
            bb = b;
        }
        if (bb > aa) {
            return 0;
        }
        long left, right, mid;
        long t;

        left = 1;
        right = aa;
        while (left <= right) {  // 所求为right
            mid = GetHalfOfNum(left + right);
            t = my_multiply(mid, bb);
            if (t > aa) {
                right = minus(mid, 1);
            } else if (t < aa) {
                left = mid + 1;
            } else {
                if (f) {
                    return negative(mid);
                }
                return mid;
            }
        }
        if (f) {
            return negative(right);
        }
        return right;
    }
};


// 面试题 17.05. 字母与数字
vector<string> findLongestSubarray(vector<string>& array)
{
    int i;
    int n = array.size();
    int t;
    int curLongest;
    map<int, pair<int, int>> data;
    map<int, vector<int>> diffData;
    pair<int, int> ansIdx = {-1, -1};
    vector<string> ans;

    curLongest = 0;
    data[0] = {0, 0};
    for (i = 0; i < n; i++) {
        if (i != 0) {
            data[i] = data[i - 1];
        }
        if (isalpha(array[i][0])) {
            data[i].first++;
            diffData[data[i].first - data[i].second].emplace_back(i);
        } else {
            data[i].second++;
            diffData[data[i].first - data[i].second].emplace_back(i);
        }
        if (data[i].first == data[i].second) {
            if (i + 1 > curLongest) {
                curLongest = i + 1;
                ansIdx = {0, i};
            }
        } else {
            t = data[i].first - data[i].second;
            if (diffData.count(t) == 1 && diffData[t].size() != 1) {
                if (i - diffData[t][0] > curLongest) {
                    curLongest = i - diffData[t][0];
                    ansIdx = {diffData[t][0] + 1, i};
                }
            }
        }
    }
    if (ansIdx.first == -1) {
        return {};
    }
    for (i = ansIdx.first; i <= ansIdx.second; i++) {
        ans.emplace_back(array[i]);
    }
    return ans;
}


// LC1617
void DFSCalcSubTree(unordered_map<int, unordered_set<int>>& treeedges, int cur, int parent, 
    set<int>& nodes, int dist, int& maxDist, set<int>& visited)
{
    visited.emplace(cur);
    maxDist = max(dist, maxDist);
    for (auto it : treeedges[cur]) {
        if (it != parent && nodes.count(it) == 1) {
            DFSCalcSubTree(treeedges, it, cur, nodes, dist + 1, maxDist, visited);
        }
    }
}
void ChooseNode(int n, int idx, int cnt,int tolcnt, vector<int>& chosenNode, vector<set<int>>& nodes)
{
    set<int> node;
    if (tolcnt == cnt) {
        for (auto n : chosenNode) {
            // cout << n << " ";
            node.emplace(n);
        }
        // cout << endl;
        nodes.emplace_back(node);
        return;
    }
    int i;
    for (i = idx; i < n; i++) {
        chosenNode.emplace_back(i + 1);
        ChooseNode(n, i + 1, cnt + 1, tolcnt, chosenNode, nodes);
        chosenNode.pop_back();
    }
}
vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges)
{
    int i, j, k;
    int maxDist;
    vector<int> ans(n - 1, 0);
    unordered_map<int, unordered_set<int>> treeedges;
    vector<int> chosenNode; 
    unordered_map<int, int> subTreeDistCount;
    vector<set<int>> nodes;
    set<int> visited;
    // bool f = false; // 是否所有点都能有效访问
    for (auto e : edges) {
        treeedges[e[0]].emplace(e[1]);
        treeedges[e[1]].emplace(e[0]);
    }
    for (i = 2; i <= n; i++) {
        nodes.clear();
        ChooseNode(n, 0, 0, i, chosenNode, nodes); // 枚举所有点的集合
        for (j = 0; j < nodes.size(); j++) {
            /* 超时
            maxDist = 0;
            for (auto it : nodes[j]) {
                visited.clear();
                f = false;
                DFSCalcSubTree(treeedges, it, -1, nodes[j], 0, maxDist, visited);
                if (visited != nodes[j]) {
                    break;
                }
                f = true;
            }
            if (f) {
                subTreeDistCount[maxDist]++;
            }
            */
            // 两次BFS求最大距离
            int t;
            int m;
            int lastNode;
            queue<int> q;
            visited.clear(); 
            q.push(*nodes[j].begin());
            while (q.size()) {
                m = q.size();
                for (k = 0; k < m; k++) {
                    t = q.front();
                    lastNode = t;
                    q.pop();
                    visited.emplace(t);
                    for (auto it : treeedges[t]) {
                        if (nodes[j].count(it) == 1 && visited.count(it) == 0) {
                            q.push(it);
                            //lastNode = it;
                        }
                    }
                }
            }
            // 判断点集是否能组成子树
            if (visited != nodes[j]) {
                continue;
            }
            //cout << lastNode << endl;
            int dist;

            q.push(lastNode);
            visited.clear();
            dist = 0;
            while (q.size()) {
                m = q.size();
                for (k = 0; k < m; k++) {
                    t = q.front();
                    q.pop();
                    visited.emplace(t);
                    for (auto it : treeedges[t]) {
                        if (nodes[j].count(it) == 1 && visited.count(it) == 0) {
                            q.push(it);
                            // lastNode = it;
                        }
                    }
                }
                if (q.size() != 0)
                    dist++;
            }

            subTreeDistCount[dist]++;
            // for (auto it : nodes[j]) {cout << it << " ";} cout << dist << endl;
        }
    }
    for (auto it : subTreeDistCount ) {
    ans[it.first - 1] = it.second;
    }
    return ans;
}


// 面试题 16.13. 平分正方形
vector<vector<double>> CalcDot(double x, vector<int>& square, pair<double, double>& center, double k)
{
    double y;
    double x1, y1;

    y = k * (x - center.first) + center.second;
    x1 = x + square[2];
    y1 = k * (x1 - center.first) + center.second;
    if (k > 0 && y < square[1]) {
        y = square[1];
        x = (y - center.second) / k + center.first;
        y1 = square[1] + square[2];
        x1 = (y1 - center.second) / k + center.first;
    }
    if (k < 0 && y > square[1] + square[2]) {
        y = square[1] + square[2];
        x = (y - center.second) / k + center.first;
        y1 = square[1];
        x1 = (y1 - center.second) / k + center.first;
    }
    return {{x, y}, {x1, y1}};
}
vector<double> cutSquares(vector<int>& square1, vector<int>& square2)
{
    double k;
    pair<double, double> dot1, dot2;

    dot1 = {square1[0] + square1[2] * 1.0 / 2, square1[1] + square1[2] * 1.0 / 2};
    dot2 = {square2[0] + square2[2] * 1.0 / 2, square2[1] + square2[2] * 1.0 / 2};

    vector<double> ans(4, 0);
    if (fabs(dot1.first - dot2.first) < 10e-5) {
        ans[0] = dot1.first;
        ans[1] = max(dot1.second + square1[2] * 1.0 / 2,
            dot2.second + square2[2] * 1.0 / 2);
        ans[2] = dot2.first;
        ans[3] = min(dot1.second - square1[2] * 1.0 / 2,
            dot2.second - square2[2] * 1.0 / 2);
        if (ans[1] > ans[3]) {
            swap(ans[1], ans[3]);
        }
        return ans;
    }
    k = (dot1.second - dot2.second) / (dot1.first - dot2.first);

    // 分别求直线穿过的4个点
    vector<vector<double>> t1, t2;
    t1 = CalcDot(square1[0], square1, dot1, k);
    t2 = CalcDot(square2[0], square2, dot2, k);
    t1.insert(t1.end(), t2.begin(), t2.end());
    sort(t1.begin(), t1.end());
    return {t1[0][0], t1[0][1], t1[3][0], t1[3][1]};
}


// LC2597
void ChooseNode(int n, int idx, int cnt, int tolcnt, int numsArr[], int numsArrSize, int chosenNode[], int k, int& ans)
{
    if (tolcnt - cnt > numsArrSize - idx) {
        return;
    }
    if (tolcnt == cnt) {
        ans++;
        return;
    }
    int i, j;
    for (i = idx; i < numsArrSize; i++) {
        if (cnt > 0) {
            for (j = 0; j < cnt; j++) {
                if (abs(chosenNode[j] - numsArr[i]) == k) {
                    break;
                }
            }
            if (j != cnt) {
                continue;
            }
        }
        chosenNode[cnt] = numsArr[i];
        ChooseNode(n, i + 1, cnt + 1, tolcnt, numsArr, numsArrSize, chosenNode, k, ans);
    }
}
int beautifulSubsets(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int chosenNode[20] = {0};
    int numsArr[20] = {0};
    for (i = 0; i < n; i++) {
        numsArr[i] = nums[i];
    }
    int ans = 0;
    for (i = 2; i <= n; i++) {
        ChooseNode(n, 0, 0, i, numsArr, n, chosenNode, k, ans);
    }
    return ans + n;
}
int beautifulSubsets_1(vector<int>& nums, int k) // 动态规划
{
    int i;
    int n = nums.size();
    int mod, t;
    int ans;
    vector<int> f(n + 1, 0);
    vector<int> v;
    unordered_map<int, vector<int>> modK;
    unordered_map<int, int> numCnt;

    sort(nums.begin(), nums.end()); 
    for (i = 0; i < n; i++) {
        numCnt[nums[i]]++;
        mod = nums[i] % k;
        if (modK.count(mod) == 0) {
            modK[mod].emplace_back(nums[i]);
        } else {
            t = modK[mod][modK[mod].size() - 1];
            if (t != nums[i]) {
                modK[mod].emplace_back(nums[i]);
            }
        }
    }
    ans = 0;
    vector<int> sum;
    for (auto it : modK) {
        v = it.second;
        f[0] = 1;
        f[1] = pow(2, numCnt[it.second[0]]);
        for (i = 1; i < v.size(); i++) {
            if (v[i] - v[i - 1] != k) {
                f[i + 1] = f[i] * pow(2, numCnt[v[i]]);
            } else {
                f[i + 1] = f[i] + f[i - 1] * (pow(2, numCnt[v[i]]) - 1);
            }
        }
        // cout << f[v.size()] << endl;;
        if (sum.size() == 0) {
            sum.emplace_back(f[v.size()] - 1);
        } else {
            n = sum.size();
            for (i = 0; i < n; i++) {
                sum.emplace_back(sum[i] * (f[v.size()] - 1));
            }
            sum.emplace_back(f[v.size()] - 1);
        }
    }
    for (auto s : sum) {
        ans += s;
    }
    return ans;
}
int beautifulSubsets_2(vector<int>& nums, int k) // 动态规划,更好方法
{
    int i;
    int n = nums.size();
    int mod, t;
    int ans;
    vector<int> f(n + 1, 0);
    vector<int> v;
    unordered_map<int, vector<int>> modK;
    unordered_map<int, int> numCnt;

    sort(nums.begin(), nums.end()); 
    for (i = 0; i < n; i++) {
        numCnt[nums[i]]++;
        mod = nums[i] % k;
        if (modK.count(mod) == 0) {
            modK[mod].emplace_back(nums[i]);
        } else {
            t = modK[mod][modK[mod].size() - 1];
            if (t != nums[i]) {
                modK[mod].emplace_back(nums[i]);
            }
        }
    }
    ans = 1;
    for (auto it : modK) {
        v = it.second;
        f[0] = 1;
        f[1] = pow(2, numCnt[it.second[0]]);
        for (i = 1; i < v.size(); i++) {
            if (v[i] - v[i - 1] != k) {
                f[i + 1] = f[i] * pow(2, numCnt[v[i]]);
            } else {
                f[i + 1] = f[i] + f[i - 1] * (pow(2, numCnt[v[i]]) - 1);
            }
        }
        // cout << f[v.size()] << endl;
        ans *= f[v.size()];
    }
    return ans - 1;
}

// LC1012
int numDupDigitsAtMostN(int n)
{
    int i, k;
    int t, x, cnt, countUnique;
    string str;
    vector<int> f(10, 0), prefixSum(n, 0);

    f[1] = 9;
    k = 9;
    for (i = 2; i <= 9; i++) {
        f[i] = f[i - 1] * k;
        k--;
    }
    prefixSum[1] = f[1];
    for (i = 2; i <= 9; i++) {
        prefixSum[i] = prefixSum[i - 1] + f[i];
    }
    t = n;
    countUnique = 0;
    str = to_string(n);
    
    k = str.size();
    //if (str[0] > '1') {
        countUnique += prefixSum[k - 1];
    //}
    t = k;
    for (i = 0; i < k; i++) {
        if (str[i] == '0') {
            continue;
        }
        x = str[i] - '0' - 1;
        if (i == k - 1) {
            x++;
        }
        countUnique += (f[t] / 9 * x);
        t--;
    }
    return n - countUnique;
}


// LC127
bool IsConnectedWord(string& a, string& b)
{
    int i;
    int n = a.size();
    int cnt;

    cnt = 0;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            cnt++;
        }
    }
    return cnt == 1;
}
int ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
    int i, j;
    int n = wordList.size();
    int ans;
    string t;
    unordered_set<string> visited;
    unordered_map<string, unordered_set<string>> edges;

    for (i = 0; i < n; i++) {
        if (wordList[i] == beginWord) {
            break;
        }
    }
    if (i == n) {
        wordList.emplace_back(beginWord);
        n = wordList.size();
    }
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (IsConnectedWord(wordList[i], wordList[j])) {
                edges[wordList[i]].emplace(wordList[j]);
                edges[wordList[j]].emplace(wordList[i]);
            }
        }
    }
    /* for (auto it1 : edges) {
        cout << it1.first << ":";
        for (auto it2 : it1.second) {
            cout << it2 << " ";
        }
        cout << endl;
    } */
    if (edges.count(endWord) == 0) {
        return 0;
    }

    queue<string> q;
    q.push(beginWord);
    ans = 1;
    while (!q.empty()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            if (t == endWord) {
                return ans;
            }
            visited.emplace(t);
            for (auto it : edges[t]) {
                if (visited.count(it) == 0) {
                    q.push(it);
                }
            }
        }
        ans++;
    }
    return 0;
}


// LC820
int minimumLengthEncoding(vector<string>& words)
{
    int n = words.size();
    int m;
    int i, j, k;
    vector<bool> canShorten(n, false);
    for (i = 0; i < n; i++) {
        if (canShorten[i]) {
            continue;
        }
        for (j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }
            if (words[i].size() > words[j].size()) {
                continue;
            }
            if (canShorten[j]) {
                continue;
            }
            m = words[j].size();
            for (k = words[i].size() - 1; k >= 0; k--) {
                if (words[i][k] != words[j][m - 1]) {
                    break;
                }
                m--;
            }
            if (k == -1) {
                canShorten[i] = true;
                break;
            }
        }
    }
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (canShorten[i] == false) {
            ans += words[i].size() + 1;
        }
    }
    return ans;
}
void DFSScanTrieTree(Trie<char> *node, int curLen, int& ans)
{
    int i;
    if (node->children.size() == 0) {
        ans += curLen + 1;
        return;
    }
    for (i = 0; i < node->children.size(); i++) {
        DFSScanTrieTree(node->children[i], curLen + 1, ans);
    }
}
int minimumLengthEncoding_1(vector<string>& words) // 后缀字典树
{
    int ans = 0;
    Trie<char> *root = new Trie<char>('/');
    for (auto word : words) {
        reverse(word.begin(), word.end());
        Trie<char>::CreateWordTrie(root, word);
    }
    DFSScanTrieTree(root, 0, ans);
    delete(root);
    return ans;
}


// LC2305
void DFSGetFullPermutation(int n, vector<int>& record, vector<bool>& visited, vector<vector<int>>& ans)
{
    int i;
    if (record.size() == n) {
        ans.emplace_back(record);
        return;
    }
    for (i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        visited[i] = true;
        record.emplace_back(i);
        DFSGetFullPermutation(n, record, visited, ans);
        visited[i] = false;
        record.pop_back();
    }
}
void DFSTrySplitWays(string& t, int idx, int cnt, vector<string>& ans)
{
    int i;
    int n = t.size();
    if (n - idx < cnt) {
        return;
    }
    if (cnt == 0) {
        ans.emplace_back('0' + t);
        return;
    }
    for (i = idx; i < n; i++) {
        t[i] = '1';
        DFSTrySplitWays(t, i + 1, cnt - 1, ans);
        t[i] = '0';
    }
}
int distributeCookies(vector<int>& cookies, int k)
{
    int i, j, l;
    int ans, curSum, unfairness;
    int n = cookies.size();
    string t(n - 1, '0');
    vector<bool> visited(n, false);

    vector<string> SplitWays;
    vector<int> record;
    vector<vector<int>> FullPermutation;

    DFSTrySplitWays(t, 0, k - 1, SplitWays);
    DFSGetFullPermutation(n, record, visited, FullPermutation);

    int pSize = FullPermutation.size();
    int swSize = SplitWays.size();

    ans = INT_MAX;
    for (i = 0; i < pSize; i++) {
        for (j = 0; j < swSize; j++) {
            curSum = 0;
            unfairness = 0;
            for (l = 0; l < n; l++) {
                if (SplitWays[j][l] == '0') {
                    curSum += cookies[FullPermutation[i][l]];
                } else {
                    unfairness = max(unfairness, curSum);
                    curSum = cookies[FullPermutation[i][l]];
                }
            }
            unfairness = max(unfairness, curSum);
            ans = min(ans, unfairness);
        }
    }
    return ans;
}


// LC711
void ScanGraphAndRecord(vector<vector<int>>& grid, int row, int col, vector<vector<bool>> &visited, vector<pair<int, int>>& points)
{
    if (row < 0 || row == grid.size() || col < 0 || col == grid[0].size() || grid[row][col] == 0 || visited[row][col] == true)
        return;

    int i;
    int direction[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

    visited[row][col] = true;
    points.push_back({row, col});
    for (i = 0; i < 4; i++)
        ScanGraphAndRecord(grid, row + direction[i][0], col + direction[i][1], visited, points);
}
int numDistinctIslands2(vector<vector<int>>& grid)
{
    int i, j, k, l;
    int minR, minC;
    set<string> shape;
    
    if (grid.size() == 0)
        return 0;

    string shapeStr;
    vector<vector<bool>> visited(grid.size(),vector<bool>(grid[0].size(), false));
    vector<pair<int, int>> points, t;
    vector<string> shapeStrColl;
    vector<vector<int>> pontsTrans = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (i = 0; i < grid.size(); i++) {
        for (j = 0; j < grid[0].size(); j++) {
            if (grid[i][j] == 1 && visited[i][j] == false) {
                points.clear();
                shapeStrColl.clear();
                ScanGraphAndRecord(grid, i, j, visited, points);

                // 求points的8种变换坐标
                // 即 (x,y),(−x,y),(x,−y),(−x,−y),(y,x),(−y,x),(y,−x),(−y,−x)
                t = points;
                for (l = 0; l < 4; l++) {
                    for (k = 0; k < points.size(); k++) {
                        t[k].first = points[k].first * pontsTrans[l][0];
                        t[k].second = points[k].second * pontsTrans[l][1];
                    }
                    // 变换后的相对坐标
                    minR = INT_MAX;
                    minC = INT_MAX;
                    for (k = 0; k < points.size(); k++) {
                        minR = min(minR, t[k].first);
                        minC = min(minC, t[k].second);
                    }
                    for (k = 0; k < points.size(); k++) {
                        t[k].first -= minR;
                        t[k].second -= minC;
                    }
                    sort(t.begin(), t.end());
                    shapeStr.clear();
                    for (auto p : t) {
                    //    printf ("(%d, %d)->",p.first, p.second);
                        shapeStr += "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
                    }
                    // cout << endl;
                    shapeStrColl.emplace_back(shapeStr);
                }
                for (k = 0; k < points.size(); k++) {
                    int tmp = points[k].first;
                    points[k].first = points[k].second;
                    points[k].second = tmp;
                }
                t = points;
                for (l = 0; l < 4; l++) {
                    for (k = 0; k < points.size(); k++) {
                        t[k].first = points[k].first * pontsTrans[l][0];
                        t[k].second = points[k].second * pontsTrans[l][1];
                    }
                    // 变换后的相对坐标
                    minR = INT_MAX;
                    minC = INT_MAX;
                    for (k = 0; k < points.size(); k++) {
                        minR = min(minR, t[k].first);
                        minC = min(minC, t[k].second);
                    }
                    for (k = 0; k < points.size(); k++) {
                        t[k].first -= minR;
                        t[k].second -= minC;
                    }
                    sort(t.begin(), t.end());
                    shapeStr.clear();
                    for (auto p : t) {
                    //    printf ("(%d, %d)->",p.first, p.second);
                        shapeStr += "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
                    }
                    // cout << endl;
                    shapeStrColl.emplace_back(shapeStr);
                }
                sort(shapeStrColl.begin(), shapeStrColl.end());
                if (shape.count(shapeStrColl[0]) == 0) {
                    shape.emplace(shapeStrColl[0]);
                }
            }
            
        }
    }
    return shape.size();
}


// LCP12
int minTime(vector<int>& time, int m)
{
    int i;
    int n = time.size();
    int t, cntDay, curMax;
    int left, right, mid;
    bool f = false;
    if (n <= m) {
        return 0;
    }
    right = 0;
    for (i = 0; i < n; i++) {
        // left = max(left, time[i]);
        right += time[i];
    }
    left = 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        t = 0;
        cntDay = 0;
        curMax = 0;
        f = false;
        for (i = 0; i < n; i++) {
            if (i == 3) {
                ;
            }
            if (t + time[i] > mid) {
                if (f) {
                    cntDay++;
                    t = time[i];
                    if (t > mid) {
                        // cntDay++;
                        t = 0;
                        f = true;
                        curMax = 0;
                        continue;
                    }
                    curMax = max(curMax, time[i]);
                    f = false;
                    continue;
                }
                if (time[i] >= curMax) {
                    f = true;
                    curMax = 0;
                } else {
                    f = true;
                    t -= curMax;
                    t += time[i];
                    curMax = 0;
                }

            } else {
                t += time[i];
                if (f == false) {
                    curMax = max(curMax, time[i]);
                }
            }
        }
        if (t > 0 || f) {
            cntDay++;
        }
        if (cntDay > m) {
            left = mid + 1;
        } else if (cntDay <= m) {
            right = mid - 1;
        }
    }
    return left;
}



// LCP45
vector<vector<int>> bicycleYard(vector<int>& position, vector<vector<int>>& terrain, vector<vector<int>>& obstacle)
{
    int i, k;
    int x, y, v, newV;
    int n = terrain.size();
    int m = terrain[0].size();
    int size;
    vector<int> pos;
    vector<vector<int>> ans;
    vector<vector<int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    set<vector<int>> visited;
    set<vector<int>> sameVPos;
    queue<vector<int>> q;
    vector<int> start = {position[0], position[1], 1, -1}; // x, y坐标, 速度, 方向

    q.push(start);
    // visited[position[0]][position[1]] = true;
    visited.insert({position[0], position[1], 1, 0});
    visited.insert({position[0], position[1], 1, 1});
    visited.insert({position[0], position[1], 1, 2});
    visited.insert({position[0], position[1], 1, 3});
    while (q.size()) {
        size = q.size();
        for (i = 0; i < size; i++) {
            pos = q.front();
            q.pop();
            for (k = 0; k < 4; k++) {
                x = pos[0] + directions[k][0];
                y = pos[1] + directions[k][1];
                v = pos[2];

                if (x < 0 || x >= n || y < 0 || y >= m) {
                    continue;
                }
                newV = v + terrain[pos[0]][pos[1]] - terrain[x][y] - obstacle[x][y];
                if (newV <= 0 || visited.count({x, y, newV, k}) == 1) {
                    continue;
                }
                visited.insert({x, y, newV, k});
                if (newV == 1) {
                    sameVPos.insert({x, y});
                }
                q.push({x, y, newV});
            }
        }
    }
    for (auto it : sameVPos) {
        ans.emplace_back(it);
    }
    return ans;
}


// LCP41
template<typename T>
void ClearQueue(queue<T>& q)
{
    queue<T> empty;
    swap(empty, q);
}
int CntChess(vector<string>& chessboard, int row, int col)
{
    int cnt, t;
    int i, j, k;
    int n = chessboard.size();
    int m = chessboard[0].size();
    vector<vector<int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    queue<pair<int, int>> q, tq;
    cnt = 0;
    for (k = 0; k < directions.size(); k++) {
        t = 0;
        i = row;
        j = col;
        while (1) {
            i += directions[k][0];
            j += directions[k][1];
            if (i < 0 || i >= n || j < 0 || j >= m) {
                ClearQueue<pair<int, int>>(q);
                break;
            }
            if (chessboard[i][j] == 'O') {
                q.push({i, j});
            } else if (chessboard[i][j] == 'X') {
                if (q.size() == 0) {
                    break;
                }
                t += q.size();
                while (!q.empty()) {
                    auto p = q.front();
                    tq.push(p);
                    chessboard[p.first][p.second] = 'X';
                    q.pop();
                }
                while (!tq.empty()) {
                    auto p = tq.front();
                    t += CntChess(chessboard, p.first, p.second);
                    tq.pop();
                }
                break;
            } else if (chessboard[i][j] == '.') {
                ClearQueue<pair<int, int>>(q);
                break;
            }
        }
        cnt += t;
    }
    return cnt;
}
int flipChess(vector<string>& chessboard)
{
    int i, j;
    int ans;
    int n = chessboard.size();
    int m = chessboard[0].size();
    vector<string> tmpChess;

    ans = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (chessboard[i][j] == '.') {
                tmpChess = chessboard;
                tmpChess[i][j] = 'X';
                ans = max(ans, CntChess(tmpChess, i, j));
            }
        }
    }
    return ans;
}


// LC300
int lengthOfLIS(vector<int> &nums)
{
    int i, j;
    int n = nums.size();
    int ans;
    vector<int> dp(n, 1); // 以i下标结尾的最长递增子序列长度

    for (i = 1; i < n; i++) {
        for (j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    ans = dp[0];
    for (i = 1; i < n; i++) {
        ans = max(ans, dp[i]);
    }
    return ans;
}
// LC673
int findNumberOfLIS(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    vector<int> t;
    int curLen, cnt;
    vector<pair<int, int>> dp(n, {0, 0}); // 以i下标结尾的最长递增子序列长度, 以及对应的序列个数

    dp[0] = {1, 1};
    for (i = 1; i < n; i++) {
        cnt = curLen = 0;
        for (j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) {
                if (dp[j].first + 1 > curLen) {
                    curLen = dp[j].first + 1;
                    cnt = dp[j].second;
                } else if (dp[j].first + 1 == curLen) {
                    cnt += dp[j].second;
                }
            }
        }
        if (curLen == 0) {
            dp[i] = {1, 1};
        } else {
            dp[i] = {curLen, cnt};
        }
    }
    int maxLen = 0;
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (dp[i].first > maxLen) {
            maxLen = dp[i].first;
            ans = dp[i].second;
        } else if (dp[i].first == maxLen) {
            ans += dp[i].second;
        }
    }
    return ans;
}


// LC1125
void DFSFindSmallestSufficientTeam(vector<string>& req_skills, int curIdx, unordered_map<string, vector<int>>& skillsPeopleKnow,
    unordered_map<int, unordered_set<string>>& peopleSkills, vector<int>& record, vector<int>& ans, int& minTeamSize)
{
    if (curIdx == req_skills.size()) {
        if (minTeamSize == -1 || record.size() < minTeamSize) {
            minTeamSize = record.size();
            ans = record;
        }
        return;
    }
    int i;
    bool f = false;
    // 判断这个技能是否已经有人在前面会了
    for (auto r : record) {
        if (peopleSkills[r].count(req_skills[curIdx]) == 1) {
            f = true;
            break;
        }
    }
    if (f) {
        DFSFindSmallestSufficientTeam(req_skills, curIdx + 1, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);
    } else {
        for (auto it : skillsPeopleKnow[req_skills[curIdx]]) {  // 这些人会req_skills[curIdx]
            record.emplace_back(it);
            if (minTeamSize != -1 && record.size() > minTeamSize) {
                record.pop_back();
                return;
            }
            DFSFindSmallestSufficientTeam(req_skills, curIdx + 1, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);
            record.pop_back();
        }
    }
}
vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people)
{
    int minTeamSize;
    unordered_map<string, vector<int>> skillsPeopleKnow; // 技能 - 会的人物列表
    unordered_map<int, unordered_set<string>> peopleSkills; // 人物 - 会的技能列表
    vector<int> ans;
    vector<int> record;
    vector<vector<int>> results;

    int i, j;
    int num = people.size();
    for (i = 0; i < num; i++) {
        for (j = 0; j < people[i].size(); j++) {
            skillsPeopleKnow[people[i][j]].emplace_back(i);
            peopleSkills[i].emplace(people[i][j]);
        }
    }
    minTeamSize = -1;
    DFSFindSmallestSufficientTeam(req_skills, 0, skillsPeopleKnow, peopleSkills, record, ans, minTeamSize);

    return ans;
}


// LC1041
bool CheckIfEqual(int len, vector<vector<int>>& source, int startIdx)
{
    int n = source.size();
    int i, j;

    for (i = 0; i < len; i++) {
        if (i + startIdx >= n) {
            break;
        }
        if (source[i + startIdx] != source[i]) {
            return false;
        }
    }
    return true;
}
bool isRobotBounded(string instructions)
{
    int i, k;
    int n = instructions.size();
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int curD, x, y;
    bool loop = false;
    vector<vector<int>> pos;
    vector<int> curPos;
    vector<int> t;

    pos.push_back({0, 0, 0});
    curD = 0;
    curPos = {0, 0};
    for (k = 0; k < 10; k++) {
        for (i = 0; i < n; i++) {
            if (instructions[i] == 'G') {
                x = curPos[0] + directions[curD][0];
                y = curPos[1] + directions[curD][1];
                t = {x, y, curD};
                curPos = {x, y};
                pos.emplace_back(t);
            } else if (instructions[i] == 'L') {
                curD = (curD + 3) % 4;
                t = {curPos[0], curPos[1], curD};
                pos.emplace_back(t);
            } else {
                curD = (curD + 1) % 4;
                t = {curPos[0], curPos[1], curD};
                pos.emplace_back(t);
            }
        }
    }
    /*for (auto p : pos) {
        for (auto a : p) {
            cout << a << " ";
        }
        cout << endl;
    } */
    n = pos.size();
    int len = 1;
    while (1) {
        loop = true;
        i = len;
        while (1) {
            if (CheckIfEqual(len, pos, i) == false) {
                loop = false;
                break;
            }
            i += len;
            if (i >= n) {
                loop = true;
                break;
            }
        }
        if (loop) {
            return true;
        }
        len++;
        if (len * 2 > n) {
            break;
        }
    }
    return false;
}


// LC2560
// 求k个不相邻最大值中的最小值
int minCapability(vector<int>& nums, int k)
{
    int i;
    int size = nums.size();
    if (size == 1) {
        return nums[0];
    } else if (size == 2) {
        return min(nums[0], nums[1]);
    }

    int left, right, mid;
    int curMaxSteal;
    vector<int> dp(size, 0);
    left = INT_MAX;
    right = INT_MIN;
    for (auto n : nums) {
        left = min(left, n);
        right = max(right, n);
    }
    // dp[i] = max(dp[i - 1], dp[i - 2] + 1) 下标i选择窃取或不窃取能得手的最大房间数
    while (left <= right) {
        mid = (right - left) / 2 + left;  // 可能的最大窃取金额
        curMaxSteal = 0;
        if (nums[0] <= mid) {
            dp[0] = 1;
            curMaxSteal = 1;
        } else {
            dp[0] = 0;
        }
        if (nums[1] <= mid) {
            dp[1] = 1;
            curMaxSteal = 1;
        } else {
            dp[1] = dp[0];
        }
        for (i = 2; i < size; i++) {
            if (nums[i] <= mid) {
                dp[i] = max(dp[i - 1], dp[i - 2] + 1);
                curMaxSteal = max(curMaxSteal, dp[i]);
            } else {
                dp[i] = dp[i - 1];
            }
        }
        if (curMaxSteal >= k) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2616
// 求p个不相邻最小值中的最大值,与上题相反
int minimizeMax(vector<int>& nums, int p)
{
    int i;
    int n = nums.size();
    vector<int> diff;

    if (p == 0) {
        return 0;
    }
    sort(nums.begin(), nums.end());
    for (i = 1; i < n; i++) {
        diff.emplace_back(nums[i] - nums[i - 1]);
    }
    n = diff.size();
    if (n == 1) {
        return diff[0];
    } else if (n == 2) {
        return min(diff[0], diff[1]);
    }

    vector<int> dp(n, 0);
    int left, right, mid;
    int curMinCnt;

    left = INT_MAX;
    right = INT_MIN;
    for (auto d : diff) {
        left = min(d, left);
        right = max(d, right);
    }
    while (left <= right) {
        mid = (right - left) / 2 + left; // mid 可能的最大值
        curMinCnt = 0;
        if (diff[0] <= mid) {
            dp[0] = 1;
            curMinCnt = 1;
        } else {
            dp[0] = 0;
        }

        if (diff[1] <= mid) {
            dp[1] = 1;
            curMinCnt = 1;
        } else {
            dp[1] = dp[0];
        }

        for (i = 2; i < n; i++) {
            if (diff[i] <= mid) {
                dp[i] = max(dp[i - 1], dp[i - 2] + 1);
                curMinCnt = max(curMinCnt, dp[i]);
            } else {
                dp[i] = dp[i - 1];
            }
        }
        if (curMinCnt >= p) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2608
int BFSFindMinLoopSize(int n, unordered_map<int, unordered_set<int>>& edges, int curNode)
{
    int d, val, loopSize, minLoopSize;
    int i, k;
    queue<pair<int, int>> q; // curNode - fromNode
    pair<int, int> t;
    vector<bool> visited(n, false);
    vector<int> distance(n);

    q.push({curNode, - 1});
    d = 0;
    minLoopSize = INT_MAX;
    while (q.size()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            t = q.front();
            q.pop();
            visited[t.first] = true;
            distance[t.first] = d;
            for (auto it : edges[t.first]) {
                if (it != t.second && visited[it] == true) {
                    loopSize = distance[it] + distance[t.first] + 1;
                    // printf ("n = %d, loopSize = %d\n", curNode, loopSize);
                    minLoopSize = min(minLoopSize, loopSize);
                    // return loopSize; 不能直接退出
                }
                if (it != t.second && visited[it] == false) {
                    q.push({it, t.first});
                }
            }
        }
        d++;
    }
    return minLoopSize == INT_MAX ? -1 : minLoopSize;
}
int findShortestCycle(int n, vector<vector<int>>& edges)
{
    unsigned int i;
    int loopSize, ans;
    unordered_map<int, unordered_set<int>> e;
    for (i = 0; i < edges.size(); i++) {
        e[edges[i][0]].emplace(edges[i][1]);
        e[edges[i][1]].emplace(edges[i][0]);
    }

    ans = INT_MAX;
    for (i = 0; i < n; i++) {
        loopSize = BFSFindMinLoopSize(n, e, i);
        if (loopSize > 0) {
            ans = min(ans, loopSize);
        }
    }
    return ans == INT_MAX ? -1 : ans;
}


// LC2064
int minimizedMaximum(int n, vector<int>& quantities)
{
    int sum;
    int left, right, mid;

    left = 1;
    right = quantities[0];
    for (auto q : quantities) {
        right = max(right, q);
    }

    while (left <= right) {
        mid = (right - left) / 2 + left;
        sum = 0;
        for (auto q : quantities) {
            sum += static_cast<int>(ceil(q * 1.0 / mid));
        }
        if (sum > n) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1042
vector<int> gardenNoAdj(int n, vector<vector<int>>& paths)
{
    int i, j;
    int curVal;
    vector<int> ans(n, 0);
    unordered_map<int, int> gardenVal;
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> s;
    bool comflict = false;

    for (auto p : paths) {
        edges[p[0]].emplace(p[1]);
        edges[p[1]].emplace(p[0]);
    }
    for (i = 1; i <= n; i++) {
        if (gardenVal.count(i) == 0) {
            gardenVal[i] = 1;
            for (auto e : edges[i]) {
                if (gardenVal.count(e) == 0) {
                    s.clear();
                    for (auto it : edges[e]) {
                        if (gardenVal.count(it) == 1) {
                            s.emplace(gardenVal[it]);
                        }
                    }
                    for (j = 1; j <= 4; j++) {
                        if (s.count(j) == 0) {
                            gardenVal[e] = j;
                            break;
                        }
                    }
                }
            }
        } else {
            s.clear();
            for (auto e : edges[i]) {
                if (gardenVal.count(e) == 1) {
                    s.emplace(gardenVal[e]);
                }
            }
            comflict = false;
            for (auto it : s) {
                if (it == gardenVal[i]) {
                    comflict = true;
                    break;
                }
            }
            if (comflict) {
                for (j = 1; j <= 4; j++) {
                    if (s.count(j) == 0) {
                        gardenVal[i] = j;
                        break;
                    }
                }
            }
        }
    }
    for (auto it : gardenVal) {
        ans[it.first - 1] = it.second;
    }
    return ans;
}
vector<int> gardenNoAdj_1(int n, vector<vector<int>>& paths)
{
    int i, j;
    int curVal;
    vector<int> ans(n, 0);
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> s;

    for (auto p : paths) {
        edges[p[0]].emplace(p[1]);
        edges[p[1]].emplace(p[0]);
    }
    for (i = 1; i <= n; i++) {
        s.clear();
        for (auto e : edges[i]) {
            if (ans[e - 1] != 0) {
                s.emplace(ans[e - 1]);
            }
        }
        for (j = 1; j <= 4; j++) {
            if (s.count(j) == 0) {
                ans[i - 1] = j;
                break;
            }
        }
    }
    return ans;
}


// LC2641
TreeNode* replaceValueInTree(TreeNode* root)
{
    int i, j;
    int n;
    queue<pair<TreeNode *, TreeNode *>> q; // pair 子 - 父
    vector<vector<pair<TreeNode *, TreeNode *>>> treeLayerLists;
    vector<pair<TreeNode *, TreeNode *>> v;
    pair<TreeNode *, TreeNode *> t;

    q.push({root, nullptr});
    while (q.size()) {
        n = q.size();
        v.clear();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            v.emplace_back(t);
            if (t.first->left != nullptr) {
                q.push({t.first->left, t.first});
            }
            if (t.first->right != nullptr) {
                q.push({t.first->right, t.first});
            }
        }
        treeLayerLists.emplace_back(v);
    }
    unordered_map<TreeNode *, int> fatherSum;
    n = treeLayerLists.size();
    vector<int> layerSum(n, 0);
    for (i = 0; i < n; i++) {
        for (j = 0; j < treeLayerLists[i].size(); j++) {
            fatherSum[treeLayerLists[i][j].second] += treeLayerLists[i][j].first->val;
            layerSum[i] += treeLayerLists[i][j].first->val;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < treeLayerLists[i].size(); j++) {
            treeLayerLists[i][j].first->val = layerSum[i] - fatherSum[treeLayerLists[i][j].second];
        }
    }
    return root;
}


// LC2642
class Graph {
public:
    int n;
    unordered_map<int, set<vector<int>>> edges;
    Graph(int n, vector<vector<int>>& edges)
    {
        for (auto e : edges) {
            (this->edges)[e[0]].insert({e[1], e[2]});
        }
        this->n = n;
    }
    
    void addEdge(vector<int> edge)
    {
        (this->edges)[edge[0]].insert({edge[1], edge[2]});
    }
    
    int shortestPath(int node1, int node2) // 超时
    {
        pair<int, int> t;
        vector<int> nodeDist(n, INT_MAX);
        queue<pair<int, int>> q;

        if (node1 == node2) {
            return 0;
        }
        q.push({node1, 0});
        while (q.size()) {
            t = q.front();
            q.pop();
            if (nodeDist[t.first] < t.second) {
                continue;
            }
            nodeDist[t.first] = t.second;
            for (auto e : edges[t.first]) {
                if (nodeDist[e[0]] > t.second + e[1]) {
                    nodeDist[e[0]] = t.second + e[1];
                    q.push({e[0], t.second + e[1]});
                }
            }
        }
        return nodeDist[node2] == INT_MAX ? -1 : nodeDist[node2];
    }
};

class Graph1 {
public:
    int n;
    vector<vector<pair<int, int>>> edges;
    Graph1(int n, vector<vector<int>>& edges)
    {
        this->n = n;
        this->edges = vector<vector<pair<int, int>>>(n);
        for (auto e : edges) {
            (this->edges)[e[0]].push_back({e[1], e[2]});
        }
    }
    
    void addEdge(vector<int> edge)
    {
        (this->edges)[edge[0]].push_back({edge[1], edge[2]});
    }
    
    int shortestPath(int node1, int node2)
    {
        pair<int, int> t;
        vector<int> nodeDist(n, INT_MAX);
        queue<pair<int, int>> q;
        if (node1 == node2) {
            return 0;
        }
        q.push({node1, 0});
        while (q.size()) {
            t = q.front();
            q.pop();
            if (nodeDist[t.first] < t.second) {
                continue;
            }
            nodeDist[t.first] = t.second;
            for (auto e : edges[t.first]) {
                if (nodeDist[e.first] > t.second + e.second) {
                    nodeDist[e.first] = t.second + e.second;
                    q.push({e.first, t.second + e.second});
                }
            }
        }
        return nodeDist[node2] == INT_MAX ? -1 : nodeDist[node2];
    }
};

// LC2646
unordered_map<int, int> tripCostData;
void DFSCalcTripCost(unordered_map<int, unordered_set<int>>& e, int from, int cur, int dest, string route, vector<int>& price, bool& find)
{
    if (find) {
        return;
    }
    if (cur == dest) {
        route[route.size() - 1] = '\0';
        
        int i;
        int node;
        int t;
        vector<string> vs = MySplit(route, '_');
        for (i = 0; i < vs.size(); i++) {
            node = atoi(vs[i].c_str());
            tripCostData[node] += price[node];
        }
        find = true;
        return;
    }
    for (auto it : e[cur]) {
        if (it != from) {
            DFSCalcTripCost(e, cur, it, dest, route + to_string(it) + "_", price, find);
        }
    }
}
int CalcMaxLessCost(unordered_map<int, unordered_set<int>>& edges, int cur, int from, bool chosen,
    unordered_map<int, int>& tripCostData, map<pair<int, bool>, int>& lessCost)
{
    if (edges[cur].size() == 1 && *edges[cur].begin() == from) {
        if (chosen) {
            lessCost[{cur, chosen}] = tripCostData[cur] / 2;
        } else {
            lessCost[{cur, chosen}] = 0;
        }
        return lessCost[{cur, chosen}];
    }
    for (auto it : edges[cur]) {
        if (it != from) {
            if (lessCost.count({it, false}) == 0) {
                lessCost[{it, false}] = CalcMaxLessCost(edges, it, cur, false, tripCostData, lessCost);
            }
            if (lessCost.count({it, true}) == 0) {
                lessCost[{it, true}] = CalcMaxLessCost(edges, it, cur, true, tripCostData, lessCost);
            }
        }
    }
    int ans = 0;
    if (chosen) {
        ans = tripCostData[cur] / 2;
        for (auto it : edges[cur]) {
            if (it != from) {
                ans += lessCost[{it, false}];
            }
        }
        lessCost[{cur, chosen}] = ans;
    } else {
        for (auto it : edges[cur]) {
            if (it != from) {
                ans += max(lessCost[{it, false}], lessCost[{it, true}]);
            }
        }
        lessCost[{cur, chosen}] = ans;
    }
    return ans;
}
int minimumTotalPrice(int n, vector<vector<int>>& edges, vector<int>& price, vector<vector<int>>& trips)
{
    int ans;
    unordered_map<int, unordered_set<int>> e;
    unordered_set<int> curChosenNode;
    bool find = false;

    if (edges.size() == 0) {
        ans = 0;
        for (auto trip : trips) {
            ans += price[trip[0]] / 2;
        }
        return ans;
    }
    for (auto edge : edges) {
        e[edge[0]].emplace(edge[1]);
        e[edge[1]].emplace(edge[0]);
    }
    string route;
    for (auto trip : trips) {
        if (trip[0] == trip[1]) {
            tripCostData[trip[0]] += price[trip[0]];
            continue;
        }
        find = false;
        route.clear();
        route = to_string(trip[0]) + "_";
        DFSCalcTripCost(e, -1, trip[0], trip[1], route, price, find);
    }

    ans = 0;
    for (auto it : tripCostData) {
        // printf ("%d : %d\n", it.first, it.second);
        ans += it.second;
    }
    // printf ("ans = %d\n", ans);
    int startNode = tripCostData.begin()->first;
    map<pair<int, bool>, int> lessCost; // <node - 是否减半> - 所节约的最大费用
    int a = CalcMaxLessCost(e, startNode, -1, true, tripCostData, lessCost); // 从startNode出发, 该点选择减半
    int b = CalcMaxLessCost(e, startNode, -1, false, tripCostData, lessCost); // 从startNode出发, 该点选择不减半
    return ans - max(a, b);
}


// LC1733
int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships)
{
    int i;
    int m = languages.size();
    int popularPerson, lessPopularPerson;
    unordered_map<int, unordered_set<int>> skill;
    unordered_map<int, int> langCnt;
    for (i = 0; i < m; i++) {
        for (auto lang : languages[i]) {
            skill[i + 1].emplace(lang);
        }
    }
    int sk;
    unordered_set<int> cantCommunicate;
    for (auto fr : friendships) {
        unordered_set<int>::iterator it;
        for (it = skill[fr[0]].begin(); it != skill[fr[0]].end(); it++) {
            if (skill[fr[1]].count(*it) == 1) {
                break; 
            } else {
                sk = *it;
            }
        }
        if (it == skill[fr[0]].end()) {
            cantCommunicate.emplace(fr[0]);
            cantCommunicate.emplace(fr[1]);
        }
    }
    for (auto it : cantCommunicate) {
        for (auto lang : skill[it]) {
            langCnt[lang]++;
        }
    }
    int t = 0;
    int lang = 0;
    for (auto it : langCnt) {
        if (t < it.second) {
            t = it.second;
            lang = it.first;
        }
    }
    int ans = 0;
    for (auto it : cantCommunicate) {
        if (skill[it].count(lang) == 0) {
            ans++;
        }
    }
    return ans;
}


// 投n个6面骰子概率分布
vector<double> dicesProbability(int n)
{
    // p[n][sum] - 投n个骰子和为sum的概率
    vector<vector<double>> p(n + 1, vector<double>(6 * n + 1, 0.0));
    int i, j, k;
    for (i = 1; i <= 6; i++) {
        p[1][i] = 1.0 / 6;
    }
    for (i = 2; i <= n; i++) {
        for (j = i - 1; j <= 6 * (i - 1); j++) {
            for (k = 1; k <= 6; k++) {
                p[i][j + k] += p[i - 1][j] * (1.0 / 6);
            }
        }
    }
    vector<double> ans;
    for (i = n; i <= n * 6; i++) {
        ans.emplace_back(p[n][i]);
    }
    return ans;
}


// LC1043
int maxSumAfterPartitioning(vector<int>& arr, int k)
{
    // dp[i][j] - 以i为下标长度为j的子数组最大和, j <= i + 1, j >= 0, j <= k
    int i, j, p;
    int t, len;
    int n = arr.size();
    vector<vector<int>> dp(n, vector<int>(k + 1, 0));

    // 从后往前找以i为下标,长度为k的子数组中的最大值
    vector<vector<int>> rangeMax(n, vector<int>(n, 0));
    for (i = n - 1; i >= 0; i--) {
        t = 0;
        for (j = i; j >= 0; j--) {
            if (j + k == i) {
                break;
            }
            t = max(t, arr[j]);
            rangeMax[j][i] = t;
        }
    }
    dp[0][1] = arr[0];
    for (i = 1; i < n; i++) {
        for (j = i; j >= 0; j--) {
            if (j + k == i) {
                break;
            }
            len = i - j + 1;
            dp[i][len] = rangeMax[j][i] * len;
            t = 0;
            for (p = 1; p <= k; p++) {
                if (i - len + 1 - p < 0) {
                    break;
                }
                t = max(t, dp[i - len][p]);
            }
            dp[i][i - j + 1] += t;
        }
    }
    int ans = 0;

    for (p = 1; p <= k; p++) {
        ans = max(ans, dp[n - 1][p]);
    }
    return ans;
}


// LC2320
int countHousePlacements(int n)
{
    // dp[i][0] - 下标i处不放置房子方案数; dp[i][1] - 下标i处放置房子方案数; 下标从1开始
    // dp[i][1] = dp[i - 1][0]; dp[i][0] = dp[i - 1][0] + dp[i - 1][1];
    int i;
    int mod = 1000000007;
    long long ans, t;
    vector<vector<long long>> dp(n + 1, vector<long long>(2, 0));

    dp[1][0] = 1;
    dp[1][1] = 1;
    for (i = 2; i <= n; i++) {
        dp[i][1] = dp[i - 1][0];
        dp[i][0] = (dp[i - 1][0] + dp[i - 1][1]) % mod;
    }
    t = dp[n][0] + dp[n][1];
    ans = t * t % mod;
    return ans;
}


// LC1178
void DFSGetPuzzlesHash(string& t, int idx, string& ori, unordered_map<string, int>& wordsHashCnt,
    unordered_map<string, int>& puzzlesCnt)
{
    int i;
    if (idx == ori.size()) {
        if (wordsHashCnt.count(t) == 1) {
            puzzlesCnt[ori] += wordsHashCnt[t];
        }
        return;
    }
    for (i = 0; i <= 1; i++) {
        t[ori[idx] - 'a'] = i + '0';
        DFSGetPuzzlesHash(t, idx + 1, ori, wordsHashCnt, puzzlesCnt);
    }
}
vector<int> findNumOfValidWords(vector<string>& words, vector<string>& puzzles)
{
    int i;
    unordered_set<char> wordHash;
    unordered_map<string, int> puzzlesCnt;
    unordered_set<string> uniquePuzzlesSet;
    unordered_map<string, int> wordsHashCnt;
    unordered_map<string, string> uniqueWordsHashMap;
    string t(26, '0');
    int n = puzzles.size();
    vector<int> ans(n, 0);

    for (auto word : words) {
        if (uniqueWordsHashMap.count(word) == 1) {
            wordsHashCnt[uniqueWordsHashMap[word]]++;
            continue;
        }
        wordHash.clear();
        t.assign(26, '0');
        for (i = 0; i < word.size(); i++) {
            t[word[i] - 'a'] = '1';
            wordHash.emplace(word[i]);
            if (wordHash.size() > 7) {
                break;
            }
        }
        if (i == word.size()) {
            wordsHashCnt[t]++;
            uniqueWordsHashMap[word] = t;
        }
    }
    for (auto p : puzzles) {
        if (uniquePuzzlesSet.count(p) == 1) {
            continue;
        }
        uniquePuzzlesSet.emplace(p);
        t.assign(26, '0');
        t[p[0] - 'a'] = '1';
        DFSGetPuzzlesHash(t, 1, p, wordsHashCnt, puzzlesCnt);
        // cout << usedPuzzlesHash.size() << endl;
    }
    for (i = 0; i < n; i++) {
        ans[i] = puzzlesCnt[puzzles[i]];
    }
    return ans;
}


// LCP72
vector<int> supplyWagon(vector<int>& supplies)
{
    int i;
    int n;
    vector<int> t, v;
    pair<int, int> p;
    vector<pair<int, int>> vp;

    v = supplies;
    while (v.size() > supplies.size() / 2) {
        t.clear();
        n = v.size();
        for (i = 1; i < n; i++) {
            vp.push_back({v[i] + v[i - 1], i - 1});
        }
        sort (vp.begin(), vp.end());
        p = vp[0];
        vp.clear();
        for (i = 0; i < n; i++) {
            if (i == p.second) {
                t.emplace_back(p.first);
                i++;
            } else {
                t.emplace_back(v[i]);
            }
        }
        v = t;
    }
    return v;
}


// LCP73
vector<string> MySplit(string& s, string separate)
{
    int idx;
    string t;
    vector<string> ans;

    t = s;
    while (1) {
        idx = t.find(separate);
        if (idx == string::npos) {
            ans.emplace_back(t);
            break;
        }
        ans.emplace_back(t.substr(0, idx));
        t = t.substr(idx + separate.size());
    }
    return ans;
}
int adventureCamp(vector<string>& expeditions)
{
    int i;
    int ans, newFound;
    int n = expeditions.size();
    vector<int> newFoundSite(n, 0);
    unordered_set<string> visitedSite;
    vector<string> vs;

    if (expeditions[0].size() > 0) {
        vs = MySplit(expeditions[0], "->");
        for (auto t : vs) {
            visitedSite.emplace(t);
        }
    }
    for (i = 1; i < n; i++) {
        if (expeditions[i].size() == 0) {
            continue;
        }
        vs = MySplit(expeditions[i], "->");
        for (auto t : vs) {
            if (visitedSite.count(t) == 0) {
                newFoundSite[i]++;
                visitedSite.emplace(t);
            }
        }
    }
    ans = -1;
    newFound = 0;
    for (i = 1; i < n; i++) {
        if (newFoundSite[i] > newFound) {
            newFound = newFoundSite[i];
            ans = i;
        }
    }
    return ans;
}


// LC2653
vector<int> getSubarrayBeauty(vector<int>& nums, int k, int x)
{
    int i;
    int n = nums.size();
    int t;
    vector<int> ans;
    map<int, int> cnt;
    for (i = 0; i < k; i++) {
        cnt[nums[i]]++;
    }
    t = 0;
    for (auto it : cnt) {
        t += it.second;
        if (t >= x) {
            ans.emplace_back(it.first < 0 ? it.first : 0);
            break;
        }
    }
    for (i = k; i < n; i++) {
        cnt[nums[i]]++;
        if (cnt[nums[i - k]] == 1) {
            cnt.erase(nums[i - k]);
        } else {
            cnt[nums[i - k]]--;
        }
        t = 0;
        for (auto it : cnt) {
            t += it.second;
            if (t >= x) {
                ans.emplace_back(it.first < 0 ? it.first : 0);
                break;
            }
        }
    }
    return ans;
}


// LC241
void DivideExpression(string& expression, vector<int>& nums)
{
    int i;
    int curIdx;
    int n = expression.size();
    unordered_set<char> sign = {'+', '-', '*'};

    curIdx = 0;
    for (i = 0; i < n; i++) {
        if (sign.count(expression[i]) == 1) {
            nums.emplace_back(atoi(expression.substr(curIdx, i - curIdx).c_str()));
            if (expression[i] == '+') {
                nums.emplace_back(-1);
            } else if (expression[i] == '-') {
                nums.emplace_back(-2);
            } else {
                nums.emplace_back(-3);
            }
            curIdx = i + 1;
        }
    }
    nums.emplace_back(atoi(expression.substr(curIdx).c_str()));
}

void Calc(vector<int>& nums, vector<int>& ans)
{
    int i, j;
    int n = nums.size();
    if (nums.size() == 3) {
        if (nums[1] == -1) {
            ans.emplace_back(nums[0] + nums[2]);
        } else if (nums[1] == -2) {
            ans.emplace_back(nums[0] - nums[2]);
        } else {
            ans.emplace_back(nums[0] * nums[2]);
        }
        return;
    }
    vector<int> v;
    int t;
    for (i = 1; i < n; i += 2) {
        if (nums[i] == -1) {
            t = nums[i - 1] + nums[i + 1];
        } else if (nums[i] == -2) {
            t = nums[i - 1] - nums[i + 1];
        } else {
            t = nums[i - 1] * nums[i + 1];
        }
        v.clear();
        for (j = 0; j < i - 1; j++) {
            v.emplace_back(nums[j]);
        }
        v.emplace_back(t);
        for (j = i + 1 + 1; j < n; j++) {
            v.emplace_back(nums[j]);
        }
        Calc(v, ans);
    }
}
vector<int> Calc1(int left, int right, vector<vector<vector<int>>>& dp, vector<int>& nums)
{
    if (left > right || left >= nums.size()) {
        return {};
    }
    if (!dp[left][right].empty()) {
        return dp[left][right];
    }
    int i, j, k;
    int m, n;
    vector<int> leftV, rightV;
    if (left == right) {
        dp[left][right].emplace_back(nums[left]);
        return dp[left][right];
    }
    for (i = left; i <= right; i += 2) {
        leftV = Calc1(left, i, dp, nums);
        rightV = Calc1(i + 2, right, dp, nums);
        m = leftV.size();
        n = rightV.size();
        for (j = 0; j < m; j++) {
            for (k = 0; k < n; k++) {
                if (nums[i + 1] == -1) {
                    dp[left][right].emplace_back(leftV[j] + rightV[k]);
                } else if (nums[i + 1] == -2) {
                    dp[left][right].emplace_back(leftV[j] - rightV[k]);
                } else {
                    dp[left][right].emplace_back(leftV[j] * rightV[k]);
                }
            }
        }
    }
    return dp[left][right];
}
vector<int> diffWaysToCompute(string expression)
{
    int i;
    int n;
    vector<int> nums;
    vector<int> ans;
    DivideExpression(expression, nums);

    n = nums.size();
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>())); // dp[l][r] 在l-r间的所有答案
    // Calc(nums, ans);
    Calc1(0, n - 1, dp, nums);
    return dp[0][n - 1];
}


// LC2439
int minimizeArrayValue(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left, right, mid;
    long long diff;

    left = right = nums[0];
    for (i = 0; i < n; i++) {
        left = min(left, nums[i]);
        right = max(right, nums[i]);
    }
    while (left <= right) {
        mid = (right - left) / 2 + left;
        diff = 0;
        for (i = 0; i < n; i++) {
            diff += static_cast<long long>(mid) - nums[i];
            if (diff < 0) {
                break;
            }
        }
        if (diff < 0) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right + 1;
}


// LC935
int knightDialer(int n)
{
    // dp[n][x] - 走n步落在x的总情况
    // 1 - (6, 8) 2 - (7, 9) 3 - (4, 8) 4 - （0, 3, 9）
    // 5 - () 6 - (0, 1, 7) 7 - (2, 6)
    // 8 - (1, 3) 9 - (2, 4) 0 - (4, 6)
    // dp[n][x] = dp[n - 1][a] + dp[n - 1][b] + ..., 所求 sum(dp[n])
    int i;
    int mod = 1000000007;
    vector<vector<long long>> dp(n + 1, vector<long long>(10, 0));
    
    for (i = 0; i < 10; i++) {
        dp[1][i] = 1;
    }
    for (i = 2; i <= n; i++) {
        dp[i][1] = (dp[i - 1][6] + dp[i - 1][8]) % mod;
        dp[i][2] = (dp[i - 1][7] + dp[i - 1][9]) % mod;
        dp[i][3] = (dp[i - 1][4] + dp[i - 1][8]) % mod;
        dp[i][4] = (dp[i - 1][0] + dp[i - 1][3] + dp[i - 1][9]) % mod;
        dp[i][6] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][7]) % mod;
        dp[i][7] = (dp[i - 1][2] + dp[i - 1][6]) % mod;
        dp[i][8] = (dp[i - 1][1] + dp[i - 1][3]) % mod;
        dp[i][9] = (dp[i - 1][2] + dp[i - 1][4]) % mod;
        dp[i][0] = (dp[i - 1][4] + dp[i - 1][6]) % mod;
    }
    long long ans = 0;
    for (i = 0; i < 10; i++) {
        ans = (ans + dp[n][i]) % mod;
    }
    return ans;
}


// LC2655
vector<vector<int>> findMaximalUncoveredRanges(int n, vector<vector<int>>& ranges)
{
    vector<vector<int>> combinedRange, ans;
    vector<int> t(2);
    int size = ranges.size();
    int i;

    if (size == 0) {
        return {{0, n - 1}};
    }
    sort(ranges.begin(), ranges.end());
    t[0] = ranges[0][0];
    t[1] = ranges[0][1];
    for (i = 1; i < size; i++) {
        if (ranges[i][0] <= t[1] + 1) {
            t[1] = max(ranges[i][1], t[1]);
        } else {
            combinedRange.emplace_back(t);
            t[0] = ranges[i][0];
            t[1] = ranges[i][1];
        }
    }
    combinedRange.emplace_back(t);

    if (combinedRange[0][0] >= 1) {
        ans.push_back({0, combinedRange[0][0] - 1});
    }
    size = combinedRange.size();
    for (i = 1; i < size; i++) {
        ans.push_back({combinedRange[i - 1][1] + 1, combinedRange[i][0] - 1});
    }
    if (combinedRange[size - 1][1] + 1 < n) {
        ans.push_back({combinedRange[size - 1][1] + 1, n - 1});
    }
    return ans;
}


// LC1334
// 迪杰斯特拉 dijstra
int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold)
{
    int i, k;
    int size = edges.size();
    int cnt;
    vector<int> dist;
    vector<int> cityCnt(n, 0);
    vector<vector<pair<int, int>>> edgeWithWeight(n);
    queue<pair<int, int>> q;
    pair<int, int> t;

    for (i = 0; i < size; i++) {
        edgeWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
        edgeWithWeight[edges[i][1]].push_back({edges[i][0], edges[i][2]});
    }

    for (i = 0; i < n; i++) {
        dist.assign(n, INT_MAX);

        q.push({i, 0});
        while (q.size()) {
            t = q.front();
            q.pop();

            if (dist[t.first] < t.second) {
                continue;
            }
            dist[t.first] = t.second;
            size = edgeWithWeight[t.first].size();
            for (auto e : edgeWithWeight[t.first]) {
                if (e.second + t.second < dist[e.first]) {
                    dist[e.first] = e.second + t.second;
                    q.push({e.first, dist[e.first]});
                }
            }
        }
        cnt = 0;
        for (k = 0; k < n; k++) {
            if (k != i && dist[k] <= distanceThreshold) {
                cnt++;
            }
        }
        cityCnt[i] = cnt;
    }
    int ans = 0;
    int cur = INT_MAX;
    for (i = 0; i < n; i++) {
        if (cityCnt[i] <= cur) {
            cur = cityCnt[i];
            ans = i;
        }
    }
    return ans;
}


// LC2658
void DFSScanGridForFish(vector<vector<int>>& grid, int row, int col, vector<vector<bool>>& visited, int& cnt)
{
    visited[row][col] = true;
    cnt += grid[row][col];

    int i;
    int directions[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int newR, newC;
    int m = grid.size();
    int n = grid[0].size();
    for (i = 0; i < 4; i++) {
        newR = row + directions[i][0];
        newC = col + directions[i][1];
        if (newR >= 0 && newR < m && newC >= 0 && newC < n && grid[newR][newC] > 0 && visited[newR][newC] == false) {
            DFSScanGridForFish(grid, newR, newC, visited, cnt);
        }
    }
}
int findMaxFish(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int ans, cnt;
    vector<vector<bool>> visited(m, vector<bool>(n, false));

    ans = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (visited[i][j] == false && grid[i][j] > 0) {
                cnt = 0;
                DFSScanGridForFish(grid, i, j, visited, cnt);
                ans = max(ans, cnt);
            }
        }
    }
    return ans;
}


// LC2662
template<typename T>
void InsertMinVal(map<T, int>& m, T key, int val) {
    if (m.count(key) == 0) {
        m[key] = val;
    } else {
        m[key] = min(m[key], val);
    }
};
int minimumCost(vector<int>& start, vector<int>& target, vector<vector<int>>& specialRoads)
{
    int i, j;
    int m = specialRoads.size();
    int ans, cost;
    long long base = 1e5;

    unordered_map<long long, int> dist;
    unordered_map<long long, unordered_set<long long>> edges;
    map<pair<long long, long long>, int> weight;

    auto cmp = [](pair<long long, int>& a, pair<long long, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, decltype(cmp)> pq(cmp);
    long long nodeStart = start[0] * base + start[1];
    long long nodeTarget = target[0] * base + target[1];

    // 对于每一条specialRoads, 它的出口都可以连接除它本身的其他specialRoads的入口, 还可以连接target
    // 将二维坐标化为一维坐标
    for (i = 0; i < m; i++) {
        auto node1Entrance = specialRoads[i][0] * base + specialRoads[i][1];
        auto node1Exit = specialRoads[i][2] * base + specialRoads[i][3];
        // start到特殊通道入口
        edges[nodeStart].emplace(node1Entrance);
        InsertMinVal(weight, {nodeStart, node1Entrance}, abs(start[0] - specialRoads[i][0]) +
            abs(start[1] - specialRoads[i][1]));

        // 特殊通道
        cost = abs(specialRoads[i][0] - specialRoads[i][2]) +
            abs(specialRoads[i][1] - specialRoads[i][3]);
        if (cost >= specialRoads[i][4]) {
            edges[node1Entrance].emplace(node1Exit);
            InsertMinVal(weight, {node1Entrance, node1Exit}, specialRoads[i][4]);
        }
        // 出口连target
        cost = abs(target[0] - specialRoads[i][2]) +
            abs(target[1] - specialRoads[i][3]);
        edges[node1Exit].emplace(nodeTarget);
        InsertMinVal(weight, {node1Exit, nodeTarget}, cost);
    }
    // 特殊通道出口连其他通道入口
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            if (i == j) {
                continue;
            }
            cost = abs(specialRoads[i][2] - specialRoads[j][0]) +
                abs(specialRoads[i][3] - specialRoads[j][1]);
            auto node1Exit = specialRoads[i][2] * base + specialRoads[i][3];
            auto node2Entrance = specialRoads[j][0] * base + specialRoads[j][1];

            edges[node1Exit].emplace(node2Entrance);
            InsertMinVal(weight, {node1Exit, node2Entrance}, cost);
        }
    }
    // start直接到target
    cost = abs(start[0] - target[0]) + abs(start[1] - target[1]);
    edges[nodeStart].emplace(nodeTarget);
    InsertMinVal(weight, {nodeStart, nodeTarget}, cost);
    
    ans = 0x3f3f3f3f;
    pq.push({nodeStart, 0});
    while (pq.size()) {
        auto p = pq.top();
        pq.pop();
        if (dist.count(p.first) && dist[p.first] < p.second) {
            continue;
        }
        dist[p.first] = p.second;
        // printf ("%d %d %d\n", p.first / base, p.first % base, p.second);
        if (p.first == nodeTarget) {
            ans = min(ans, p.second);
            break;
        }
        for (auto it : edges[p.first]) {
            if (dist.count(it) == 0) {
                dist[it] = p.second + weight[{p.first, it}];
                pq.push({it, dist[it]});
            } else {
                if (dist[it] <= p.second + weight[{p.first, it}]) {
                    continue;
                }
                dist[it] = p.second + weight[{p.first, it}];
                pq.push({it, dist[it]});
            }
        }
    }
    return ans;
}


// LC2659 (有问题)
long long countOperationsToEmptyArray(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int curIdx;
    long long ans = 0;
    pair<int, int> p;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    for (i = 0; i < n; i++) {
        pq.push({nums[i], i});
    }
    curIdx = 0;
    while (pq.size()) {
        p = pq.top();
        pq.pop();

        if (p.second > curIdx) {
            ans += static_cast<long long>(p.second - curIdx);
        }
    }
    return ans;
}


// LCP78
int rampartDefensiveLine(vector<vector<int>>& rampart)
{
    vector<int> diff;
    int i;
    int n = rampart.size();
    int curLeft;
    int left, right, mid;
    bool tooLarge = false;

    left = 1;
    right = 0;
    for (i = 1; i < n; i++) {
        diff.emplace_back(rampart[i][0] - rampart[i - 1][1]);
        right = max(right, rampart[i][0] + rampart[i - 1][1]);
    }
    n = diff.size();
    if (n == 2) {
        return diff[0] + diff[1];
    }
    int t = 0;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curLeft = diff[0];
        tooLarge = false;
        for (i = 1; i < n; i++) {
            if (curLeft >= mid) {
                curLeft = diff[i];
                continue;
            }
            curLeft = curLeft + diff[i] - mid;
            if (curLeft < 0) {
                right = mid - 1;
                tooLarge = true;
                break;
            }
        }
        if (tooLarge) {
            continue;
        }
        left = mid + 1;
    }
    return right;
}



// LCP80
string CalcEvolutionaryRoute(unordered_map<int, unordered_set<int>>& edges, int cur, unordered_map<int, string>& nodeRoute)
{
    if (edges.count(cur) == 0) {
        return "";
    }
    vector<string> routes;
    string route, ans;
    for (auto it : edges[cur]) {
        route.clear();
        if (nodeRoute.count(it) == 0) {
            route += '0';
            route += CalcEvolutionaryRoute(edges, it, nodeRoute);
            route += '1';
            nodeRoute[it] = route;
        }
        // printf ("%d: %s\n", cur, nodeRoute[it].c_str());
        routes.emplace_back(nodeRoute[it]);
    }
    sort(routes.begin(), routes.end());
    for (auto r : routes) {
        ans += r;
    }
    return ans;
}
string evolutionaryRecord(vector<int>& parents)
{
    int i;
    int n = parents.size();
    int root;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, string> nodeRoute;
    string ans;
    // 建图
    for (i = 0; i < n; i++) {
        if (parents[i] != -1) {
            edges[parents[i]].emplace(i);
        } else {
            root = i;
        }
    }
    ans = CalcEvolutionaryRoute(edges, root, nodeRoute);

    int len = ans.size();
    int idx = len;
    for (i = len - 1; i >= 0; i--) {
        if (ans[i] == '0') {
            idx = i;
            break;
        }
    }
    ans = ans.substr(0, idx + 1);
    return ans;
}


// LC2061
int numberOfCleanRooms(vector<vector<int>>& room)
{
    int m = room.size();
    int n = room[0].size();
    int nx, ny;
    int ans;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<int> cur;
    set<vector<int>> visited;

    cur = {0, 0, 0};
    visited.insert({0, 0, 0});
    while (1) {
        room[cur[0]][cur[1]] = 2;
        nx = cur[0] + directions[cur[2]][0];
        ny = cur[1] + directions[cur[2]][1];

        if (nx < 0 || nx >= m || ny < 0 || ny >= n || room[nx][ny] == 1) {
            cur[2] = (cur[2] + 1) % 4;
        } else {
            cur = {nx, ny, cur[2]};
        }
        if (visited.count(cur) == 1) {
            break;
        }
        visited.insert(cur);
    }
    ans = 0;
    for (auto r : room) {
        for (auto c : r) {
            if (c == 2) {
                ans++;
            }
        }
    }
    return ans;
}


// LC2049
int DFSCountNodeSubNode(int cur, unordered_map<int, vector<int>>& edges, unordered_map<int, pair<int, int>>& nodeCnt)
{
    if (edges.count(cur) == 0) {
        return 1;
    }
    int t;
    if (nodeCnt.count(cur) == 0) {
        t = DFSCountNodeSubNode(edges[cur][0], edges, nodeCnt);
        nodeCnt[cur].first = t;
        if (edges[cur].size() == 2) {
            t = DFSCountNodeSubNode(edges[cur][1], edges, nodeCnt);
            nodeCnt[cur].second = t;
        } else {
            nodeCnt[cur].second = 0;
        }
    }
    return nodeCnt[cur].first + nodeCnt[cur].second + 1;
}
int countHighestScoreNodes(vector<int>& parents)
{
    int i;
    int n = parents.size();
    int leftNodes, rightNodes, parentNodes;
    int root;
    map<long long, int, greater<>> multiply;
    unordered_map<int, vector<int>> edges; // node - {letf, right}
    unordered_map<int, pair<int, int>> nodeCnt; // node - {leftNodeCnt, rightNodeCnt}
    for (i = 0; i < n; i++) {
        if (parents[i] == -1) {
            root = i;
        } else {
            edges[parents[i]].push_back(i);
        }
    }
    DFSCountNodeSubNode(root, edges, nodeCnt);
    /* for (auto it : nodeCnt) {
        printf ("%d : %d %d\n", it.first, it.second.first, it.second.second);
    } */
    for (i = 0; i < n; i++) {
        if (i == root) {
            leftNodes = nodeCnt[i].first;
            rightNodes = nodeCnt[i].second == 0 ? 1 : nodeCnt[i].second;
            multiply[static_cast<long long>(leftNodes) * rightNodes]++;
            continue;
        }
        if (edges.count(i) == 0) { // 叶子节点
            multiply[n - 1]++;
            continue;
        }
        parentNodes = n - nodeCnt[i].first - nodeCnt[i].second - 1;
        leftNodes = nodeCnt[i].first;
        rightNodes = nodeCnt[i].second == 0 ? 1 : nodeCnt[i].second;
        multiply[static_cast<long long>(leftNodes) * rightNodes * parentNodes]++;
    }
    return multiply.begin()->second;
}


// LC2036
long long maximumAlternatingSubarraySum(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<vector<long long>> dp(n, vector<long long>(2, 0)); // dp[i][0] 以i结尾且nums[i] 为正
    long long ans;

    dp[0][0] = nums[0];
    dp[0][1] = -0x3f3f3f3f;
    ans = nums[0];

    for (i = 1; i < n; i++) {
        dp[i][0] = dp[i - 1][1] > 0 ? dp[i - 1][1] + nums[i] : nums[i];
        dp[i][1] = dp[i - 1][0] - nums[i];
        ans = max(ans, dp[i][0]);
        ans = max(ans, dp[i][1]);
    }
    return ans;
}


// LC1218
int longestSubsequence(vector<int>& arr, int difference)
{
    int i;
    int n = arr.size();
    unordered_map<int, int> dp; // dp[arr[i]] - 以arr[i]结尾的最大长度
    int ans ;
    unordered_map<int, vector<int>> arrCnt;
   
    dp[arr[0]] = 1;
    for (i = 1; i < n; i++) {
        if (dp.count(arr[i] - difference) == 1) {
            dp[arr[i]] = dp[arr[i] - difference] + 1;
        } else {
            dp[arr[i]] = 1;
        }
    }
    ans = 0;
    for (auto it : dp) {
        ans = max(ans, it.second);
    }
    return ans;
}


// LC1073
vector<int> addNegabinary(vector<int>& arr1, vector<int>& arr2)
{
    int i;
    int idx;
    int m = arr1.size();
    int n = arr2.size();
    bool f = false;
    vector<int> t(1002, 0);

    reverse(arr1.begin(), arr1.end());
    reverse(arr2.begin(), arr2.end());
    idx = 0;
    while (idx < m && idx < n) {
        t[idx] = t[idx] + arr1[idx] + arr2[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        }
        idx++;
    }
    while (idx < m) {
        t[idx] = t[idx] + arr1[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        }
        idx++;
    }
    while (idx < n) {
        t[idx] = t[idx] + arr2[idx];
        if (t[idx] == 2) {
            t[idx] = 0;
            t[idx + 1] = -1;
        } else if (t[idx] == -1) {
            t[idx] = 1;
            t[idx + 1] = 1;
        } else if (t[idx] == 3) {
            t[idx] = 1;
            t[idx + 1] = -1;
        }
        idx++;
    }
    if (t[idx] == -1) {
         t[idx] = 1;
        t[idx + 1] = 1;
    }
    vector<int> ans;
    for (i = t.size() - 1; i >= 0; i--) {
        if (f == false && t[i] == 1) {
            f = true;
        }
        if (f) {
            ans.emplace_back(t[i]);
        }
    }
    if (f == false) {
        return {0};
    }
    return ans;
}


// LC1049
int lastStoneWeightII(vector<int>& stones)
{
    int i, j;
    int sum;
    int n = stones.size();

    sum = 0;
    for (auto s : stones) {
        sum += s;
    }
    vector<vector<int>> dp(n + 1, vector<int>(sum / 2 + 1, 0)); // dp[i][j] 前i个石子之和不超过j的最大石子之和
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= sum / 2; j++) {
            if (stones[i - 1] > j) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - stones[i - 1]] + stones[i  - 1]);
            }
        }
    }
    return sum - dp[n][sum / 2] * 2;
}


// LC2434
string robotWithString(string s)
{
    int i;
    int n = s.size();
    string ans;
    char t;
    stack<char> st;
    vector<char> f(n);  // f[i] 表示从i  -  n - 1最小字符

    f[n - 1] = s[n  - 1];
    for (i = n - 2; i >= 0; i--) {
        f[i] = min(f[i + 1], s[i]);
    }
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(s[i]);
            continue;
        }
        t = st.top();
        while (t <= f[i]) {
            ans += t;
            st.pop();
            if (st.empty()) {
                break;
            }
            t = st.top();
        }
        st.push(s[i]);
    }
    while (!st.empty()) {
        ans += st.top();
        st.pop();
    }
    return ans;
}


// LC1377
void DFSFindNode(unordered_map<int, unordered_set<int>>& e, int cur, int from, int target, 
    int cnt, int& step, bool& find, bool& IsLeaf, vector<int>& r, vector<int>& route)
{
    if (find) {
        return;
    }
    if (cur == target) {
        step = cnt;
        find = true;
        if (e[cur].size() == 1 && *e[cur].begin() == from) {
            IsLeaf = true;
        }
        // for (auto it : route) cout << it << " ";
        route = r;
        return;
    }
    for (auto it : e[cur]) {
        if (it != from) {
            r.emplace_back(it);
            DFSFindNode(e, it, cur, target, cnt + 1, step, find, IsLeaf, r, route);
            r.pop_back();
        }
    }
}
double frogPosition(int n, vector<vector<int>>& edges, int t, int target)
{
    if (n == 1) {
        return 1.0;
    }

    int i;
    int ans;
    int step, ways;
    bool IsLeaf, find;
    vector<int> r, route;
    unordered_map<int, unordered_set<int>> e;
    for (auto edge : edges) {
        e[edge[0]].emplace(edge[1]);
        e[edge[1]].emplace(edge[0]);
    }
    step = 0;
    ways = 1;
    IsLeaf = false;
    find = false;
    r.emplace_back(1);
    DFSFindNode(e, 1, -1, target, 0, step, find, IsLeaf, r, route);
    if (step > t) {
        return 0.0;
    }
    if (!IsLeaf && step < t) {
        return 0.0;
    }
    ways *= e[1].size();
    for (i = 1; i < route.size(); i++) {
        ways *= e[route[i]].size() - 1;
    }
    return 1.0 / ways;
}


// LC640
string solveEquation(string equation) // 垃圾题
{
    int i, k;
    int n;
    int idx, m, p;
    int constNum, xNum;
    bool findEqual;
    char sign;
    string t;

    if (equation[0] == '-') {
        equation = "0" + equation;
    }
    idx = equation.find('=');
    if (equation[idx + 1] == '-') {
        equation = equation.substr(0, idx + 1) + "0" + equation.substr(idx + 1);
    }
    idx = 0;
    constNum = xNum = 0;
    findEqual = false;
    sign = '+';
    n = equation.size();
    for (i = 0; i < n; i++) {
        if (equation[i] == '+' || equation[i] == '-') {
            t = equation.substr(idx, i - idx);
            m = t.size();
            if (t[m - 1] == 'x') {
                p = atoi(equation.substr(idx, i - idx).c_str());
                p = p == 0 ? 1 : p;
                if (t[0] == '0') {
                    p = 0;
                }
                for (k = 0; k < m - 1; k++) {
                    if (equation[k] != '0') {
                        break;
                    }
                }
                if (k == m) {

                }
                if (sign == '-') {
                    if (findEqual) {
                        xNum += p;
                    } else {
                        xNum -= p;
                    }
                } else {
                    if (findEqual) {
                        xNum -= p;
                    } else {
                        xNum += p;
                    }
                }
            } else {
                p = atoi(equation.substr(idx, i - idx).c_str());
                if (sign == '-') {
                    if (findEqual) {
                        constNum += p;
                    } else {
                        constNum -= p;
                    }
                } else {
                    if (findEqual) {
                        constNum -= p;
                    } else {
                        constNum += p;
                    }
                }
            }
            sign = equation[i];
            idx = i + 1;
        } else if (equation[i] == '=') {
            findEqual = true;
            t = equation.substr(idx, i - idx);
            m = t.size();
            if (t[m - 1] == 'x') {
                p = atoi(equation.substr(idx, i - idx).c_str());
                p = p == 0 ? 1 : p;
                if (t[0] == '0') {
                    p = 0;
                }
                if (sign == '-') {
                    xNum -= p;
                } else {
                    xNum += p;
                }
            } else {
                p = atoi(equation.substr(idx, i - idx).c_str());
                if (sign == '-') {
                    constNum -= p;
                } else {
                    constNum += p;
                }
            }
            // sign = equation[i + 1] == '-' ? '-' : '+';
            sign = '+';
            idx = i + 1;
        }
    }
    
    t = equation.substr(idx, n - idx);
    m = t.size();
    if (t[m - 1] == 'x') {
        p = atoi(equation.substr(idx, i - idx).c_str());
        p = p == 0 ? 1 : p;
        if (t[0] == '0') {
            p = 0;
        }
        if (sign == '-') {
            xNum += p;
        } else {
            xNum -= p;
        }
    } else {
        p = atoi(equation.substr(idx, i - idx).c_str());
        if (sign == '-') {
            constNum += p;
        } else {
            constNum -= p;
        }
    }

    if (xNum == 0) {
        if (constNum != 0) {
            return "No solution";
        } else {
            return "Infinite solutions";
        }
    }

    int x = constNum * - 1 / xNum;
    string ans = "x=" + to_string(x);
    // cout << ans << endl;
    return ans;
}


// LC2052
int minimumCost(string sentence, int k)
{
    vector<string> words = MySplit(sentence, ' ');
    int i, j;
    int t, ans;
    int n = words.size();
    vector<int> dp(n + 1, 0); // dp[i] - 以第i个单词作为行尾的最小cost
    vector<int> wordLen;

    for (auto w : words) {
        wordLen.emplace_back(w.size());
    }
    dp[1] = (k - wordLen[0]) * (k - wordLen[0]);
    for (i = 2; i <= n; i++) {
        dp[i] = (k - wordLen[i - 1]) * (k - wordLen[i - 1]) + dp[i - 1];
        t = wordLen[i - 1];
        for (j = i - 1; j >= 1; j--) {
            t += 1 + wordLen[j - 1];
            if (t <= k) {
                dp[i] = min(dp[i], (k - t) * (k - t) + dp[j - 1]);
            }
        }
    }
    for (auto d : dp) cout << d << " "; cout << endl;
    t = wordLen[n - 1];
    ans = dp[n - 1];
    for (i = n - 2; i >= 0; i--) {
        t += 1 + wordLen[i];
        if (t <= k) {
            ans = min(ans, dp[i]);
        } else {
            break;
        }
    }
    return ans;
}



// 面试题 04.09. 二叉搜索树序列
vector<vector<int>> BSTSequences(TreeNode* root)
{
    if (root == nullptr) {
        return {{}};
    }

    int i, j;
    int n;
    int nodeNum = TreeNodesNum(root);
    vector<vector<int>> ans;
    vector<int> serial(nodeNum);
    pair<vector<TreeNode *>, unordered_set<TreeNode *>> t; // 已加入点 - 访问点
    queue<pair<vector<TreeNode *>, unordered_set<TreeNode *>>> q;

    q.push({{root}, {root}});
    while (q.size()) {
        n = q.size();
        for (i = 0; i < n; i++) {
            t = q.front();
            q.pop();
            auto v = t.first;
            auto s = t.second;
            if (v.size() == nodeNum) {
                for (j = 0; j < v.size(); j++) {
                    serial[j] = v[j]->val;
                }
                ans.emplace_back(serial);
                continue;
            }
            for (j = 0; j < v.size(); j++) {
                if (v[j]->left != nullptr && s.count(v[j]->left) == 0) {
                    v.emplace_back(v[j]->left);
                    s.emplace(v[j]->left);
                    q.push({v, s});
                    v.pop_back();
                    s.erase(v[j]->left);
                }
                if (v[j]->right != nullptr && s.count(v[j]->right) == 0) {
                    v.emplace_back(v[j]->right);
                    s.emplace(v[j]->right);
                    q.push({v, s});
                    v.pop_back();
                    s.erase(v[j]->right);
                }
            }
        }
    }
    return ans;
}


// LC135
int candy(vector<int>& ratings)
{
    int i;
    int n = ratings.size();
    int ans;

    if (n == 1) {
        return 1;
    }
    vector<int> t1(n, 1), t2(n, 1);
    for (i = 1; i < n; i++) {
        if (ratings[i] > ratings[i - 1]) {
            t1[i] = t1[i - 1] + 1;
        }
    }
    // for (auto n : t1) cout << n << " "; cout << endl;

    for (i = n - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            t2[i] = t2[i + 1] + 1;
        }
    }
    // for (auto n : t2) cout << n << " "; cout << endl;

    ans = 0;
    for (i = 0; i < n; i++) {
        ans += max(t1[i], t2[i]);
    }
    return ans;
}


// LC722
vector<string> removeComments(vector<string>& source)
{
    int i, j, k;
    int n = source.size();
    int pos1, pos2, pos3;
    int startIdx;
    vector<string> t = source;
    vector<string> vs;
    // 先处理块注释
    for (i = 0; i < n; i++) {
        pos1 = t[i].find("/*");
        if (pos1 != string::npos) {
            pos2 = t[i].find("//");
            if (pos2 != string::npos) {
                if (pos2 < pos1) {
                    t[i] = t[i].substr(0, pos2);
                    continue;
                }
            }
            startIdx = 0;
            for (j = i; j < n; j++) {
                pos3 = t[j].find("*/", startIdx);
                if (pos3 != string::npos) {
                    if (j == i) {
                        if (pos3 < pos1 + 2) {
                            startIdx = pos1 + 2;
                            j--;
                            continue;
                        }
                        if (pos1 + 1 != pos3) {
                            t[i] = t[i].substr(0, pos1) + t[i].substr(pos3 + 2);
                            i--;
                            break;
                        }
                    } else {
                        t[i] = t[i].substr(0, pos1);
                        t[j] = t[j].substr(pos3 + 2);
                        t[i] += t[j];
                        for (k = i + 1; k <= j; k++) {
                            t[k] = "";
                        }
                        i = j - 1;
                        break;
                    }
                } else {
                    startIdx = 0;
                }
            }
        }
    }
    for (auto line : t) {
        if (line.size() > 0) {
            vs.emplace_back(line);
        }
    }
    for (auto v : vs) cout << v << endl;

    t = vs;
    // 处理行注释
    n = t.size();
    for (i = 0; i < n; i++) {
        pos2 = t[i].find("//");
        if (pos2 != string::npos) {
            t[i] = t[i].substr(0, pos2);
        }
    }
    vs.clear();
    for (auto line : t) {
        if (line.size() > 0) {
            vs.emplace_back(line);
        }
    }
    return vs;
}


// LC1240
void SetRectangle(vector<vector<bool>>& visited, int row, int col, int size, bool val)
{
    int i, j;
    for (i = row; i < row + size; i++) {
        for (j = col; j < col + size; j++) {
            visited[i][j] = val;
        }
    }
}
bool CanTile(vector<vector<bool>>& visited, int row, int col, int size)
{
    int i, j, k;
    int n = visited.size();
    int m = visited[0].size();

    if (row + size > n || col + size > m) {
        return false;
    }
    for (i = row; i < row + size; i++) {
        for (j = col; j < col + size; j++) {
            if (visited[i][j]) {
                return false;
            }
        }
    }
    return true;
}
void Tiling(vector<vector<bool>>& visited, int row, int col, int edgeSize, int cnt, int cntAera, int& ans)
{
    int k;
    int n = visited.size();
    int m = visited[0].size();

    if (cntAera == m * n) {
        ans = min(ans, cnt);
        return;
    }
    if (cnt >= ans || row == n) {
        return;
    }
    if (col == m) {
        Tiling(visited, row + 1, 0, k, cnt, cntAera, ans);
        return;
    }
    if (visited[row][col]) {
        Tiling(visited, row, col + 1, k, cnt, cntAera, ans);
        return;
    }
    for (k = edgeSize; k >= 1; k--) {
        if (CanTile(visited, row, col, k)) {
            SetRectangle(visited, row, col, k, true);
            Tiling(visited, row, col + k, edgeSize, cnt + 1, cntAera + k * k, ans);
            SetRectangle(visited, row, col, k, false);
        }
    }
}
int tilingRectangle(int n, int m)
{
    int small = min(n, m);

    if (small == 1) {
        return n * m;
    }

    int ans = INT_MAX;
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    Tiling(visited, 0, 0, small, 0, 0, ans);
    return ans;
}


// LC132
int minCut(string s)
{
    // dp[i] 表示i结尾的字符串分割为回文子串的最小切割次数
    // dp[0] = 0; 
    int i, j;
    int n = s.size();
    vector<vector<bool>> IsPalindrome(n, vector<bool>(n, false)); // IsPalindrome[i][j] 是否是回文串
    vector<int> dp(n, 0x3f3f3f3f);

    for (j = 1; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                IsPalindrome[i][j] = true;
                continue;
            }
            if (s[i] == s[j]) {
                if (i + 1 == j) {
                    IsPalindrome[i][j] = true;
                } else {
                    IsPalindrome[i][j] = IsPalindrome[i + 1][j - 1];
                }
            } else {
                IsPalindrome[i][j] = false;
            }
        }
    }
    dp[0] = 0;
    for (j = 1; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (IsPalindrome[i][j]) {
                if (i != 0) {
                    dp[j] = min(dp[j], dp[i - 1] + 1);
                } else {
                    dp[j] = 0;
                }
            }
        }
    }
    return dp[n - 1];
}


// LC2731
int sumDistance(vector<int>& nums, string s, int d)
{
    int i;
    int n = s.size();
    int mod = 1000000007;
    vector<long long> prefixSum(n, 0);

    for (i = 0; i < n; i++) {
        nums[i] = static_cast<long long>(nums[i]) + (s[i] == 'R' ? d : -d);
    }
    sort (nums.begin(), nums.end());

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = nums[i] + prefixSum[i - 1];
    }
    long long ans = 0;
    for (i = 0; i < n - 1; i++) {
        ans = (ans + ((prefixSum[n - 1] - prefixSum[i] - static_cast<long long>(n - i - 1) * nums[i] + mod) % mod)) % mod;
    }
    return ans;
}


// LC2735
long long minCost(vector<int>& nums, int x)
{
    int i, k;
    int n = nums.size();
    int t;
    vector<int> idx;
    unordered_map<int, int> val;
    long long sum, s1;
    long long ans = LONG_MAX;

    sum = 0;
    // sum 不旋转的cost
    for (i = 0; i < n; i++) {
        idx.emplace_back(i);
        val[i] = nums[i];
        sum += nums[i];
    }
    ans = min(sum, ans);
    k = 0;
    s1 = 0;
    while (k < n) {
        s1 += x;
        // 旋转idx
        t = idx[0];
        for (i = 1; i < n; i++) {
            idx[i - 1] = idx[i];
        }
        idx[n - 1] = t;

        sum = 0;
        for (i = 0; i < n; i++) {
            if (val[idx[i]] > nums[i]) {
                val[idx[i]] = nums[i];
            }
            sum += val[idx[i]];
        }
        ans = min(ans, sum + s1);
        k++;
    }
    return ans;
}


// LC2737
int minimumDistance(int n, vector<vector<int>>& edges, int s, vector<int>& marked)
{
    int ans;
    vector<int> dist(n, 0x3f3f3f3f);
    unordered_map<int, set<pair<int, int>>> e;
    queue<pair<int, int>> q;

    for (auto edge : edges) {
        e[edge[0]].insert({edge[1], edge[2]});
    }
    q.push({s, 0});
    while (q.size()) {
        auto p = q.front();
        q.pop();
        if (dist[p.first] < p.second) {
            continue;
        }
        dist[p.first] = p.second;
        for (auto edge : e[p.first]) {
            if (edge.second + dist[p.first] < dist[edge.first]) {
                dist[edge.first] = edge.second + dist[p.first];
                q.push({edge.first, dist[edge.first]});
            }
        }
    }
    ans = 0x3f3f3f3f;
    for (auto m : marked) {
        ans = min(ans, dist[m]);
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC588
vector<string> FileSystem::ls(string path)
{
    int i, j;
    int n;
    int m;
    vector<string> ans;
    Trie<string> *node = FileSystem::root;

    if (path == "/") {
        for (i = 0; i < node->children.size(); i++) {
            ans.emplace_back(node->children[i]->val);
        }
        sort (ans.begin(), ans.end());
        return ans;
    }
    vector<string> dirs = MySplit(path, '/');
    n = dirs.size();
    if (FileSystem::fileContent.count(path) == 1) {
        return {dirs[n - 1]};
    }
    for (i = 1; i < n; i++) { // 去掉第一个'/'
        m = node->children.size();
        for (j = 0; j < m; j++) {
            if (node->children[j]->val == dirs[i]) {
                node = node->children[j];
                break;
            }
        }
    }
    for (i = 0; i < node->children.size(); i++) {
        ans.emplace_back(node->children[i]->val);
    }
    sort (ans.begin(), ans.end());
    return ans;
}

void FileSystem::mkdir(string path)
{
    int i, j;
    Trie<string> *node = FileSystem::root;
    vector<string> dirs = MySplit(path, '/');
    int n = dirs.size();
    int m;

    for (i = 1; i < n; i++) { // 去掉第一个'/'
        m = node->children.size();
        for (j = 0; j < m; j++) {
            if (node->children[j]->val == dirs[i]) {
                node = node->children[j];
                break;
            }
        }
        if (j == m) {
            Trie<string> *dir = new Trie<string>(dirs[i]);
            node->children.emplace_back(dir);
            node = dir;
        }
    }
}

void FileSystem::addContentToFile(string filePath, string content)
{
    if (FileSystem::fileContent.count(filePath) == 1) {
        FileSystem::fileContent[filePath] += content;
        return;
    }
    FileSystem::mkdir(filePath);
    FileSystem::fileContent[filePath] = content;
}

string FileSystem::readContentFromFile(string filePath)
{
    return FileSystem::fileContent[filePath];
}


// LC1039
// dp[i][j] - 从顶点i到顶点j构成的多边形最小乘积之和
// dp[i][i + 1] = 0, 所求dp[i][n - 1]
int minScoreTriangulation(vector<int>& values)
{
    int i, j, k;
    int n = values.size();
    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));

    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            if (j == i + 1) {
                dp[i][j] = 0;
            } else if (j == i + 2) {
                dp[i][j] = values[i] * values[i + 1] * values[j];
            } else {
                for (k = i + 1; k < j; k++) {
                    if (k == i + 1) {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[k][j]);
                    } else if (k == j - 1) {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[i][k]);
                    } else {
                        dp[i][j] = min(dp[i][j], values[i] * values[j] * values[k] + dp[i][k] + dp[k][j]);
                    }
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC2762
long long continuousSubarrays(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left;
    int small, big;
    long long ans = 0;
    multiset<int> ms;

    left = 0;
    for (i = 0; i < n; i++) {
        ms.insert(nums[i]);
        small = *ms.begin();
        big = *ms.rbegin();

        while (big - small > 2) {
            ms.erase(ms.find(nums[left])); // C++17新特性可用ms.extract(nums[left]);
            left++;
            small = *ms.begin();
            big = *ms.rbegin();
        }
        ans += i - left + 1;
    }
    return ans;
}


// LC42
int trap(vector<int>& height)
{
    int i;
    int ans;
    int startIdx;
    int n = height.size();
    stack<int> st;

    startIdx = 0;
    for (i = 0; i < n; i++) {
        if (height[i] != 0) {
            startIdx = i;
            break;
        }
    }

    ans = 0;
    for (i = startIdx; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        while (height[i] >= height[st.top()]) {
            if (height[i] == height[st.top()]) {
                st.pop();
                if (st.empty()) {
                    break;
                }
                continue;
            }

            auto t = st.top();
            st.pop();
            if (st.empty()) {
                break;
            }
            auto left = st.top(); // 左边界
            ans += (min(height[left], height[i]) - height[t]) * (i - left - 1);
        }
        st.push(i);
    }
    return ans;
}


// LC2812
int maximumSafenessFactor(vector<vector<int>>& grid)
{
    int i, j;
    int k;
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> safety(m, vector<int>(n, -1));
    if (grid[0][0] == 1 || grid[m - 1][n - 1] == 1) {
        return 0;
    }
    
    unordered_set<pair<int, int>, MyHash<int, int, int>> thieves;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                thieves.emplace(make_pair(i, j));
            }
        }
    }
    queue<pair<int, int>> q;
    for (auto t : thieves) {
        q.push(t);
    }
    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int nr, nc;
    int curSafety = 0;
    while (q.size()) {
        k = q.size();
        for (i = 0; i < k; i++) {
            auto p = q.front();
            q.pop();
            if (p.first == 5 && p.second == 0) {
                    auto t = 12;
            }
            if (safety[p.first][p.second] == - 1 || safety[p.first][p.second] > curSafety) {
                safety[p.first][p.second] = curSafety;
            } else {
                continue;
            }

            for (j = 0; j < 4; j++) {
                nr = p.first + directions[j][0];
                nc = p.second + directions[j][1];
                if (nr == 5 && nc == 0) {
                    auto t = 12;
                }
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || thieves.count({nr, nc}) == 1 || 
                    (safety[nr][nc] != -1 && safety[nr][nc] <= curSafety)) {
                    continue;
                }
                q.push({nr, nc});
            }
        }
        curSafety++;
    }
    /* for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            cout << safety[{i, j}] << " ";
        }
        cout << endl;
    } */

    bool canAccess;
    int left, right, mid;

    left = 0;
    right = m + n;
    vector<vector<int>> visited(m, vector<int>(n, -1));
    while (left <= right) {
        mid = (right - left) / 2 + left;
        canAccess = false;
        q.push({0, 0});
        while (q.size()) {
            k = q.size();
            for (i = 0; i < k; i++) {
                auto p = q.front();
                q.pop();

                if (safety[p.first][p.second] < mid || visited[p.first][p.second] == mid) {
                    continue;
                }
                visited[p.first][p.second] = mid;
                if (p == make_pair(m - 1, n - 1)) {
                    canAccess = true;
                    ClearQueue(q);
                    break;
                }
                for (j = 0; j < 4; j++) {
                    nr = p.first + directions[j][0];
                    nc = p.second + directions[j][1];

                    if (nr < 0 || nr >= m || nc < 0 || nc >= n || thieves.count({nr, nc}) == 1
                        || visited[nr][nc] == mid) {
                        continue;
                    }
                    q.push({nr, nc});
                }
            }
        }
        if (canAccess) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right < 0 ? 0 : right;
}


// LC2431
int maxTastiness(vector<int>& price, vector<int>& tastiness, int maxAmount, int maxCoupons)
{
    int i, j, k;
    int n = price.size();
    int ans;
    // dp[i][j][k] 0 - i水果用j金额使用k次优惠券得到的最大甜度
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(maxAmount + 1, vector<int>(maxCoupons + 1, 0)));

    // 买0号水果两种情况
    for (j = price[0]; j <= maxAmount; j++) {
        dp[0][j][0] = tastiness[0];
    }
    for (k = 1; k <= maxCoupons; k++) {
        for (j = price[0] / 2; j <= maxAmount; j++) {
            dp[0][j][k] = tastiness[0];
        }
    }
    for (i = 1; i < n; i++) {
        for (j = 0; j <= maxAmount; j++) {
            for (k = 0; k <= maxCoupons; k++) {
                dp[i][j][k] = dp[i - 1][j][k]; // 不买
                if (j - price[i] >= 0) { // 不用优惠券
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - price[i]][k] + tastiness[i]);
                }
                if (j - price[i] / 2 >= 0 && k > 0) { // 使用优惠券
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - price[i] / 2][k - 1] + tastiness[i]);
                }
            }
        }
    }
    return dp[n - 1][maxAmount][maxCoupons];
}


class Probability {
public:
    double a = 0;
    double b = 0;
    unordered_map<int, int> leftNum, rightNum;
    double f(double t)
    {
        double ans = 1.0;
        while (t > 0) {
            ans *= t;
            t--;
        }
        return ans;
    }
    void DFS(vector<int>& balls, int idx, int leftCnt, int leftLen, int rightCnt, int rightLen, double tol)
    {
        int i;
        double l, r;
        if (idx == balls.size()) {
            l = tol;
            for (auto it : leftNum) {
                l /= f(it.second);
            }
            r = tol;
            for (auto it : rightNum) {
                r /= f(it.second);
            }
            if (leftNum.size() == rightNum.size()) {
                a += l * r;
            }
            b += l * r;

            // cout << left << " " << right << " " << t << endl; 
            // cout << l << " " << r << endl;
            return;
        }
        for (i = 0; i <= balls[idx]; i++) {
            if (leftCnt + i <= leftLen && rightCnt + balls[idx] - i <= rightLen) {
                if (i) {
                    leftNum[idx] += i;
                }
                if (balls[idx] - i) {
                    rightNum[idx] += balls[idx] - i;
                }
                DFS(balls, idx + 1, leftCnt + i, leftLen, rightCnt + balls[idx] - i, rightLen, tol);
                leftNum[idx] == i ? leftNum.erase(idx) : leftNum[idx] -= i;
                rightNum[idx] == balls[idx] - i ? rightNum.erase(idx) : rightNum[idx] -= (balls[idx] - i);
            }
        }
    }
    double getProbability(vector<int>& balls)
    {
        int i;
        int sum;

        sum  = 0;
        for (i = 0; i < balls.size(); i++) {
            sum += balls[i];
        }

        int len = sum / 2;
        DFS(balls, 0, 0, len, 0, len, f(len));
        cout << a << " " << b << endl;
        return a / b;
    }
};

// LC2861
int maxNumberOfAlloys(int n, int k, int budget, vector<vector<int>>& composition, vector<int>& stock, vector<int>& cost)
{
    int i, j;
    int left, right, mid;
    int ans = 0;
    long long val;
    for (i = 0; i < k; i++) {
        left = 0;
        right = 2 * 10e8;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            val = 0;
            for (j = 0; j < n; j++) {
                if (composition[i][j] * static_cast<long long>(mid) > stock[j]) {
                    val += static_cast<long long>(cost[j]) * (composition[i][j] * static_cast<long long>(mid) - stock[j]);
                }
            }
            if (val > budget) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        ans = max(ans, right);
    }
    return ans;
}


// LC2866
void Erase(map<int, int, greater<>>& m, int t)
{
    for (auto it = m.begin(); it != m.end();) {
        if (it->first > t) {
            m.erase(it++);
        } else {
            break;
        }
    }
}
long long maximumSumOfHeights(vector<int>& maxHeights)
{
    int i;
    int curMin, curMax;
    int n = maxHeights.size();
    vector<long long> prefix(n, 0), suffix(n, 0);
    map<int, int, greater<>> m;
    prefix[0] = maxHeights[0];
    curMin = curMax = maxHeights[0];
    m[maxHeights[0]] = 0;
    for (i = 1; i < n; i++) {
        if (maxHeights[i] >= curMax) {
            prefix[i] = prefix[i - 1] + maxHeights[i];
            m[maxHeights[i]] = i;
            if (m.count(maxHeights[i]) == 0) {
                m[maxHeights[i]] = i;
            }
            curMax = maxHeights[i];
        } else {
            if (maxHeights[i] <= curMin) {
                prefix[i] = static_cast<long long>(i + 1) * maxHeights[i];
                curMin = maxHeights[i];
                curMax = maxHeights[i];
                m.clear();
            } else {
                auto it = m.lower_bound(maxHeights[i]);
                curMax = maxHeights[i];
                prefix[i] = prefix[it->second] + static_cast<long long>(i - (it->second + 1) + 1) * maxHeights[i];
                Erase(m, curMax);
            }
        }
        m[maxHeights[i]] = i;
    }
    m.clear();
    suffix[n - 1] = maxHeights[n - 1];
    m[maxHeights[n - 1]] = n - 1;
    curMin = curMax = maxHeights[n - 1];
    for (i = n - 2; i >= 0; i--) {
        if (maxHeights[i] >= curMax) {
            suffix[i] = suffix[i + 1] + maxHeights[i];
            if (m.count(maxHeights[i]) == 0) {
                m[maxHeights[i]] = i;
            }
            curMax = maxHeights[i];
        } else {
            if (maxHeights[i] <= curMin) {
                suffix[i] = static_cast<long long>(n - i) * maxHeights[i];
                curMin = maxHeights[i];
                curMax = maxHeights[i];
                m.clear();
            } else {
                auto it = m.lower_bound(maxHeights[i]);
                curMax = maxHeights[i];
                suffix[i] = suffix[it->second] + static_cast<long long>(it->second - i) * maxHeights[i];
                Erase(m, curMax);
            }
        }
        m[maxHeights[i]] = i;
    }
    long long ans = 0;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            ans = max(ans, suffix[i]);
        } else if (i == n - 1) {
            ans = max(ans, prefix[i]);
        } else {
            ans = max(ans, prefix[i] + suffix[i] - maxHeights[i]);
        }
    }
    // for (auto a : prefix) cout << a << " "; cout << endl;
    // for (auto a : suffix) cout << a << " "; cout << endl;
    return ans;
}


// LC8
int myAtoi(string s)
{
    // 前导空格
    auto lambda1 = [](string &s) {
        int cnt = 0;
        for (auto ch : s) {
            if (ch != ' ') {
                break;
            } else {
                cnt++;
            }
        }
        s = s.substr(cnt);
    };
    function<void(string &)> Trim = lambda1;
    Trim(s);

    // 前导0
    auto lambda2 = [](string &s) {
        int cnt = 0;
        for (auto ch : s) {
            if (ch != '0') {
                break;
            } else {
                cnt++;
            }
        }
        s = s.substr(cnt);
        if (cnt == 0) {
            return false;
        }
        return true;
    };
    function<bool(string &)> Trim0 = lambda2;
    bool operTrim0 = Trim0(s);

    if (s.empty() || s[0] == ' ' || isalpha(s[0])) {
        return 0;
    }

    bool isNegetive = false;
    int startIdx = 0;
    int i;
    if (s[0] == '-') {
        if (operTrim0) {
            return 0;
        }
        isNegetive = true;
        startIdx++;
    } else if (s[0] == '+') {
        if (operTrim0) {
            return 0;
        }
        startIdx++;
    }
    string t;
    for (i = startIdx; i < s.size(); i++) {
        if (isdigit(s[i])) {
            t += s[i];
        } else {
            break;
        }
    }
    if (t.empty()) {
        return 0;
    }
    long long ans = 0;
    for (i = 0; i < t.size(); i++) {
        ans = t[i] - '0' + ans * 10;
        if (isNegetive && ans * -1 < INT_MIN) {
            return INT_MIN;
        }
        if (!isNegetive && ans > INT_MAX) {
            return INT_MAX;
        }
    }
    if (isNegetive) {
        ans *= -1;
    }
    return ans;
}


// LC156
TreeNode* upsideDownBinaryTree(TreeNode* root)
{
    if (root == nullptr) {
        return nullptr;
    }
    stack<pair<TreeNode *, TreeNode *>> st;
    // 递归lambda捕获列表要加上自身引用
    function<void(TreeNode *node, TreeNode *parent)> f = [&st, &f](TreeNode *node, TreeNode *parent) {
        if (node == nullptr) {
            return;
        }
        st.push({node, parent});
        f(node->left, node);
    };

    f(root, nullptr);

    bool flag = true;
    TreeNode *ans = nullptr;
    while (!st.empty()) {
        auto p = st.top();
        st.pop();
        
        if (flag) {
            flag = false;
            ans = p.first;
        }
        p.first->right = p.second;
        if (p.second != nullptr) {
            if (p.second->right != nullptr) {      
                p.first->left = p.second->right;
            }
            p.second->left = nullptr;
            p.second->right = nullptr;
        }
    }
    return ans;
}


// LC174
// dp[i][j] - 从{i, j}走到终点的最小健康点数
int calculateMinimumHP(vector<vector<int>>& dungeon)
{
    int i, j;
    auto m = dungeon.size();
    auto n = dungeon[0].size();
    vector<vector<int>> dp(m, vector<int>(n));

    dp[m - 1][n - 1] = dungeon[m - 1][n - 1] <= 0 ? abs(dungeon[m - 1][n - 1]) + 1 : 1;
    for (j = n - 2; j >= 0; j--) {
        dp[m - 1][j] = max(1, dp[m - 1][j + 1] - dungeon[m - 1][j]);
    }
    for (i = m - 2; i >= 0; i--) {
        dp[i][n - 1] = max(1, dp[i + 1][n - 1] - dungeon[i][n - 1]);
    }
    for (i = m - 2; i >= 0; i--) {
        for (j = n - 2; j >= 0; j--) {
            dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]);
        }
    }
    return dp[0][0];
}


// LC410
// 动态规划
int splitArray(vector<int>& nums, int k)
{
    int i, j, p;
    int n = nums.size();
    // dp[i][k] - 以第i位结束分为k段的最大子数组和的最小值
    vector<vector<int>> dp(n, vector<int>(k + 1, INT_MAX));
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    dp[0][1] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
        dp[i][1] = prefixSum[i];
    }
    for (i = 1; i < n; i++) {
        for (j = 2; j <= k; j++) {
            if (j > i + 1) {
                break;
            }
            for (p = 0; p < i; p++) {
                if (j - 1 > p + 1) {
                    continue;
                }
                dp[i][j] = min(dp[i][j], max(dp[p][j - 1], prefixSum[i] - prefixSum[p]));
            }
        }
    }
    return dp[n - 1][k];
}

// 二分
int splitArray_1(vector<int>& nums, int k)
{
    int i;
    int left, right, mid;
    int cnt, idx;
    int n = nums.size();
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    left = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
        left = max(left, nums[i]);
    }
    right = prefixSum[n - 1];
    while (left <= right) {
        mid = (right - left) / 2 + left;
        cnt = 0;
        idx = 0;
        for (i = 0; i < n; i++) {
            if (cnt == 0) {
                if (prefixSum[i] > mid) {
                    i--;
                    idx = i;
                    cnt++;
                }
            } else {
                if (prefixSum[i] - prefixSum[idx] > mid) {
                    i--;
                    idx = i;
                    cnt++;
                }
            }
        }
        if (cnt >= k) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1094
bool carPooling(vector<vector<int>>& trips, int capacity)
{
    int i, k;
    int n = trips.size();

    sort (trips.begin(), trips.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] != b[1]) {
            return a[1] < b[1];
        }
        return a[2] < b[2]; });
    for (k = 0; k <= 1000; k++) {
        for (i = 0; i < n; i++) {
            if (trips[i][2] == k) {
                capacity += trips[i][0];
            }
            if (trips[i][1] == k) {
                capacity -= trips[i][0];
                if (capacity < 0) {
                    return false;
                }
            }
        }
    }
    return true;
}
// 差分数组
bool carPooling_1(vector<vector<int>>& trips, int capacity)
{
    int i;
    int n = trips.size();
    vector<int> diff(1001, 0);
    // diff[n] = a[n] - a[n - 1];
    for (i = 0; i < n; i++) {
        diff[trips[i][1]] += trips[i][0];
        diff[trips[i][2]] -= trips[i][0];
    }
    vector<int> num(1000, 0);
    num[0] = diff[0];
    if (num[0] > capacity) {
        return false;
    }
    for (i = 1; i < 1000; i++) {
        num[i] = diff[i] + num[i - 1];
        if (num[i] > capacity) {
            return false;
        }
    }
    return true;
}


// LC1488
vector<int> avoidFlood(vector<int>& rains)
{
    int i;
    int n = rains.size();
    unordered_map<int, int> lake; // 第n个湖泊下雨的日期
    set<int> st; // 可抽水的日期
    vector<int> ans(n, 1);
    for (i = 0; i < n; i++) {
        if (rains[i] == 0) {
            st.emplace(i);
        } else {
            ans[i] = -1;
            if (lake.count(rains[i]) == 0) {
                lake[rains[i]] = i;
            } else {
                auto it = st.lower_bound(lake[rains[i]]);
                if (it == st.end()) {
                    ans.clear();
                    return ans;
                }
                ans[*it] = rains[i];
                st.erase(*it);
                lake[rains[i]] = i;
            }
        }
    }
    return ans;
}


// LC188
// dp[i][0 - 1][1 - k] 第i天持有股票已交易k次的最大价值
int maxProfit(int k, vector<int>& prices)
{
    int i, p;
    int n = prices.size();
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(2, vector<int>(k + 1, -0x3f3f3f3f)));

    dp[0][1][1] = -prices[0];
    dp[0][0][0] = 0;
    for (i = 1; i < n; i++) {
        dp[i][0][0] = 0;
        for (p = 1; p <= k; p++) {
            dp[i][0][p] = max(dp[i - 1][0][p], dp[i - 1][1][p] + prices[i]);
            dp[i][1][p] = max(dp[i - 1][1][p], dp[i - 1][0][p - 1] - prices[i]);
        }
    }
    int ans = 0;
    for (i = 0; i < n; i++) {
        for (p = 1; p <= k; p++) {
            // printf ("dp[%d][0][%d] = %d\n", i, p, dp[i][0][p]);
            // printf ("dp[%d][1][%d] = %d\n", i, p, dp[i][1][p]);
            ans = max(ans, dp[i][0][p]);
        }
    }
    return ans;
}


double MySqrt(double num)
{
    if (num < 0) {
        return INFINITY;
    }
    double left = 0.0;
    double right = num;
    if (num < 1) {
        right = 1.0;
    }
    while (right - left > 1e-10) { // 使用更小的阈值
        double mid = left + (right - left) / 2;
        double t = mid * mid;
        if (t > num) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return right;
}


// LC517
int findMinMoves(vector<int>& machines)
{
    int i;
    int n = machines.size();
    int sum = 0;

    for (auto m : machines) {
        sum += m;
    }
    if (sum % n != 0) {
        return -1;
    }
    int ans = 0;
    int avg = sum / n;
    vector<int> need = machines;
    for (i = 0; i < n; i++) {
        need[i] -= avg;
        ans = max(ans, need[i]);
    }
    for (i = 0; i < n; i++) {
        ans = max(ans, abs(need[i]));
        if (i < n - 1) {
            if (need[i] <= 0) {
                need[i + 1] -= abs(need[i]);
            } else {
                need[i + 1] += need[i];
            }
        }
    }
    return ans;
}


// LC115
int numDistinct(string s, string t)
{
    int i, j;
    int m = s.size();
    int n = t.size();
    int mod = 1000000007;
    if (m < n) {
        return 0;
    }
    // dp[i][j] - s的前i位包含t前j位的子序列个数
    vector<vector<long long>> dp(m + 1, vector<long long>(n + 1, 0));
    for (i = 0; i < m; i++) {
        if (s[i] == t[0]) {
            dp[i + 1][1] = dp[i][1] + 1;
        } else {
            dp[i + 1][1] = dp[i][1];
        }
    }
    for (i = 0; i < m; i++) {
        for (j = 1; j < n; j++) {
            if (i < j) {
                break;
            }
            if (s[i] == t[j]) {
                dp[i + 1][j + 1] = (dp[i][j + 1] + dp[i][j]) % mod;
            } else {
                dp[i + 1][j + 1] = dp[i][j + 1];
            }
        }
    }
    return dp[m][n];
}


// LC1363
string largestMultipleOfThree(vector<int>& digits)
{
    int sum = 0;
    string ans;
    vector<int> remainder1;
    vector<int> remainder2;
    map<int, int, greater<>> data;
    for (auto d : digits) {
        sum += d;
        data[d]++;
        if (d % 3 == 1) {
            remainder1.emplace_back(d);
        } else if (d % 3 == 2) {
            remainder2.emplace_back(d);
        }
    }
    sort (remainder1.begin(), remainder1.end());
    sort (remainder2.begin(), remainder2.end());

    auto func = [&data]() {
        int i;
        string ans;
        for (auto it : data) {
            for (i = 0; i < it.second; i++) {
                ans += it.first + '0';
            }
        }
        return ans;
    };

    if (sum % 3 == 0) {
        ans = func();
    } else if (sum % 3 == 1) { // 去掉一个余1或两个余2
        if (remainder1.size() > 0) {
            auto num = remainder1[0];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }
            ans = func();
        } else if (remainder2.size() > 1) {
            auto num = remainder2[0];
            auto num1 = remainder2[1];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }

            if (data[num1] == 1) {
                data.erase(num1);
            } else {
                data[num1]--;
            }
            ans = func();
        }
    } else { // 去掉一个余2或两个余1
        if (remainder2.size() > 0) {
            auto num = remainder2[0];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }
            ans = func();
        } else if (remainder1.size() > 1) {
            auto num = remainder1[0];
            auto num1 = remainder1[1];
            if (data[num] == 1) {
                data.erase(num);
            } else {
                data[num]--;
            }

            if (data[num1] == 1) {
                data.erase(num1);
            } else {
                data[num1]--;
            }
            ans = func();
        }
    }
    if (ans[0] == '0') {
        return "0";
    }
    return ans;
}


// LC1458
int maxDotProduct(vector<int>& nums1, vector<int>& nums2)
{
    // dp[i][j] - 以nums1前i位 nums2前j位 最大子序列点积
    int i, j;
    int m = nums1.size();
    int n = nums2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, -0x3f3f3f3f));

    dp[0][0] = 0;
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            // 4种情况
            dp[i][j] = max({dp[i][j], nums1[i - 1] * nums2[j - 1] + dp[i - 1][j - 1],
                        nums1[i - 1] * nums2[j - 1],
                        dp[i - 1][j],
                        dp[i][j - 1]});
        }
    }
    return dp[m][n];
}


// LC1298
int maxCandies(vector<int>& status, vector<int>& candies, vector<vector<int>>& keys, 
    vector<vector<int>>& containedBoxes, vector<int>& initialBoxes)
{
    int i, k;
    int box;
    int ans = 0;
    unordered_set<int> cantOpen;
    queue<int> q;

    for (auto init : initialBoxes) {
        q.push(init);
    }
    while (q.size()) {
        int size = q.size();
        for (i = 0; i < size; i++) {
            box = q.front();
            q.pop();
            if (status[box] == 1) {
                ans += candies[box];
            } else {
                q.push(box);
                if (cantOpen.count(box) == 1) {
                    goto maxCandiesEND;
                }
                cantOpen.emplace(box);
                continue;
            }
            if (keys[box].size() > 0) {
                for (k = 0; k < keys[box].size(); k++) {
                    status[keys[box][k]] = 1;
                }
            }
            if (containedBoxes[box].size() > 0) {
                for (k = 0; k < containedBoxes[box].size(); k++) {
                    q.push(containedBoxes[box][k]);
                }
            }
        }
    }
maxCandiesEND:
    return ans;
}


// LC1269
int numWays(int steps, int arrLen)
{
    // dp[i][x] - 第i步停留在下标x的方案数, 所求dp[steps][0];
    int i, j;
    int mod = 1000000007;
    vector<vector<long long>> dp(steps + 1, vector<long long>(steps + 2, 0));
    dp[1][0] = 1; // 不动
    if (arrLen > 1) {
        dp[1][1] = 1; // 向右走一步
    }
    for (i = 2; i <= steps; i++) {
        for (j = 0; j < arrLen; j++) {
            if (i < j) {
                break;
            }
            if (j == 0) {
                dp[i][j] = (dp[i - 1][j] + dp[i - 1][j + 1]) % mod;
            } else if (j > 0 && j < arrLen - 1) {
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] + dp[i - 1][j + 1]) % mod;
            } else {
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod;
            }
        }
    }
    return dp[steps][0];
}


// LC2332
int latestTimeCatchTheBus(vector<int>& buses, vector<int>& passengers, int capacity)
{
    int i;
    int left, right, mid;
    int k;
    int bus = buses.size();
    int n = passengers.size();
    unordered_set<int> arriveTime;

    sort(buses.begin(), buses.end());
    sort(passengers.begin(), passengers.end());

    for (auto passenger : passengers) {
        arriveTime.emplace(passenger);
    }

    left = passengers[0];
    right = buses[bus - 1];
    while (left <= right) {
        mid = (right - left) / 2 + left;
        auto busCnt = 0;
        k = 0;
        for (i = 0; i < n; i++) {
            if (passengers[i] <= mid) {
                if (passengers[i] <= buses[busCnt]) {
                    k++;
                    if (k == capacity) {
                        busCnt++;
                        k = 0;
                        if (busCnt == bus) {
                            right = mid - 1;
                            break;
                        }
                    }
                } else {
                    busCnt++;
                    if (busCnt == bus) {
                        right = mid - 1;
                        break;
                    }
                    k = 0;
                    i--;
                    continue;
                }
            } else {
                left = mid + 1;
            }
        }
        if (busCnt < bus) {
            left = mid + 1;
        }
    }
    int ans = right;
    while (arriveTime.count(ans) == 1) {
        ans--;
    }
    return ans;
}


// LC939
int minAreaRect(vector<vector<int>>& points)
{
    int i, j;
    int size = points.size();
    int area = INT_MAX;
    int r = 40000;
    unordered_set<int> dots;

    for (auto p : points) {
        dots.emplace(p[0] * r + p[1]);
    }
    for (i = 0; i < size - 1; i++) {
        for (j = i + 1; j < size; j++) {
            if ((points[i][0] != points[j][0] && points[i][1] != points[j][1]) &&
                (dots.count(points[i][0] * r + points[j][1]) && dots.count(points[j][0] * r + points[i][1]))) {
                area = min(area, abs(points[i][0] - points[j][0]) * abs(points[i][1] - points[j][1]));
            }
        }
    }
    return area == INT_MAX ? 0 : area;
}


// LC593
bool validSquare(vector<int>& p1, vector<int>& p2, vector<int>& p3, vector<int>& p4)
{
    vector<vector<int>> points;
    points.emplace_back(p1);
    points.emplace_back(p2);
    points.emplace_back(p3);
    points.emplace_back(p4);
    double k1, k2;

    vector<vector<int>> tries = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    for (auto tr : tries) {
        auto len1 = (points[tr[0]][0] - points[tr[1]][0]) * (points[tr[0]][0] - points[tr[1]][0]) +
                    (points[tr[0]][1] - points[tr[1]][1]) * (points[tr[0]][1] - points[tr[1]][1]);
        auto len2 = (points[tr[2]][0] - points[tr[3]][0]) * (points[tr[2]][0] - points[tr[3]][0]) +
                    (points[tr[2]][1] - points[tr[3]][1]) * (points[tr[2]][1] - points[tr[3]][1]);
        // 对角线相等
        if (len1 != len2) {
            continue;
        }
        // 中心点一致
        if ((points[tr[0]][0] + points[tr[1]][0] != points[tr[2]][0] + points[tr[3]][0]) || 
            (points[tr[0]][1] + points[tr[1]][1] != points[tr[2]][1] + points[tr[3]][1])) {
            continue;
        }
        // 对角线垂直
        if (points[tr[0]][0] == points[tr[1]][0]) {
            k1 = INT_MAX;
        } else {
            k1 = (points[tr[0]][1] - points[tr[1]][1]) * 1.0 / (points[tr[0]][0] - points[tr[1]][0]);
        }
        if (points[tr[2]][0] == points[tr[3]][0]) {
            k2 = INT_MAX;
        } else {
            k2 = (points[tr[2]][1] - points[tr[3]][1]) * 1.0 / (points[tr[2]][0] - points[tr[3]][0]);
        }
        if ((k1 == 0 && k2 == INT_MAX) || (k1 == INT_MAX && k2 == 0) || (k1 * k2 + 1 < 10e-7)) {
            return true;
        }
    }
    return false;
}


// LC1712
int waysToSplit(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int mod = 1000000007;
    int left, right, mid;
    int idx1, idx2;
    int leftSum, midSum, rightSum;
    long long ans;
    vector<int> prefixSum(n, 0);

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = nums[i] + prefixSum[i - 1];
    }
    ans = 0;
    for (i = 0; i < n - 2; i++) {
        // 找到中间子数组大于左边子数组最小结束下标
        leftSum = prefixSum[i];
        left = i + 1;
        right = n - 2;
        // 所求为left
        while (left <= right)
        {
            mid = (right - left) / 2 + left;
            midSum = prefixSum[mid] - leftSum;
            if (midSum >= leftSum) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        midSum = prefixSum[left] - leftSum;
        rightSum = prefixSum[n - 1] - midSum - leftSum;
        if (rightSum < midSum) {
            continue; // 不能用break
        }
        idx1 = left;

        // 找到中间子数组大于左边子数组最大结束下标
        // leftSum = prefixSum[i];
        left = idx1;
        right = n - 2;
        // 所求为right
        while (left <= right)
        {
            mid = (right - left) / 2 + left;
            midSum = prefixSum[mid] - leftSum;
            rightSum = prefixSum[n - 1] - midSum - leftSum;
            if (midSum <= rightSum) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        idx2 = right;
        ans = (ans + idx2 - idx1 + 1) % mod;
        printf ("i = %d, idx1 = %d, idx2 = %d\n", i, idx1, idx2);
        printf ("%d %d %d\n", prefixSum[i], prefixSum[idx1] - prefixSum[i], prefixSum[n - 1] - prefixSum[idx1]);
    }
    return ans;
}


// LC2398
int maximumRobots(vector<int>& chargeTimes, vector<int>& runningCosts, long long budget)
{
    int i;
    int n = chargeTimes.size();
    int left, right, mid;
    int curTopCharge;
    long long curSum, curCost;
    priority_queue<pair<int, int>, vector<pair<int, int>>> pq;

    left = 1;
    right = n;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curSum = 0;
        for (i = 0; i < mid; i++) {
            pq.push({chargeTimes[i], i});
            curSum += runningCosts[i];
        }
        while (1) {
            auto t = pq.top();
            if (t.second >= mid) {
                pq.pop();
            } else {
                curTopCharge = t.first;
                break;
            }
        }
        curCost = curTopCharge + curSum * mid;
        if (curCost <= budget) {
            left = mid + 1;
            continue;
        }
        for (i = mid; i < n; i++) {
            curSum = curSum - runningCosts[i - mid] + runningCosts[i];
            pq.push({chargeTimes[i], i});
            while (1) {
                auto t = pq.top();
                if (t.second <= i - mid || t.second > i) {
                    pq.pop();
                } else {
                    curTopCharge = t.first;
                    break;
                }
            }
            curCost = curTopCharge + curSum * mid;
            if (curCost <= budget) {
                break;
            }
        }
        if (i == n) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return right;
}


// LC1424
vector<int> findDiagonalOrder(vector<vector<int>>& nums)
{
    int i, j;
    vector<int> ans;
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first + a.second != b.first + b.second) {
            return a.first + a.second > b.first + b.second;
        }
        return a.first < b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);

    for (i = 0; i < nums.size(); i++) {
        for (j = 0; j < nums[i].size(); j++) {
            pq.push({i, j});
        }
    }
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        ans.emplace_back(nums[p.first][p.second]);
    }
    return ans;
}


// LC1191
int kConcatenationMaxSum(vector<int>& arr, int k)
{
    int mod = 1000000007;
    int i;
    int n = arr.size();
    long long curMax, suffixMax;
    vector<long long> prefix(n , 0), suffix(n, 0);

    prefix[0] = arr[0];
    for (i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + arr[i];
    }
    suffix[n - 1] = arr[n - 1];
    suffixMax = arr[n - 1];
    for (i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] + arr[i];
        suffixMax = max(suffixMax, suffix[i]);
    }
    vector<long long> dp(n, 0);
    dp[0] = arr[0];
    curMax = arr[0];
    for (i = 1; i < n; i++) {
        dp[i] = dp[i - 1] > 0 ? dp[i - 1] + arr[i] : arr[i];
        curMax = max(curMax, dp[i]);
    }
    if (k == 1) {
        return curMax < 0 ? 0 : curMax % mod;
    }
    auto t = LONG_LONG_MIN;
    for (i = 0; i < n; i++) {
        if (prefix[n - 1] < 0) {
            t = max({prefix[i] + suffixMax, dp[i], t});
        } else {
            t = max({prefix[i] + suffixMax + (k - 2) * prefix[n - 1], dp[i], t});
        }
    }
    return t < 0 ? 0 : t % mod;
}


// LC1510
bool winnerSquareGame(int n)
{
    // dp[n][0] - 剩余n个石子,alice行动是否必胜
    int i, k;
    vector<vector<bool>> dp(n + 1, vector<bool>(2, false));

    dp[0][0] = false;
    dp[0][1] = true;
    for (i = 1; i <= n; i++) {
        for (k = 1; k <= 320; k++) {
            if (i - k * k >= 0) {
                dp[i][0] = dp[i][0] | dp[i - k * k][1];
                if (dp[i][0]) {
                    break;
                }
            } else {
                break;
            }
        }
        dp[i][1] = !dp[i][0];
    }
    return dp[n][0];
}


// LC1312
int minInsertions(string s)
{
    int i, j;
    int n = s.size();
    // dp[i][j] - 下标i到j的子字符串变成回文串的最小插入字符数
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 0;
            } else if (i + 1 == j) {
                dp[i][j] = (s[i] == s[j] ? 0 : 1);
            } else {
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC1102
int maximumMinimumPath(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    int left, right, mid;

    right = left = grid[0][0];
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            left = min(left, grid[i][j]);
            right = max(right, grid[i][j]);
        }
    }
    bool access = false;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        access = false;
        auto t = grid;
        function<void (vector<vector<int>>& grid, int row, int col, bool& access)> DFS = 
            [&mid, &DFS](vector<vector<int>>& grid, int row, int col, bool& access) {
            int i;
            int m = grid.size();
            int n = grid[0].size();
            int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
            if (access) {
                return;
            }
            if (grid[row][col] < mid) {
                return;
            }
            if (row == m - 1 && col == n - 1) {
                access = true;
                return;
            }
            grid[row][col] = -1;
            for (i = 0; i < 4; i++) {
                auto nr = row + directions[i][0];
                auto nc = col + directions[i][1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] == -1 || grid[nr][nc] < mid) {
                    continue;
                }
                DFS(grid, nr, nc, access);
            }
        };
        DFS(t, 0, 0, access);
        if (access) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC1416
int numberOfArrays(string s, int k)
{
    int i, j;
    int n = s.size();
    int mod = 1000000007;
    long long t;
    vector<long long> dp(n + 1, 0);

    if (s[0] - '0' > k) {
        return 0;
    }

    dp[0] = 1;
    dp[1] = 1;
    for (i = 2; i <= n; i++) {
        if (s[i - 1] != '0') {
            dp[i] = dp[i - 1];
        }
        j = i - 2;
        while (j >= 0) {
            if (i - j > 10) {
                break;
            }
            if (s[j] != '0') {
                t = atol(s.substr(j, i - j).c_str());
                if (t > k) {
                    break;
                }
                dp[i] = (dp[i] + dp[j]) % mod;
            }
            j--;
        }
    }
    return dp[n];
}


// LC1301
// BFS超时
vector<int> pathsWithMaxScore_1(vector<string>& board)
{
    int i, j;
    int n = board.size();
    int size;
    int mod = 1000000007;
    int directions[3][2] = {{-1, 0}, {0, -1}, {-1, -1}};
    int points;
    long long ways;
    vector<vector<vector<int>>> boardData(n, vector<vector<int>>(n, vector<int>(2, 0)));
    queue<tuple<int, int, int>> q;

    board[0][0] = '0';
    board[n - 1][n - 1] = '0';
    q.push({n * n - 1, 0, 1});
    while (q.size()) {
        size = q.size();
        for (i = 0; i < size; i++) {
            auto tp = q.front();
            q.pop();

            auto x = get<0>(tp) / n;
            auto y = get<0>(tp) % n;
            points = get<1>(tp);
            ways = get<2>(tp);
            if (points > boardData[x][y][0]) {
                boardData[x][y][0] = points;
                boardData[x][y][1] = ways;
            } else if (points == boardData[x][y][0]) {
                boardData[x][y][1] = (boardData[x][y][1] + ways) % mod;
            } else {
                continue;
            }
            for (j = 0; j < 3; j++) {
                auto nx = x + directions[j][0];
                auto ny = y + directions[j][1];

                if (nx < 0 || ny < 0 || board[nx][ny] == 'X') {
                    continue;
                }
                q.push({nx * n + ny, board[nx][ny] - '0' + points, ways});
            }
        }
    }
    return boardData[0][0];
}

// dp
vector<int> pathsWithMaxScore(vector<string>& board)
{
    int i, j;
    int n = board.size();
    int mod = 1000000007;
    vector<vector<vector<int>>> boardData(n, vector<vector<int>>(n, vector<int>(2, 0)));
    
    // 判断连通性
    bool access = false;
    auto b = board;
    function<void (vector<string>& board, int pos, bool& access)> CanAccess = [&CanAccess](vector<string>& board, int pos, bool& access) {
        if (access) {
            return;
        }
        int i;
        int n = board.size();
        int directions[3][2] = {{-1, 0}, {0, -1}, {-1, -1}};
        int row = pos / n;
        int col = pos % n;

        if (row == 0 && col == 0) {
            access = true;
            return;
        }
        board[row][col] = 'X';
        for (i = 0; i < 3; i++) {
            auto nr = row + directions[i][0];
            auto nc = col + directions[i][1];
            if (nr < 0 || nc < 0 || board[nr][nc] == 'X') {
                continue;
            }
            CanAccess(board, nr * n + nc, access);
        }
    };
    CanAccess(b, n * n - 1, access);
    if (!access) {
        return {0, 0};
    }

    // 预处理
    board[0][0] = '0';
    boardData[n - 1][n - 1] = {0, 1};
    for (i = n - 2; i >= 0; i--) {
        if (board[i][n - 1] != 'X') {
            boardData[i][n - 1] = {board[i][n - 1] - '0' + boardData[i + 1][n - 1][0], 1};
        } else {
            break;
        }
    }
    for (j = n - 2; j >= 0; j--) {
        if (board[n - 1][j] != 'X') {
            boardData[n - 1][j] = {board[n - 1][j] - '0' + boardData[n - 1][j + 1][0], 1};
        } else {
            break;
        }
    }
    int maxPoint;
    vector<vector<int>> waysData;
    for (i = n - 2; i >= 0; i--) {
        for (j = n - 2; j >= 0; j--) {
            if (board[i][j] == 'X') {
                continue;
            }
            waysData.clear();
            if (board[i + 1][j] != 'X' && boardData[i + 1][j][1] > 0) {
                waysData.emplace_back(boardData[i + 1][j]);
            }
            if (board[i][j + 1] != 'X' && boardData[i][j + 1][1] > 0) {
                waysData.emplace_back(boardData[i][j + 1]);
            }
            if (board[i + 1][j + 1] != 'X' && boardData[i + 1][j + 1][1] > 0) {
                waysData.emplace_back(boardData[i + 1][j + 1]);
            }
            // 类似
            // 5X
            // XX
            if (waysData.empty()) {
                continue;
            }
            boardData[i][j][0] = board[i][j] - '0';
            maxPoint = 0;
            for (auto w : waysData) {
                maxPoint = max(maxPoint, w[0]);
            }
            boardData[i][j][0] += maxPoint;
            for (auto w : waysData) {
                if (w[0] == maxPoint) {
                    boardData[i][j][1] = (static_cast<long long>(boardData[i][j][1]) + w[1]) % mod;
                }
            }
        }
    }
    return boardData[0][0];
}


// LC1326
int minTaps(int n, vector<int>& ranges)
{
    int i, j;
    int left, right, prevLeft, prevRight;
    int cnt;
    vector<vector<int>> r;
    for (i = 0; i <= n; i++) {
        if (ranges[i] == 0) {
            continue;
        }
        r.push_back({max(0, i - ranges[i]), min(n, i + ranges[i])});
    }
    if (r.empty()) {
        return -1;
    }
    sort(r.begin(), r.end());

    if (r[0][0] != 0) {
        return -1;
    }
    cnt = 1;
    if (r[0][1] == n) {
        return cnt;
    }
    left = r[0][0];
    right = r[0][1];
    prevRight = -1;
    for (i = 1; i < r.size(); i++) {
        if (r[i][0] == left) {
            right = r[i][1];
        } else if (r[i][0] > right) {
            return -1;
        } else if (r[i][1] > right) {
            if (r[i][0] > prevRight) {
                prevRight = right;
                prevLeft = left;
                right = r[i][1];
                left = r[i][0];
                cnt++;
            } else {
                right = r[i][1];
            }
        }
        if (r[i][1] == n) {
            break;
        }
    }
    return right < n ? - 1 : cnt;
}

// LC1964
// 求最长不降子序列, 二分法
vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles)
{
    int i;
    int n = obstacles.size();
    int m;
    int left, right, mid;
    vector<int> low;
    vector<int> ans(n, 0);
    low.emplace_back(obstacles[0]);
    ans[0] = 1;
    for (i = 1; i < n; i++) {
        m = low.size();
        if (obstacles[i] >= low[m - 1]) {
            low.emplace_back(obstacles[i]);
            ans[i] = low.size();
            continue;
        }
        left = 0;
        right = m - 1;
        while (left <= right) { // 所求为left
            mid = ((right - left) >> 1) + left;
            if (low[mid] <= obstacles[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        low[left] = obstacles[i];
        ans[i] = left + 1;
    }
    return ans;
}


// LC907
// 样例 [71,55,82,55] - 593
int sumSubarrayMins(vector<int>& arr)
{
    // 分别找某一个数左右第一个小于它的值, 两个单调递增栈
    int i, idx;
    int n = arr.size();
    unordered_map<int, pair<int, int>> idxBoundary; // 下标 - {左长度, 右长度}
    stack<int> stRight, stLeft;

    // 右边
    for (i = 0; i < n; i++) {
        if (stRight.empty()) {
            stRight.push(i);
            continue;
        }
        idx = stRight.top();
        if (arr[idx] < arr[i]) {
            stRight.push(i);
        } else {
            while (arr[idx] >= arr[i]) {
                idxBoundary[idx].second = i - idx;
                stRight.pop();
                if (stRight.empty()) {
                    break;
                }
                idx = stRight.top();
            }
            stRight.push(i);
        }
    }
    while (!stRight.empty()) {
        idx = stRight.top();
        idxBoundary[idx].second = n - idx;
        stRight.pop();
    }

    // 左边
    for (i = n - 1; i >= 0; i--) {
        if (stLeft.empty()) {
            stLeft.push(i);
            continue;
        }
        idx = stLeft.top();
        if (arr[i] >= arr[idx]) { // 注意此处判断关系与上面不同,是为了避免有相同元素重复记录子数组
            stLeft.push(i);
        } else {
            while (arr[i] < arr[idx]) {  // 这里也是
                idxBoundary[idx].first = idx - i;
                stLeft.pop();
                if (stLeft.empty()) {
                    break;
                }
                idx = stLeft.top();
            }
            stLeft.push(i);
        }
    }
    while (!stLeft.empty()) {
        idx = stLeft.top();
        idxBoundary[idx].first = idx + 1;
        stLeft.pop();
    }
    long long ans = 0;
    for (auto it : idxBoundary) {
        ans += static_cast<long long>(it.second.first) * it.second.second * arr[it.first];
    }
    return ans % 1000000007;
}


// LC755
vector<int> pourWater(vector<int>& heights, int volume, int k)
{
    int i;
    int width = heights.size();
    int curHeight, curIdx;
    while (volume) {
        curHeight = heights[k] + 1;
        curIdx = k;
        for (i = k - 1; i >= 0; i--) {
            if (heights[i] + 1 < curHeight) {
                curHeight = heights[i] + 1;
                curIdx = i;
            } else if (heights[i] >= curHeight) {
                break;
            }
        }
        if (curHeight == heights[k] + 1) {
            curIdx = k;
        }
        if (curIdx == k) {
            for (i = k + 1; i < width; i++) {
                if (heights[i] + 1 < curHeight) {
                    curHeight = heights[i] + 1;
                    curIdx = i;
                } else if (heights[i] >= curHeight) {
                    break;
                }
            }
        }
        heights[curIdx] = curHeight;
        volume--;
    }
    return heights;
}


// LC828
int uniqueLetterString(string s)
{
    int i;
    int n = s.size();
    vector<vector<int>> alphaIdx(26, vector<int>(2, -1)); // 上一个字符出现位置
    vector<int> f(n, 0); // f[i] - 以s[i]结尾的子字符串的唯一字符的个数
    vector<int> dp(n, 0); // dp[i] - 到s[i]为止所有子字符串唯一字符的个数

    f[0] = 1;
    dp[0] = 1;
    alphaIdx[s[0] - 'A'][0] = 0;
    for (i = 1; i < n; i++) {
        if (alphaIdx[s[i] - 'A'][0] == -1) {
            f[i] = f[i - 1] + i + 1;
            alphaIdx[s[i] - 'A'][0] = i;
        } else {
            if (alphaIdx[s[i] - 'A'][1] == -1) {
                f[i] = f[i - 1] + i + 1 - 2 * (alphaIdx[s[i] - 'A'][0] + 1);
                alphaIdx[s[i] - 'A'][1] = alphaIdx[s[i] - 'A'][0];
                alphaIdx[s[i] - 'A'][0] = i;
            } else {
                auto len1 = alphaIdx[s[i] - 'A'][0] + 1;
                auto len2 = alphaIdx[s[i] - 'A'][1] + 1;
                f[i] = f[i - 1] + i + 1 - 2 * (len1 - len2) - len2;
                alphaIdx[s[i] - 'A'][1] = alphaIdx[s[i] - 'A'][0];
                alphaIdx[s[i] - 'A'][0] = i;
            }
        }
        dp[i] = dp[i - 1] + f[i];
    }
    return dp[n - 1];
}


// LC1411
int numOfWays(int n)
{
    int i;
    int mod = 1000000007;
    // 对于每一行, 可以有12种方式, 其中2色6种, 3色6种;
    // 2色可衍生出5种配对, 3色2种, 2色3种; 3色可衍生4种, 各2种
    // dp[i][0] - 第i层以双色结尾的方案数
    vector<vector<long long>> dp(n, vector<long long>(2, 0));

    dp[0][0] = 6;
    dp[0][1] = 6;

    for (i = 1; i < n; i++) {
        dp[i][0] = (dp[i - 1][0] * 3 + dp[i - 1][1] * 2) % mod;
        dp[i][1] = (dp[i - 1][0] * 2 + dp[i - 1][1] * 2) % mod;
    }
    return (dp[n - 1][0] + dp[n - 1][1]) % mod;
}


// LC790
int numTilings(int n)
{
    int i;
    int mod = 1000000007;
    vector<long long> dp(n + 1, 0);

    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return 2;
    }
    // 长为n的地板可分为n-i和i两部分, 其中i具有唯一性, 比如i=2的情况只能是上下两横向,两竖向则与i=1重复计数了
    // f[n] = f[n - 1] + f[n - 2] + 2 * f[n - 3] + 2 * f[n - 4] + ... + 2 * f[0]
    // f[0] = 1;
    dp[0] = 1;
    dp[1] = 1;
    dp[2] = 2;
    long long prefixSum = dp[0];
    for (i = 3; i <= n; i++) {
        dp[i] = (dp[i - 1] + dp[i - 2] + prefixSum * 2) % mod;
        prefixSum = (prefixSum + dp[i - 2]) % mod;
    }
    return dp[n];
}


// LC1406
string stoneGameIII(vector<int>& stoneValue)
{
    int i;
    int a, b, c;
    int n = stoneValue.size();
    // dp[i] - alice从第i处的石堆开始, 能获得的最大和
    vector<int> dp(n, INT_MIN);
    // 后缀和, 方便计算
    vector<int> suffixSum(n, 0);
    suffixSum[n - 1] = stoneValue[n - 1];
    for (i = n - 2; i >= 0; i--) {
        suffixSum[i] = suffixSum[i + 1] + stoneValue[i];
    }

    dp[n - 1] = stoneValue[n - 1];
    for (i = n - 2; i >= 0; i--) {
        a = INT_MIN;
        if (i + 1 < n) {
            a = suffixSum[i] - dp[i + 1];
        }
        b = INT_MIN;
        if (i + 2 < n) {
            b = suffixSum[i] - dp[i + 2];
        }
        c = INT_MIN;
        if (i + 3 < n) {
            c = suffixSum[i] - dp[i + 3];
        }
        dp[i] = max({dp[i], a, b, c});
        if (i + 3 >= n) { // 直接取完
            dp[i] = max(dp[i], suffixSum[i]);
        } 
    }
    int alice = dp[0];
    int bob = suffixSum[0] - alice;

    if (alice > bob) {
        return "Alice";
    } else if (alice < bob) {
        return "Bob";
    }
    return "Tie";
}


// LC1686
int stoneGameVI(vector<int>& aliceValues, vector<int>& bobValues)
{
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first + a.second < b.first + b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);

    int i;
    int n = aliceValues.size();
    for (i = 0; i < n; i++) {
        pq.push({aliceValues[i], bobValues[i]});
    }
    int cnt;
    int alice, bob;

    cnt = alice = bob = 0;
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        if (cnt % 2 == 0) {
            alice += p.first;
        } else {
            bob += p.second;
        }
        cnt++;
    }
    if (alice > bob) {
        return 1;
    } else if (alice < bob) {
        return -1;
    }
    return 0;
}


// LC1690
int stoneGameVII(vector<int>& stones)
{
    int i, j;
    int head, tail;
    int n = stones.size();
    vector<vector<int>> dpMax(n, vector<int>(n, 0)); // dpMax[i][j] alice先手在区间[i,j]得到的最大分差

    vector<int> prefixSum(n, 0);
    prefixSum[0] = stones[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + stones[i];
    }
    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dpMax[i][j] = 0;
                continue;
            } else if (i + 1 == j) {
                dpMax[i][j] = max(stones[i], stones[j]);
                continue;
            }
            if (i == 0) {
                head = prefixSum[j - 1]; // i -> j - 1
            } else {
                head = prefixSum[j - 1] - prefixSum[i - 1];
            }
            tail = prefixSum[j] - prefixSum[i]; // i + 1 -> j
            dpMax[i][j] = max(head - dpMax[i][j - 1], tail - dpMax[i + 1][j]); // 此处减去dpMax为bob的最佳策略
        }
    }
    return dpMax[0][n - 1];
}


// LC1562
// 从后往前二分
int findLatestStep(vector<int>& arr, int m)
{
    int i;
    int n = arr.size();
    set<int> blankInc;
    set<int, greater<>> blankDec;
    if (m == n) {
        return n;
    }
    for (i = 0; i < n; i++) {
        arr[i]--;
    }
    blankInc.emplace(arr[n - 1]);
    blankDec.emplace(arr[n - 1]);
    if (arr[n - 1] == m || n - 1 - arr[n - 1] == m) {
        return n - 1;
    }
    int len;
    for (i = n - 2; i >= 0; i--) {
        // 找最近的左右端点
        // 右端点
        auto it = blankInc.lower_bound(arr[i]);
        if (it == blankInc.end()) {
            len = n - 1 - arr[i];
        } else {
            len = *it - 1 - arr[i];
        }
        if (len == m) {
            return i;
        }
        // 左端点
        it = blankDec.lower_bound(arr[i]);
        if (it == blankDec.end()) {
            len = arr[i];
        } else {
            len = arr[i] - 1 - *it;
        }
        if (len == m) {
            return i;
        }
        blankInc.emplace(arr[i]);
        blankDec.emplace(arr[i]);
    }
    return -1;
}


// LC774
double minmaxGasDist(vector<int>& stations, int k)
{
    int i;
    int n = stations.size();
    long long curPos, cnt, t;
    long long left, right, mid;
    vector<long long> st(n);

    for (i = 0; i < n; i++) {
        st[i] = stations[i] * 1e5;
    }

    left = 1;
    right = 1e8 * 1e5;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curPos = st[0];
        t = k;
        for (i = 1; i < n; i++) {
            if (st[i] - curPos > mid) {
                if ((st[i] - curPos) % mid == 0) {
                    cnt = (st[i] - curPos) / mid - 1;
                } else {
                    cnt = (st[i] - curPos) / mid;
                }
                curPos = st[i];
                t -= cnt;
                if (t < 0) {
                    left = mid + 1;
                    break;
                }
            } else {
                curPos = st[i];
            }
        }
        if (i == n) {
            right = mid - 1;
        }
    }
    return left / 1e5;
}


// LC776
// 如果当前节点 root->val <= target, 那它及左子树都应该被划分到less树, 但root的右子树中也有可能存在小于等于target的节点,
// 而这些节点可以通过 p = splitBST(root->right, target) 返回出来, 而由于是右子树分出来的, 子树p[0]中所有结点的值肯定也比root->val大,
// 因此接在root的右边 root->right = p[0];
// root->val > target时同理, root及其右子树直接划分出来到more, 递归左子树找漏掉的大值节点
vector<TreeNode*> splitBST(TreeNode* root, int target)
{
    if (root == nullptr) {
        return {nullptr, nullptr};
    }
    TreeNode *less, *greater;
    if (root->val <= target) {
        less = root;
        auto p = splitBST(root->right, target);
        root->right = p[0];
        greater = p[1];
    } else {
        greater = root;
        auto p = splitBST(root->left, target);
        root->left = p[1];
        less = p[0];
    }
    return {less, greater};
}


// LC801
int minSwap(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int n = nums1.size();
    int a1, a2, b1, b2;
    vector<vector<int>> dp(n, vector<int>(2, 0x3f3f3f3f));

    // dp[n][0] - 第n位不交换使序列严格递增的最小交换次数
    dp[0][0] = 0;
    a1 = nums1[0];
    b1 = nums2[0];
    dp[0][1] = 1;
    a2 = nums2[0];
    b2 = nums1[0];

    for (i = 1; i < n; i++) {
        if (nums1[i] > a1 && nums2[i] > b1) {
            dp[i][0] = dp[i - 1][0];
        }
        if (nums1[i] > a2 && nums2[i] > b2) {
            dp[i][0] = min(dp[i - 1][1], dp[i][0]);
        }

        if (nums2[i] > a1 && nums1[i] > b1) {
            dp[i][1] = dp[i - 1][0] + 1;
        }
        if (nums2[i] > a2 && nums1[i] > b2) {
            dp[i][1] = min(dp[i - 1][1] + 1, dp[i][1]);
        }
        a1 = nums1[i];
        b1 = nums2[i];
        a2 = nums2[i];
        b2 = nums1[i];
    }
    return min(dp[n - 1][0], dp[n - 1][1]);
}


// LC1621
int numberOfSets(int n, int k)
{
    int mod = 1000000007;
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, 0));
    vector<vector<long long>> sum(n + 1, vector<long long>(k + 1, 0));

    // dp[n][1] = (n - 1) + (n - 2) + ... + 1 = n(n - 1) / 2 (n >= 2)
    // dp[n][k] = dp[n - 1][k - 1] + dp[n - 2][k - 1] + ... + dp[k][k - 1] + (dp[n - 2][k - 1] + ... + dp[k][k - 1]) + ...   (k > 1)
    // 例如
    // dp[5][2] = (dp[4][1] + dp[3][1] + dp[2][1]) + (dp[3][1] + dp[2][1]) + (dp[2][1]) = 15
    // dp[4][2] = (dp[3][1] + dp[2][1]) + (dp[2][1]) = 5
    // dp[5][2] = dp[4][2] + sum(dp[5 - 1][2 - 1], dp[5 - 2][2 - 1], dp[5 - 3][2 - 1])
    // 再例如 dp[6][3] = (dp[5][2] + dp[4][2] + dp[3][2]) + (dp[4][2] + dp[3][2]) + (dp[3][2]) = dp[5][3] +
    // sum(dp[5][2], dp[4][2], dp[3][2])
    int i, j;
    for (i = 2; i <= n; i++) {
        for (j = 1; j <= k; j++) {
            if (j == 1) {
                dp[i][j] = (i - 1) * i / 2 % mod;
                if (i == 2) {
                    sum[i][j] = dp[i][j];
                } else {
                    sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
                }
                continue;
            } else if (i == j + 1) {
                dp[i][j] = 1;
                if (i == 2) {
                    sum[i][j] = dp[i][j];
                } else {
                    sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
                }
                continue;
            }
            if (j >= i) {
                break;
            }
           dp[i][j] = (sum[i - 1][j - 1] + dp[i - 1][j]) % mod;
           sum[i][j] = (sum[i - 1][j] + dp[i][j]) % mod;
        }
    }
    return dp[n][k];
}


// LC2477
long long minimumFuelCost(vector<vector<int>>& roads, int seats)
{
    if (roads.empty()) {
        return 0;
    }
    int i;
    int size = roads.size();
    int parent;
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, int> nodeData; // 节点 - 人数
    for (i = 0; i <= size; i++) {
        nodeData[i] = 1;
    }
    for (auto r : roads) {
        edges[r[0]].emplace(r[1]);
        edges[r[1]].emplace(r[0]);
    }
    queue<pair<int, int>> q;
    for (auto it : edges) {
        // 叶子节点
        if (it.second.size() == 1) {
            q.push({it.first, -1});
        }
    }

    while (q.size()) {
        auto p = q.front();
        q.pop();
        if (p.first == 0) {
            continue;
        }
        parent = *edges[p.first].begin();
        edges[parent].erase(p.first);
        nodeData[parent] += nodeData[p.first];
        if (edges[parent].size() == 1) {
            q.push({parent, p.first});
        }
    }
    long long ans = 0;
    for (auto it : nodeData) {
        // printf ("%d %d\n", it.first, it.second);
        if (it.first == 0 || it.first == -1) {
            continue;
        }
        if (it.second % seats == 0) {
            ans += it.second / seats;
        } else {
            ans += it.second / seats + 1;
        }
    }
    return ans;
}


// LC2955
vector<int> sameEndSubstringCount(string s, vector<vector<int>>& queries)
{
    // unordered_map<char, vector<int>> idx; 超时
    vector<vector<int>> idx(26);
    int i;
    int n = s.size();
    int left, right, mid;
    int leftBound, rightBound, sum;
    vector<int> ans;
    for (i = 0; i < n; i++) {
        idx[s[i] - 'a'].emplace_back(i);
    }

    for (auto q : queries) {
        sum = 0;
        for (auto ch = 'a'; ch <= 'z'; ch++) {
            if (idx[ch - 'a'].size() == 0) {
                continue;
            }
            // 第一个大于等于q[0]的idx[ch], left
            left = 0;
            right = idx[ch - 'a'].size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (idx[ch - 'a'][mid] < q[0]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            if (left >= idx[ch - 'a'].size()) {
                continue;
            }
            leftBound = left;

            // 第一个小于等于q[1]的idx[ch], right
            left = 0;
            right = idx[ch - 'a'].size() - 1;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (idx[ch - 'a'][mid] > q[1]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            if (right < 0) {
                continue;
            }
            rightBound = right;
            auto t = (rightBound - leftBound + 1) * (rightBound - leftBound + 2) / 2;
            sum += t;
        }
        ans.emplace_back(sum);
    }
    return ans;
}


// LC2267
bool hasValidPath(vector<vector<char>>& grid)
{
    int m = grid.size();
    int n = grid[0].size();
    unordered_map<int, unordered_set<int>> route;
    if (grid[0][0] == ')' || grid[m - 1][n - 1] == '(') {
        return false;
    }
    if ((m + n - 1) % 2 == 1) { // 左右括号数量要匹配
        return false;
    }
    function<void (vector<vector<char>>&, int, int, int, int, bool&)> DFS =
    [&DFS, &route](vector<vector<char>>& grid, int row, int col, int cnt1, int cnt2, bool& find) {
        if (find) {
            return;
        }

        int k;
        int m = grid.size();
        int n = grid[0].size();
        int directions[2][2] = {{0, 1}, {1, 0}};

        if (row == m - 1 && col == n - 1 && cnt1 == cnt2) {
            find = true;
            return;
        }
        if (route.count(row * n + col) && route[row * n + col].count(cnt1 - cnt2)) {
            return;
        }
        route[row * n + col].emplace(cnt1 - cnt2);
        auto t = m - 1 - row + n - 1 - col;
        if (cnt1 < cnt2 || cnt1 * 2 > m + n || (cnt1 == cnt2 && t % 2 != 0) || (cnt1 > cnt2 && t < cnt1 - cnt2)) {
            return;
        }

        for (k = 0; k < 2; k++) {
            auto nr = row + directions[k][0];
            auto nc = col + directions[k][1];
            if (nr < m && nc < n) {
                if (grid[nr][nc] == '(') {
                    DFS(grid, nr, nc, cnt1 + 1, cnt2, find);
                } else {
                    DFS(grid, nr, nc, cnt1, cnt2 + 1, find);
                }
            }
        }
    };
    bool find = false;
    DFS(grid, 0, 0, 1, 0, find);
    return find;
}


// LC2763
int sumImbalanceNumbers(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int ans, cnt, curMaxVal, curMinVal;
    map<int, int> dataLess;
    map<int, int, greater<>> dataGreater;

    ans = 0;
    for (i = 0; i < n; i++) {
        dataLess.clear();
        dataGreater.clear();
        cnt = 0;
        for (j = i; j < n; j++) {
            if (j == i) {
                dataLess[nums[j]]++;
                dataGreater[nums[j]]++;
                continue;
            }
            if (dataLess.count(nums[j])) {
                dataLess[nums[j]]++;
                dataGreater[nums[j]]++;
                ans += cnt;
                continue;
            }
            if (nums[j] < dataLess.begin()->first) { // 最小值
                curMinVal = dataLess.begin()->first;
                if (curMinVal - nums[j] > 1) {
                    cnt++;
                }
            } else if (nums[j] > dataGreater.begin()->first) { // 最大值
                curMaxVal = dataGreater.begin()->first;
                if (nums[j] - curMaxVal > 1) {
                    cnt++;
                }
            } else {
                auto rightIter = dataLess.upper_bound(nums[j]);
                auto leftIter = dataGreater.upper_bound(nums[j]);
                if (nums[j] - leftIter->first > 1) {
                    if (rightIter->first - nums[j] > 1) {
                        cnt++;
                    } else {
                        // cnt不变
                    }
                } else {
                    if (rightIter->first - nums[j] > 1) {
                        // cnt不变
                    } else {
                        cnt--;
                    }
                }
            }
            dataLess[nums[j]]++;
            dataGreater[nums[j]]++;
            ans += cnt;
        }
    }
    return ans;
}


// LC2008
long long maxTaxiEarnings(int n, vector<vector<int>>& rides)
{
    int i, j;
    vector<long long> dp(n + 1, 0);

    sort(rides.begin(), rides.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] != b[1]) {
            return a[1] < b[1];
        }
        return a[0] < b[0];
    });
    j = 0;
    for (i = 2; i <= n; i++) {
        if (j == rides.size() || i < rides[j][1]) {
            dp[i] = max(dp[i], dp[i - 1]);
        } else {
            dp[i] = max(dp[i], dp[rides[j][0]] + rides[j][1] - rides[j][0] + rides[j][2]);
            i--;
            j++;
        }
    }
    return dp[n];
}


// LC1671
int minimumMountainRemovals(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    vector<int> dpUp(n, 1), dpDown(n, 1);

    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (nums[i] > nums[j]) {
                dpUp[i] = max(dpUp[i], dpUp[j] + 1);
            }
        }
    }
    for (i = n - 2; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            if (nums[i] > nums[j]) {
                dpDown[i] = max(dpDown[i], dpDown[j] + 1);
            }
        }
    }
    int ans = INT_MAX;
    for (i = 1; i < n - 1; i++) {
        if (dpUp[i] != 1 && dpDown[i] != 1) {
            ans = min(ans, n - (dpUp[i] + dpDown[i] - 1));
        }
    }
    return ans;
}


// LC2959
// 迪杰斯特拉
int numberOfSets(int n, int maxDistance, vector<vector<int>>& roads)
{
    // 一共有 2^n 种情况
    int i, j, k;
    int tmp, idx;
    int ans = 0;
    vector<int> status(n);
    vector<int> dist(n);
    vector<vector<pair<int, int>>> edgeWithWeight(n);
    queue<pair<int, int>> q;
    pair<int, int> t;
    for (k = 0; k < pow(2, n); k++) {
        tmp = k;
        idx = 0;
        while (tmp) {
            status[idx] = tmp % 2;
            tmp /= 2;
            idx++;
        }
        for (i = 0; i < n; i++) {
            edgeWithWeight[i].clear();
        }
        for (auto r : roads) {
            if (status[r[0]] && status[r[1]]) { // 当前情况存在路径
                edgeWithWeight[r[0]].push_back({r[1], r[2]});
                edgeWithWeight[r[1]].push_back({r[0], r[2]});
            }
        }
        for (i = 0; i < n; i++) {
            if (status[i] == 0) {
                continue;
            }
            dist.assign(n, INT_MAX);
            q.push({i, 0});
            while (q.size()) {
                t = q.front();
                q.pop();

                if (dist[t.first] < t.second) {
                    continue;
                }
                dist[t.first] = t.second;
                auto size = edgeWithWeight[t.first].size();
                for (auto e : edgeWithWeight[t.first]) {
                    if (e.second + t.second < dist[e.first]) {
                        dist[e.first] = e.second + t.second;
                        q.push({e.first, dist[e.first]});
                    }
                }
            }
            for (j = 0; j < n; j++) {
                if (j != i && status[j] && dist[j] > maxDistance) {
                    goto numberOfSetsENDLOOP;
                }
            }
        }
        ans++;
numberOfSetsENDLOOP:
        ;
    }
    return ans;
}


// LC1631
int minimumEffortPath(vector<vector<int>>& heights)
{
    int i, j;
    int left, right, mid;
    int m = heights.size();
    int n = heights[0].size();
    vector<vector<bool>> visited(heights.size(), vector<bool>(heights[0].size(), false));
    bool reach = false;

    left = right = 0;
    for (auto height : heights) {
        for (auto h : height) {
            right = max(right, h);
        }
    }
    while (left <= right) {
        mid = (right - left) / 2 + left;
        function<void (int, int, vector<vector<bool>>&, bool&)> DFS = 
            [&DFS, &heights, &mid](int row, int col, vector<vector<bool>>& visited, bool& reach) -> void {

            int k;
            int m = heights.size();
            int n = heights[0].size();
            int direction[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            if (reach) {
                return;
            }
            if (row == m - 1 && col == n - 1) {
                reach = true;
                return;
            }
            visited[row][col] = true;
            for (k = 0; k < 4; k++) {
                auto nr = row + direction[k][0];
                auto nc = col + direction[k][1];

                if (nr < 0 || nr >= m || nc < 0 || nc >= n || 
                    visited[nr][nc] || abs(heights[row][col] - heights[nr][nc]) > mid) {
                    continue;
                }
                DFS(nr, nc, visited, reach);
            }
        };
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                visited[i][j] = false;
            }
        }
        reach = false;
        DFS(0, 0, visited, reach);
        if (reach) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC1818
int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int n = nums1.size();
    long long sum;
    set<int> less;
    set<int, greater<>> greater;

    sum = 0;
    for (i = 0; i < n; i++) {
        sum += abs(nums1[i] - nums2[i]);
        less.emplace(nums1[i]);
        greater.emplace(nums1[i]);
    }
    long long ans = sum;
    for (i = 0; i < n; i++) {
        sum -= abs(nums1[i] - nums2[i]);
        auto it1 = less.lower_bound(nums2[i]);
        if (it1 != less.end()) {
            ans = min(ans, sum + abs(*it1 - nums2[i]));
        }
        auto it2 = greater.lower_bound(nums2[i]);
        if (it2 != greater.end()) {
            ans = min(ans, sum + abs(*it2 - nums2[i]));
        }
        // 还原
        sum += abs(nums1[i] - nums2[i]);
    }
    return ans % 1000000007;
}


// LC2002
int maxProduct(string s)
{
    // 一共有 2^n - 1 种情况
    int i, j, k;
    int tmp, idx;
    int n = s.size();
    vector<int> status(n);
    vector<int> idxes;
    vector<pair<string, vector<int>>> palindromes;
    string t;
    for (k = 1; k < pow(2, n); k++) {
        tmp = k;
        idx = 0;
        while (tmp) {
            status[idx] = tmp % 2;
            tmp /= 2;
            idx++;
        }
        t.clear();
        idxes.clear();
        for (i = 0; i < n; i++) {
            if (status[i]) {
                t += s[i];
                idxes.emplace_back(i);
            }
        }
        if (IsPalindrome(t)) {
            palindromes.push_back({t, idxes});
        }
    }

    int p, q;
    int ans;
    bool findSameIndex = false;

    ans = 0;
    n = palindromes.size();
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            p = 0;
            q = 0;
            findSameIndex = false;
            while (p < palindromes[i].second.size() && q < palindromes[j].second.size()) {
                if (palindromes[i].second[p] < palindromes[j].second[q]) {
                    p++;
                } else if (palindromes[i].second[p] > palindromes[j].second[q]) {
                    q++;
                } else {
                    findSameIndex = true;
                    break;
                }
            }
            if (!findSameIndex) {
                tmp = palindromes[i].first.size() * palindromes[j].first.size();
                ans = max(ans, tmp);
            }
        }
    }
    return ans;
}


// LC1898
int maximumRemovals(string s, string p, vector<int>& removable)
{
    int i, j, k;
    int left, right, mid;
    string t, tt;
    bool f = false;

    left = 0;
    right = removable.size();

    while (left <= right) {
        mid = (right - left) / 2 + left;
        t = s;
        for (k = 0; k < mid; k++) {
            t[removable[k]] = '0';
        }
        tt.clear();
        for (auto ch : t) {
            if (ch != '0') {
                tt += ch;
            }
        }
        f = false;
        i = j = 0;
        while (i < tt.size() && j < p.size()) {
            if (tt[i] == p[j]) {
                i++;
                j++;
                if (j == p.size()) {
                    f = true;
                    break;
                }
            } else {
                i++;
            }
        }
        if (f) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC2454
vector<int> secondGreaterElement(vector<int>& nums)
{
    // 比一个数大的下一个的下一个, 1 -> 3 -> 2, 则是2
    // 单调递减栈
    int i, idx;
    int n = nums.size();
    unordered_map<int, int> nextGreaterIdx; // 下标 - 下一个更大下标
    set<int> s;
    stack<int> stRight;

    for (i = 0; i < n; i++) {
        if (stRight.empty()) {
            stRight.push(i);
            continue;
        }
        idx = stRight.top();
        if (nums[idx] >= nums[i]) {
            stRight.push(i);
        } else {
            while (nums[idx] < nums[i]) {
                if (nextGreaterIdx.count(idx) == 0) {
                    s.emplace(idx);
                    nextGreaterIdx[idx] = -1;
                } else {
                    nextGreaterIdx[idx] = i;
                }
                stRight.pop();
                if (stRight.empty()) {
                    break;
                }
                idx = stRight.top();
            }
            stRight.push(i);
            for (auto it : s) {
                stRight.push(it);
            }
            s.clear();
        }
    }

    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        if (nextGreaterIdx.count(i) && nextGreaterIdx[i] != -1) {
            ans[i] = nums[nextGreaterIdx[i]];
        } else {
            ans[i] = -1;
        }
    }
    return ans;
}


// LC1092
string shortestCommonSupersequence(string str1, string str2)
{
    int i, j;
    int m = str1.size();
    int n = str2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (str1[i] == str2[j]) {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            } else {
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }
    // 求最长子序列
    string lcs;
    function<void (int, int)> print_lcs = [&dp, &str1, &str2, &lcs, &print_lcs](int i, int j) {
        if(i == 0 || j == 0) {
            return;
        }
        //相等则添加
        if (str1[i - 1] == str2[j - 1]) {
            print_lcs(i - 1, j - 1);
            lcs += str1[i - 1];
        } else if(dp[i - 1][j] >= dp[i][j - 1]) {
            print_lcs(i - 1, j);
        } else {
            print_lcs(i, j - 1);
        }
    };
    print_lcs(m, n);

    // cout << "lcs = " << lcs << endl;

    int cur;
    int size = lcs.size();
    string ans;
    vector<string> part1, part2;

    j = 0;
    cur = 0;
    for (i = 0; i < m; i++) {
        if (str1[i] == lcs[j]) {
            part1.emplace_back(str1.substr(cur, i - cur));
            cur = i + 1;
            j++;
            if (j == size) {
                break;
            }
        }
    }
    if (cur != m) {
        part1.emplace_back(str1.substr(cur));
    }
    j = 0;
    cur = 0;
    for (i = 0; i < n; i++) {
        if (str2[i] == lcs[j]) {
            part2.emplace_back(str2.substr(cur, i - cur));
            cur = i + 1;
            j++;
            if (j == size) {
                break;
            }
        }
    }
    if (cur != n) {
        part2.emplace_back(str2.substr(cur));
    }
    for (i = 0; i < size; i++) {
        ans += part1[i] + part2[i] + lcs[i];
    }
    if (i < part1.size()) {
        ans += part1[part1.size() - 1];
    }
    if (i < part2.size()) {
        ans += part2[part2.size() - 1];
    }
    return ans;
}


// LC1722
int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps)
{
    int i, k;
    int n = source.size();
    unordered_map<int, unordered_set<int>> edges;
    unordered_map<int, int> sourceNum;
    unordered_map<int, int> targetNum;
    vector<bool> visited(n, false);
    vector<int> idx;

    for (auto as : allowedSwaps) {
        edges[as[0]].emplace(as[1]);
        edges[as[1]].emplace(as[0]);
    }

    function<void (int)> DFS = [&DFS, &visited, &edges, &idx](int cur) {
        visited[cur] = true;
        idx.emplace_back(cur);
        for (auto it : edges[cur]) {
            if (visited[it] == false) {
                DFS(it);
            }
        }
    };
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (visited[i] == false) {
            idx.clear();
            DFS(i);
            sourceNum.clear();
            targetNum.clear();
            for (auto id : idx) {
                sourceNum[source[id]]++;
                targetNum[target[id]]++;
            }
            for (auto it : sourceNum) {
                for (k = 0; k < it.second; k++) {
                    if (targetNum.count(it.first)) {
                        if (targetNum[it.first] == 1) {
                            targetNum.erase(it.first);
                        } else {
                            targetNum[it.first]--;
                        }
                    } else {
                        ans++;
                    }
                }
            }
        }
    }
    return ans;
}


// LC1937
long long maxPoints(vector<vector<int>>& points)
{
    int i, j;
    int m = points.size();
    int n = points[0].size();
    vector<vector<long long>> dp(m, vector<long long>(n, 0)); // dp[i][j] - 在points[i][j]能得到的最大分数

    // dp[i][j] = max{f[i − 1][j′] − abs(j − j′)} + points[i][j] ->
    // max{f[i − 1][j′] − j′} + points[i][j] + j (j < j') or  右边
    // max{f[i − 1][j′] + j′} + points[i][j] - j (j >= j')  左边
    // 用两个数组分别记录 f[i − 1][j′] - j′ 和 f[i − 1][j′] + j′的最大值
    vector<vector<long long>> left(m, vector<long long>(n, 0));
    vector<vector<long long>> right(m, vector<long long>(n, 0));
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0) {
                dp[i][j] = points[i][j];
                continue;
            }
            dp[i][j] = max(left[i - 1][j] + points[i][j] - j, right[i - 1][j] + points[i][j] + j);
        }
        left[i][0] = dp[i][0] + 0; 
        for (j = 1; j < n; j++) {
            left[i][j] = dp[i][j] + j > left[i][j - 1] ? dp[i][j] + j : left[i][j - 1];
        }
        right[i][n - 1] = dp[i][n - 1] - (n - 1);
        for (j = n - 2; j >= 0; j--) {
            right[i][j] = dp[i][j] - j > right[i][j + 1] ? dp[i][j] - j : right[i][j + 1];
        }
    }
    return *max_element(dp[m - 1].begin(), dp[m - 1].end());
}


// LC871
int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations)
{
    int i;
    int n;
    int location, leftFuel;
    int ans;
    priority_queue<int, vector<int>> pq;

    leftFuel = startFuel;
    n = stations.size();
    ans = 0;
    location = 0;
    for (i = 0; i < n; i++) {
        if (stations[i][0] - location > leftFuel) {
            while (stations[i][0] - location > leftFuel) {
                if (pq.empty()) {
                    return -1;
                }
                auto p = pq.top();
                pq.pop();
                ans++;
                leftFuel += p;
            }
            
        }
        leftFuel -= stations[i][0] - location;
        pq.push(stations[i][1]);
        location = stations[i][0];
    }

    while (target - location > leftFuel) {
        if (pq.empty()) {
            return -1;
        }
        auto p = pq.top();
        pq.pop();
        leftFuel += p;
        ans++;
    }
    return ans;
}


// LC294
unordered_map<string, bool> canWinStatus; // 字符串先手是否必胜 例如staus["++"] = true
bool canWin(string s)
{
    int i;
    int n = s.size();
    if (canWinStatus.count(s)) {
        return canWinStatus[s];
    }

    for (i = 0; i < n - 1; i++) {
        if (s[i] == s[i + 1] && s[i] == '+') {
            s[i] = '-';
            s[i + 1] = '-';
            if (canWin(s) == false) { // 翻转后的s仍可以获胜,说明后手会赢,反之先手赢
                return true;
            }
            s[i] = '+';
            s[i + 1] = '+';
        }
    }
    canWinStatus[s] = false; // 循环完到此处说明s不能先手赢,否则在循环中就直接return了
    return canWinStatus[s];
}


// LC464
// 数字可以重复取的情况
bool canIWin1(int maxChoosableInteger, int desiredTotal)
{
    if (maxChoosableInteger >= desiredTotal) {
        return true;
    }

    int i, j;
    vector<int> dp(desiredTotal + 1, false); // dp[i] - 剩余i先手能否赢

    for (i = 1; i <= maxChoosableInteger; i++) {
        dp[i] = true;
    }
    for (i = maxChoosableInteger + 1; i <= desiredTotal; i++) {
        for (j = 1; j <= maxChoosableInteger; j++) {
            if (dp[i - j] == false) {
                dp[i] = true;
                break;
            }
        }
        if (j == maxChoosableInteger + 1) {
            dp[i] = false;
        }
    }
    return dp[desiredTotal];
}
// 不可以重复取
bool canIWin(int maxChoosableInteger, int desiredTotal)
{
    if (maxChoosableInteger >= desiredTotal) {
        return true;
    }

    int i;
    int sum = 0;
    for (i = 1; i <= maxChoosableInteger; i++) {
        sum += i;
    }
    if (sum < desiredTotal) {
        return false;
    }

    unordered_map<int, bool> status; // 表示当前取数的状态
    function<bool (int, int, int, int)> DFS = [&status, &DFS](int usedNum, int cur, int maxChoosableInteger, int desiredTotal) {
        if (status.count(usedNum)) {
            return status[usedNum];
        }
        int i;
        for (i = maxChoosableInteger; i >= 1; i--) {
            if ((usedNum & (1 << i)) == 0) {
                if (cur + i >= desiredTotal) {
                    status[usedNum] = true;
                    return true;
                }
                if (DFS(usedNum + (1 << i), cur + i, maxChoosableInteger, desiredTotal) == false) {
                    status[usedNum] = true;
                    return true;
                }
            }
        }
        status[usedNum] = false;
        return false;
    };
    int usedNum = 0;
    return DFS(usedNum, 0, maxChoosableInteger, desiredTotal);
}


// LC375
int getMoneyAmount(int n)
{
    // dp[i][j] - 区间[i, j]猜到数字的最小花费
    int i, j, k;
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0x3f3f3f3f));

    for (j = 1; j <= n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 0;
            } else if (i + 1 == j) {
                dp[i][j] = min(i, j);
            } else {
                // 假设所猜数字是k
                for (k = i + 1; k <= j - 1; k++) {
                    dp[i][j] = min(dp[i][j], max(dp[k + 1][j], dp[i][k - 1]) + k);
                }
            }
        }
    }
    return dp[1][n];
}


// LC1871
// BFS超时
bool canReach1(string s, int minJump, int maxJump)
{
    int i;
    int n = s.size();
    vector<bool> visited(n, false);

    if (s[n - 1] == '1') {
        return false;
    }
    vector<int> idx;
    for (i = 1; i < n; i++) {
        if (s[i] == '0') {
            idx.emplace_back(i);
        }
    }
    vector<int> idx1 = idx;
    reverse(idx1.begin(), idx1.end());
    queue<int> q;
    q.push(0);
    while (!q.empty()) {
        auto t = q.front();
        q.pop();

        visited[t] = true;
        if (t == n - 1) {
            return true;
        }
        auto left = lower_bound(idx.begin(), idx.end(), t + minJump);
        auto right = lower_bound(idx1.begin(), idx1.end(), t + maxJump, greater<int>());
        if (left != idx.end() && right != idx1.end()) {
            for (i = left - idx.begin(); i <= right - idx1.begin(); i++) {
                if (!visited[idx[i]]) {
                    q.push(idx[i]);
                }
            }
        }
    }
    return false;
}
bool canReach(string s, int minJump, int maxJump)
{
    int i;
    int n = s.size();
    int left, right;
    vector<int> prefixSum(n);
    vector<bool> dp(n, false); // dp[i] - 第i位能否到达

    if (s[n - 1] == '1') {
        return false;
    }

    for (i = 0; i < minJump; i++) {
        prefixSum[i] = 1;
    }
    // 同步更新前缀和
    for (i = minJump; i < n; i++) {
        if (s[i] == '0') {
            left = i - maxJump;
            right = i - minJump;
            if (right == 0) {
                dp[i] = true;
                prefixSum[i] = prefixSum[i - 1] + 1;
                continue;
            }
            if (left < 0) {
                left = 0;
            }
            if (left == 0) {
                if (prefixSum[right] > 0) {
                    prefixSum[i] = prefixSum[i - 1] + 1;
                    dp[i] = true;
                }
            } else {
                if (prefixSum[right] - prefixSum[left - 1] > 0) {
                    prefixSum[i] = prefixSum[i - 1] + 1;
                    dp[i] = true;
                }
            }
            if (dp[i] == false) {
                prefixSum[i] = prefixSum[i - 1];
            }
        } else {
            prefixSum[i] = prefixSum[i - 1];
        }
    }
    return dp[n - 1];
}


// LC2964
int divisibleTripletCount(vector<int>& nums, int d)
{
    int i, j;
    int a, b;
    int ans;
    int left, right, mid;
    int n = nums.size();
    unordered_map<int, vector<int>> um;

    for (i = 0; i < n; i++) {
        um[nums[i] % d].emplace_back(i);
    }
    ans = 0;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            a = (nums[i] + nums[j]) % d;
            b = (d - a) % d;
            vector<int> v = um[b];
            left = 0;
            right = v.size() - 1;
            // 所求为left;
            while (left <= right) {
                mid = (right - left) / 2 + left;
                if (v[mid] <= j) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            ans += v.size() - left;
        }
    }
    return ans;
}


// LC2967
void createPalindrome(string& num, int index, set<int>& resultLess, 
    set<int, greater<>>& resultGreater)
{
    int i;
    int n = num.size();

    if ((n % 2 == 0 && index >= n / 2) || (n % 2 == 1 && index > n / 2)) {
        resultLess.emplace(atoi(num.c_str()));
        resultGreater.emplace(atoi(num.c_str()));
        return;
    }
    for (i = 0; i <= 9; i++) {
        if (index == 0 && i == 0 ) {
            continue;
        }

        num[index] = i + '0';
        num[n - 1 - index] = num[index];
        createPalindrome(num, index + 1, resultLess, resultGreater);
    }
}
long long minimumCost(vector<int>& nums)
{
    sort(nums.begin(), nums.end());

    int i;
    int k;
    int n = nums.size();
    int target;
    long long ans = 0;

    k = n / 2;
    string t = to_string(nums[k]);
    if (IsPalindrome(t)) {
        for (i = 0; i < n; i++) {
            ans += abs(nums[i] - nums[k]);
        }
        return ans;
    }

    set<int> resultLess;
    set<int, greater<>> resultGreater;
    string num = t;
    createPalindrome(num, 0, resultLess, resultGreater);

    // 第一个大于等于nums[k]的回文数
    target = *resultLess.lower_bound(nums[k]);
    for (i = 0; i < n; i++) {
        ans += abs(nums[i] - target);
    }

    // 第一个小于等于nums[k]的回文数
    long long tmp = 0;
    auto it = resultGreater.lower_bound(nums[k]);
    if (it == resultGreater.end()) {
        target = nums[k] - 1;
    } else {
        target = *it;
    }
    for (i = 0; i < n; i++) {
        tmp += abs(nums[i] - target);
    }
    return min(ans, tmp);
}


// LC2968
int maxFrequencyScore(vector<int>& nums, long long k)
{
    sort(nums.begin(), nums.end());

    int i;
    int n = nums.size();
    int left, right, mid;
    long long median; // 中位数
    long long need;
    vector<long long> prefixSum(n);

    prefixSum[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + nums[i];
    }

    left = 1;
    right = nums.size();
    while (left <= right) {
        mid = (right - left) / 2 + left;
        need = 0;
        // 滑动窗口
        for (i = 0; i <= n - mid; i++) {
            if (mid % 2 == 1) {
                median = nums[(i + i + mid - 1) / 2];
            } else {
                median = (nums[(i + i + mid - 1) / 2] + nums[(i + i + mid - 1) / 2 + 1]) / 2;
            }
            if (i == 0) {
                need = (mid  + 1) / 2 * median - prefixSum[(i + i + mid - 1) / 2] + 
                    prefixSum[i + mid - 1] - prefixSum[(i + i + mid - 1) / 2] - mid / 2 * median;
            } else {
                need = (mid  + 1) / 2 * median - (prefixSum[(i + i + mid - 1) / 2]  - prefixSum[i - 1]) + 
                    prefixSum[i + mid - 1] - prefixSum[(i + i + mid - 1) / 2] - mid / 2 * median;
            }
            // printf ("i = %d, median = %d, need = %d, mid = %d\n", i, median, need, mid);
            if (need <= k) {
                left = mid + 1;
                break;
            }
        }
        if (i == n - mid + 1) {
            right = mid - 1;
        }
    }
    return right;
}


// LC2276
// to do
class CountIntervals {
public:
    set<vector<int>> s;
    int cnt;
    CountIntervals()
    {
        cnt = 0;
    }
    
    void add(int left, int right)
    {
        if (s.empty()) {
            s.insert({left, right});
            cnt += right - left + 1;
            return;
        }
        auto it = s.lower_bound({left, right});
    }
    
    int count()
    {
        return cnt;
    }
};


// LC458
// N进制编码问题
int poorPigs(int buckets, int minutesToDie, int minutesToTest)
{
    // 有n次实验机会, 就转换成n + 1进制
    int N = minutesToTest / minutesToDie + 1;
    int t;
    int cnt = 0;

    t = buckets;
    while (t) {
        cnt++;
        t /= N;
    }
    if (buckets == pow(2, cnt - 1)) {
        cnt--;
    }
    return cnt;
}


// LC1901
vector<int> findPeakGrid(vector<vector<int>>& mat)
{
    int m = mat.size();
    int n = mat[0].size();
    int left, right, mid;
    int j;

    left = 0;
    right = m - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        j = max_element(mat[mid].begin(), mat[mid].end()) - mat[mid].begin();
        if (mid > 0 && mat[mid -1][j] > mat[mid][j]) {
            right = mid - 1;
        } else if (mid < m - 1 && mat[mid + 1][j] > mat[mid][j]) {
            left = mid + 1;
        } else {
            break;
        }
    }
    return {mid, j};
}


// LC1359
int countOrders(int n)
{
    // 一对一对pd的增加
    int mod = 1000000007;
    int i;
    vector<long long> dp(n + 1);

    dp[1] = 1;
    for (i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] * ((2 * (i - 1) + 1) + 1) * (2 * i - 1) / 2 % mod;
    }
    return dp[n];
}


// LC1220
int countVowelPermutation(int n)
{
    int mod = 1000000007;
    int i;

    vector<vector<long long>> dp(n, vector<long long>(5));
    // dp[i][0] - 第i位以'a'结尾的序列数目

    for (i = 0; i < 5; i++) {
        dp[0][i] = 1;
    }
    for (i = 1; i < n; i++) {
        // 每个元音 'a' 后面都只能跟着 'e'
        // 每个元音 'e' 后面只能跟着 'a' 或者是 'i'
        // 每个元音 'i' 后面 不能 再跟着另一个 'i'
        // 每个元音 'o' 后面只能跟着 'i' 或者是 'u'
        // 每个元音 'u' 后面只能跟着 'a'
        dp[i][0] = (dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][4]) % mod; // [e, i, u] a
        dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % mod; // [a, i] e
        dp[i][2] = (dp[i - 1][1] + dp[i - 1][3]) % mod; // [e, o] i
        dp[i][3] = dp[i - 1][2]; // [i] o
        dp[i][4] = (dp[i - 1][2] + dp[i - 1][3]) % mod; // [i, o] u
    }
    long long ans = 0;
    for (auto it : dp[n - 1]) {
        ans = (ans + it) % mod;
    }
    return ans;
}


// LC1111
vector<int> maxDepthAfterSplit(string seq)
{
    stack<pair<int, int>> st;
    int i;
    int n = seq.size();
    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push({i, 0});
            continue;
        }
        if (seq[i] == ')') {
            auto p = st.top();
            st.pop();
            ans[p.first] = p.second;
            ans[i] = p.second;
        } else {
            auto p = st.top();
            st.push({i, 1 - p.second});
        }
    }
    return ans;
}


// LC2311
int longestSubsequence(string s, int k)
{
    int i, j;
    int n = s.size();
    long long ans;
    long long t;
    bool flag = false;
    // dp[i] - 以s[i]结尾且表示不超过k的最长子序列
    vector<long long> dp(n, 0);

    dp[0] = 1;
    ans = 1;
    for (i = 1; i < n; i++) {
        dp[i] = 1;
        t = s[i] - '0';
        flag = false;
        for (j = i - 1; j >= 0; j--) {
            if (s[j] == '0') {
                dp[i]++;
            } else {
                if (flag) {
                    continue;
                }
                if (i - j > 30) {
                    flag = true;
                    continue;
                }
                t += 1ll << (i - j);
                if (t > k) {
                    flag = true;
                } else {
                    dp[i]++;
                }
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC852
int peakIndexInMountainArray(vector<int>& arr)
{
    // 山脉数组
    int left, right, mid;
    int n = arr.size();

    left = 0;
    right = n - 1;

    while (left < right) {
        mid = (right - left) / 2 + left;
        if (mid < n - 1 && arr[mid] > arr[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2832
vector<int> maximumLengthOfRanges(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int top, topIdx;
    vector<int> ans(n, 1);
    stack<int> st;

    // 右边界 - 单调递减栈
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        topIdx = st.top();
        top = nums[topIdx];
        if (top >= nums[i]) {
            st.push(i);
            continue;
        }
        while (top < nums[i]) {
            ans[topIdx] += i - topIdx - 1;
            st.pop();
            if (st.empty()) {
                break;
            }
            topIdx = st.top();
            top = nums[topIdx];
        }
        st.push(i);
    }
    while (!st.empty()) {
        topIdx = st.top();
        st.pop();
        ans[topIdx] += n - 1 - topIdx;
    }

    // 左边界 - 单调递减栈
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        topIdx = st.top();
        top = nums[topIdx];
        if (top >= nums[i]) {
            st.push(i);
            continue;
        }
        while (top < nums[i]) {
            ans[topIdx] += topIdx - i - 1;
            st.pop();
            if (st.empty()) {
                break;
            }
            topIdx = st.top();
            top = nums[topIdx];
        }
        st.push(i);
    }
    while (!st.empty()) {
        topIdx = st.top();
        st.pop();
        ans[topIdx] += topIdx;
    }
    return ans;
}


// LC1478
int minDistance(vector<int>& houses, int k)
{
    int i, j, p;
    int mid;
    int n = houses.size();
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, INT_MAX));

    // 预处理 - 任意两个下标点有一个邮箱的距离和
    vector<vector<long long>> dist(n, vector<long long>(n, 0));
    sort(houses.begin(), houses.end());
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if ((j - i) % 2 == 0) {
                mid = houses[(i + j) / 2];
            } else {
                mid = (houses[(i + j) / 2] + houses[(i + j + 1) / 2]) / 2;
            }
            for (p = i; p <= j; p++) {
                dist[i][j] += abs(houses[p] - mid);
            }
        }
    }
    for (j = 1; j <= k; j++) {
        for (i = 1; i <= n; i++) {
            if (j >= i) {
                dp[i][j] = 0;
                continue;
            }
            if (j == 1) {
                dp[i][j] = dist[0][i - 1];
            } else {
                for (p = 1; p < i; p++) {
                    dp[i][j] = min(dp[i][j], dp[p][j - 1] + dist[p][i - 1]);
                }
            }
        }
    }
    return dp[n][k];
}


// LC1477
int minSumOfLengths(vector<int>& arr, int target)
{
    int i;
    int n = arr.size();
    unordered_map<int, int> sum;
    vector<int> prefixSum(n, 0);
    vector<vector<int>> range;
    vector<int> minRangeDiff;

    prefixSum[0] = arr[0];
    sum[arr[0]] = 0;
    if (arr[0] == target) {
        range.push_back({0, 0});
    }
    for (i = 1; i < n; i++) {
        prefixSum[i] = arr[i] + prefixSum[i - 1];
        sum[prefixSum[i]] = i;
        if (prefixSum[i] == target) {
            range.push_back({0, i});
        }
        if (sum.count(prefixSum[i] - target)) {
            range.push_back({sum[prefixSum[i] - target] + 1, i});
        }
    }
    if (range.size() < 2) {
        return -1;
    }

    n = range.size();
    minRangeDiff.resize(n);
    minRangeDiff[0] = range[0][1] - range[0][0] + 1;
    for (i = 1; i < n; i++) {
        auto t = range[i][1] - range[i][0] + 1;
        if (t < minRangeDiff[i - 1]) {
            minRangeDiff[i] = t;
        } else {
            minRangeDiff[i] = minRangeDiff[i - 1];
        }
    }

    int left, right, mid;
    int ans = 0x3f3f3f3f;
    for (i = 1; i < n; i++) {
        left = 0;
        right = i - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) /2 + left;
            if (range[mid][1] < range[i][0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            continue;
        }
        ans = min(ans, range[i][1] - range[i][0] + 1 + minRangeDiff[right]);
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC2971
long long largestPerimeter(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left, right, mid;

    sort(nums.begin(), nums.end());

    long long sum = 0;
    long long ans = -1;
    for (i = 0; i < n - 1; i++) {
        sum += nums[i];
        if (i < 1) {
            continue;
        }
        left = i + 1;
        right = n - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) /2 + left;
            if (sum > nums[mid]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right <= i) {
            continue;
        }
        ans = max(ans, sum + nums[right]);
    }
    return ans;
}


// LC2972
long long incremovableSubarrayCount(vector<int>& nums)
{
    int i;
    int n = nums.size();
    long long ans;
    int len = 1;
    
    vector<int> front, rear;
    front.emplace_back(nums[0]);
    for (i = 1; i < n; i++) {
        if (nums[i] > nums[i - 1]) {
            front.emplace_back(nums[i]);
            len++;
        } else {
            break;
        }
    }
    if (len == n) {
        return (long long)n * (n + 1) / 2;
    }
    ans = 1;
    ans += len;

    len = 1;
    rear.emplace_back(nums[n - 1]);
    for (i = n - 2; i >= 0; i--) {
        if (nums[i + 1] > nums[i]) {
            rear.emplace_back(nums[i]);
            len++;
        } else {
            break;
        }
    }
    ans += len;

    n = front.size();
    int left, right, mid;
    // cout << ans << endl;
    reverse(rear.begin(), rear.end());
    for (i = 0; i < n; i++) {
        left = 0;
        right = rear.size() - 1;
        // 所求为left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (rear[mid] < front[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left >= rear.size()) {
            continue;
        }
        // cout << left << endl;
        if (rear[left] == front[i]) {
            ans += rear.size() - left - 1;
        } else {
            ans += rear.size() - left;
        }
    }

    return ans;
}


// LC2975
int maximizeSquareArea(int m, int n, vector<int>& hFences, vector<int>& vFences)
{
    sort(hFences.begin(), hFences.end());
    sort(vFences.begin(), vFences.end());

    int i, j;
    long long ans = -1;
    vector<long long> hvector;
    vector<long long> vvector;
    unordered_set<long long> diffHSet;

    hvector.emplace_back(1);
    for (auto h : hFences) {
        hvector.emplace_back(h);
    }
    hvector.emplace_back(m);

    vvector.emplace_back(1);
    for (auto v : vFences) {
        vvector.emplace_back(v);
    }
    vvector.emplace_back(n);

    for (i = 0; i < hvector.size() - 1; i++) {
        for (j = i + 1; j < hvector.size(); j++) {
            diffHSet.emplace(hvector[j] - hvector[i]);
        }
    }
    // cout << diffHSet.size() << endl;
    for (i = 0; i < vvector.size() - 1; i++) {
        for (j = i + 1; j < vvector.size(); j++) {
            if (diffHSet.count(vvector[j] - vvector[i])) {
                ans = max(ans, (vvector[j] - vvector[i]) * (vvector[j] - vvector[i]));
            }
        }
    }
    return ans > 0 ? ans % 1000000007 : ans;
}


// LC924 - 删除某点不删除其网络
// LC928 - 删除某点同时也删除其网络结构
int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial)
{
    int i, k;
    int n = graph.size();
    int kind = initial.size();
    int minNode = INT_MAX;
    int cnt, ans;
    unordered_set<int> visited;
    queue<int> q;

    sort(initial.begin(), initial.end());
    for (k = 0; k < kind; k++) {
        for (i = 0; i < kind; i++) {
            if (i != k) {
                q.push(initial[i]);
            }
        }
        cnt = 0;
        visited.clear();
        while (!q.empty()) {
            auto t = q.front();
            q.pop();

            if (visited.count(t)) {
                continue;
            }
            cnt++;
            visited.emplace(t);
            for (i = 0; i < n; i++) {
                // LC924
                // if (graph[t][i] == 1 && i != t && visited.count(i) == 0) {
                // LC928
                if (graph[t][i] == 1 && i != t && i != initial[k] && visited.count(i) == 0) {
                    q.push(i);
                }
            }
        }
        if (cnt < minNode) {
            minNode = cnt;
            ans = initial[k];
        }
    }
    return ans;
}


// LC2296
// 暴力,超时
class TextEditor1 {
public:
    TextEditor1()
    {
        cursorPos = 0;
    }

    void addText(string text)
    {
        if (this->text.empty()) {
            this->text = text;
            cursorPos = text.size();
            return;
        }
        if (cursorPos == this->text.size()) {
            this->text += text;
            cursorPos += text.size();
            return;
        } else if (cursorPos == 0) {
            this->text = text + this->text;
            cursorPos += text.size();
            return;
        }
        int len = this->text.size();
        string front = this->text.substr(0, cursorPos);
        string rear = this->text.substr(cursorPos);
        this->text = front + text + rear;
        cursorPos += text.size();
    }

    int deleteText(int k)
    {
        int ans;
        if (k >= cursorPos) {
            ans = cursorPos;
            this->text = this->text.substr(cursorPos);
            cursorPos = 0;
            return ans;
        }
        string front = this->text.substr(0, cursorPos - k);
        string rear = this->text.substr(cursorPos);
        this->text = front + rear;
        cursorPos -= k;
        return k;
    }

    string cursorLeft(int k)
    {
        if (k >= cursorPos) {
            cursorPos = 0;
            return "";
        }
        cursorPos -= k;
        if (cursorPos <= 10) {
            return this->text.substr(0, cursorPos);
        }
        return this->text.substr(cursorPos - 10, 10);
    }

    string cursorRight(int k)
    {
        if (k + cursorPos >= this->text.size()) {
            cursorPos = this->text.size();
            if (cursorPos <= 10) {
                return this->text.substr(0, cursorPos);
            }
            return this->text.substr(cursorPos - 10, 10);
        }
        cursorPos += k;
        if (cursorPos <= 10) {
            return this->text.substr(0, cursorPos);
        }
        return this->text.substr(cursorPos - 10, 10);
    }
private:
    int cursorPos;
    string text;
};
// 双向链表 更好的方法
class TextEditor {
public:
    struct ListNode {
        char val;
        ListNode *prev;
        ListNode *next;
        ListNode(char v) : val(v), prev(nullptr), next(nullptr) {};
    };
    TextEditor()
    {
        head = new ListNode('@');
        tail = new ListNode('#');
        head->next = tail;
        tail->prev = head;
        cursor = tail;
    }
    ~TextEditor()
    {
        ListNode *t = head;
        ListNode *cur = t;
        while (t != tail) {
            t = t->next;
            delete cur;
            cur = t;
        }
        delete cur;
    }

    void addText(string text)
    {
        for (auto ch : text) {
            ListNode *node = new ListNode(ch);
            auto prevNode = cursor->prev;
            prevNode->next = node;
            node->prev = prevNode;
            node->next = cursor;
            cursor->prev = node;
        }
    }

    int deleteText(int k)
    {
        int i;
        int ans = 0;

        for (i = 0; i < k; i++) {
            auto prevNode = cursor->prev; // 要删的点
            if (prevNode == head) {
                break;
            }
            auto prevprevNode = prevNode->prev;
            delete prevNode;
            prevprevNode->next = cursor;
            cursor->prev = prevprevNode;
            ans++;
        }
        return ans;
    }

    string cursorLeft(int k)
    {
        int i;
        while (k) {
            cursor = cursor->prev;
            if (cursor == head) {
                cursor = head->next;
                return "";
            }
            k--;
        }
        ListNode *point = cursor->prev;
        string ans;
        for (i = 0; i < 10; i++) {
            if (point == head) {
                break;
            }
            ans += point->val;
            point = point->prev;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }

    string cursorRight(int k)
    {
        int i;
        while (k) {
            if (cursor == tail) {
                break;
            }
            cursor = cursor->next;
            k--;
        }
        ListNode *point = cursor->prev;
        string ans;
        for (i = 0; i < 10; i++) {
            if (point == head) {
                break;
            }
            ans += point->val;
            point = point->prev;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
private:
    ListNode *head;
    ListNode *tail;
    ListNode *cursor;
};


// LC87
bool isScramble(string s1, string s2)
{
    int i, j, k, p;
    int len = s1.size();
    bool f1, f2;

    // dp[i][j][k] - s1以i下标开始, s2以j下标开始长度为k的子字符串是否可以相互转化
    // 所求dp[0][0][len];
    vector<vector<vector<bool>>> dp(len, vector<vector<bool>>(len, vector<bool>(len + 1, false)));

    for (k = 1; k <= len; k++) {
        for (i = 0; i < len; i++) {
            for (j = 0; j < len; j++) {
                if (i + k > len || j + k > len) {
                    break;
                }
                if (k == 1) {
                    dp[i][j][k] = s1[i] == s2[j];
                    continue;
                }
                for (p = 1; p < k; p++) {
                    f1 = dp[i][j][p] && dp[i + p][j + p][k - p]; // 不交换
                    f2 = dp[i][k + j - p][p] && dp[i + p][j][k - p]; // 交换
                    dp[i][j][k] = dp[i][j][k] || f1 || f2;
                }
            }
        }
    }
    return dp[0][0][len];
}


// LC305
vector<int> numIslands2(int m, int n, vector<vector<int>>& positions)
{
    int k;
    int no, land, oriLand;
    int nrow, ncol, p, np;
    unordered_map<int, vector<int>> lands;
    unordered_map<int, int> posNo;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<int> ans;

    no = 1;
    for (auto pos : positions) {
        p = pos[0] * n + pos[1];
        for (k = 0; k < 4; k++) {
            nrow = pos[0] + directions[k][0];
            ncol = pos[1] + directions[k][1];
            if (nrow < 0 || nrow >= m || ncol < 0 || ncol >= n) {
                continue;
            }
            np = nrow * n + ncol;
            if (posNo.count(np)) {
                if (posNo.count(p) == 0) {
                    land = posNo[np];
                    lands[land].emplace_back(p);
                    posNo[p] = land;
                } else {
                    if (posNo[p] == posNo[np]) {
                        continue;
                    }
                    land = posNo[np];
                    auto v = lands[land];
                    oriLand = posNo[p];
                    for (auto vv : v) {
                        lands[oriLand].emplace_back(vv);
                        posNo[vv] = oriLand;
                    }
                    lands.erase(land);
                }
            }
        }
        if (posNo.count(p) == 0) {
            posNo[p] = no;
            lands[no].emplace_back(p);
            no++;
        }
        ans.emplace_back(lands.size());
    }
    return ans;
}


// LC403
bool canCross(vector<int>& stones)
{
    if (stones[1] - stones[0] != 1) {
        return false;
    }

    int i, j;
    int n = stones.size();
    int dist;
    // dp[i][j] - 青蛙是否能从j跳到i
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    unordered_map<int, unordered_set<int>> steps; // 跳到i的所有步距
    dp[1][0] = true;
    steps[1].emplace(1);
    for (i = 2; i < n; i++) {
        for (j = i - 1; j >= 1; j--) {
            dist = stones[i] - stones[j];
            if (steps[j].count(dist) || steps[j].count(dist - 1) || steps[j].count(dist + 1)) {
                dp[i][j] = true;
                steps[i].emplace(dist);
            }
        }
    }
    for (auto f : dp[n - 1]) {
        if (f) {
            return true;
        }
    }
    return false;
}


// LC1235
int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit)
{
    int i;
    int n = startTime.size();
    int ans;
    int left, right, mid;
    vector<vector<int>> range;
    vector<int> prefixMax(n); // 前缀最大值

    for (i = 0; i < n; i++) {
        range.push_back({startTime[i], endTime[i], profit[i]});
    }
    sort(range.begin(), range.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    // dp[i] - 以第i份兼职结束的最大收益
    vector<int> dp(n, 0);

    dp[0] = range[0][2];
    prefixMax[0] = dp[0];
    ans = dp[0];
    for (i = 1; i < n; i++) {
        left = 0;
        right = i - 1;
        // 所求为right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (range[mid][1] <= range[i][0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            dp[i] = range[i][2];
        } else {
            dp[i] = range[i][2] + prefixMax[right];
        }
        prefixMax[i] = dp[i] > prefixMax[i - 1] ? dp[i] : prefixMax[i - 1];
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC2251
vector<int> fullBloomFlowers(vector<vector<int>>& flowers, vector<int>& people)
{
    int i;
    int n;
    int left, right, mid;
    int a, b;
    vector<int> ans;
    vector<vector<int>> f1, f2;

    f1 = flowers;
    sort(f1.begin(), f1.end());

    f2 = flowers;
    sort(f2.begin(), f2.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    n = people.size();
    for (i = 0; i < n; i++) {
        left = 0;
        right = f1.size() - 1;
        // f1中前区间第一个小于等于people[i]的值, right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (f1[mid][0] > people[i]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        a = right + 1; // 在people[i]时有a朵花开放
        // f2中后区间第一个小于people[i]的值, right
        left = 0;
        right = f2.size() - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (f2[mid][1] >= people[i]) {
                right = mid - 1;
                
            } else {
                left = mid + 1;
            }
        }
        b = right + 1; // 在people[i]时有b朵花凋零
        ans.emplace_back(a - b);
    }
    return ans;
}


// LC1231
int maximizeSweetness(vector<int>& sweetness, int k)
{
    int i;
    int n = sweetness.size();
    int cnt, curSum;
    int left, right, mid;

    left = *min_element(sweetness.begin(), sweetness.end());
    right = accumulate(sweetness.begin(), sweetness.end(), 0);
    while (left <= right) {
        mid = (right - left) / 2 + left;
        curSum = cnt = 0;
        for (i = 0; i < n; i++) {
            curSum += sweetness[i];
            if (curSum >= mid) {
                cnt++;
                curSum = 0;
            }
        }
        if (cnt >= k + 1) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC149
int maxPoints_LC149(vector<vector<int>>& points)
{
    int i, j;
    int n = points.size();
    int a, b;
    unordered_map<int, map<vector<int>, int>> pointK;
    vector<int> k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                continue;
            }
            if (points[i][0] == points[j][0]) {
                pointK[i][{0, 0}]++;
            } else if (points[i][1] == points[j][1]) {
                pointK[i][{0, 1}]++;
            } else {
                a = points[j][1] - points[i][1];
                b = points[j][0] - points[i][0];
                k = {a / gcd(a, b), b / gcd(a, b)};
                pointK[i][k]++;
            }
        }
    }
    int ans = 1;
    for (auto it : pointK) {
        for (auto it1 : it.second) {
            ans = max(ans, it1.second + 1);
        }
    }
    return ans;
}


// LC2271
int maximumWhiteTiles(vector<vector<int>>& tiles, int carpetLen)
{
    int i;
    int n = tiles.size();
    int left, right, mid;
    int ans, t;
    int start;
    vector<int> prefixSum(n, 0);

    sort(tiles.begin(), tiles.end());
    prefixSum[0] = tiles[0][1] - tiles[0][0] + 1;
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + tiles[i][1] - tiles[i][0] + 1;
    }

    ans = 0;
    for (i = 0; i < n; i++) {
        left = 0;
        right = i;
        // 从后往前覆盖, 避免被浪费
        start = tiles[i][1] - carpetLen + 1;
        // 查找第一个比start小的tiles[i][1], 所求为right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (tiles[mid][1] >= start) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (right < 0) {
            t = prefixSum[i];
        } else {
            t = prefixSum[i] - prefixSum[right];
        }
        if (start > tiles[right + 1][0]) {
            t -= start - tiles[right + 1][0];
        }
        ans = max(ans, t);
    }
    return ans;
}


// LC2952
int minimumAddedCoins(vector<int>& coins, int target)
{
    int i;
    int value, idx;
    int n = coins.size();
    int cnt;

    sort(coins.begin(), coins.end());
    value = 1;
    idx = cnt = 0;
    while (value <= target) {
        if (idx == n) {
            value *= 2;
            cnt++;
            continue;
        }
        if (value < coins[idx]) {
            value *= 2;
            cnt++;
        } else {
            value += coins[idx];
            idx++;
        }
    }
    return cnt;
}


// LC2907
int maxProfit_LC2907(vector<int>& prices, vector<int>& profits)
{
    int i, j;
    int n = prices.size();
    int ans;
    // dp[i][k] - 取i商品且一共选k个的最大利润
    vector<vector<int>> dp(n, vector<int>(4, 0));
    for (i = 0; i < n; i++) {
        dp[i][1] = profits[i];
    }
    for (i = 1; i < n; i++) {
        for (j = i - 1; j >= 0; j--) {
            if (prices[i] > prices[j]) {
                dp[i][2] = max(dp[i][2], profits[i] + dp[j][1]);
            }
        }
    }
    ans = -1;
    for (i = 2; i < n; i++) {
        for (j = i - 1; j >= 1; j--) {
            if (prices[i] > prices[j] && dp[j][2] != 0) {
                dp[i][3] = max(dp[i][3], profits[i] + dp[j][2]);
                ans = max(ans, dp[i][3]);
            }
        }
    }
    return ans;
}


// LC2992
int selfDivisiblePermutationCount(int n)
{
    int ans;
    vector<bool> visited(n + 1, false);
    vector<int> permutation(n + 1);

    function<void (int, int)> CreatePermutation = [&CreatePermutation, &visited, &permutation, &ans](int idx, int n) {

        int i;
        if (idx > n) {
            ans++;
            return;
        }
        for (i = 1; i <= n; i++) {
            if (visited[i] == false && (i % idx == 0 || idx % i == 0)) {
                permutation[idx] = i;
                visited[i] = true;
                CreatePermutation(idx + 1, n);
                visited[i] = false;
            }
        }
    };
    ans = 0;
    CreatePermutation(1, n);
    return ans;
}


// LC639
int numDecodings(string s)
{
    int i;
    int n = s.size();
    int mod = 1000000007;
    vector<long long> dp(n + 1, 0);

    dp[0] = 1;
    if (s[0] == '0') {
        return 0;
    }
    if (s[0] == '*') {
        dp[1] = 9;
    } else {
        dp[1] = 1;
    }
    for (i = 1; i < n; i++) {
        if (s[i - 1] == '*') {
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 15) % mod; // 1 - 9, 11 - 19, 21 - 26
            } else if (s[i] == '0') {
                dp[i + 1] = dp[i - 1] * 2 % mod; // 10, 20
            } else if (s[i] > '6') {
                dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 7 - 9, 17 - 19
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1] * 2) % mod; // 7 - 9, 17 - 19, 27 - 29
            }
        } else if (s[i - 1] == '0' || s[i - 1] > '2') {
            if (s[i] == '*') {
                dp[i + 1] = dp[i] * 9 % mod; // 1 - 9
            } else {
                if (s[i] == '0') {
                    return 0;
                }
                dp[i + 1] = dp[i];
            }
        } else if (s[i - 1] == '2') {
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 6) % mod; // 1 - 9, 21 - 26
            } else {
                if (s[i] > '6') {
                    dp[i + 1] = dp[i]; // 7 - 9
                } else if (s[i] == '0') {
                    dp[i + 1] = dp[i - 1]; // 20
                } else {
                    dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 1 - 6, 21 - 26
                }
            }
        } else { // s[i - 1] == '1'
            if (s[i] == '*') {
                dp[i + 1] = (dp[i] * 9 + dp[i - 1] * 9) % mod; // 1 - 9, 11 - 19
            } else {
                if (s[i] == '0') {
                    dp[i + 1] = dp[i - 1]; // 10
                } else {
                    dp[i + 1] = (dp[i] + dp[i - 1]) % mod; // 1 - 9, 11 - 19
                }
            }
        }
    }
    return dp[n];
}


// LC91
int numDecodings_LC91(string s)
{
    int i;
    int n = s.size();
    vector<long long> dp(n + 1, 0);

    dp[0] = 1;
    if (s[0] == '0') {
        return 0;
    }
    dp[1] = 1;
    for (i = 1; i < n; i++) {
        if (s[i - 1] == '0' || s[i - 1] > '2') {
            if (s[i] == '0') {
                    return 0;
            }
            dp[i + 1] = dp[i];
        } else if (s[i - 1] == '2') {
            if (s[i] > '6') {
                dp[i + 1] = dp[i]; // 7 - 9
            } else if (s[i] == '0') {
                dp[i + 1] = dp[i - 1]; // 20
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1]); // 1 - 6, 21 - 26
            }
        } else { // s[i - 1] == '1'
            if (s[i] == '0') {
                dp[i + 1] = dp[i - 1]; // 10
            } else {
                dp[i + 1] = (dp[i] + dp[i - 1]); // 1 - 9, 11 - 19
            }
        }
    }
    return dp[n];
}


// LC1944
vector<int> canSeePersonsCount(vector<int>& heights)
{
    int i;
    int idx;
    int n = heights.size();
    vector<int> ans(n, 0);
    // 单调递减栈
    stack<int> st;

    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        idx = st.top();
        while (heights[i] > heights[idx]) {
            ans[idx]++;
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        if (!st.empty()) {
            ans[idx]++;
        }
        st.push(i);
    }
    return ans;
}


// LC3002
int maximumSetSize(vector<int>& nums1, vector<int>& nums2)
{
    unordered_set<int> s1(nums1.begin(), nums1.end());
    unordered_set<int> s2(nums2.begin(), nums2.end());

    int n1 = s1.size();
    int n2 = s2.size();
    int common = 0; // 相同元素个数
    for (auto it : s1) {
        if (s2.count(it)) {
            common++;
        }
    }
    int unique1 = n1 - common;
    int unique2 = n2 - common;
    int n = nums1.size();
    int m = n / 2;
    int ans = 0;

    if (unique1 >= m) {
        ans += m;
        if (unique2 >= m) {
            ans += m;
        } else {
            ans += min(unique2 + common, m);
        }
    } else {
        if (unique1 + common >= m) {
            ans += m;
            if (unique2 >= m) {
                ans += m;
            } else {
                ans += unique2 + min(common - (m - unique1), m - unique2);
            }
        } else {
            ans += unique1 + common;
            if (unique2 >= m) {
                ans += m;
            } else {
                ans += unique2;
            }
        }
    }

    return ans;
}


// LC545
vector<int> boundaryOfBinaryTree(TreeNode* root)
{
    vector<int> ans;
    list<TreeNode *> left;
    unordered_set<TreeNode *> maybeRightNode, maybeLeftNode;
    // 左边界
    TreeNode * t = root;
    if (t->left != nullptr) {
        while (1) {
            if (t->left) {
                t = t->left;
                left.emplace_back(t);
            } else if (t->right) {
                t = t->right;
                left.emplace_back(t);
                // 此节点可能会占用右节点, 记录
                maybeRightNode.emplace(t);
            } else {
                break;
            }
        }
    }
    // 右边界
    list<TreeNode *> right;
    t = root;
    if (t->right != nullptr) {
        while (1) {
            if (t->right) {
                t = t->right;
                right.emplace_back(t);
            } else if (t->left) {
                t = t->left;
                right.emplace_back(t);
                // 此节点可能会占用左节点, 记录
                maybeLeftNode.emplace(t);
            } else {
                break;
            }
        }
    }
    // 去除左边界与右边界重复的点
    for (auto it : left) {
        // 如果这个节点在右边界中, 删除
        auto Right_iter = find(right.begin(), right.end(), it);
        if (Right_iter != right.end() && maybeLeftNode.count(it)) {
            right.erase(Right_iter);
        }
    }
    for (auto it : right) {
        // 如果这个节点在左边界中, 删除
        auto Left_iter = find(left.begin(), left.end(), it);
        if (Left_iter != left.end() && maybeRightNode.count(it)) {
            left.erase(Left_iter);
        }
    }
    // 将右边界逆序
    reverse(right.begin(), right.end());
    // 叶子节点
    list<TreeNode *> leaf;
    function<void (TreeNode *)> Preorder = [&Preorder, &leaf, &root](TreeNode *node) {
        if (node == nullptr) {
            return;
        }
        if (node != root && node->left == nullptr && node->right == nullptr) {
            leaf.emplace_back(node);
            return;
        }
        Preorder(node->left);
        Preorder(node->right);
    };
    Preorder(root);
    // 去除leaf里面与right和left重复的点
    for (auto it = leaf.begin(); it != leaf.end();) {
        auto Left_iter = find(left.begin(), left.end(), *it);
        auto Right_iter = find(right.begin(), right.end(), *it);
        if (Left_iter != left.end()) {
            leaf.erase(it++);
        } else if (Right_iter != right.end()) {
            leaf.erase(it++);
        } else {
            it++;
        }
    }
    // 组合答案
    ans.emplace_back(root->val);
    for (auto it : left) {
        ans.emplace_back(it->val);
    }
    for (auto it : leaf) {
        ans.emplace_back(it->val);
    }
    for (auto it : right) {
        ans.emplace_back(it->val);
    }
    return ans;
}


// LC2182
string repeatLimitedString(string s, int repeatLimit)
{
    map<char, int, greater<>> m;
    for (auto ch : s) {
        m[ch]++;
    }
    string t;
    char ch;
    while (1) {
        auto it = m.begin();
        if (it == m.end()) {
            goto repeatLimitedStringEND;
        }
        for (; it != m.end();) {
            if (t.empty()) {
                if (it->second <= repeatLimit) {
                    t.append(it->second, it->first);
                    m.erase(it++);
                } else {
                    t.append(repeatLimit, it->first);
                    m[it->first] -= repeatLimit;
                }
                break;
            } else {
                ch = t[t.size() - 1];
                if (ch == it->first) {
                    it++;
                    if (it == m.end()) {
                        goto repeatLimitedStringEND;
                    }
                    continue;
                }
                if (it == m.begin()) {
                    if (it->second <= repeatLimit) {
                        t.append(it->second, it->first);
                        m.erase(it++);
                    } else {
                        t.append(repeatLimit, it->first);
                        m[it->first] -= repeatLimit;
                    }
                } else {
                    t.append(1, it->first);
                    if (it->second == 1) {
                        m.erase(it++);
                    } else {
                        it->second--;
                    }
                }
                break;
            }
        }
    }
repeatLimitedStringEND:
    return t;
}


// LC998
TreeNode* insertIntoMaxTree(TreeNode* root, int val)
{
    vector<int> v;
    // 重建原数组
    function<vector<int> (TreeNode *)> RebuildVector = [&RebuildVector](TreeNode *node) {
        if (node == nullptr) {
            return vector<int>();
        }
        vector<int> v;
        vector<int> left = RebuildVector(node->left);
        vector<int> right = RebuildVector(node->right);
        v.insert(v.end(), left.begin(), left.end());
        v.emplace_back(node->val);
        v.insert(v.end(), right.begin(), right.end());
        return v;
    };
    v = RebuildVector(root);

    v.emplace_back(val);
    // 从数组构建新树
    TreeNode *newroot = nullptr;
    function<TreeNode *(vector<int>&)> BuildTree = [&BuildTree](vector<int>& v) {
        if (v.empty()) {
            return static_cast<TreeNode *>(nullptr);
        }
        vector<int> left;
        vector<int> right;
        int maxVal = *max_element(v.begin(), v.end());
        int i, j;
        int n = v.size();
        for (i = 0; i < n; i++) {
            if (v[i] != maxVal) {
                left.emplace_back(v[i]);
            } else {
                j = i + 1;
                break;
            }
        }
        for (i = j; i < n; i++) {
            right.emplace_back(v[i]);
        }
        TreeNode *node = new TreeNode(maxVal);
        node->left = BuildTree(left);
        node->right = BuildTree(right);
        return node;
    };
    newroot = BuildTree(v);
    return newroot;
}


// LC2979
int mostExpensiveItem(int primeOne, int primeTwo)
{
    int i, j, k;
    int n = primeOne * primeTwo;
    vector<int> dp(n + 1, 0);

    dp[0] = 0;
    i = 0;
    j = 0;
    for (k = 1; k <= n; k++) {
        dp[k] = min(dp[i] + primeOne, dp[j] + primeTwo);
        if (dp[k] == dp[i] + primeOne) {
            i++;
        }
        if (dp[k] == dp[j] + primeTwo) {
            j++;
        }
    }
    for (k = n - 1; k >= 0; k--) {
        if (dp[k] + 1 != dp[k + 1]) {
            return dp[k] + 1;
        }
    }
    return 0;
}


// LC777
bool canTransform(string start, string end)
{
    int i;
    int m = start.size();
    int n = end.size();

    if (m != n) {
        return false;
    }
    string t1, t2;
    vector<int> idxStart, idxEnd;

    i = 0;
    for (auto ch : start) {
        if (ch != 'X') {
            t1 += ch;
            idxStart.emplace_back(i);
        }
        i++;
    }
    i = 0;
    for (auto ch : end) {
        if (ch != 'X') {
            t2 += ch;
            idxEnd.emplace_back(i);
        }
        i++;
    }
    if (t1 != t2) {
        return false;
    }
    for (i = 0; i < idxStart.size(); i++) {
        if (start[idxStart[i]] == 'L' && idxStart[i] < idxEnd[i]) {
            return false;
        }
        if (start[idxStart[i]] == 'R' && idxStart[i] > idxEnd[i]) {
            return false;
        }
    }
    return true;
}


// LC2488
int countSubarrays(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int ans = 0;
    unordered_map<int, int> um;
    vector<int> prefixSum(n, 0);

    auto idx = find(nums.begin(), nums.end(), k) - nums.begin();
    if (nums[0] < k) {
        prefixSum[0] = -1;
        um[-1] = 1;
    } else if (nums[0] > k) {
        prefixSum[0] = 1;
        um[1] = 1;
    } else {
        prefixSum[0] = 0;
        ans++;
    }
    for (i = 1; i < n; i++) {
        if (nums[i] < k) {
            prefixSum[i] = prefixSum[i - 1] - 1;
        } else if (nums[i] > k) {
            prefixSum[i] = prefixSum[i - 1] + 1;
        } else {
            prefixSum[i] = prefixSum[i - 1];
        }
        if (i < idx) {
            um[prefixSum[i]]++;
            continue;
        }
        if (prefixSum[i] == 0 || prefixSum[i] == 1) {
            ans++;
        }
        if (um.count(prefixSum[i] - 1)) {
            ans += um[prefixSum[i] - 1];
        }
        if (um.count(prefixSum[i])) {
            ans += um[prefixSum[i]];
        }
        // um[prefixSum[i]]++; 不再记录, 因为nums[idx]必须包含
    }
    return ans;
}


// LC2054
int maxTwoEvents(vector<vector<int>>& events)
{
    sort(events.begin(), events.end(), [](vector<int>& a, vector<int>& b) {
            if (a[1] != b[1]) {
                return a[1] < b[1];
            }
            return a[0] < b[0];
        }
    );

    int i;
    int n = events.size();
    int ans;
    vector<int> dp(n, 0);

    // pair<int, int> - {开始时间, val} 按val从大到小, 开始时间从小到大排
    auto CMP = [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        }
        return a.first > b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);
    
    for (i = 0; i < n; i++) {
        pq.push({events[i][0], events[i][2]});
    }
    ans = 0;
    for (i = 0; i < n; i++) {
        dp[i] = events[i][2];
        while (!pq.empty()) {
            auto top = pq.top();
            if (top.first > events[i][1]) {
                dp[i] += top.second;
                break;
            }
            pq.pop();
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC2171
long long minimumRemoval(vector<int>& beans)
{
    sort(beans.begin(), beans.end());

    int i;
    int n = beans.size();
    vector<long long> prefixSum(n, 0);
    unordered_map<int, int> first; // 某个值第一次出现时的下标
    unordered_map<int, int> last; // 某个值最后一次出现时的下标

    prefixSum[0] = beans[0];
    first[beans[0]] = 0;
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + beans[i];
        if (beans[i] != beans[i - 1]) {
            first[beans[i]] = i;
            last[beans[i - 1]] = i - 1;
        }
    }
    last[beans[n - 1]] = n - 1;

    long long ans = LONG_LONG_MAX;
    long long cur, left, right;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            cur = prefixSum[n - 1] - prefixSum[last[beans[i]]] - static_cast<long long>(beans[i]) * (n - 1 - last[beans[i]]);
        } else if (i == n - 1) {
            if (first[beans[i]] == 0) {
                cur = 0;
                continue;
            }
            cur = prefixSum[first[beans[i]] - 1];
        } else {
            if (first[beans[i]] == 0) {
                left = 0;
            } else {
                left = prefixSum[first[beans[i]] - 1];
            }
            if (last[beans[i]] == n - 1) {
                right = 0;
            } else {
                right = prefixSum[n - 1] - prefixSum[last[beans[i]]] - static_cast<long long>(beans[i]) * (n - 1 - last[beans[i]]);
            }
            cur = left + right;
        }
        ans = min(ans, cur);
    }
    return ans;
}


// LC1223
int dieSimulator(int n, vector<int>& rollMax)
{
    int mod = 1e9 + 7;
    // dp[i][j][k] - 第i次投骰子为j且已经连续投k次j的总序列数
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(7, vector<long long>(16, 0)));

    int i, j, k;
    int jCur, jPrev;
    for (i = 1; i <= 6; i++) {
        dp[0][i][1] = 1;
    }
    for (i = 1; i < n; i++) {
        for (jCur = 1; jCur <= 6; jCur++) {
            for (jPrev = 1; jPrev <= 6; jPrev++) {
                for (k = 1; k <= 15; k++) {
                    if (k > i + 1 || (k == rollMax[jPrev - 1] && jCur == jPrev)) {
                        break;
                    }
                    if (jCur == jPrev) {
                        dp[i][jCur][k + 1] = (dp[i][jCur][k + 1] + dp[i - 1][jPrev][k]) % mod;
                    } else {
                        dp[i][jCur][1] = (dp[i][jCur][1] + dp[i - 1][jPrev][k]) % mod;
                    }
                }
            }
        }
    }
    long long ans = 0;
    for (i = 1; i <= 6; i++) {
        for (j = 1; j <= rollMax[i - 1]; j++) {
            ans = (ans + dp[n - 1][i][j]) % mod;
        }
    }
    return ans;
}


// LC1361
bool validateBinaryTreeNodes(int n, vector<int>& leftChild, vector<int>& rightChild)
{
    int i;
    unordered_map<int, int> outDegree, inDegree;
    unordered_map<int, vector<int>> edges;

    for (i = 0; i < n; i++) {
        if (leftChild[i] != -1) {
            if (leftChild[leftChild[i]] == i || rightChild[leftChild[i]] == i) { // 双向边
                return false;
            }
            outDegree[i]++;
            inDegree[leftChild[i]]++;
            edges[i].emplace_back(leftChild[i]);
        }
        if (rightChild[i] != -1) {
            if (leftChild[rightChild[i]] == i || rightChild[rightChild[i]] == i) { // 双向边
                return false;
            }
            outDegree[i]++;
            inDegree[rightChild[i]]++;
            edges[i].emplace_back(rightChild[i]);
        }
    }
    int m = inDegree.size();
    if (n - m != 1) { // 入度为0(根)不止一个或没有根
        return false;
    }
    for (auto it : inDegree) {
        if (it.second > 1) { // 图
            return false;
        }
    }
    for (auto it : outDegree) {
        if (it.second > 2) { // 多个子节点
            return false;
        }
    }
    // 判断环
    vector<int> visited(n, 0);
    function<void (int, vector<int>&, bool&)> Loop = [&edges, &Loop](int cur, vector<int>& visited, bool& find) {
        if (find) {
            return;
        }
        if (visited[cur] == 2) {
            find = true;
            return;
        }
        visited[cur] = 2; // 正在被访问
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                Loop(it, visited, find);
            }
        }
        visited[cur] = 1; // 访问完成
    };
    bool find = false;
    for (i = 0; i < n; i++) {
        if (visited[i] == 0) {
            find = false;
            Loop(i, visited, find);
            if (find) {
                return false;
            }
        }
    }
    return true;
}


// LC1229
vector<int> minAvailableDuration(vector<vector<int>>& slots1, vector<vector<int>>& slots2, int duration)
{
    sort(slots1.begin(), slots1.end());
    sort(slots2.begin(), slots2.end());

    int i, j;
    int start, end;
    int m = slots1.size();
    int n = slots2.size();

    i = j = 0;
    while (i < m && j < n) {
        if (slots2[j][0] > slots1[i][1]) {
            i++;
        } else if (slots1[i][0] > slots2[j][1]) {
            j++;
        } else {
            if (slots2[j][0] <= slots1[i][0]) {
                start = slots1[i][0];
                end = min(slots1[i][1], slots2[j][1]);
            } else if (slots2[j][1] < slots1[i][1]) {
                start = slots2[j][0];
                end = slots2[j][1];
            } else {
                start = slots2[j][0];
                end = slots1[i][1];
            }
            if (end - start >= duration) {
                return {start, start + duration};
            } else {
                if (slots2[j][1] > slots1[i][1]) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }
    return {};
}


// LC3008 LC1392
// KMP
void GenerateNextArr(string& s, vector<int>& next)
{
    int i, j;
    int n = s.size();
    next.resize(n);

    next[0] = 0;
    j = 0;
    for (i = 1; i < n; i++) {
        while (s[i] != s[j]) {
            if (j == 0) {
                break;
            }
            j = next[j - 1];
        }
        if (s[i] == s[j]) {
            next[i] = j + 1;
            j++;
        } else {
            next[i] = 0;
        }
    }
}
vector<int> beautifulIndices(string s, string a, string b, int k)
{
    int i, j;
    vector<int> nextA, nextB;

    GenerateNextArr(a, nextA);
    GenerateNextArr(b, nextB);

    vector<int> idxa, idxb;
    int n = s.size();
    int m = a.size();

    j = 0;
    for (i = 0; i < n; i++) {
        while (s[i] != a[j]) {
            if (j == 0) {
                break;
            }
            j = nextA[j - 1];
        }
        if (s[i] == a[j]) {
            j++;
        }
        if (j == m) {
            idxa.emplace_back(i - j + 1);
        }
    }

    j = 0;
    m = b.size();
    for (i = 0; i < n; i++) {
        while (s[i] != b[j]) {
            if (j == 0) {
                break;
            }
            j = nextB[j - 1];
        }
        if (s[i] == b[j]) {
            j++;
        }
        if (j == m) {
            idxb.emplace_back(i - j + 1);
        }
    }
    vector<int> ans;
    int left, right, mid;
    for (i = 0; i < idxa.size(); i++) {
        left = 0;
        right = idxb.size() - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (idxb[mid] < idxa[i] && abs(idxb[mid] - idxa[i]) > k) {
                left = mid + 1;
            } else if (idxb[mid] > idxa[i] && abs(idxb[mid] - idxa[i]) > k) {
                right = mid - 1;
            } else {
                ans.emplace_back(idxa[i]);
                break;
            }
        }
    }
    return ans;
}


// LC3011
bool canSortArray(vector<int>& nums)
{
    unordered_map<int, int> bitCnt;

    int cnt;
    int n = nums.size();
    int i;
    for (i = 0; i < n; i++) {
        cnt = 0;
        auto t = nums[i];
        while (t) {
            if (t % 2 == 1) {
                cnt++;
            }
            t /= 2;
        }
        bitCnt[nums[i]] = cnt;
    }
    for (i = 1; i < n; i++) {
        while (nums[i] < nums[i - 1]) {
            if (bitCnt[nums[i]] != bitCnt[nums[i - 1]]) {
                return false;
            }
            swap(nums[i], nums[i - 1]);
            i--;
            if (i == 0) {
                break;
            }
        }
    }
    return true;
}


// LC3012
int minimumArrayLength(vector<int>& nums)
{
    map<int, int> cnt;
    for (auto n : nums) {
        cnt[n]++;
    }

    vector<int> v;
    for (auto it : cnt) {
        v.emplace_back(it.first);
    }
    if (cnt[v[0]] == 1) {
        return 1;
    }

    int i;
    int n = v.size();
    for (i = 1; i < n; i++) {
        if (v[i] % v[0] != 0) {
            return 1;
        }
    }
    return (cnt[v[0]] + 1) / 2;
}


// LC1156
int maxRepOpt1(string text)
{
    int i, k;
    int n = text.size();
    int ans, size;
    int start, end;
    bool f = false;
    unordered_set<int> us;
    vector<pair<int, int>> vp;
    for (auto ch : text) {
        us.emplace(ch);
    }
    ans = 0;
    for (auto it : us) {
        vp.clear();
        f = false;
        for (i = 0; i < n; i++) {
            if (text[i] == it) {
                if (f == false) {
                    f = true;
                    start = i;
                }
            } else {
                if (f) {
                    f = false;
                    end = i - 1;
                    vp.push_back({start, end});
                }
            }
        }
        if (f) {
            vp.push_back({start, n - 1});
        }

        size = vp.size();
        if (size == 1) {
            ans = max(ans, vp[0].second - vp[0].first + 1);
        } else {
            ans = max(ans, 1 + vp[0].second - vp[0].first + 1);
        }
        for (k = 1; k < size; k++) {
            if (vp[k - 1].second + 2 == vp[k].first) {
                if (size == 2) {
                    ans = max(ans, 
                    vp[k - 1].second - vp[k - 1].first + 1 + vp[k].second - vp[k].first + 1);
                } else { // 从第3个字符串"借"一个
                    ans = max(ans, 1 +
                    vp[k - 1].second - vp[k - 1].first + 1 + vp[k].second - vp[k].first + 1);
                }
            } else {
                ans = max(ans, 1 + vp[k].second - vp[k].first + 1);
            }
        }
    }
    return ans;
}


// LC2121
vector<long long> getDistances(vector<int>& arr)
{
    int i;
    int n = arr.size();
    unordered_map<int, vector<int>> idx;

    for (i = 0; i < n; i++) {
        idx[arr[i]].emplace_back(i);
    }
    // 计算差值前缀和
    unordered_map<int, vector<long long>> diffPreSum;
    for (auto it : idx) {
        auto v = it.second;
        n = v.size();
        diffPreSum[it.first].emplace_back(0);
        for (i = 1; i < n; i++) {
            diffPreSum[it.first].emplace_back(
                diffPreSum[it.first][diffPreSum[it.first].size() - 1] + idx[it.first][i] - idx[it.first][0]);
        }
    }
    // 计算差值后缀和
    unordered_map<int, vector<long long>> diffSufSum;
    for (auto it : idx) {
        auto v = it.second;
        n = v.size();
        diffSufSum[it.first].emplace_back(0);
        for (i = n - 2; i >= 0; i--) {
            diffSufSum[it.first].emplace_back(
                diffSufSum[it.first][diffSufSum[it.first].size() - 1] + idx[it.first][n - 1] - idx[it.first][i]);
        }
    }

    unordered_map<int, int> um; // 遍历时arr[i]出现的次数
    n = arr.size();
    vector<long long> ans(n);
    long long pre, suf;
    for (i = 0; i < n; i++) {
        if (idx[arr[i]].size() == 1) {
            ans[i] = 0;
        } else {
            pre = diffPreSum[arr[i]][diffPreSum[arr[i]].size() - 1];
            suf = diffSufSum[arr[i]][diffSufSum[arr[i]].size() - 1];
            if (um[arr[i]] == 0) {
                ans[i] = pre;
            } else if (um[arr[i]] == diffPreSum[arr[i]].size() - 1) {
                ans[i] = suf;
            } else {
                pre = static_cast<long long>(um[arr[i]]) * (idx[arr[i]][um[arr[i]]] - idx[arr[i]][0]) - diffPreSum[arr[i]][um[arr[i]] - 1];
                suf = static_cast<long long>(idx[arr[i]].size() - 1 - um[arr[i]]) * 
                    (idx[arr[i]][idx[arr[i]].size() - 1] - idx[arr[i]][um[arr[i]]]) - diffSufSum[arr[i]][idx[arr[i]].size() - 1 - um[arr[i]] - 1];
                ans[i] = pre + suf;
            }
            um[arr[i]]++;
        }
    }
    return ans;
}


// LC3004
int maximumSubtreeSize(vector<vector<int>>& e, vector<int>& colors)
{
    unordered_map<int, vector<int>> edges;

    if (e.size() == 0) {
        return 1;
    }
    for (auto it : e) {
        edges[it[0]].emplace_back(it[1]);
        edges[it[1]].emplace_back(it[0]);
    }

    unordered_map<int, int> colorCnt;
    function<int (unordered_map<int, vector<int>>&, int, int)> find = 
        [&colors, &colorCnt, &find](unordered_map<int, vector<int>>& edges, int cur, int parent) {
        bool flag = true;
        if (edges[cur].size() == 1 && edges[cur][0] == parent) {
            colorCnt[cur] = 1;
            return 1;
        }
        for (auto it : edges[cur]) {
            if (it != parent) {
                colorCnt[it] = find(edges, it, cur);
                if (colors[cur] == colors[it] && flag) {
                    if (colorCnt[it] != 0) {
                        colorCnt[cur] += colorCnt[it];
                    } else {
                        colorCnt[cur] = 0;
                        flag = false;
                    }
                } else {
                    flag = false;
                    colorCnt[cur] = 0;
                }
            }
        }
        if (colorCnt[cur] != 0) {
            colorCnt[cur]++;
        }
        return colorCnt[cur];
    };
    find(edges, 0, -1);

    int ans = 1;
    for (auto it : colorCnt) {
        ans = max(ans, it.second);
    }
    return ans;
}


// LC780
bool reachingPoints(int sx, int sy, int tx, int ty)
{
    if (sx > tx || sy > ty) {
        return false;
    }
    if (sx == tx && sy == ty) {
        return true;
    }
    int tmpX, tmpY;
    while (1) {
        if (tx > ty) {
            tmpX = (tx - sx) % ty + sx;
            if (tmpX == tx) {
                return false;
            }
            tx = tmpX;
        } else if (ty > tx) {
            tmpY = (ty - sy) % tx + sy;
            if (tmpY == ty) {
                return false;
            }
            ty = tmpY;
        } else {
            return false;
        }
        if (sx == tx && sy == ty) {
            return true;
        }
        if (tx < sx || ty < sy) {
            break;
        }
    }
    return false;
}


// LC2438
vector<int> productQueries(int n, vector<vector<int>>& queries)
{
    vector<long long> powers;
    long long i = 1;
    int mod = 1e9 + 7;
    while (i <= n) {
        if ((n & i) == i) {
            powers.emplace_back(i);
        }
        i <<= 1;
    }
    long long t;
    vector<int> ans;
    for (auto q : queries) {
        t = 1;
        for (i = q[0]; i <= q[1]; i++) {
            t = t * powers[i] % mod;
        }
        ans.emplace_back(t);
    }
    return ans;
}


// LC2585
int waysToReachTarget(int target, vector<vector<int>>& types)
{
    int i, j, k;
    int n = types.size();
    int mod = 1e9 + 7;
    vector<vector<long long>> dp(target + 1, vector<long long>(n, 0)); // dp[i][j] - 前j个题目获得i分的方法数

    for (j = 0; j < n; j++) {
        dp[0][j] = 1;
    }
    for (j = 0; j < n; j++) {
        if (j == 0) {
            for (k = 1; k <= types[j][0]; k++) {
                if (k * types[j][1] <= target) {
                    dp[k * types[j][1]][j] = 1;
                } else {
                    break;
                }
            }
            continue;
        }
        for (i = 1; i <= target; i++) {
            for (k = 1; k <= types[j][0]; k++) {
                if (i - k * types[j][1] >= 0) {
                    dp[i][j] = (dp[i][j] + dp[i - k * types[j][1]][j - 1]) % mod;
                } else {
                    break;
                }
            }
            dp[i][j] = (dp[i][j] + dp[i][j - 1]) % mod; // 如果k = 0开始循环, 此句不要
        }
    }
    return dp[target][n - 1];
}


// LC1187
int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2)
{
    int i, j, k;
    int idx;
    int n = arr1.size();
    set<int> arr2Set;

    for (auto a : arr2) {
        arr2Set.emplace(a);
    }
    arr2.clear();
    for (auto it : arr2Set) {
        arr2.emplace_back(it);
    }

    int m = arr2.size();
    // dp[i][0] - arr1[i]不替换最小操作数
    // dp[i][j] (j >= 1) - arr1[i]被arr2[j - 1]替换最小操作数
    vector<vector<int>> dp(n, vector<int>(m + 1, 0x3f3f3f3f));

    dp[0][0] = 0;
    for (j = 1; j <= m; j++) {
        dp[0][j] = 1;
    }
    for (i = 1; i < n; i++) {
        // i 不操作 i - 1 也不操作
        if (arr1[i] > arr1[i - 1]) {
            dp[i][0] = min(dp[i][0], dp[i - 1][0]);
        }
        // i 不操作 i - 1 操作
        idx = lower_bound(arr2.begin(), arr2.end(), arr1[i]) - arr2.begin();
        // cout << "1:" << idx << endl;
        for (j = 1; j <= idx; j++) {
            dp[i][0] = min(dp[i][0], dp[i - 1][j]);
        }
        // i 操作, i - 1 不操作
        idx = upper_bound(arr2.begin(), arr2.end(), arr1[i - 1]) - arr2.begin();
        // cout << "2:" << idx << endl;
        for (j = idx + 1; j <= m; j++) {
            dp[i][j] = min(dp[i][j], dp[i - 1][0] + 1);
        }
        // i - 1 与 i 都操作
        for (j = 2; j <= m; j++) {
            dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1);
        }
    }
    int ans = 0x3f3f3f3f;
    ans = min(ans, *min_element(dp[n - 1].begin(), dp[n - 1].end()));
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC768
int maxChunksToSorted_LC768(vector<int>& arr)
{
    map<int, int> arrCnt;

    for (auto a : arr) {
        arrCnt[a]++;
    }
    int curMaxNum, ans;

    curMaxNum = -1;
    ans = 0;
    for (auto num : arr) {
        curMaxNum = max(curMaxNum, num);
        if (arrCnt[num] == 1) {
            arrCnt.erase(num);
        } else {
            arrCnt[num]--;
        }
        if (arrCnt.empty() || curMaxNum <= arrCnt.begin()->first) {
            ans++;
        }
    }
    return ans;
}


// LC955
int minDeletionSize(vector<string>& strs)
{
    int i;
    int n = strs.size();
    int len = strs[0].size();
    int ans;
    vector<int> deleteChar(len, 0);

    auto Check = [&deleteChar](string& a, string& b) {
        int i;
        int len = a.size();
        bool del = false;
        for (i = 0; i < len; i++) {
            if (deleteChar[i] == 1) {
                continue;
            }
            if (a[i] > b[i]) {
                deleteChar[i] = 1;
                del = true;
            } else if (a[i] < b[i]) {
                break;
            } else {
                continue;
            }
        }
        return !del;
    };
    for (i = 1; i < n; i++) {
        if (Check(strs[i - 1], strs[i]) == false) {
            i = 0;
        }
    }
    ans = 0;
    for (auto one : deleteChar) {
        if (one) {
            ans++;
        }
    }
    return ans;
}


// LC1121
bool canDivideIntoSubsequences(vector<int>& nums, int k)
{
    int size = nums.size();
    unordered_map<int, int> um;
    for (auto n : nums) {
        um[n]++;
    }
    int maxFreq = 0;
    for (auto it : um) {
        maxFreq = max(maxFreq, it.second);
    }
    if (maxFreq * k > size) {
        return false;
    }
    return true;
}


// LC2321
int maximumsSplicedArray(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int n = nums1.size();
    vector<int> diff1(n), diff2(n);
    vector<int> prefixSum1(n), prefixSum2(n);
    vector<int> dp1(n), dp2(n), len1(n), len2(n); // len[i] - 满足dp[i]的子数组长度, dp[i] - diff[i]的最大子数组和
    for (i = 0; i < n; i++) {
        diff1[i] = nums1[i] - nums2[i];
        diff2[i] = nums2[i] - nums1[i];
        if (i == 0) {
            prefixSum1[i] = nums1[i];
            prefixSum2[i] = nums2[i];
        } else {
            prefixSum1[i] = prefixSum1[i - 1] + nums1[i];
            prefixSum2[i] = prefixSum2[i - 1] + nums2[i];
        }
    }
    int ans = max(prefixSum1[n - 1], prefixSum2[n - 1]);
    int t, idx;
    dp1[0] = diff1[0];
    len1[0] = 1;
    dp2[0] = diff2[0];
    len2[0] = 1;
    for (i = 1; i < n; i++) {
        if (dp1[i - 1] >= 0) {
            dp1[i] = dp1[i - 1] + diff1[i];
            len1[i] = len1[i - 1] + 1;
        } else {
            dp1[i] = diff1[i];
            len1[i] = 1;
        }

        if (dp2[i - 1] >= 0) {
            dp2[i] = dp2[i - 1] + diff2[i];
            len2[i] = len2[i - 1] + 1;
        } else {
            dp2[i] = diff2[i];
            len2[i] = 1;
        }
    }
    for (i = 0; i < n; i++) {
        idx = i + 1 - len2[i];
        if (idx == 0) {
            t = prefixSum2[i] + prefixSum1[n - 1] - prefixSum1[i];
        } else {
            t = prefixSum2[i] - prefixSum2[idx - 1] + prefixSum1[n - 1] - (prefixSum1[i] - prefixSum1[idx - 1]);
        }
        ans = max(ans, t);
        idx = i + 1 - len1[i];
        if (idx == 0) {
            t = prefixSum2[n - 1] - prefixSum2[i] + prefixSum1[i];
        } else {
            t = prefixSum2[n - 1] - prefixSum2[i] + prefixSum2[idx - 1] + prefixSum1[i] - prefixSum1[idx - 1];
        }
        ans = max(ans, t);
    }
    return ans;
}


// LC2892
int minArrayLength(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int cnt;
    long long cur;
    bool f = false; // 是否进行过乘法操作

    cur = 1;
    cnt = 0;
    for (i = 0; i < n; i++) {
        if (nums[i] == 0) {
            return 1;
        } else if (nums[i] > k) {
            if (f) {
                cnt += 2;
            } else {
                cnt++;
            }
            f = false;
            cur = 1;
            continue;
        }
        cur *= nums[i];
        f = true;
        if (cur > k) {
            cur = nums[i];
            cnt++;
        }
    }
    if (f) {
        cnt++;
    }
    return cnt;
}


// LC2420
vector<int> goodIndices(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    vector<int> less(n , 0), greater(n , 0);
    vector<int> ans;

    less[0] = 1;
    for (i = 1; i < n; i++) {
        if (nums[i] <= nums[i - 1]) {
            less[i] = less[i - 1] + 1;
        } else {
            less[i] = 1;
        }
    }
    greater[n - 1] = 1;
    for (i = n - 2; i >= 0; i--) {
        if (nums[i + 1] >= nums[i]) {
            greater[i] = greater[i + 1] + 1;
        } else {
            greater[i] = 1;
        }
    }
    for (i = k; i < n - k; i++) {
        if (less[i - 1] >= k && greater[i + 1] >= k) {
            ans.emplace_back(i);
        }
    }
    return ans;
}


// LC3031
unsigned long long FastPow(size_t a, size_t b) // 重载一下
{
    unsigned long long ans = 1, base = a;
    while (b != 0) {
        if ((b & 1) != 0) {
            ans *= base;
        }
        base *= base;
        b >>= 1;
    }
    return ans;
}
int minimumTimeToInitialState(string word, int k)
{
    int i;
    int n = word.size();
    vector<unsigned long long> prefixHash(n), suffixHash(n);

    prefixHash[0] = word[0] - 'a' + 1;
    for (i = 1; i < n; i++) {
        prefixHash[i] = prefixHash[i - 1] * 1337 + word[i] - 'a' + 1;
    }
    suffixHash[n - 1] = word[n - 1] - 'a' + 1;
    for (i = n - 2; i >= 0; i--) {
        suffixHash[i] = FastPow(1337, n - i - 1) * (word[i] - 'a' + 1) + suffixHash[i + 1];
    }
    int ans = 1;
    while (1) {
        if (ans * k >= n) {
            break;
        }
        if (suffixHash[ans * k] == prefixHash[n - 1 - ans * k]) {
            break;
        }
        ans++;
    }
    return ans;
}


// LC3035
int maxPalindromesAfterOperations(vector<string>& words)
{
    unordered_map<char, int> data;
    int i;
    int n = words.size();
    priority_queue<int, vector<int>> pq;

    for (auto w : words) {
        for (auto ch : w) {
            data[ch]++;
        }
    }
    // pq只存放偶数个数
    int single = 0;
    for (auto it : data) {
        if (it.second > 1) {
            if (it.second % 2 == 1) {
                single++;
                pq.push(it.second - 1);
            } else {
                pq.push(it.second);
            }
        } else {
            single++;
        }
    }
    vector<int> wordsLenData;
    for (auto w : words) {
        wordsLenData.emplace_back(w.size());
    }
    sort(wordsLenData.begin(), wordsLenData.end());

    int cnt = 0;
    bool canBuild = true;
    for (i = 0; i < n; i++) {
        auto len = wordsLenData[i];
        while (len) {
            if (len == 1) {
                if (single > 0) {
                    single--;
                } else {
                    if (pq.empty()) {
                        canBuild = false;
                    } else {
                        auto top = pq.top();
                        pq.pop();
                        if (top == 2) {
                            single++;
                        } else {
                            pq.push(top - 2);
                            single++;
                        }
                    }
                }
                break;
            } else {
                if (pq.empty()) {
                    canBuild = false;
                    break;
                }
                auto top = pq.top();
                pq.pop();
                if (top > 2) {
                    pq.push(top - 2);
                }
                len -= 2;
            }
        }

        if (canBuild == false) {
            break;
        }
        cnt++;
    }
    return cnt;
}


// LC3045
long long countPrefixSuffixPairs(vector<string>& words)
{
    int i, k;
    int n;
    int size = words.size();
    int base1, base2;
    int mod1, mod2;
    long long ans;
    // 双哈希
    unordered_map<pair<long long, long long>, int, MyHash<long long, long long, long long>> hashCnt;
    vector<pair<long long, long long>> prefixHash(1e5), suffixHash(1e5);

    base1 = 137;
    base2 = 1337;
    mod1 = 1e9 + 7;
    mod2 = 1e9 + 9;
    prefixHash[0] = {words[0][0] - 'a' + 1, words[0][0] - 'a' + 1};
    n = words[0].size();
    for (i = 1; i < n; i++) {
        prefixHash[i] = {(prefixHash[i - 1].first * base1 + (words[0][i] - 'a' + 1)) % mod1,
            (prefixHash[i - 1].second * base2 + (words[0][i] - 'a' + 1)) % mod2};
    }

    hashCnt[prefixHash[n - 1]]++;
    ans = 0;
    for (k = 1; k < size; k++) {
        prefixHash[0] = {words[k][0] - 'a' + 1, words[k][0] - 'a' + 1};
        n = words[k].size();
        for (i = 1; i < n; i++) {
            prefixHash[i] = {(prefixHash[i - 1].first * base1 + (words[k][i] - 'a' + 1)) % mod1,
                (prefixHash[i - 1].second * base2 + (words[k][i] - 'a' + 1)) % mod2};
        }
        suffixHash[n - 1] = {words[k][n - 1] - 'a' + 1, words[k][n - 1] - 'a' + 1};
        for (i = n - 2; i >= 0; i--) {
            suffixHash[i] = {(FastPow(base1, n - i - 1, mod1) * (words[k][i] - 'a' + 1) + suffixHash[i + 1].first) % mod1,
                (FastPow(base2, n - i - 1, mod2) * (words[k][i] - 'a' + 1) + suffixHash[i + 1].second) % mod2};
        }
        for (i = 0; i < n; i++) {
            if (prefixHash[i] == suffixHash[n - 1 - i] && hashCnt.count(prefixHash[i])) {
                ans += hashCnt[prefixHash[i]];
                // printf ("k = %d : %d\n", k, hashCnt[prefixHash[i]]);
            }
        }
        hashCnt[prefixHash[n - 1]]++;
    }
    return ans;
}


// LC741
int cherryPickup(vector<vector<int>>& grid)
{
    // 考虑成走两次摘取最多樱桃
    int i, j, k;
    int n = grid.size();
    // dp[i][j][k] - 从(i, j)走到(n - 1, n - 1)和从(k, i + j - k)走到(n - 1, n - 1)的最大摘取樱桃数
    // 所求dp[0][0][0]
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(n, vector<int>(n, -1)));

    dp[n - 1][n - 1][n - 1] = grid[n - 1][n - 1];
    for (i = n - 1; i >= 0; i--) {
        for (j = n - 1; j >= 0; j--) {
            for (k = n - 1; k >= 0; k--) {
                if (i == n - 1 && j == n - 1 && k == n - 1) {
                    continue;
                }
                if (i + j - k >= n || i + j - k < 0 || grid[i][j] == -1 ||
                    grid[k][i + j - k] == -1) {
                    continue;
                }
                if (i + 1 < n) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i + 1][j][k]);
                    if (k + 1 < n) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i + 1][j][k + 1]);
                    }
                }
                if (j + 1 < n) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j + 1][k]);
                    if (k + 1 < n) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i][j + 1][k + 1]);
                    }
                }
                if (dp[i][j][k] == -1) {
                    continue;
                }
                if (i == k) {
                    dp[i][j][k] += grid[i][j];
                } else {
                    dp[i][j][k] += grid[i][j] + grid[k][i + j - k];
                }
            }
        }
    }
    return dp[0][0][0] == -1 ? 0 : dp[0][0][0];
}


// LC1463
int cherryPickup_II(vector<vector<int>>& grid)
{
    int i, j, k;
    int n = grid.size();
    int m = grid[0].size();
    int ans = -1;
    // dp[i][j][k] - 两机器人分别走到(i, j)和(i, k)的最大樱桃摘取数
    // 所求max(dp[n - 1][j][k])
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(m, vector<int>(m, -1)));

    dp[0][0][m - 1] = grid[0][0] + grid[0][m - 1];
    for (i = 1; i < n; i++) {
        for (j = 0; j < m; j++) {
            for (k = 0; k < m; k++) {
                // 9种情况
                if (j > 0) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - 1][k]);
                    if (k > 0) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - 1][k - 1]);
                    }
                    if (k < m - 1) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - 1][k + 1]);
                    }
                }
                if (j < m - 1) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j + 1][k]);
                    if (k > 0) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j + 1][k - 1]);
                    }
                    if (k < m - 1) {
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j + 1][k + 1]);
                    }
                }
                if (k > 0) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k - 1]);
                }
                if (k < m - 1) {
                    dp[i][j][k] = max(dp[i][j][k],  dp[i - 1][j][k + 1]);
                }
                dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k]);
                if (dp[i][j][k] == -1) {
                    continue;
                }
                if (j == k) {
                    dp[i][j][k] += grid[i][j];
                } else {
                    dp[i][j][k] += grid[i][j] + grid[i][k];
                }
                if (i == n - 1) {
                    ans = max(ans, dp[i][j][k]);
                }
            }
        }
    }
    return ans;
}


// LC889
TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder)
{
    if (preorder.empty()) {
        return static_cast<TreeNode *>(nullptr);
    }
    if (preorder.size() == 1) {
        TreeNode *node = new TreeNode(preorder[0]);
        return node;
    }
    int i, j;
    int n = preorder.size();
    TreeNode *root = new TreeNode(preorder[0]);
    vector<int> a, b, ta, tb;
    vector<int> a1, b1;

    for (i = 1; i < n; i++) {
        a.emplace_back(preorder[i]);
        b.emplace_back(postorder[i - 1]);
        ta = a;
        tb = b;
        sort (ta.begin(), ta.end());
        sort (tb.begin(), tb.end());
        if (ta == tb) {
            root->left = constructFromPrePost(a, b);
            for (j = i + 1; j < n; j++) {
                a1.emplace_back(preorder[j]);
            }
            for (j = i; j < n - 1; j++) {
                b1.emplace_back(postorder[j]);
            }
            root->right = constructFromPrePost(a1, b1);
            break;
        }
    }
    return root;
}


// LC1814
int countNicePairs(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int mod = 1e9 + 7;
    vector<int> rev_nums(n);
    long long ans;

    for (i = 0; i < n; i++) {
        string t = to_string(nums[i]);
        reverse(t.begin(), t.end());
        rev_nums[i] = atoi(t.c_str());
    }
    unordered_map<int, int> diff;
    int val;

    ans = 0;
    for (i = 0; i < n; i++) {
        val = nums[i] - rev_nums[i];
        if (diff.count(val)) {
            ans = (ans + diff[val]) % mod;
        }
        diff[val]++;
    }
    return ans;
}


// LC1737
int minCharacters(string a, string b)
{
    int ch;
    int cnt;
    int ans = INT_MAX;
    // 换成同一字符
    for (ch = 'a'; ch <= 'z'; ch++) {
        cnt = 0;
        for (auto c : a) {
            if (c != ch) {
                cnt++;
            }
        }
        for (auto c : b) {
            if (c != ch) {
                cnt++;
            }
        }
        ans = min(ans, cnt);
    }
    // string a 严格小于 string b
    // ch - a 中最大字符
    for (ch = 'a'; ch <= 'y'; ch++) {
        cnt = 0;
        for (auto c : a) {
            if (c > ch) {
                cnt++;
            }
        }
        for (auto c : b) {
            if (c <= ch) {
                cnt++;
            }
        }
        if (cnt == 1) {
            auto tt = 12;
        }
        ans = min(ans, cnt);
    }
    // string a 严格大于 string b
    // ch - a 中最小字符
    for (ch = 'b'; ch <= 'z'; ch++) {
        cnt = 0;
        for (auto c : a) {
            if (c < ch) {
                cnt++;
            }
        }
        for (auto c : b) {
            if (c >= ch) {
                cnt++;
            }
        }
        ans = min(ans, cnt);
    }
    return ans;
}


// LC3049
int earliestSecondToMarkIndices(vector<int>& nums, vector<int>& changeIndices)
{
    int i;
    int n = nums.size();
    int m = changeIndices.size();
    int left, right, mid;
    int start, cnt;
    unordered_map<int, int> idx; // 每个标记下标最后出现的位置
    unordered_set<int> number;

    vector<int> _nums;
    vector<int> _changeIndices;
    vector<pair<int, int>> vp; // {最后出现下标位置changeIndices, 最后出现的下标}

    _nums.emplace_back(-1);
    _nums.insert(_nums.end(), nums.begin(), nums.end());
    _changeIndices.emplace_back(-1);
    _changeIndices.insert(_changeIndices.end(), changeIndices.begin(), changeIndices.end());

    left = 1;
    right = m;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        number.clear();
        for (i = 1; i <= mid; i++) {
            number.emplace(_changeIndices[i]);
        }
        // 不能全部标记
        if (number.size() != n) {
            left = mid + 1;
            if (left > m) {
                return -1;
            }
            continue;
        }

        idx.clear();
        for (i = 1; i <= mid; i++) {
            idx[_changeIndices[i]] = i;
        }
        vp.clear();
        for (auto it : idx) {
            vp.push_back({it.second, it.first});
        }
        sort(vp.begin(), vp.end());
        start = 1;
        for (i = 0; i < vp.size(); i++) {
            cnt = _nums[vp[i].second];
            start += cnt;
            if (start > vp[i].first) {
                break;
            }
            start++; // 自身标记
        }
        if (i == vp.size()) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    if (left > m) {
        return -1;
    }
    return left;
}


// LC2187
long long minimumTime(vector<int>& time, int totalTrips)
{
    int i;
    int n = time.size();
    long long left, right, mid;
    long long cnt;

    left = 1;
    right = LONG_LONG_MAX - 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        cnt = 0;
        for (i = 0; i < n; i++) {
            cnt += mid / time[i];
            if (cnt >= totalTrips) {
                break;
            }
        }
        if (i < n) {
            right = mid - 1;
            continue;
        }
        left = mid + 1;
    }
    return left;
}


// LC2188
int minimumFinishTime(vector<vector<int>>& tires, int changeTime, int numLaps)
{
    int i, j;
    int n = tires.size();
    // 2 ^ 18 = 262144 > 2e5
    vector<vector<long long>> tireCost(n, vector<long long>(18 + 1, 0));
    vector<vector<long long>> prefixSum(n, vector<long long>(18 + 1, 0));
    vector<long long> minCostUsingSameTire(18 + 1, INT_MAX);

    for (j = 0; j <= 18; j++) {
        for (i = 0; i < n; i++) {
            if (j == 0) {
                tireCost[i][j] = tires[i][0];
                prefixSum[i][j] = tires[i][0] + changeTime;
            } else {
                tireCost[i][j] = min(static_cast<long long>(INT_MAX), tireCost[i][j - 1] * tires[i][1]);
                prefixSum[i][j] = min(static_cast<long long>(INT_MAX), prefixSum[i][j - 1] + tireCost[i][j]);
            }
            minCostUsingSameTire[j] = min(minCostUsingSameTire[j], prefixSum[i][j]);
        }
    }

    vector<long long> dp(numLaps, INT_MAX); // dp[i] - 第i圈最小cost
    for (i = 0; i <= min(18, numLaps - 1); i++) {
        dp[i] = minCostUsingSameTire[i];
    }
    for (i = 1; i < numLaps; i++) {
        for (j = 1; j <= 18; j++) {
            if (i - j >= 0) {
                dp[i] = min(dp[i], dp[i - j] + minCostUsingSameTire[j - 1]);
            } else {
                break;
            }
        }
    }
    return dp[numLaps - 1] - changeTime; // 第一圈不需要换胎
}


// LC3067
vector<int> countPairsOfConnectableServers(vector<vector<int>>& edge, int signalSpeed)
{
    int i, k;
    int n;
    unordered_map<int, vector<pair<int, int>>> edges;

    n = edge.size() + 1;
    vector<int> ans(n, 0);
    for (auto e : edge) {
        edges[e[0]].push_back({e[1], e[2]});
        edges[e[1]].push_back({e[0], e[2]});
    }
    vector<int> suitNodes;
    function<void (int, int, int, unordered_map<int, vector<pair<int, int>>>&, int, int&)> DFS = 
        [&DFS, signalSpeed, &suitNodes]
        (int root, int cur, int parent, unordered_map<int, vector<pair<int, int>>>& edges, int dist, int &cnt) {

        if (dist != 0 && dist % signalSpeed == 0) {
            cnt++;
        }
        int i;
        for (i = 0; i < edges[cur].size(); i++) {
            if (edges[cur][i].first == parent) {
                continue;
            }
            DFS(root, edges[cur][i].first, cur, edges, dist + edges[cur][i].second, cnt);
            
            if (cur == root) {
                suitNodes.emplace_back(cnt);
                // cout << "cnt = " << cnt << endl;
                cnt = 0;
            }
        }
    };
    int cnt;
    for (i = 0; i < n; i++) {
        suitNodes.clear();
        cnt = 0;
        DFS(i, i, -1, edges, 0, cnt);
        if (suitNodes.size() > 1) {
            int t = 0;
            int sum = 0;
            for (k = 0; k < suitNodes.size(); k++) {
                sum += suitNodes[k];
            }
            for (k = 0; k < suitNodes.size(); k++) {
                t += (sum - suitNodes[k]) * suitNodes[k];
            }
            ans[i] = t / 2;
        }
    }
    return ans;
}


// LC799
double champagneTower(int poured, int query_row, int query_glass)
{
    // 第i层j列的香槟满了只会向champagne[i + 1][j]和champagne[i + 1][j + 1]流动
    vector<vector<double>> champagne(101, vector<double>(101, 0.0));
    int i, j;
    double t;

    champagne[0][0] = poured;
    for (i = 0; i <= query_row; i++) {
        for (j = 0; j <= i; j++) {
            if (champagne[i][j] > 1) {
                t = (champagne[i][j] - 1) / 2;
                champagne[i + 1][j] += t;
                champagne[i + 1][j + 1] += t;
            }
        }
    }
    return champagne[query_row][query_glass] > 1 ? 1 : champagne[query_row][query_glass];
}


// LC839
int numSimilarGroups(vector<string>& strs)
{
    int i, j;
    int n = strs.size();

    auto Check = [](string& a, string& b) {
        int i;
        int n = a.size();
        int cnt = 0;
        char aa, bb;
        for (i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                if (cnt > 2) {
                    return false;
                } else if (cnt == 0) {
                    aa = a[i];
                    bb = b[i];
                    cnt++;
                } else {
                    if (a[i] == bb && b[i] == aa) {
                        cnt++;
                    } else {
                        return false;
                    }
                }
            }
        }
        if (cnt == 1) {
            return false;
        }
        return true;
    };

    // 建图求连通分量个数
    unordered_map<int, unordered_set<int>> edges;
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            if (Check(strs[i], strs[j])) {
                edges[i].emplace(j);
                edges[j].emplace(i);
            }
        }
    }
    vector<bool> visited(n, false);
    function<void (unordered_map<int, unordered_set<int>>&, int, int)> DFS = 
        [&visited, &DFS](unordered_map<int, unordered_set<int>>& edges, int cur, int parent) {

        visited[cur] = true;
        if (edges.count(cur) == 0) {
            return;
        }
        for (auto it : edges[cur]) {
            if (it != parent && visited[it] == false) {
                DFS(edges, it, cur);
            }
        }
    };
    int cnt = 0;
    for (i = 0; i < n; i++) {
        if (visited[i] == false) {
            DFS(edges, i, -1);
            cnt++;
        }
    }
    return cnt;
}


// LC895
class FreqStack {
public:
    unordered_map<int, int> valFrequency;
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>> pq; // {frequency, timestamp, val}
    int timestamp;

    FreqStack()
    {
        timestamp = 0;
    }
    void push(int val)
    {
        timestamp++;
        valFrequency[val]++;
        pq.push({valFrequency[val], timestamp, val});
    }
    int pop()
    {
        int val;
        int freq;
        while (1) {
            auto t = pq.top();
            pq.pop();
            val = get<2>(t);
            freq = get<0>(t);
            if (valFrequency.count(val) == 0 || valFrequency[val] != freq) {
                continue;
            } else {
                if (freq == 1) {
                    valFrequency.erase(val);
                } else {
                    valFrequency[val]--;
                }
                break;
            }
        }
        return val;
    }
};


// LC1537
int maxSum(vector<int>& nums1, vector<int>& nums2)
{
    int i;
    int mod = 1e9 + 7;
    int m = nums1.size();
    int n = nums2.size();
    int size;
    vector<long long> prefixSum1(m, 0);
    vector<long long> prefixSum2(n, 0);
    map<int, int> idx1, idx2;

    prefixSum1[0] = nums1[0];
    idx1[nums1[0]] = 0;
    for (i = 1; i < m; i++) {
        idx1[nums1[i]] = i;
        prefixSum1[i] = prefixSum1[i - 1] + nums1[i];
    }
    prefixSum2[0] = nums2[0];
    idx2[nums2[0]] = 0;
    for (i = 1; i < n; i++) {
        idx2[nums2[i]] = i;
        prefixSum2[i] = prefixSum2[i - 1] + nums2[i];
    }

    vector<int> sameValIdx1, sameValIdx2;
    for (auto it : idx1) {
        if (idx2.count(it.first)) {
            sameValIdx1.emplace_back(it.second);
            sameValIdx2.emplace_back(idx2[it.first]);
        }
    }
    if (sameValIdx1.empty()) {
        return max(prefixSum1[m - 1], prefixSum2[n - 1]) % mod;
    }

    long long ans;
    long long t1, t2;

    size = sameValIdx1.size();
    ans = max(prefixSum1[sameValIdx1[0]], prefixSum2[sameValIdx2[0]]);
    for (i = 1; i < size; i++) {
        t1 = prefixSum1[sameValIdx1[i]] - prefixSum1[sameValIdx1[i - 1]];
        t2 = prefixSum2[sameValIdx2[i]] - prefixSum2[sameValIdx2[i - 1]];
        ans += max(t1, t2);
    }
    t1 = prefixSum1[m - 1] - prefixSum1[sameValIdx1[size - 1]];
    t2 = prefixSum2[n - 1] - prefixSum2[sameValIdx2[size - 1]];
    ans += max(t1, t2);
    return ans % mod;
}


// LC3073
int maximumTripletValue(vector<int>& nums)
{
    int i;
    int third;
    int n = nums.size();
    int ans = INT_MIN;
    multiset<int, greater<>> front;
    map<int, int, greater<>> tail;

    for (i = n - 1; i >= 2; i--) {
        tail[nums[i]]++;
    }
    front.emplace(nums[0]);
    for (i = 1; i < n - 1; i++) {
        auto it = front.upper_bound(nums[i]);

        third = tail.begin()->first;
        if (it != front.end() && third > nums[i]) {
            ans = max(ans, *it - nums[i] + third);
        }
        front.emplace(nums[i]);
        if (tail[nums[i + 1]] == 1) {
            tail.erase(nums[i + 1]);
        } else {
            tail[nums[i + 1]]--;
        }
    }
    return ans;
}


// LC555
string splitLoopedString(vector<string>& strs)
{
    int i, j;
    int n = strs.size();
    string t, rev;
    vector<string> vs;

    for (i = 0; i < n; i++) {
        t = strs[i];
        reverse(t.begin(), t.end());
        strs[i] = max(strs[i], t);
        vs.emplace_back(strs[i]);
    }
    string ans;
    vector<string> possibleRes;
    for (i = 0; i < n; i++) {
        // 考虑每一个strs[i]及其翻转作为头的情况
        // 中间部分
        ans = "";
        for (j = i + 1; j < n; j++) {
            ans += vs[j];
        }
        for (j = 0; j < i; j++) {
            ans += vs[j];
        }
        // 拼接头尾
        for (j = 0; j < strs[i].size(); j++) {
            t = strs[i].substr(j) + ans + strs[i].substr(0, j);
            possibleRes.emplace_back(t);
        }
        rev = strs[i];
        reverse(rev.begin(), rev.end());
        for (j = 0; j < rev.size(); j++) {
            t = rev.substr(j) + ans + rev.substr(0, j);
            possibleRes.emplace_back(t);
        }
    }
    sort(possibleRes.rbegin(), possibleRes.rend());
    return possibleRes[0];
}


// LC1526
int minNumberOperations(vector<int>& target)
{
    int i;
    int n = target.size();
    vector<int> dp(n, 0);

    dp[0] = target[0];
    for (i = 1; i < n; i++) {
        if (target[i] > target[i - 1]) {
            dp[i] = dp[i - 1] + target[i] - target[i - 1];
        } else {
            dp[i] = dp[i - 1];
        }
    }
    return dp[n - 1];
}


// LC3076
vector<string> shortestSubstrings(vector<string>& arr)
{
    unordered_map<string, unordered_set<int>> allSubstrings;
    auto cmp = [](const string& a, const string& b) {
        if (a.size() == b.size()) {
            return a < b;
        }
        return a.size() < b.size();
    };
    unordered_map<string, vector<string>> substrings;
    int i, j, k;
    int len;
    string t;
    for (k = 0; k < arr.size(); k++) {
        len = arr[k].size();
        for (i = 0; i < len; i++) {
            t = arr[k][i];
            allSubstrings[t].emplace(k);
            substrings[arr[k]].emplace_back(t);
            for (j = i + 1; j < len; j++) {
                t += arr[k][j];
                allSubstrings[t].emplace(k);
                substrings[arr[k]].emplace_back(t);
            }
        }
        sort(substrings[arr[k]].begin(), substrings[arr[k]].end(), cmp);
        /*for (auto w : substrings[arr[k]]) {
            cout << w << " ";
        }
        cout << endl;
        */
    }
    vector<string> ans(arr.size(), "");
    for (i = 0; i < arr.size(); i++) {
        for (auto it : substrings[arr[i]]) {
            if (allSubstrings[it].size() == 1) {
                ans[i] = it;
                break;
            }
        }
    }
    return ans;
}


// LC815
int numBusesToDestination(vector<vector<int>>& routes, int source, int target)
{
    int i, j, k;
    int n = routes.size();
    unordered_set<int> A;
    unordered_map<int, unordered_set<int>> edges;
    unordered_set<int> sourceStation;
    unordered_set<int> targetStation;

    if (source == target) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        A.clear();
        for (auto r : routes[i]) {
            A.emplace(r);
            if (r == source) {
                sourceStation.emplace(i);
            }
            if (r == target) {
                targetStation.emplace(i);
            }
        }
        if (sourceStation.count(i) && targetStation.count(i)) { // 起点到终点在一条线路上
            return 1;
        }
        for (j = i + 1; j < n; j++) {
            for (k = 0; k < routes[j].size(); k++) {
                if (A.count(routes[j][k])) {
                    edges[i].emplace(j);
                    edges[j].emplace(i);
                    break;
                }
            }
        }
    }

    int ans = INT_MAX;
    int cnt, size;
    queue<int> q;
    vector<int> dist(n, INT_MAX);
    for (auto it : sourceStation) {
        cnt = 1;
        q.push(it);
        while (!q.empty()) {
            size = q.size();
            for (i = 0; i < size; i++) {
                auto node = q.front();
                q.pop();
                if (dist[node] < cnt) {
                    continue;
                }
                dist[it] = cnt;
                if (targetStation.count(node)) {
                    ans = min(ans, cnt);
                    goto QUITQUEUE; // 也可以遍历完, 效率上差距不大
                }
                for (auto it : edges[node]) {
                    if (dist[it] > cnt + 1) {
                        dist[it] = cnt + 1;
                        q.push(it);
                    }
                }
            }
            cnt++;
        }
QUITQUEUE:
        if (!q.empty()) {
            while (q.size()) {
                q.pop();
            }
        }
    }
    return ans == INT_MAX ? -1 : ans;
}


// LC1642
int furthestBuilding(vector<int>& heights, int bricks, int ladders)
{
    int i;
    int n = heights.size();
    int needBricks, ans;
    long long sum;
    priority_queue<int, vector<int>> pq;

    sum = 0;
    ans = n - 1;
    for (i = 1; i < n; i++) {
        if (heights[i] > heights[i - 1]) {
            needBricks = heights[i] - heights[i - 1];
            pq.push(needBricks);
            sum += needBricks;
            if (sum > bricks) {
                if (ladders > 0) {
                    auto t = pq.top();
                    pq.pop();
                    sum -= t;
                    ladders--;
                } else {
                    ans = i - 1;
                    break;
                }
            }
        }
    }
    return ans;
}


// LC892
int surfaceArea(vector<vector<int>>& grid)
{
    int i, j, k;
    int n = grid.size();
    int cnt;
    set<tuple<int, int, int>> brick;

    cnt = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 0) {
                continue;
            }
            cnt += grid[i][j];
            for (k = 0; k < grid[i][j]; k++) {
                brick.insert({i, j, k});
            }
        }
    }
    cnt *= 6;
    // 扣除相交的面
    int a, b, c;
    for (auto it : brick) {
        a = get<0>(it);
        b = get<1>(it);
        c = get<2>(it);

        if (brick.count({a - 1, b, c})) {
            cnt--;
        }
        if (brick.count({a + 1, b, c})) {
            cnt--;
        }
        if (brick.count({a, b - 1, c})) {
            cnt--;
        }
        if (brick.count({a, b + 1, c})) {
            cnt--;
        }
        if (brick.count({a, b, c - 1})) {
            cnt--;
        }
        if (brick.count({a, b, c + 1})) {
            cnt--;
        }
    }
    return cnt;
}


// LC805
bool splitArraySameAverage(vector<int>& nums)
{
    int i, k;
    int t;
    int n = nums.size();
    int sum;

    if (n <= 1) {
        return false;
    } else if (n == 2) {
        return nums[0] == nums[1];
    }

    sum = 0;
    for (auto num : nums) {
        sum += num;
    }

    int sum1, sum2;
    vector<int> part1, part2;

    sum1 = sum2 = 0;
    for (i = 0; i < n / 2; i++) {
        part1.emplace_back(nums[i] * n - sum);
        sum1 += nums[i] * n - sum;
    }
    for (i = n / 2; i < n; i++) {
        part2.emplace_back(nums[i] * n - sum);
        sum2 += nums[i] * n - sum;
    }

    vector<int> record;
    int choose;
    set<pair<int, int>> chooseSum1, chooseSum2;

    // 30个数, 一边最多15个, 2^15数量级是可接受的
    record.resize(n / 2);
    for (i = 1; i < static_cast<int>(pow(2, n / 2)); i++) {
        k = 0;
        t = i;
        while (k < n / 2) {
            if (t % 2 == 1) {
                record[k] = 1;
            } else {
                record[k] = 0;
            }
            t /= 2;
            k++;
        }
        choose = 0;
        for (k = 0; k < n / 2; k++) {
            if (record[k]) {
                choose += part1[k];
            }
        }
        chooseSum1.insert({choose, sum1 - choose});
    }
    record.resize(n - n / 2);
    for (i = 1; i < static_cast<int>(pow(2, n - n / 2)); i++) {
        k = 0;
        t = i;
        while (k < n - n / 2) {
            if (t % 2 == 1) {
                record[k] = 1;
            } else {
                record[k] = 0;
            }
            t /= 2;
            k++;
        }
        choose = 0;
        for (k = 0; k < n - n / 2; k++) {
            if (record[k]) {
                choose += part2[k];
            }
        }
        chooseSum2.insert({choose, sum2 - choose});
    }
    for (auto it : chooseSum1) {
        if (chooseSum2.count({it.second * -1, it.first * -1})) {
            return true;
        }
    }
    return false;
}


// LC1771
// 基础版最长回文子序列
// 返回 - dp数组
vector<vector<int>> longestPalindrome(string& word)
{
    int i, j;
    int n = word.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 1;
                continue;
            }
            if (word[i] == word[j]) {
                if (i + 1 == j) {
                    dp[i][j] = 2;
                } else {
                    dp[i][j] = 2 + dp[i + 1][j - 1];
                }
            } else {
                if (i + 1 == j) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
    }
    return dp;
}
int longestPalindrome(string word1, string word2)
{
    int i, j;
    int m = word1.size();
    int n = word2.size();
    int ans;

    string word = word1 + word2;
    vector<vector<int>> dp = longestPalindrome(word);

    ans = 0;
    for (i = m - 1; i >= 0; i--) {
        for (j = m; j < n + m; j++) {
            if (word[i] == word[j]) {
                if (i + 1 == j) {
                    ans = max(ans, 2);
                } else {
                    ans = max(ans, 2 + dp[i + 1][j - 1]);
                }
            }
        }
    }
    return ans;
}


// LC1382
TreeNode* balanceBST(TreeNode* root)
{
    vector<TreeNode *> nodes;
    function<void (TreeNode *)> Inorder = [&Inorder, &nodes](TreeNode *node) {
        if (node == nullptr) {
            return;
        }
        Inorder(node->left);
        nodes.emplace_back(node);
        Inorder(node->right);
    };

    Inorder(root);

    function<TreeNode* (vector<TreeNode *>&)> CreateBinaryTree = [&CreateBinaryTree](vector<TreeNode *>& nodes) {
        if (nodes.size() == 1) {
            nodes[0]->left = nullptr;
            nodes[0]->right = nullptr;
            return nodes[0];
        } else if (nodes.empty()) {
            return static_cast<TreeNode *>(nullptr);
        }

        int i;
        int n = nodes.size();
        int mid;
        vector<TreeNode *> left, right;

        mid = nodes.size() / 2;
        for (i = 0; i <= mid - 1; i++) {
            left.emplace_back(nodes[i]);
        }
        for (i = mid + 1; i < n; i++) {
            right.emplace_back(nodes[i]);
        }
        nodes[mid]->left = CreateBinaryTree(left);
        nodes[mid]->right = CreateBinaryTree(right);
        return nodes[mid];
    };

    return CreateBinaryTree(nodes);
}


// LC1770
int maximumScore(vector<int>& nums, vector<int>& multipliers)
{
    int i;
    int n = nums.size();
    int m = multipliers.size();
    int ans;
    pair<int, int> np;
    map<pair<int, int>, int> record, t;

    if (m == 1) {
        return multipliers[0] * max(nums[0], nums[n - 1]);
    }

    record[{0, 0}] = nums[0] * multipliers[0];
    record[{-1, -1}] = nums[n - 1] * multipliers[0];
    ans = INT_MIN;
    for (i = 1; i < m; i++) {
        // cout << record.size() << endl;
        for (auto it : record) {
            auto p = it.first;
            np = {p.first - 1, p.second};
            if (t.count(np)) {
                t[np] = max(t[np], it.second + multipliers[i] * nums[n + p.first - 1]);
            } else {
                t[np] = it.second + multipliers[i] * nums[n + p.first - 1];
            }
            if (i == m - 1) {
                ans = max(ans, t[np]);
            }

            np = {p.first, p.second + 1};
            if (t.count(np)) {
                t[np] = max(t[np], it.second + multipliers[i] * nums[p.second + 1]);
            } else {
                t[np] = it.second + multipliers[i] * nums[p.second + 1];
            }
            if (i == m - 1) {
                ans = max(ans, t[np]);
            }
        }
        record = t;
        t.clear();
    }
    return ans;
}


// LC1793
int maximumScore(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();
    int curMin, ans;

    i = j = k;
    curMin = ans = nums[k];
    while (1) {
        if (i - 1 >= 0) {
            if (j + 1 < n) {
                if (nums[j + 1] >= nums[i - 1]) {
                    j++;
                    curMin = min(curMin, nums[j]);
                } else {
                    i--;
                    curMin = min(curMin, nums[i]);
                }
                ans = max(ans, curMin * (j - i + 1));
            } else {
                i--;
                curMin = min(curMin, nums[i]);
                ans = max(ans, curMin * (j - i + 1));
            }
        } else {
            if (j + 1 < n) {
                j++;
                curMin = min(curMin, nums[j]);
                ans = max(ans, curMin * (j - i + 1));
            } else {
                break;
            }
        }
    }
    return ans;
}


// LC2092
vector<int> findAllPeople(int n, vector<vector<int>>& meetings, int firstPerson)
{
    sort (meetings.begin(), meetings.end(), [](const vector<int>& a, const vector<int>& b){
        return a[2] < b[2];
    });

    int i;
    int m = meetings.size();
    unordered_set<int> known;
    unordered_set<int> start, visited;
    unordered_map<int, unordered_set<int>> edges;

    function<void (unordered_map<int, unordered_set<int>>&, int, int)> DFS = 
        [&visited, &DFS](unordered_map<int, unordered_set<int>>& edges, int cur, int parent) {

        visited.emplace(cur);
        if (edges.count(cur) == 0) {
            return;
        }
        for (auto it : edges[cur]) {
            if (it != parent && visited.count(it) == 0) {
                DFS(edges, it, cur);
            }
        }
    };

    known.emplace(0);
    known.emplace(firstPerson);
    for (i = 0; i < m; i++) {
        if (i == 0) {
            edges[meetings[i][0]].emplace(meetings[i][1]);
            edges[meetings[i][1]].emplace(meetings[i][0]);
            if (known.count(meetings[i][0])) {
                start.emplace(meetings[i][0]);
            }
            if (known.count(meetings[i][1])) {
                start.emplace(meetings[i][1]);
            }
            continue;
        }
        if (i > 0 && meetings[i][2] == meetings[i - 1][2]) {
            edges[meetings[i][0]].emplace(meetings[i][1]);
            edges[meetings[i][1]].emplace(meetings[i][0]);
            if (known.count(meetings[i][0])) {
                start.emplace(meetings[i][0]);
            }
            if (known.count(meetings[i][1])) {
                start.emplace(meetings[i][1]);
            }
        } else {
            if (edges.size() == 0) {
                edges[meetings[i][0]].emplace(meetings[i][1]);
                edges[meetings[i][1]].emplace(meetings[i][0]);
                if (known.count(meetings[i][0])) {
                    start.emplace(meetings[i][0]);
                }
                if (known.count(meetings[i][1])) {
                    start.emplace(meetings[i][1]);
                }
                continue;
            }
            visited.clear();
            for (auto node : start) {
                DFS(edges, node, -1);
            }
            known.insert(visited.begin(), visited.end());
            edges.clear();
            start.clear();
            edges[meetings[i][0]].emplace(meetings[i][1]);
            edges[meetings[i][1]].emplace(meetings[i][0]);
            if (known.count(meetings[i][0])) {
                start.emplace(meetings[i][0]);
            }
            if (known.count(meetings[i][1])) {
                start.emplace(meetings[i][1]);
            }
        }
    }
    if (edges.size()) {
        visited.clear();
        for (auto node : start) {
            DFS(edges, node, -1);
        }
        known.insert(visited.begin(), visited.end());
    }
    vector<int> ans;
    for (auto it : known) {
        ans.emplace_back(it);
    }
    return ans;
}


// LC986
vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList)
{
    int i, j;
    int left, right;
    int m = firstList.size();
    int n = secondList.size();
    vector<vector<int>> ans;

    i = j = 0;
    while (i < m && j < n) {
        if (firstList[i][1] < secondList[j][0]) {
            i++;
            continue;
        }
        if (secondList[j][1] < firstList[i][0]) {
            j++;
            continue;
        }
        left = max(firstList[i][0], secondList[j][0]);
        right = min(firstList[i][1], secondList[j][1]);
        ans.push_back({left, right});
        if (firstList[i][1] == secondList[j][1]) {
            i++;
            j++;
        } else if (firstList[i][1] > secondList[j][1]) {
            j++;
        } else {
            i++;
        }
    }
    return ans;
}


// LC1058
string minimizeError(vector<string>& prices, int target)
{
    int i;
    int minVal, maxVal;
    int diff, num;
    double res;
    vector<pair<double, double>> r;

    minVal = maxVal = 0;
    for (auto p : prices) {
        num = static_cast<int>(stod(p));
        if (stod(p) - num < 1e-5) {
            target -= num;
            continue;
        }
        minVal += num;
        maxVal += num + 1;
        r.push_back({stod(p) - num, num + 1 - stod(p)});
    }
    if (maxVal < target || minVal > target) {
        return "-1";
    }

    diff = maxVal - target; // 把diff个数调小
    sort(r.begin(), r.end());
    res = 0.0;
    for (i = 0; i < r.size(); i++) {
        if (i < diff) {
            res += r[i].first;
        } else {
            res += r[i].second;
        }
    }
    char s[100] = {0};
    sprintf (s, "%.3lf", res);
    return string(s);
}


// LC3078
vector<int> findPattern(vector<vector<int>>& board, vector<string>& pattern)
{
    int i, j, k;
    int m, n, a, b;
    string p;

    m = pattern.size();
    n = pattern[0].size();
    for (i = 0; i < m; i++) {
        p += pattern[i];
    }

    a = board.size();
    b = board[0].size();
    if (m > a || n > b) {
        return {-1, -1};
    }

    string t;
    vector<string> s;
    for (i = 0; i < a; i++) {
        t.clear();
        for (j = 0; j < b; j++) {
            t += board[i][j] + '0';
        }
        s.emplace_back(t);
    }
    int len;
    int row;
    unordered_map<char, char> dict1, dict2; // 字母 - 数字  数字 - 字母
    for (i = 0; i <= a - m; i++) {
        for (j = 0; j <= b - n; j++) {
            t.clear();
            row = i;
            for (k = 0; k < m; k++) {
                t += s[row].substr(j, n);
                row++;
            }
            len = p.size();
            dict1.clear();
            dict2.clear();
            for (k = 0; k < len; k++) {
                if (t[k] != p[k]) {
                    if (p[k] >= '0' && p[k] <= '9') {
                        break;
                    }
                    if (dict1.count(p[k]) == 0 && dict2.count(t[k]) == 0) {
                        dict1[p[k]] = t[k];
                        dict2[t[k]] = p[k];
                    } else if (
                        (dict1.count(p[k]) && dict1[p[k]] != t[k]) || 
                        (dict1.count(p[k]) == 0 && dict2.count(t[k]) != 0)
                        ) {
                        break;
                    }
                }
            }
            if (k == len) {
                return {i, j};
            }
        }
    }
    return {-1, -1};
}


// LC2345
int visibleMountains(vector<vector<int>>& peaks)
{
    int i;
    int n = peaks.size();

    // 转换成底边的起始位置, 顶点为(x0, y0), 则两底边点为(x0 - y0, 0), (x0 + y0, 0)
    vector<pair<int, int>> vp;
    for (auto p : peaks) {
        vp.emplace_back(make_pair(p[0] - p[1], p[0] + p[1]));
    }
    sort(vp.begin(), vp.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second > b.second;
    });

    bool same;
    int left, right;
    int cnt, cntSame;

    cnt = 1;
    left = vp[0].first;
    right = vp[0].second;
    same = false;
    cntSame = 0;
    for (i = 1; i < n; i++) {
        if (vp[i].second <= right) {
            if (left == vp[i].first && right == vp[i].second) {
                if (same == false) {
                    same = true;
                    cntSame++;
                }
            }
            continue;
        } else {
            same = false;
            cnt++;
            left = vp[i].first;
            right = vp[i].second;
        }
    }
    return cnt - cntSame;
}


// LC2865
int maxSubarrayLength(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int ans;

    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(nums[i], i));
    }
    sort(vp.rbegin(), vp.rend());

    int minIdx = n;
    ans = 0;
    for (i = 0; i < n; i++) {
        if (minIdx > vp[i].second) {
            minIdx = vp[i].second;
        } else {
            if (minIdx != n) {
                ans = max(ans, vp[i].second - minIdx + 1);
            }
        }
    }
    return ans;
}


// LC1063
int validSubarrays(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int ans;
    stack<int> st; // 单调递增栈
    
    ans = 0;
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        auto top = st.top();
        while (nums[top] > nums[i]) {
            ans += i - top;
            st.pop();
            if (st.empty()) {
                break;
            }
            top = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        ans += n - st.top();
        st.pop();
    }
    return ans;
}

// LC3093
vector<int> stringIndices(vector<string>& wordsContainer, vector<string>& wordsQuery)
{
    int i, j, id;
    int m;
    vector<pair<string, int>> vp;

    i = 0;
    for (auto w : wordsContainer) {
        vp.emplace_back(make_pair(w, i));
        i++;
    }
    auto cmp = [](const pair<string, int>& a, const pair<string, int>& b) {
        if (a.first.size() != b.first.size()) {
            return a.first.size() < b.first.size();
        }
        return a.second < b.second;
    };
    sort (vp.begin(), vp.end(), cmp);
    vector<pair<string, int>> vp1;
    // 去重
    vp1.emplace_back(vp[0]);
    for (i = 1; i < vp.size(); i++) {
        if (vp[i].first != vp[i - 1].first) {
            vp1.emplace_back(vp[i]);
        }
    }
    vp = vp1;

    Trie<char> *root = new Trie<char>('/');
    Trie<char> *t;
    for (auto p : vp) {
        reverse(p.first.begin(), p.first.end());
        t = root;
        Trie<char>::CreateWordTrie(t, p.first, p.second);
    }

    int n = wordsQuery.size();
    vector<int> ans(n, -1);
    
    id = 0;
    for (auto q : wordsQuery) {
        // reverse(q.begin(), q.end());
        t = root;
        for (i = q.size() - 1; i >= 0; i--) {
            m = t->children.size();
            for (j = 0; j < m; j++) {
                if (t->children[j]->val == q[i]) {
                    t = t->children[j];
                    break;
                }
            }
            if (j < m) {
                continue;
            } else {
                if (t->val == '/') {
                    ans[id] = vp[0].second;
                } else {
                    ans[id] = t->timestamp;
                }
                break;
            }
        }
        if (i < 0) {
            ans[id] = t->timestamp;
        }
        id++;
    }
    delete(root);
    return ans;
}


// LC22
vector<string> generateParenthesis(int n)
{
    vector<string> ans;

    function<void (string&, int, int)> DFS = [&DFS, &ans](string& cur, int left, int right) {
        if (left == right && left == 0) {
            ans.emplace_back(cur);
            return;
        }
        if (left == right) {
            cur += '(';
            DFS(cur, left - 1, right);
            cur.pop_back();
        } else if (right > left) {
            if (left > 0) {
                cur += '(';
                DFS(cur, left - 1, right);
                cur.pop_back();
            }
            cur += ')';
            DFS(cur, left, right - 1);
            cur.pop_back();
        }
    };
    string cur;
    DFS(cur, n, n);
    return ans;
}


// LC878
int nthMagicalNumber(int n, int a, int b)
{
    int mod = 1e9 + 7;
    int g;
    long long cnt;
    long long left, right, mid;

    g = gcd(a, b);
    a /= g;
    b /= g;
    left = 1;
    right = LLONG_MAX >> 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        cnt = mid / a + mid / b - mid / (a * b);
        if (cnt >= n) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left * g % mod;
}


// LC1788
int maximumBeauty(vector<int>& flowers)
{
    int i;
    int n = flowers.size();
    int left, right;
    vector<int> prefixSum(n), prefixMinusSum(n);
    unordered_map<int, vector<int>> flowersIdx;

    prefixSum[0] = flowers[0];
    prefixMinusSum[0] = flowers[0] < 0 ? flowers[0] : 0;
    flowersIdx[flowers[0]].emplace_back(0);
    for (i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + flowers[i];
        prefixMinusSum[i] = prefixMinusSum[i - 1] + (flowers[i] < 0 ? flowers[i] : 0);
        flowersIdx[flowers[i]].emplace_back(i);
    }
    int ans = INT_MIN;
    int cur;
    for (auto it : flowersIdx) {
        if (it.second.size() == 1) {
            continue;
        }
        left = it.second[0];
        right = it.second[it.second.size() - 1];
        if (left == 0) {
            cur = prefixSum[right] - (prefixMinusSum[right - 1] - prefixMinusSum[left]);
        } else {
            cur = prefixSum[right] - prefixSum[left - 1] - (prefixMinusSum[right - 1] - prefixMinusSum[left]);
        }
        ans = max(ans, cur);
    }
    return ans;
}


// LC3088
string makeAntiPalindrome(string s)
{
    sort(s.begin(), s.end());

    int i, j;
    int idx;
    int n = s.size();
    unordered_map<char, set<int>> charIdx;
    vector<int> charCnt(26, 0);
    for (i = 0; i < n; i++) {
        charCnt[s[i] - 'a']++;
        if (charCnt[s[i] - 'a'] > n / 2) {
            return "-1";
        }
        charIdx[s[i]].emplace(i);
    }
    for (i = n / 2 - 1; i >= 0; i--) {
        if (s[i] == s[n - 1 - i]) {
            idx = n;
            for (j = s[i]; j <= 'z'; j++) {
                if (j == s[i] || charIdx.count(j) == 0) {
                    continue;
                }
                auto it = charIdx[j].upper_bound(n - 1 - i);
                if (it == charIdx[j].end()) {
                    continue;
                }
                idx = *it;
                break;
            }
            if (idx == n) {
                return "-1";
            }
            charIdx[s[n - 1 - i]].erase(n - 1 - i);
            // charIdx[s[n - 1 - i]].emplace(idx);
            charIdx[s[idx]].erase(idx);
            // charIdx[s[idx]].emplace(n - 1 - i);
            swap(s[n - 1 - i], s[idx]);
        }
    }
    return s;
}


// n个点的最大曼哈顿距离
int GetManhattan(vector<vector<int>>& p)
{
    int ans = 0;
    int Min, Max;
    int i, j, k;
    for (i = 0; i < (1 << (2)); i++) {    // 用二进制形式遍历所有可能的运算情况
        Min = 1e9, Max = -1e9;
        for (j = 0; j < p.size(); j++) {    // 遍历每一个点
            int sum = 0;
            for (k = 0; k < 2; k++) {
                // 提取当前运算符
                int t = i & 1 << k;    // 1为+，0为-
                if (t) {
                    sum += p[j][k];
                } else {
                    sum -= p[j][k];
                }
            }
            if (sum > Max) {
                Max = sum;
            }
            if (sum < Min) {
                Min = sum;
            }
        }
        if (Max - Min > ans) {
            ans = Max - Min;
        }
    }
    return ans;
}


// LC494
int findTargetSumWays(vector<int> &nums, int target)
{
    int ans = 0;
    int i, n = nums.size();
    // 同奇偶
    int sum = 0;
    for (auto num : nums) {
        sum += num;
    }
    if (sum % 2 != target % 2) {
        return 0;
    }
    unordered_map<int, int> data1, data2;
    function<void (vector<int>&, int, int, unordered_map<int, int>&)> DFS = 
        [&DFS](vector<int>& nums, int idx, int cur, unordered_map<int, int>& data) {
        if (idx == nums.size()) {
            data[cur]++;
            return;
        }
        int i;
        for (i = -1; i <= 1; i += 2) {
            cur += nums[idx] * i;
            DFS(nums, idx + 1, cur, data);
            cur -= nums[idx] * i;
        }
    };
    if (nums.size() == 1) {
        return abs(nums[0]) == abs(target) ? 1 : 0;
    }
    vector<int> nums1, nums2;
    for (i = 0; i < n; i++) {
        if (i < n / 2) {
            nums1.emplace_back(nums[i]);
        } else {
            nums2.emplace_back(nums[i]);
        }
    }
    DFS(nums1, 0, 0, data1);
    DFS(nums2, 0, 0, data2);
    for (auto it : data1) {
        if (data2.count(target - it.first)) {
            ans += it.second * data2[target - it.first];
        }
    }
    return ans;
}


// LC3108
vector<int> minimumCost(int n, vector<vector<int>>& edge, vector<vector<int>>& query)
{
    unordered_map<int, unordered_map<int, int>> edges;

    for (auto e : edge) {
        if (edges[e[0]].count(e[1])) {
            edges[e[0]][e[1]] &= e[2];
        } else {
            edges[e[0]][e[1]] = e[2];
        }
        if (edges[e[1]].count(e[0])) {
            edges[e[1]][e[0]] &= e[2];
        } else {
            edges[e[1]][e[0]] = e[2];
        }
    }
    int i;
    int area;
    int dist;
    unordered_set<int> connected;
    unordered_map<int, pair<int, int>> nodeData;
    vector<bool> visited(n, false);

    function<void (unordered_map<int, unordered_map<int, int>>&, int)> DFS = 
        [&DFS, &connected, &visited](unordered_map<int, unordered_map<int, int>>& edges, int node) {

        if (edges.count(node) == 0) {
            return;
        }
        connected.emplace(node);
        visited[node] = true;
        for (auto it : edges[node]) {
            if (connected.count(it.first) == 0) {
                DFS(edges, it.first);
            }
        }
    };

    area = 0;
    for (i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        connected.clear();
        DFS(edges, i);
        dist = 0xfffff; // 注意必须每位为1
        if (connected.empty()) {
            nodeData[i] = {area, 0};
            area++;
            continue;
        }
        for (auto it : connected) {
            for (auto e : edges[it]) {
                dist &= e.second;
            }
        }
        for (auto it : connected) {
            nodeData[it] = {area, dist};
        }
        area++;
    }
    vector<int> ans;
    for (auto q : query) {
        if (q[0] == q[1]) {
            ans.emplace_back(0);
            continue;
        }
        if (nodeData[q[0]].first == nodeData[q[1]].first) {
            ans.emplace_back(nodeData[q[0]].second);
        } else {
            ans.emplace_back(-1);
        }
    }
    return ans;
}


// LC2009
int minOperations(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int left, right, mid, lastnum;
    int ans = 0x3f3f3f3f;
    int cur;
    vector<int> offset(n, 0); // 偏移量数组 offset[i] - nums[i]及以前重复数字个数

    sort(nums.begin(), nums.end());
    for (i = 1; i < n; i++) {
        if (nums[i] == nums[i - 1]) {
            offset[i] = offset[i - 1] + 1;
        } else {
            offset[i] = offset[i - 1];
        }
    }
    // 以nums[i]作为第一个数
    for (i = 0; i < n; i++) {
        cur = i;
        lastnum = nums[i] + n - 1;
        left = i;
        right = n - 1;
        // nums中第一个小于等于lastNum的位置
        // 所求right
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (nums[mid] <= lastnum) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < i) { // nums最大值小于lastNum
            // do nothing
        } else {
            cur += n - 1 - right + (offset[right] - offset[i]);
        }
        ans = min(ans, cur);
    }
    return ans;
}


// LC1766
vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& e)
{
    int i, j;
    unordered_map<int, unordered_set<int>> edges;
    // nums 限定范围从1 - 50, 一条分支上最多50种情况
    vector<stack<int>> sts(51); // sts[i] 表示 nums[i]出现层数的栈,栈顶一定是最新的
    unordered_map<int, unordered_set<int>> coprime; // 每一个数(1-50)的互质数
    for (i = 1; i <= 50; i++) {
        for (j = 1; j <= 50; j++) {
            if (gcd(i, j) == 1) {
                coprime[i].emplace(j);
            }
        }
    }
    for (auto edge : e) {
        edges[edge[0]].emplace(edge[1]);
        edges[edge[1]].emplace(edge[0]);
    }
    vector<int> ans(nums.size(), -1);
    function<void (int, int, int, vector<int>&)> DFS = 
        [&DFS, &edges, &sts, &coprime, &nums, &ans](int cur, int parent, int layer, vector<int>& route) {

        int tLayer;

        // 遍历所有互斥数
        tLayer = -1;
        for (auto it : coprime[nums[cur]]) {
            if (sts[it].size() > 0) {
                if (sts[it].top() > tLayer) {
                    tLayer = sts[it].top();
                    ans[cur] = route[tLayer];
                }
            }
        }
        if (tLayer == -1) {
            ans[cur] = -1;
        }
        sts[nums[cur]].push(layer);
        route.emplace_back(cur);
        for (auto it : edges[cur]) {
            if (it != parent) {
                DFS(it, cur, layer + 1, route);
            }
        }
        sts[nums[cur]].pop();
        route.pop_back();
    };

    vector<int> route;
    DFS(0, -1, 0, route);
    return ans;
}


// LC3113
long long numberOfSubarrays(vector<int>& nums)
{
    int i;
    int n = nums.size();
    long long ans;
    stack<int> st;
    unordered_map<int, long long> um;

    ans = 0;
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(nums[i]);
            
            continue;
        }
        auto t = st.top();
        while (t < nums[i]) {
            st.pop();
            um[t]++;
            if (st.empty()) {
                break;
            }
            t = st.top();
        }
        st.push(nums[i]);
        for (auto it : um) {
            ans += (it.second + 1) * it.second / 2;
        }
        um.clear();
    }
    while (!st.empty()) {
        um[st.top()]++;
        st.pop();
    }
    for (auto it : um) {
        ans += (it.second + 1) * it.second / 2;
    }
    return ans;
}


// LC3116
// 多重容斥原理
long long findKthSmallest(vector<int>& coins, int k)
{
    int i, j;
    int n = coins.size();
    int size = 1 << n; // size - 1 种选币可能, 2进制

    sort(coins.begin(), coins.end());
    long long left, right, mid;
    long long cur, t;
    int cnt, cntOne;
    int lastIdx;
    vector<int> bits(n);
    vector<vector<long long>> lcm(n + 1); // n个数最小公倍数 Least Common Multiple

    for (i = 1; i < size; i++) {
        bits.assign(n, 0);
        cnt = cntOne = 0;
        t = i;
        cur = 1;
        while (cnt < n) {
            bits[cnt] = t % 2;
            if (bits[cnt] == 1) {
                cur = cur * coins[cnt] / gcd(cur, coins[cnt]);
                if (cur > INT_MAX) {
                    cur = INT_MAX;
                }
                cntOne++;
            }
            t /= 2;
            cnt++;
        }
        lcm[cntOne].emplace_back(cur);
    }
    /*for (auto lc : lcm) {
        for (auto l : lc) {
            cout << l << " ";
        }
        cout << endl;
    }*/
    left = coins[0];
    right = LLONG_MAX >> 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        cur = 0;
        for (i = 1; i <= n; i++) {
            for (j = 0; j < lcm[i].size(); j++)
            if (i % 2 == 1) {
                cur += mid / lcm[i][j];
            } else {
                cur -= mid / lcm[i][j];
            }
        }
        if (cur >= k) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2018
bool placeWordInCrossword(vector<vector<char>>& board, string word)
{
    int i, j;
    int m = board.size();
    int n = board[0].size();

    // 在board圈外再加一层'#'
    vector<vector<char>> nboard(m + 2, vector<char>(n + 2, '#'));
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            nboard[i][j] = board[i - 1][j - 1];
        }
    }

    string a = '#' + word + '#';
    reverse(word.begin(), word.end());
    string b = '#' + word + '#';

    string t;
    vector<int> idx;

    auto f = [](vector<int>& idx, string& t, string& a) {
        int i, k;
        int p;
        for (i = 1; i < idx.size(); i++) {
            p = 1;
            if (idx[i] - idx[i - 1] - 1 == a.size() - 2) {
                for (k = idx[i - 1] + 1; k <= idx[i] - 1; k++) {
                    if (t[k] == ' ' || t[k] == a[p]) {
                        p++;
                        continue;
                    } else {
                        break;
                    }
                }
                if (k == idx[i]) {
                    return true;
                }
            }
        }
        return false;
    };

    // 横向
    for (i = 0; i <= m + 1; i++) {
        t.clear();
        idx.clear();
        for (j = 0; j <= n + 1; j++) {
            t += nboard[i][j];
            if (nboard[i][j] == '#') {
                idx.emplace_back(j);
            }
        }
        if (f(idx, t, a) || f(idx, t, b)) {
            return true;
        }
    }

    // 纵向
    for (j = 0; j <= n + 1; j++) {
        t.clear();
        idx.clear();
        for (i = 0; i <= m + 1; i++) {
            t += nboard[i][j];
            if (nboard[i][j] == '#') {
                idx.emplace_back(i);
            }
        }
        if (f(idx, t, a) || f(idx, t, b)) {
            return true;
        }
    }
    return false;
}


// LC730
int countPalindromicSubsequences(string s)
{
    int i, j;
    int l, r;
    int mod = 1e9 + 7;
    int n = s.size();
    vector<vector<long long>> dp(n, vector<long long>(n, 0));
    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 1;
                continue;
            }
            if (i + 1 == j) {
                dp[i][j] = 2;
                continue;
            }
            if (s[i] != s[j]) {
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + mod - dp[i + 1][j - 1]) % mod;
            } else {
                dp[i][j] = dp[i + 1][j - 1] * 2;
                l = i + 1;
                r = j - 1;
                while (l <= j - 1) {
                    if (s[l] == s[i]) {
                        break;
                    }
                    l++;
                }
                while (r >= i + 1) {
                    if (s[r] == s[i]) {
                        break;
                    }
                    r--;
                }
                if (l > r) {
                    dp[i][j] = (dp[i][j] + 2) % mod;
                } else if (l == r) {
                    dp[i][j] = (dp[i][j] + 1) % mod;
                } else {
                    dp[i][j] = (dp[i][j] + mod - dp[l + 1][r - 1]) % mod;
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC2930
int stringCount(int n)
{
    int i;
    int mod = 1e9 + 7;
    // dp[n][0 - 4][0 - 2] - 前n位字符串有0 - 4个满足要求的字符的总字符串数, 且之前已包含0 - 2个'e'字符
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(5, vector<long long>(3, 0)));
    // l e e t四个字符
    dp[0][0][0] = 23;
    dp[0][1][0] = 2;
    dp[0][1][1] = 1;
    for (i = 1; i < n; i++) {
        dp[i][0][0] = dp[i - 1][0][0] * 23 % mod;

        dp[i][1][0] = (dp[i - 1][0][0] * 2 + dp[i - 1][1][0]* (23 + 1)) % mod; // l和t
        dp[i][1][1] = (dp[i - 1][0][0] + dp[i - 1][1][1] * 23) % mod; // e

        dp[i][2][0] = (dp[i - 1][1][0] + dp[i - 1][2][0] * (23 + 2)) % mod; // l和t
        dp[i][2][1] = (dp[i - 1][1][0] + dp[i - 1][1][1] * 2 + dp[i - 1][2][1] * (23 + 1)) % mod; // 只有一个e
        dp[i][2][2] = (dp[i - 1][1][1] + dp[i - 1][2][2] * (23 + 1)) % mod;

        dp[i][3][1] = (dp[i - 1][2][0] + dp[i - 1][2][1] + dp[i - 1][3][1] * (23 + 2)) % mod;
        dp[i][3][2] = (dp[i - 1][2][1] + dp[i - 1][2][2] * 2 + dp[i - 1][3][2] * (23 + 2)) % mod;

        dp[i][4][2] = (dp[i - 1][3][1] + dp[i - 1][3][2] + dp[i - 1][4][2] * 26) % mod;
    }
    return dp[n - 1][4][2];
}


// LC216
vector<vector<int>> combinationSum3(int k, int n)
{
    // 枚举迭代
    int i, j;
    int cnt;
    int curSum;
    vector<vector<int>> ans;
    vector<int> record;

    for (i = 1; i <= 511; i++) {
        cnt = curSum = 0;
        record.clear();
        for (j = 0; j < 9; j++) {
            if ((i & 1 << j) == (1 << j)) {
                cnt++;
                curSum += j + 1;
                if (curSum > n || cnt > k || (cnt == k && curSum != n)) {
                    break;
                }
                record.emplace_back(j + 1);
            }
        }
        if (cnt == k && curSum == n) {
            ans.emplace_back(record);
        }
    }
    return ans;
}


// LC3122
int minimumOperations(vector<vector<int>>& grid)
{
    int i, j, k;
    int n = grid.size();
    int m = grid[0].size();
    vector<vector<int>> costs(m, vector<int>(10, 0)); // costs[i][0 - 9] 第i列全变成0 - 9的花费
    vector<int> cnt(10);

    for (j = 0; j < m; j++) {
        cnt.assign(10, 0);
        for (i = 0; i < n; i++) {
            cnt[grid[i][j]]++;
        }
        for (i = 0; i < 10; i++) {
            costs[j][i] = n - cnt[i];
        }
    }

    vector<vector<int>> dp(m, vector<int>(10, INT_MAX)); // dp[i][0 - 9] 第i列全变成0 - 9的总花销最小值
    for (i = 0; i < 10; i++) {
        dp[0][i] = costs[0][i];
    }
    for (j = 1; j < m; j++) {
        for (k = 0; k < 10; k++) {
            for (i = 0; i < 10; i++) {
                if (k == i) {
                    continue;
                }
                dp[j][k] = min(dp[j][k], costs[j][k] + dp[j - 1][i]);
            }
        }
    }
    return *min_element(dp[m - 1].begin(), dp[m - 1].end());
}


// LC3123
vector<bool> findAnswer(int n, vector<vector<int>>& edges)
{
    int i;
    int size = edges.size();
    vector<int> dist;
    vector<bool> ans(size, false);
    vector<vector<pair<int, int>>> edgeWithWeight(n);
    priority_queue<pair<int, int>, vector<pair<int, int>>> q;
    pair<int, int> t;
    map<vector<int>, int> edgesIdx;

    for (i = 0; i < size; i++) {
        edgeWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
        edgeWithWeight[edges[i][1]].push_back({edges[i][0], edges[i][2]});
        edgesIdx[{edges[i][0], edges[i][1]}] = i;
        edgesIdx[{edges[i][1], edges[i][0]}] = i;
    }

    dist.assign(n, INT_MAX);
    q.push({0, 0});
    while (q.size()) {
        t = q.top();
        q.pop();

        if (dist[t.first] < t.second) {
            continue;
        }
        dist[t.first] = t.second;
        size = edgeWithWeight[t.first].size();
        for (auto e : edgeWithWeight[t.first]) {
            if (e.second + t.second < dist[e.first]) {
                dist[e.first] = e.second + t.second;
                q.push({e.first, dist[e.first]});
            }
        }
    }
    /*for (auto d : dist) {
        cout << d << " ";
    }*/
    vector<bool> visited(n, false);
    vector<int> record;
    record.emplace_back(0);
    function<void (vector<vector<pair<int, int>>>&, int, int, int, int)> dfs = 
        [&dfs, &ans, &dist, &edgesIdx, &visited, &record](vector<vector<pair<int, int>>>& edgeWithWeight, int cur, int parent, int target, int curDist) {

        int i;
        if (cur == target) {
            if (curDist == dist[cur]) {
                /*for (auto r : record) {
                    cout << r << "->";
                }
                cout << endl;
                */
                for (i = 1; i < record.size(); i++) {
                    ans[edgesIdx[{record[i - 1], record[i]}]] = true;
                }
            }
            return;
        }
        if (visited[cur]) {
            return;
        }
        visited[cur] = true;
        for (i = 0; i < edgeWithWeight[cur].size(); i++) {
            if (edgeWithWeight[cur][i].first != parent && 
                curDist + edgeWithWeight[cur][i].second == dist[edgeWithWeight[cur][i].first]) {
                record.emplace_back(edgeWithWeight[cur][i].first);
                dfs(edgeWithWeight, edgeWithWeight[cur][i].first, cur, target, curDist + edgeWithWeight[cur][i].second);
                record.pop_back();
            }
        }
        visited[cur] = false;
    };

    dfs(edgeWithWeight, 0, -1, n - 1, 0);
    return ans;
}


// LC1274
class Sea {
public:
    bool hasShips(vector<int> topRight, vector<int> bottomLeft)
    {
        return true;
    }
};
int countShips(Sea sea, vector<int> topRight, vector<int> bottomLeft)
{
    function<int (vector<int>, vector<int>)> find = 
        [&find, &sea](vector<int> topRight, vector<int> bottomLeft) {
        if (bottomLeft[0] > topRight[0] || bottomLeft[1] > topRight[1]) {
            return 0;
        }
        int midx, midy;
        if (sea.hasShips(topRight, bottomLeft)) {
            if (topRight == bottomLeft) {
                return 1;
            }
            midx = (topRight[0] + bottomLeft[0]) / 2;
            midy = (topRight[1] + bottomLeft[1]) / 2;
            // 分成四个区间
            return find({midx, midy}, bottomLeft) + find(topRight, {midx + 1, midy + 1}) + 
                find({midx, topRight[1]}, {bottomLeft[0], midy + 1}) + 
                find({topRight[0], midy}, {midx + 1, bottomLeft[1]});
        } else {
            return 0;
        }
    };
    return find(topRight, bottomLeft);
}


// LC576
int findPaths(int m, int n, int maxMove, int startRow, int startColumn)
{
    // dp[i][j][n] - 在(i, j)移动n步的出界方案数
    vector<vector<vector<long long>>> dp(m, vector<vector<long long>>(n, vector<long long>(maxMove + 1, 0)));
    int i, j, k, p;
    int mod = 1e9 + 7;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    if (maxMove == 0) {
        return 0;
    }
    for (i = 0; i < m; i++) {
        dp[i][0][1]++;
        dp[i][n - 1][1]++;
    }
    for (j = 0; j < n; j++) {
        dp[0][j][1]++;
        dp[m - 1][j][1]++;
    }

    for (k = 2; k <= maxMove; k++) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                for (p = 0; p < 4; p++) {
                    auto nrow = i + directions[p][0];
                    auto ncol = j + directions[p][1];
                    if (nrow < 0 || nrow >= m || ncol < 0 || ncol >= n) {
                        continue;
                    }
                    dp[i][j][k] = (dp[i][j][k] + dp[nrow][ncol][k - 1]) % mod;
                }
            }
        }
    }
    long long ans = 0;
    for (auto val : dp[startRow][startColumn]) {
        ans = (ans + val) % mod;
    }
    return ans;
}


// LC778
int swimInWater(vector<vector<int>>& grid)
{
    int i;
    int n = grid.size();
    int left, right, mid;
    bool canReach = false;
    vector<vector<int>> curGrid(n, vector<int>(n));
    vector<vector<bool>> visited(n, vector<bool>(n, false));

    auto AssignAltitude = [](vector<vector<int>>& grid, int time, vector<vector<int>>& curGrid) {
        int i, j;
        int n = grid.size();
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                curGrid[i][j] = max(time, grid[i][j]);
            }
        }
    };
    function<void (vector<vector<int>>&, int, int, int, bool&)> CanReach = 
        [&visited, &CanReach](vector<vector<int>>& curGrid, int row, int col, int curHeight, bool& canReach) {
        int i;
        int n = curGrid.size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        if (canReach) {
            return;
        }
        if (row == n - 1 && col == n - 1) {
            canReach = true;
            return;
        }

        visited[row][col] = true;
        for (i = 0; i < 4; i++) {
            auto nrow = row + directions[i][0];
            auto ncol = col + directions[i][1];
            if (nrow < 0 || nrow >= n || ncol < 0 || ncol >= n || visited[nrow][ncol] || curGrid[nrow][ncol] > curHeight) {
                continue;
            }
            CanReach(curGrid, nrow, ncol, curHeight, canReach);
        }

    };
    left = grid[0][0];
    right = 2500; 
    while (left <= right) {
        mid = (right - left) / 2 + left;
        canReach = false;
        for (i = 0; i < n; i++) {
            visited[i].assign(n, false);
        }
        AssignAltitude(grid, mid, curGrid);
        CanReach(curGrid, 0, 0, mid, canReach);
        if (canReach) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return left;
}


// LC2435
int numberOfPaths(vector<vector<int>>& grid, int k)
{
    int mod = 1e9 + 7;
    int i, j, p;
    int m = grid.size();
    int n = grid[0].size();
    int sum, r;
    // dp[i][j][k] - (i, j) 余数为k的路径总数
    vector<vector<vector<long long>>> dp(m, vector<vector<long long>>(n, vector<long long>(k, -1)));

    dp[0][0][grid[0][0] % k] = 1;
    sum = grid[0][0];
    for (j = 1; j < n; j++) {
        sum += grid[0][j];
        dp[0][j][sum % k] = 1;
    }
    sum = grid[0][0];
    for (i = 1; i < m; i++) {
        sum += grid[i][0];
        dp[i][0][sum % k] = 1;
    }

    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            for (p = 0; p < k; p++) {
                if (dp[i - 1][j][p] != -1) {
                    r = (p + grid[i][j]) % k;
                    if (dp[i][j][r] == -1) {
                        dp[i][j][r] = dp[i - 1][j][p];
                    } else {
                        dp[i][j][r] = (dp[i][j][r] + dp[i - 1][j][p]) % mod;
                    }
                }
                if (dp[i][j - 1][p] != -1) {
                    r = (p + grid[i][j]) % k;
                    if (dp[i][j][r] == -1) {
                        dp[i][j][r] = dp[i][j - 1][p];
                    } else {
                        dp[i][j][r] = (dp[i][j][r] + dp[i][j - 1][p]) % mod;
                    }
                }
            }
        }
    }
    return dp[m - 1][n - 1][0] == -1 ? 0 : dp[m - 1][n - 1][0];
}


// LC351
// 手机九键解锁
int numberOfPatterns(int m, int n)
{
    if (m > n) {
        return 0;
    }
    vector<bool> visited(10, false);
    unordered_map<int, vector<pair<int, int>>> edges; // edges[i] - {j, k} i到j需要经过k, 不经过其他点就是0
    // 暴力建图
    edges[1].emplace_back(make_pair(2, 0));
    edges[1].emplace_back(make_pair(3, 2));
    edges[1].emplace_back(make_pair(4, 0));
    edges[1].emplace_back(make_pair(5, 0));
    edges[1].emplace_back(make_pair(6, 0));
    edges[1].emplace_back(make_pair(7, 4));
    edges[1].emplace_back(make_pair(8, 0));
    edges[1].emplace_back(make_pair(9, 5));

    edges[2].emplace_back(make_pair(1, 0));
    edges[2].emplace_back(make_pair(3, 0));
    edges[2].emplace_back(make_pair(4, 0));
    edges[2].emplace_back(make_pair(5, 0));
    edges[2].emplace_back(make_pair(6, 0));
    edges[2].emplace_back(make_pair(7, 0));
    edges[2].emplace_back(make_pair(8, 5));
    edges[2].emplace_back(make_pair(9, 0));

    edges[3].emplace_back(make_pair(1, 2));
    edges[3].emplace_back(make_pair(2, 0));
    edges[3].emplace_back(make_pair(4, 0));
    edges[3].emplace_back(make_pair(5, 0));
    edges[3].emplace_back(make_pair(6, 0));
    edges[3].emplace_back(make_pair(7, 5));
    edges[3].emplace_back(make_pair(8, 0));
    edges[3].emplace_back(make_pair(9, 6));

    edges[4].emplace_back(make_pair(1, 0));
    edges[4].emplace_back(make_pair(2, 0));
    edges[4].emplace_back(make_pair(3, 0));
    edges[4].emplace_back(make_pair(5, 0));
    edges[4].emplace_back(make_pair(6, 5));
    edges[4].emplace_back(make_pair(7, 0));
    edges[4].emplace_back(make_pair(8, 0));
    edges[4].emplace_back(make_pair(9, 0));

    edges[5].emplace_back(make_pair(1, 0));
    edges[5].emplace_back(make_pair(2, 0));
    edges[5].emplace_back(make_pair(3, 0));
    edges[5].emplace_back(make_pair(4, 0));
    edges[5].emplace_back(make_pair(6, 0));
    edges[5].emplace_back(make_pair(7, 0));
    edges[5].emplace_back(make_pair(8, 0));
    edges[5].emplace_back(make_pair(9, 0));

    edges[6].emplace_back(make_pair(1, 0));
    edges[6].emplace_back(make_pair(2, 0));
    edges[6].emplace_back(make_pair(3, 0));
    edges[6].emplace_back(make_pair(4, 5));
    edges[6].emplace_back(make_pair(5, 0));
    edges[6].emplace_back(make_pair(7, 0));
    edges[6].emplace_back(make_pair(8, 0));
    edges[6].emplace_back(make_pair(9, 0));

    edges[7].emplace_back(make_pair(1, 4));
    edges[7].emplace_back(make_pair(2, 0));
    edges[7].emplace_back(make_pair(3, 5));
    edges[7].emplace_back(make_pair(4, 0));
    edges[7].emplace_back(make_pair(5, 0));
    edges[7].emplace_back(make_pair(6, 0));
    edges[7].emplace_back(make_pair(8, 0));
    edges[7].emplace_back(make_pair(9, 8));

    edges[8].emplace_back(make_pair(1, 0));
    edges[8].emplace_back(make_pair(2, 5));
    edges[8].emplace_back(make_pair(3, 0));
    edges[8].emplace_back(make_pair(4, 0));
    edges[8].emplace_back(make_pair(5, 0));
    edges[8].emplace_back(make_pair(6, 0));
    edges[8].emplace_back(make_pair(7, 0));
    edges[8].emplace_back(make_pair(9, 0));

    edges[9].emplace_back(make_pair(1, 5));
    edges[9].emplace_back(make_pair(2, 0));
    edges[9].emplace_back(make_pair(3, 6));
    edges[9].emplace_back(make_pair(4, 0));
    edges[9].emplace_back(make_pair(5, 0));
    edges[9].emplace_back(make_pair(6, 0));
    edges[9].emplace_back(make_pair(7, 8));
    edges[9].emplace_back(make_pair(8, 0));

    int ans;
    function<void (unordered_map<int, vector<pair<int, int>>>&, int, int, int)> dfs = 
        [&dfs, &visited, &ans](unordered_map<int, vector<pair<int, int>>>& edges, int cur, int cnt, int len) {
        if (cnt == len) {
            ans++;
            return;
        }
        int i;
        visited[cur] = true;
        for (i = 0; i < edges[cur].size(); i++) {
            auto p = edges[cur][i];
            if ((p.second == 0 && visited[p.first] == false) || (p.second != 0 && visited[p.second] && visited[p.first] == false)) {
                visited[p.first] = true;
                dfs(edges, p.first, cnt + 1, len);
                visited[p.first] = false;
            }
        }
        visited[cur] = false;
    };

    int tol = 0;
    int i, j;
    for (i = m; i <= n; i++) {
        for (j = 1; j <= 9; j++) {
            ans = 0;
            dfs(edges, j, 1, i);
            tol += ans;
        }
    }
    return tol;
}


// LC857
double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int k)
{
    int i, n;
    int tolQuality;
    vector<pair<int, int>> vp;
    double ans;

    n = wage.size();
    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(wage[i], quality[i]));
    }
    sort(vp.begin(), vp.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first * b.second < a.second * b.first;
    });

    priority_queue<int, vector<int>> pq; // 维持一个大小为k的最大堆
    ans = INT_MAX;
    tolQuality = 0;
    for (i = 0; i < n; i++) {
        if (pq.size() < k) {
            tolQuality += vp[i].second;
            pq.push(vp[i].second);
            if (pq.size() == k) {
                ans = min(ans, tolQuality * 1.0 * vp[i].first / vp[i].second);
            }
        } else {
            auto t = pq.top();
            if (t > vp[i].second) {
                tolQuality = tolQuality - t + vp[i].second;
                pq.pop();
                pq.push(vp[i].second);
                ans = min(ans, tolQuality * 1.0 * vp[i].first / vp[i].second);
            }
        }
    }
    return ans;
}


// LC3138
int minAnagramLength(string s)
{
    int i;
    int n = s.size();
    vector<vector<int>> prefix(n, vector<int>(26, 0));

    for (i = 0; i < n; i++) {
        if (i != 0) {
            prefix[i] = prefix[i - 1];
        }
        prefix[i][s[i] - 'a']++;
    }

    auto check = [](const vector<int>& a, const vector<int>& b) {
        int i;
        int n = a.size();
        for (i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    };
    auto minus = [](const vector<int>& a, const vector<int>& b) {
        int i;
        int n = a.size();
        vector<int> ans(n);
        for (i = 0; i < n; i++) {
            ans[i] = b[i] - a[i];
        }
        return ans;
    };
    int len = 1;
    vector<int> start, t;
    while (len <= n) {
        if (n % len) {
            len++;
            continue;
        }
        for (i = 0; i < n; i += len) {
            if (i == 0) {
                start = prefix[len - 1];
                continue;
            }
            t = minus(prefix[i - 1], prefix[i - 1 + len]);
            if (check(start, t) == false) {
                break;
            }
        }
        if (i == n) {
            break;
        }
        len++;
    }
    return len;
}


// LC3104
int maxSubstringLength(string s)
{
    int i, j;
    int n = s.size();
    vector<vector<int>> prefix(n, vector<int>(26, 0));
    vector<vector<int>> idxes(26, vector<int>(2, -1));
    for (i = 0; i < n; i++) {
        if (i != 0) {
            prefix[i] = prefix[i - 1];
        }
        prefix[i][s[i] - 'a']++;
        if (idxes[s[i] - 'a'][0] == -1) {
            idxes[s[i] - 'a'][0] = i;
            idxes[s[i] - 'a'][1] = i;
        } else {
            idxes[s[i] - 'a'][1] = i;
        }
    }

    vector<int> a, b;
    auto check = [](const vector<int>& a, const vector<int>& b) {
        int i;
        int n = a.size();
        for (i = 0; i < n; i++) {
            if (a[i] != 0 && b[i] != 0) {
                return false;
            }
        }
        return true;
    };
    auto minus = [](const vector<int>& a, const vector<int>& b) {
        int i;
        int n = a.size();
        vector<int> ans(n);
        for (i = 0; i < n; i++) {
            ans[i] = b[i] - a[i];
        }
        return ans;
    };

    // 枚举每一个字符的起始与结尾下标位置
    int ans = -1;
    for (i = 0; i < 26; i++) {
        if (idxes[i][0] == -1) {
            continue;
        }
        for (j = 0; j < 26; j++) {
            if (idxes[j][1] - idxes[i][0] + 1 == n) {
                continue;
            }
            if (idxes[j][0] >= idxes[i][0] && idxes[j][1] >= idxes[i][1]) {
                if (idxes[i][0] == 0) {
                    a = prefix[idxes[j][1]];
                    b = minus(a, prefix[n - 1]);
                } else {
                    a = minus(prefix[idxes[i][0] - 1], prefix[idxes[j][1]]);
                    b = minus(a, prefix[n - 1]);
                }
                if (check(a, b)) {
                    ans = max(ans, idxes[j][1] - idxes[i][0] + 1);
                }
            }
        }
    }

    return ans;
}


// LC3144
int minimumSubstringsInPartition(string s)
{
    int i, j;
    int n = s.size();
    vector<vector<int>> prefix(n, vector<int>(26, 0));

    for (i = 0; i < n; i++) {
        if (i != 0) {
            prefix[i] = prefix[i - 1];
        }
        prefix[i][s[i] - 'a']++;
    }

    auto check = [](const vector<int>& a) {
        int i;
        int n = a.size();
        int freq;
        bool f = false;

        for (i = 0; i < n; i++) {
            if (a[i] != 0 && f == false) {
                f = true;
                freq = a[i];
                continue;
            }
            if (f && a[i] != 0) {
                if (a[i] != freq) {
                    return false;
                }
            }
        }
        return true;
    };
    vector<int> ans(n);
    auto minus = [&ans](const vector<int>& a, const vector<int>& b) {
        int i;
        int n = a.size();
        for (i = 0; i < n; i++) {
            ans[i] = b[i] - a[i];
        }
    };
    vector<int> dp(n, 0x3f3f3f3f); // dp[i] - 以s[i]结尾的最小分割次数
    dp[0] = 1;
    for (i = 1; i < n; i++) {
        for (j = i; j >= 0; j--) {
            if (j == 0) {
                if (check(prefix[i])) {
                    dp[i] = 1;
                }
            } else {
                minus(prefix[j - 1], prefix[i]);
                if (check(ans)) {
                    dp[i] = min(dp[i], dp[j - 1] + 1);
                }
            }
        }
    }
    /* for (auto d : dp) {
        cout << d << " ";
    } */
    return dp[n - 1];
}


// LC994
int orangesRotting(vector<vector<int>>& grid)
{
    int i, j;
    int size;
    int m = grid.size();
    int n = grid[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    bool f = false;
    queue<pair<int, int>> q;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 2) {
                q.push({i, j});
            } else if (grid[i][j] == 1) {
                f = true;
            }
        }
    }
    if (q.empty()) {
        if (f) {
            return -1;
        }
        return 0;
    }

    int minute = 0;
    while (q.size()) {
        size = q.size();
        for (i = 0; i < size; i++) {
            auto p = q.front();
            q.pop();

            for (j = 0; j < 4; j++) {
                auto ni = p.first + directions[j][0];
                auto nj = p.second + directions[j][1];
                if (ni < 0 || ni >= m || nj < 0 || nj >= n || grid[ni][nj] != 1) {
                    continue;
                }
                grid[ni][nj] = 2;
                q.push({ni, nj});
            }
        }
        minute++;
    }
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                return -1;
            }
        }
    }
    return minute - 1;
}


// LC3129、LC3130
int numberOfStableArrays(int zero, int one, int limit)
{
    int mod = 1e9 + 7;
    int i, j;

    // dp[i][j][0 - 1] - i个0, j个1且最后一位是0、1的符合条件的子字符串
    vector<vector<vector<long long>>> dp(zero + 1, vector<vector<long long>>(one + 1, vector<long long>(2, 0)));

    for (i = 0; i <= zero; i++) {
        for (j = 0; j <= one; j++) {
            if (i == 0 && j == 0) {
                continue;
            }
            if (i == 0) {
                if (j <= limit) {
                    dp[i][j][1] = 1;
                }
                continue;
            }
            if (j == 0) {
                if (i <= limit) {
                    dp[i][j][0] = 1;
                }
                continue;
            }
            // 保证i - limit j - limit是0、1
            dp[i][j][0] = (dp[i - 1][j][0] + dp[i - 1][j][1] + mod - (i - limit > 0 ? dp[i - limit - 1][j][1] : 0)) % mod;
            dp[i][j][1] = (dp[i][j - 1][0] + dp[i][j - 1][1] + mod - (j - limit > 0 ? dp[i][j - limit - 1][0] : 0)) % mod;
        }
    }
    return (dp[zero][one][0] + dp[zero][one][1]) % mod;
}


// LC1202
string smallestStringWithSwaps(string s, vector<vector<int>>& pairs)
{
    int i, j;
    int n = pairs.size();
    unordered_map<int, unordered_set<int>> edges;

    for (i = 0; i < n; i++) {
        edges[pairs[i][0]].emplace(pairs[i][1]);
        edges[pairs[i][1]].emplace(pairs[i][0]);
    }
    vector<bool> visited(n, false);
    vector<pair<char, int>> record;
    vector<int> idx;

    function<void (int)> dfs = [&dfs, &visited, &record, &edges, &s](int cur) {
        visited[cur] = true;
        record.emplace_back(make_pair(s[cur], cur));
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                if (visited[it] == false) {
                    dfs(it);
                }
            }
        }
    };

    for (i = 0; i < n; i++) {
        if (visited[i] == false) {
            record.clear();
            idx.clear();
            dfs(i);
            for (auto r : record) {
                idx.emplace_back(r.second);
            }
            sort(record.begin(), record.end());
            sort(idx.begin(), idx.end());
            for (j = 0; j < record.size(); j++) {
                s[idx[j]] = record[j].first;
            }
        }
    }
    return s;
}


// LC2307
bool checkContradictions(vector<vector<string>>& equations, vector<double>& values)
{
    unordered_map<string, vector<pair<string, double>>> edges;
    unordered_set<string> node;
    map<pair<string, string>, double> weight;
    int i;
    int n = values.size();
    for (i = 0; i < n; i++) {
        edges[equations[i][0]].emplace_back(make_pair(equations[i][1], values[i]));
        weight[{equations[i][0], equations[i][1]}] = values[i];
        weight[{equations[i][1], equations[i][0]}] = 1.0 / values[i];
        weight[{equations[i][0], equations[i][0]}] = 1.0;
        weight[{equations[i][1], equations[i][1]}] = 1.0;
        node.emplace(equations[i][0]);
        node.emplace(equations[i][1]);
    }
    unordered_set<string> visited;
    bool conflict;
    function<void (string, string, double, bool&)> dfs = [&dfs, &visited, &edges, &weight](string node, string root, double val, bool& conflict) {
        if (conflict) {
            return;
        }
        if (weight.count({root, node}) && fabs(weight[{root, node}] - val) > 1e-5) {
            conflict = true;
            return;
        }
        if (visited.count(node)) {
            return;
        }
        visited.emplace(node);
        if (edges.count(node)) {
            for (auto it : edges[node]) {
                dfs(it.first, root, val * it.second, conflict);
            }
        }
    };
    for (auto it : node) {
        visited.clear();
        conflict = false;
        dfs(it, it, 1.0, conflict);
        if (conflict) {
            return true;
        }
    }
    return false;
}


// LC2328
int countPaths(vector<vector<int>>& grid)
{
    int mod = 1e9 + 7;
    int i, j;
    int m = grid.size();
    int n = grid[0].size();

    vector<vector<long long>> dp(m, vector<long long>(n, -1));
    // dp[i][j]和f(i, j) 从(i, j)开始的递增路径个数
    function<long long (int, int, int)> f = [&dp, &f, &grid, mod](int i, int j, int prevVal) {
        int m = dp.size();
        int n = dp[0].size();
        if (i < 0 || i >= m || j < 0 || j >= n) {
            return 0ll;
        }
        if (grid[i][j] <= prevVal) {
            return 0ll;
        }
        if (dp[i][j] != -1) {
            return dp[i][j];
        }
        dp[i][j] = (1 + f(i - 1, j, grid[i][j]) + f(i + 1, j, grid[i][j]) + 
                f(i, j - 1, grid[i][j]) + f(i, j + 1, grid[i][j])) % mod;
        return dp[i][j];
    };
    int ans = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            dp[i][j] = f(i, j, -1);
            ans = (ans + dp[i][j]) % mod;
        }
    }
    return ans;
}


// LC3154
int waysToReachStair(int k)
{
    // 跳了j次，退了i次
    // 1 + 2 + 4 + ... + 2^(j - 1) - i + 1(初始) = k
    // 2^j - i = k; 0 <= i <= j + 1 (j > 0)

    int i, j;
    vector<long long> base(31);
    for (i = 0; i <= 30; i++) {
        base[i] = 1ll << i;
    }
    int ans = 0;
    map<pair<int, int>, long long> combineData;
    for (j = 0; j <= 30; j++) {
        for (i = j + 1; i >= 0; i--) {
            if (base[j] - i == k) {
                ans += Combine(j + 1, i, combineData);
            }
        }
    }
    return ans;
}


// LC1542
int longestAwesome(string s)
{
    int i, j;
    int cnt;
    int n = s.size();
    int ans;
    string t, tmp;
    unordered_map<string, vector<int>> um;

    vector<string> codes(10);
    cnt = 0;
    for (i = 0; i < 10; i++) {
        codes[i] = "0000000000";
        codes[i][cnt] = '1';
        cnt++;
    }
    auto StringXor = [](string& a, string& b) {
        int i;
        int n = a.size();
        string ans(n, ' ');
        for (i = 0; i < n; i++) {
            if (a[i] == b[i]) {
                ans[i] = '0';
            } else {
                ans[i] = '1';
            }
        }
        return ans;
    };
    ans = 0;
    t = "0000000000";
    for (i = 0; i < n; i++) {
        t[s[i] - '0'] = (t[s[i] - '0'] - '0' + 1) % 2 + '0';
        um[t].emplace_back(i);
        if (t == "0000000000") {
            ans = i + 1;
        } else {
            if (um.count(t)) {
                ans = max(ans, i - um[t][0]);
            }
            for (j = 0; j < 10; j++) {
                auto tmp = StringXor(t, codes[j]);
                if (tmp == "0000000000") {
                    ans = i + 1;
                    continue;
                }
                if (um.count(tmp)) {
                    ans = max(ans, i - um[tmp][0]);
                }
            }
        }
    }
    return ans;
}


// LC664
int strangePrinter(string s)
{
    int i, j, k;
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0x3f3f3f3f));

    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = 1;
            } else if (i + 1 == j) {
                dp[i][j] = (s[i] == s[j] ? 1 : 2);
            } else {
                if (s[i] == s[i + 1]) {
                    dp[i][j] = dp[i + 1][j];
                } else {
                    dp[i][j] = 1 + dp[i + 1][j];
                    for (k = i + 1; k <= j; k++) {
                        if (s[i] == s[k]) {
                            dp[i][j] = min(dp[i][j], 
                                dp[i + 1][k - 1] + dp[k][j]);
                        }
                    }
                }
            }
        }
    }
    return dp[0][n - 1];
}


// LC919
class CBTInserter {
public:
    vector<vector<pair<TreeNode *, TreeNode *>>> nodes;
    TreeNode *root;
    int layer;
    CBTInserter(TreeNode* root)
    {
        int i, n;
        this->root = root;
        layer = 0;
        queue<pair<TreeNode *, TreeNode *>> q;
        vector<pair<TreeNode *, TreeNode *>> v;

        q.push({root, nullptr});
        while (q.size()) {
            n = q.size();
            v.clear();
            for (i = 0; i < n; i++) {
                auto t = q.front();
                q.pop();
                v.emplace_back(t);
                if (t.first->left != nullptr) {
                    q.push({t.first->left, t.first});
                }
                if (t.first->right != nullptr) {
                    q.push({t.first->right, t.first});
                }
            }
            layer++;
            nodes.emplace_back(v);
        }
    }

    int insert(int v)
    {
        int size = nodes.back().size();
        // 第layer层已满
        if (size == (1ll << (layer - 1))) {
            auto node = nodes.back()[0].first;
            node->left = new TreeNode(v);
            nodes.push_back({{node->left, node}});
            layer++;
            return node->val;
        }
        size++;
        auto node = nodes[nodes.size() - 2][size % 2 == 1 ? size / 2 : size / 2 - 1].first;
        if (node->left == nullptr) {
            node->left = new TreeNode(v);
            nodes.back().push_back({node->left, node});
        } else {
            node->right = new TreeNode(v);
            nodes.back().push_back({node->right, node});
        }
        return node->val;
    }

    TreeNode* get_root()
    {
        return this->root;
    }
};


// LC1584
int minCostConnectPoints(vector<vector<int>>& points)
{
    // 最小生成树, Prim
    int i, j;
    int cnt, ans;
    int n = points.size();
    unordered_map<int, unordered_map<int, int>> edgesWithWeight;
    vector<int> dist;

    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            edgesWithWeight[i].insert({j, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])});
            edgesWithWeight[j].insert({i, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])});
        }
    }

    vector<bool> visited(n, false);
    auto CMP = [](pair<int, int>& a, pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(CMP)> pq(CMP);

    for (auto it : edgesWithWeight[0]) {
        pq.push(it);
    }

    visited[0] = true;
    cnt = 1;
    ans = 0;
    while (cnt < n) {
        auto p = pq.top();
        pq.pop();
        if (visited[p.first]) {
            continue;
        }
        visited[p.first] = true;
        ans += p.second;
        for (auto it : edgesWithWeight[p.first]) {
            if (!visited[it.first]) {
                pq.push(it);
            }
        }
        cnt++;
    }
    return ans;
}


// LC1135
int minimumCost(int n, vector<vector<int>>& connections)
{
    int i;
    // 判断连通性
    unordered_map<int, unordered_set<int>> edges;
    for (auto c : connections) {
        edges[c[0]].emplace(c[1]);
        edges[c[1]].emplace(c[0]);
    }
    vector<bool> visited(n + 1, false);
    function<void (int)> dfs = [&edges, &visited, &dfs](int cur) {
        visited[cur] = true;
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                if (visited[it] == false) {
                    dfs(it);
                }
            }
        }
    };
    dfs(1);
    for (i = 1; i <= n; i++) {
        if (visited[i] == false) {
            return -1;
        }
    }
    // 最小生成树, Kruskal
    sort(connections.begin(), connections.end(), [](vector<int>& a, vector<int>& b) {
        return a[2] < b[2];
    });

    unordered_map<int, unordered_set<int>> um;
    unordered_map<int, int> nodeArea;
    int size = connections.size();
    int area = 1;
    int ans = 0;
    for (i = 0; i < size; i++) {
        if (nodeArea.count(connections[i][0]) && nodeArea.count(connections[i][1]) &&
            nodeArea[connections[i][0]] == nodeArea[connections[i][1]]) {
            continue;
        }
        ans += connections[i][2];
        if (nodeArea.count(connections[i][0])) {
            if (nodeArea.count(connections[i][1]) == 0) {
                nodeArea[connections[i][1]] = nodeArea[connections[i][0]];
                um[nodeArea[connections[i][1]]].emplace(connections[i][1]);
            } else {
                auto t = nodeArea[connections[i][1]];
                for (auto it : um[nodeArea[connections[i][1]]]) {
                    nodeArea[it] = nodeArea[connections[i][0]];
                    um[nodeArea[connections[i][0]]].emplace(it);
                }
                um.erase(t);
            }
        } else {
            if (nodeArea.count(connections[i][1]) == 0) {
                nodeArea[connections[i][1]] = area;
                nodeArea[connections[i][0]] = area;
                um[area].emplace(connections[i][0]);
                um[area].emplace(connections[i][1]);
                area++;
            } else {
                nodeArea[connections[i][0]] = nodeArea[connections[i][1]];
                um[nodeArea[connections[i][1]]].emplace(connections[i][0]);
            }
        }
    }
    return ans;
}


// LC548
bool splitArray(vector<int>& nums)
{
    int i, j, k;
    int n = nums.size();
    vector<int> prefix(n);
    vector<int> suffix(n);
    unordered_map<int, vector<int>> prefixSumIdx;
    unordered_map<int, vector<int>> suffixSumIdx;

    prefix[0] = nums[0];
    prefixSumIdx[prefix[0]].emplace_back(0);
    for (i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + nums[i];
        prefixSumIdx[prefix[i]].emplace_back(i);
    }
    suffix[n - 1] = nums[n - 1];
    suffixSumIdx[suffix[n - 1]].emplace_back(n - 1);
    for (i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] + nums[i];
        suffixSumIdx[suffix[i]].emplace_back(i);
    }

    int sum;
    int left, right;
    for (auto it : prefixSumIdx) {
        if (suffixSumIdx.count(it.first) == 0) {
            continue;
        }
        for (i = 0; i < it.second.size(); i++) {
            for (j = 0; j < suffixSumIdx[it.first].size(); j++) {
                if (it.second[i] + 5 >= suffixSumIdx[it.first][j]) { // 至少需要5的数组长度才能保证中间可分割
                    break;
                }
                sum = it.first;
                left = it.second[i] + 2;
                right = suffixSumIdx[it.first][j] - 2;
                for (k = left + 1; k <= right - 1; k++) {
                    if (prefix[k - 1] - prefix[left - 1] == sum && 
                        prefix[k - 1] - prefix[left - 1] == prefix[right] - prefix[k]) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}


// LC1482
int minDays(vector<int>& bloomDay, int m, int k)
{
    int i;
    int n = bloomDay.size();
    vector<int> flowers(n, 0);
    int cntSet, cnt;
    int left = *min_element(bloomDay.begin(), bloomDay.end());
    int right = *max_element(bloomDay.begin(), bloomDay.end());
    int mid;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        flowers.assign(n, 0);
        for (i = 0; i < n; i++) {
            if (bloomDay[i] <= mid) {
                flowers[i] = 1;
            }
        }
        cntSet = 0;
        cnt = 0;
        for (i = 0; i < n; i++) {
            if (flowers[i]) {
                cnt++;
            } else {
                cntSet += cnt / k;
                cnt = 0;
                if (cntSet >= m) {
                    right = mid - 1;
                    break;
                }
            }
        }
        cntSet += cnt / k;
        if (cntSet >= m) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    if (left == *max_element(bloomDay.begin(), bloomDay.end()) + 1) {
        return -1;
    }
    return left;
}


// LC2136
int earliestFullBloom(vector<int>& plantTime, vector<int>& growTime)
{
    vector<pair<int, int>> vp;
    int i;
    int n = plantTime.size();

    for (i = 0; i < n; i++) {
        vp.emplace_back(make_pair(plantTime[i], growTime[i]));
    }
    sort(vp.begin(), vp.end(), [](pair<int, int>& a, pair<int, int>& b) {
        return a.second > b.second;
    });

    int ans = 0;
    int sumOfPlantTime = 0;
    for (i = 0; i < n; i++) {
        sumOfPlantTime += vp[i].first;
        ans = max(ans, sumOfPlantTime + vp[i].second);
    }
    return ans;
}


// LC1168
int minCostToSupplyWater(int n, vector<int>& wells, vector<vector<int>>& pipes)
{
    // 把每一个wills看作一个虚拟节点到每个点的cost
    int i;
    unordered_map<int, vector<pair<int, int>>> edges; // wells1 - {wells2, dist}

    for (auto p : pipes) {
        edges[p[0]].emplace_back(make_pair(p[1], p[2]));
        edges[p[1]].emplace_back(make_pair(p[0], p[2]));
    }
    for (i = 0; i < n; i++) {
        edges[0].emplace_back(make_pair(i + 1, wells[i]));
        edges[i + 1].emplace_back(make_pair(0, wells[i]));
    }

    auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);
    vector<bool> visited(n + 1, false);
    
    int cost, curWell;
    bool find = false;

    cost = 0;
    visited[0] = true;
    curWell = 0;
    while (1) {
        if (edges.count(curWell)) {
            for (auto it : edges[curWell]) {
                if (visited[it.first] == false) {
                    pq.push(it);
                }
            }
            pair<int, int> tp;
            find = false;
            while (!pq.empty()) {
                tp = pq.top();
                if (visited[tp.first] || tp.second > wells[tp.first - 1]) {
                    pq.pop();
                    continue;
                } else {
                    pq.pop();
                    find = true;
                    break;
                }
            }
            if (find == false) {
                break;
            }
            cost += tp.second;
            curWell = tp.first;
            visited[curWell] = true;
        } else {
            break;
        }
    }
    return cost;
}


// LC3180
int maxTotalReward(vector<int>& rewardValues)
{
    sort(rewardValues.begin(), rewardValues.end());

    int i, j;
    int n = rewardValues.size();
    int ans;
    // dp[i][j] - 前i个rewardValues能否组成j
    int maxPoint = rewardValues[n - 1] * 2 - 1;
    vector<vector<bool>> dp(n + 1, vector<bool>(maxPoint + 1, false));

    dp[0][0] = true;
    ans = 0;
    for (i = 1; i <= n; i++) {
        for (j = 0; j <= maxPoint; j++) {
            if (j >= rewardValues[i - 1] * 2) {
                break;
            }
            dp[i][j] = (dp[i][j] || dp[i - 1][j]); // 不选rewardValues[i - 1]
            if (dp[i][j]) {
                ans = max(ans, j);
                continue;
            }
            if (j - rewardValues[i - 1] >= 0 && j - rewardValues[i - 1] < rewardValues[i - 1]) {
                dp[i][j] = (dp[i][j] || dp[i - 1][j - rewardValues[i - 1]]);
            }
            if (dp[i][j]) {
                ans = max(ans, j);
            }
        }
    }
    return ans;
}


// LC2106
int maxTotalFruits(vector<vector<int>>& fruits, int startPos, int k)
{
    vector<int> pos(2e5 + 1, 0);

    for (auto f : fruits) {
        pos[f[0]] = f[1];
    }
    int i;
    int n = pos.size();
    vector<int> prefix(n);

    prefix[0] = pos[0];
    for (i = 1; i < n; i++) {
        prefix[i] = pos[i] + prefix[i - 1];
    }

    int ans = 0;
    int left, right, stepLeft;

    // 向左走
    for (i = 0; i <= k; i++) {
        if (startPos - i < 0) {
            break;
        }
        left = startPos - i;
        right = left + k - i > startPos ? left + k - i : startPos;
        if (right >= n) {
            right = n - 1;
        }
        if (left == 0) {
            ans = max(ans, prefix[right]);
        } else {
            ans = max(ans, prefix[right] - prefix[left - 1]);
        }
    }
    // 向右走
    for (i = 0; i <= k; i++) {
        if (startPos + i == n) {
            break;
        }
        right = startPos + i;
        left = right - (k - i) < startPos ? right - (k - i) : startPos;
        if (left < 0) {
            left = 0;
        }
        if (left == 0) {
            ans = max(ans, prefix[right]);
        } else {
            ans = max(ans, prefix[right] - prefix[left - 1]);
        }
    }
    return ans;
}


// LC261
bool validTree(int n, vector<vector<int>>& edges)
{
    unordered_map<int, unordered_set<int>> edge;
    for (auto e : edges) {
        edge[e[0]].emplace(e[1]);
        edge[e[1]].emplace(e[0]);
    }
    vector<int> visited(n, 0);
    function<void (int, int, bool&)> dfs = [&dfs, &edge, &visited](int cur, int parent, bool &loop) {
        if (loop) {
            return;
        }
        if (visited[cur] == 1) {
            loop = true;
            return;
        }
        visited[cur] = 1;
        if (edge.count(cur)) {
            for (auto it : edge[cur]) {
                if (it != parent) {
                    dfs(it, cur, loop);
                }
            }
        }
        visited[cur] = 2;
    };

    bool loop = false;
    dfs(0, -1, loop);
    if (loop) {
        return false;
    }
    for (auto v : visited) {
        if (v == 0) {
            return false;
        }
    }
    return true;
}


// LC3164
long long numberOfPairs(vector<int>& nums1, vector<int>& nums2, int k)
{
    int i;
    int a, b;
    unordered_map<int, long long> factors;

    for (auto num : nums1) {
        if (num % k != 0) {
            continue;
        } else {
            num /= k;
        }
        for (i = 1; i <= static_cast<int>(sqrt(num)); i++) {
            if (num % i == 0) {
                a = num / i;
                factors[i]++;
                if (a != i) {
                    factors[a]++;
                }
            }
        }
    }
    long long ans = 0;
    for (auto num : nums2) {
        if (factors.count(num)) {
            ans += factors[num];
        }
    }
    return ans;
}


// LC843
class Master {
public:
    int guess(string word)
    {
        return 0;
    }
};

void findSecretWord(vector<string>& words, Master& master)
{
    int i;
    int len, cnt_s;
    int n = words.size();
    unordered_set<string> unvisited;
    vector<bool> visited(n, false);
    unordered_map<string, int> wordIdx;
    unordered_map<string, unordered_map<int, unordered_set<string>>> wordsSimilarity;

    for (i = 0; i < n; i++) {
        wordIdx[words[i]] = i;
    }
    auto CreateSimilarity = [&visited, &wordsSimilarity](vector<string>& words) {
        int i, j, k;
        int len, cnt_s;
        int n = words.size();
        // unordered_map<string, unordered_map<int, unordered_set<string>>> wordsSimilarity;
        wordsSimilarity.clear();
        for (i = 0; i < n; i++) {
            if (visited[i]) {
                continue;
            }
            for (j = 0; j < n; j++) {
                if (visited[j]) {
                    continue;
                }
                if (i == j) {
                    continue;
                }
                len = words[i].size();
                cnt_s = 0;
                for (k = 0; k < len; k++) {
                    if (words[i][k] == words[j][k]) {
                        cnt_s++;
                    }
                }
                wordsSimilarity[words[i]][cnt_s].emplace(words[j]);
            }
        }
        // return wordsSimilarity;
    };

    string word;
    unordered_set<string> wordset;
    int maxLen;
    while (1) {
        CreateSimilarity(words);
        // 找到wordsSimilarity中最大的word集合
        maxLen = 0;
        for (auto it : wordsSimilarity) {
            len = 0;
            for (auto p : it.second) {
                len += p.second.size();
            }
            if (len > maxLen) {
                maxLen = len;
                word = it.first;
            }
        }
        int t = master.guess(word);
        if (t == 6) {
            return;
        }
        // 大于等于similarity的都不符合
        int similarity = t + 1;
        visited[wordIdx[word]] = true;
        while (similarity < 6) {
            if (wordsSimilarity[word].count(similarity)) {
                for (auto s : wordsSimilarity[word][similarity]) {
                    visited[wordIdx[s]] = true;
                }
            }
            similarity++;
        }
        // wordset以外的都不符合
        visited[wordIdx[word]] = true;
        wordset = wordsSimilarity[word][t];
        for (i = 0; i < n; i++) {
            if (wordset.count(words[i]) == 0) {
                visited[i] = true;
            }
        }
        cnt_s = 0;
        int idx;
        for (i = 0; i < n; i++) {
            if (!visited[i]) {
                idx = i;
                cnt_s++;
            }
        }
        if (cnt_s == 1) {
            master.guess(words[idx]);
            return;
        }
    }
}


// LC3177
int maximumLength(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();
    int pos;
    int ans = 1;

    vector<vector<int>> dp(n, vector<int>(k + 1, -1));
    vector<vector<int>> record(k + 1, vector<int>(2, 0)); // 记录改变k次得到最大长度的子序列和次大长度最后一个数字的下标
    unordered_map<int, int> idx; // 记录上一个nums[i]的下标

    idx[nums[0]] = 0;
    dp[0][0] = 1;
    for (i = 1; i < n; i++) {
        if (idx.count(nums[i])) {
            dp[i][0] = dp[idx[nums[i]]][0] + 1;
            for (j = 1; j <= k; j++) {
                if (record[j - 1][0] == idx[nums[i]]) {
                    pos = record[j - 1][1];
                } else {
                    pos = record[j - 1][0];
                }
                dp[i][j] = max(dp[idx[nums[i]]][j] + 1, dp[pos][j - 1] + 1);
            }
        } else {
            dp[i][0] = 1;
            for (j = 1; j <= k; j++) {
                pos = record[j - 1][0];
                dp[i][j] = dp[pos][j - 1] + 1;
            }
        }
        // 更新record的值
        for (j = 0; j <= k; j++) {
            if (dp[record[j][0]][j] <= dp[i][j]) {
                record[j][1] = record[j][0];
                record[j][0] = i;
            } else if (dp[record[j][1]][j] <= dp[i][j]) {
                record[j][1] = i;
            }
            ans = max(ans, dp[i][j]);
        }
        idx[nums[i]] = i;
    }
    return ans;
}


// LC5
string longestPalindrome_LC5(string s)
{
    int i, j;
    int n = s.size();
    int start, maxLen;
    vector<vector<bool>> dp(n, vector<bool>(n, false));

    maxLen = 0;
    start = -1;
    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (s[i] != s[j]) {
                continue;
            }
            if (i == j || i + 1 == j) {
                dp[i][j] = true;
            } else if (dp[i + 1][j - 1]) {
                dp[i][j] = true;
            }
            if (dp[i][j] && j - i + 1 > maxLen) {
                start = i;
                maxLen = j - i + 1;
            }
        }
    }
    string ans = s.substr(start, maxLen);
    return ans;
}


// LC3102
int minimumDistance_LC3102(vector<vector<int>>& points)
{
    int i;
    int n = points.size();
    /*
    |x1 - x2| + |y1 - y2| = max((x1 - x2), (x2 - x1)) + max((y1 - y2), (y2 - y1))

    = max(x1 - x2 + y1 - y2, x1 - x2 + y2 - y1, x2 - x1 + y1 - y2, x2 - x1 + y2 - y1)

    = max((x1 + y1) - (x2 + y2), x1 - y1 - (x2 - y2), -(x1 - y1) + (x2 - y2), -(x1 + y1) + (x2 + y2))

    令 x1 + y1 = X1, x1 - y1 = X2,
    x2 + y2 = Y1, x2 - y2 = Y2,
    则上式
    = max(X1 - Y1, X2 - Y2, -X2 + Y2, -X1 + Y1) = max(|X1 - Y1|, |X2 - Y2|)
    */
    int ans = INT_MAX;
    vector<pair<int, int>> vp, vpt;
    for (i = 0; i < n; i++) {
        vp.push_back({points[i][0] + points[i][1], points[i][0] - points[i][1]});
    }
    auto t = vp;
    sort (t.begin(), t.end());

    // 分情况讨论
    // 看pair.first
    // 去掉第一项
    auto dist1 = t[n - 1].first - t[1].first;
    for (i = 1; i < n; i++) {
        vpt.emplace_back(t[i]);
    }
    sort(vpt.begin(), vpt.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.second < b.second;
    });
    auto dist2 = vpt[vpt.size() - 1].second - vpt[0].second;
    ans = min(ans, max(dist1, dist2));

    // 去掉最后一项
    vpt.clear();
    dist1 = t[n - 2].first - t[0].first;
    for (i = 0; i < n - 1; i++) {
        vpt.emplace_back(t[i]);
    }
    sort(vpt.begin(), vpt.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.second < b.second;
    });
    dist2 = vpt[vpt.size() - 1].second - vpt[0].second;
    ans = min(ans, max(dist1, dist2));

    // 看pair.second
    sort(t.begin(), t.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.second < b.second;
    });
    // 去掉第一项
    vpt.clear();
    dist1 = t[n - 1].second - t[1].second;
    for (i = 1; i < n; i++) {
        vpt.emplace_back(t[i]);
    }
    sort(vpt.begin(), vpt.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.first < b.first;
    });
    dist2 = vpt[vpt.size() - 1].first - vpt[0].first;
    ans = min(ans, max(dist1, dist2));

    // 去掉最后一项
    vpt.clear();
    dist1 = t[n - 2].second - t[0].second;
    for (i = 0; i < n - 1; i++) {
        vpt.emplace_back(t[i]);
    }
    sort(vpt.begin(), vpt.end(), [](const pair<int, int>& a, const pair<int, int>& b){
        return a.first < b.first;
    });
    dist2 = vpt[vpt.size() - 1].first - vpt[0].first;
    ans = min(ans, max(dist1, dist2));

    return ans;
}


// LC721
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts)
{
    int i, j;
    int n;
    int area, cur, prev;
    unordered_map<int, set<string>> mails;
    unordered_map<int, string> user;
    unordered_map<string, int> accountArea;

    n = accounts.size();
    area = 1;
    for (i = 1; i < accounts[0].size(); i++) {
        mails[area].emplace(accounts[0][i]);
        accountArea[accounts[0][i]] = area;
        user[area] = accounts[0][0];
    }
    area++;
    for (i = 1; i < n; i++) {
        cur = -1;
        for (j = 1; j < accounts[i].size(); j++) {
            if (accountArea.count(accounts[i][j])) {
                if (cur == -1) {
                    cur = accountArea[accounts[i][j]];
                } else if (cur != accountArea[accounts[i][j]]) {
                    prev = accountArea[accounts[i][j]];
                    auto s = mails[prev];
                    mails.erase(prev);
                    for (auto it : s) {
                        accountArea[it] = cur;
                        mails[cur].emplace(it);
                    }
                }
            }
        }
        if (cur == -1) {
            for (j = 1; j < accounts[i].size(); j++) {
                mails[area].emplace(accounts[i][j]);
                accountArea[accounts[i][j]] = area;
            }
            user[area] = accounts[i][0];
            area++;
        } else {
            for (j = 1; j < accounts[i].size(); j++) {
                mails[cur].emplace(accounts[i][j]);
                accountArea[accounts[i][j]] = cur;
            }
        }
    }
    vector<vector<string>> ans;
    vector<string> t;
    for (auto it : mails) {
        t.clear();
        t.emplace_back(user[it.first]);
        for (auto mail : it.second) {
            t.emplace_back(mail);
        }
        ans.emplace_back(t);
    }
    return ans;
}


// LC1163
// 后缀数组
string lastSubstring(string s)
{
    int i, j;
    int n = s.size();
    int len;
    string ans;

    i = 0;
    j = 1;
    len = 0;
    while (i + len < n && j + len < n) {
        if (s[i + len] == s[j + len]) {
            len++;
        } else if (s[i + len] > s[j + len]) {
            j = j + len + 1;
            len = 0;
        } else {
            i = i + len + 1;
            if (i >= j) {
                j = i + 1;
            }
            len = 0;
        }
    }
    ans = s.substr(i);
    return ans;
}


// LC3196
long long maximumTotalCost(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<vector<long long>> dp(n, vector<long long>(2, 0)); // dp[i][0] 以i结尾且nums[i] 为正

    dp[0][0] = nums[0];
    dp[0][1] = -0x3f3f3f3f;

    for (i = 1; i < n; i++) {
        dp[i][0] = max(dp[i - 1][0] + nums[i], dp[i - 1][1] + nums[i]);
        dp[i][1] = dp[i - 1][0] - nums[i];
    }

    return max(dp[n - 1][0], dp[n - 1][1]);
}


// LC3221
long long maxScore(vector<int>& nums)
{
    // i < k < j
    // (k - i) * nums[k] + (j - k) * nums[j] > (j - i) * nums[j] =>
    // (k - i) * (nums[k] - nums[j]) > 0

    int i, idx;
    int n = nums.size();
    long long ans = 0;

    idx = n - 1;
    for (i = n - 2; i >= 0; i--) {
        if (nums[i] > nums[idx]) {
            ans += (idx - i) * static_cast<long long>(nums[idx]);
            idx = i;
        }
    }
    ans += (idx - 0) * static_cast<long long>(nums[idx]);
    return ans;
}


// LC1838
int maxFrequency(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int ans;
    int left, right, mid;
    long long need;
    vector<long long> prefix(n);

    sort(nums.begin(), nums.end());

    prefix[0] = nums[0];
    for (i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + nums[i];
    }
    ans = 1;
    for (i = n - 1; i >= 0; i--) {
        left = 0;
        right = i;
        // 所求left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (mid == 0) {
                need = nums[i] * static_cast<long long>(i - mid + 1) - prefix[i];
            } else {
                need = nums[i] * static_cast<long long>(i - mid + 1) - (prefix[i] - prefix[mid - 1]);
            }
            if (need <= k) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        ans = max(ans, i - left + 1);
    }
    return ans;
}


// LC756
void CreatePyramid(string& s, int idx, string& cur, unordered_map<string, vector<char>>& edge, bool& find)
{
    if (find) {
        return;
    }
    if (idx == s.size()) {
        if (cur.size() == 1) {
            find = true;
            return;
        }
        string newCur;
        // 满足条件, 继续递归
        CreatePyramid(cur, 1, newCur, edge, find);
        return;
    }
    string t = s.substr(idx - 1, 2);
    if (edge.count(t)) {
        for (auto it : edge[t]) {
            cur.push_back(it);
            // 剪枝
            if (cur.size() == 1 || (cur.size() > 1 && edge.count(cur.substr(cur.size() - 2)) == 1)) {
                CreatePyramid(s, idx + 1, cur, edge, find);
            }
            cur.pop_back();
        }
    }
}
bool pyramidTransition(string bottom, vector<string>& allowed)
{
    unordered_map<string, vector<char>> edge;
    for (auto a : allowed) {
        edge[a.substr(0, 2)].emplace_back(a[2]);
    }

    string cur;
    bool find = false;

    CreatePyramid(bottom, 1, cur, edge, find);

    return find;
}


// LC2547
int minCost_LC2547(vector<int>& nums, int k)
{
    int i, j;
    int n = nums.size();
    int len;
    vector<int> cnt(n, 0);
    vector<long long> dp(n, LLONG_MAX);

    dp[0] = k;
    cnt[nums[0]] = 1;
    len = 0;
    for (i = 1; i < n; i++) {
        cnt[nums[i]]++;
        if (cnt[nums[i]] == 2) {
            len += 2;
        } else if (cnt[nums[i]] > 2) {
            len++;
        }
        auto tCnt = cnt;
        auto tLen = len;
        dp[i] = k + len;
        for (j = 0; j < i; j++) {
            if (tCnt[nums[j]] == 2) {
                tCnt[nums[j]]--;
                tLen -= 2;
            } else if (tCnt[nums[j]] > 2) {
                tCnt[nums[j]]--;
                tLen--;
            }
            dp[i] = min(dp[i], dp[j] + k + tLen);
        }
    }
    return dp[n - 1];
}


// LC2318
int distinctSequences(int n)
{
    // dp[i][p1][p2] - 第i次投, 其最后一个数是p1, 倒数第二个数是p2的总序列数
    // 所求sum(dp[i]);
    vector<vector<vector<long long>>> dp(n + 1, vector<vector<long long>>(7, vector<long long>(7 , 0)));
    long long mod = 1e9 + 7;
    int p, i, j, k;

    if (n == 1) {
        // dp[1][i][j] 无意义, 不存在
        return 6;
    }
    for (i = 1; i <= 6; i++) {
        for (j = 1; j <= 6; j++) {
            if (i != j && gcd(i, j) == 1) {
                dp[2][i][j]++;
            }
        }
    }
    for (p = 2; p <= n - 1; p++) {
        for (i = 1; i <= 6; i++) {
            for (j = 1; j <= 6; j++) {
                if (i == j || gcd(i, j) != 1) {
                    continue;
                }
                for (k = 1; k <= 6; k++) {
                    if (i == k || j == k || gcd(j, k) != 1) {
                        continue;
                    }
                    dp[p + 1][i][j] = (dp[p][j][k] + dp[p + 1][i][j]) % mod;
                }
            }
        }
    }
    long long ans = 0;
    for (i = 1; i <= 6; i++) {
        for (j = 1; j <= 6; j++) {
            ans = (ans + dp[n][i][j]) % mod;
        }
    }
    return ans;
}


// LC2067
int equalCountSubstrings(string s, int count)
{
    auto Check = [](vector<int>& alphabet, int count) {
        int i;
        int n = alphabet.size();
        bool find = false;
        for (i = 0; i < n; i++) {
            if (alphabet[i] != 0) { 
                if (alphabet[i] < count) {
                    return -1;
                } else if (alphabet[i] > count) {
                    return 1;
                }
                find = true;
            }
        }
        if (find) {
            return 0;
        }
        return -2;
    };

    int i, k;
    int n = s.size();
    int len, ans;
    vector<int> alphabet(26, 0);

    k = 1;
    ans = 0;
    while (k <= 26) {
        len = count * k;
        if (len > n) {
            break;
        }
        alphabet.assign(26, 0);
        for (i = 0; i < n; i++) {
            if (i < len) {
                alphabet[s[i] - 'a']++;
                if (i == len - 1 && Check(alphabet, count) == 0) {
                    ans++;
                }
            } else {
                alphabet[s[i] - 'a']++;
                alphabet[s[i - len] - 'a']--;
                if (Check(alphabet, count) == 0) {
                    ans++;
                }
            }
        }
        k++;
    }
    return ans;
}


// LC2953 (类似LC2067)
int countCompleteSubstrings(string word, int k)
{
    auto Check = [](vector<int>& alphabet, int count) {
        int i;
        int n = alphabet.size();
        bool find = false;
        for (i = 0; i < n; i++) {
            if (alphabet[i] != 0) { 
                if (alphabet[i] < count) {
                    return -1;
                } else if (alphabet[i] > count) {
                    return 1;
                }
                find = true;
            }
        }
        if (find) {
            return 0;
        }
        return -2;
    };

    int i, j;
    int n = word.size();
    int len, ans;
    int suit;
    vector<int> alphabet(26, 0);

    j = 1;
    suit = ans = 0;
    while (j <= 26) {
        len = k * j;
        if (len > n) {
            break;
        }
        alphabet.assign(26, 0);
        suit = 1;
        for (i = 0; i < n; i++) {
            if (i > 0) {
                if (abs(word[i] - word[i - 1]) > 2) {
                    suit--;
                } else {
                    suit++;
                }
            }
            if (i < len) {
                alphabet[word[i] - 'a']++;
                if (i == len - 1 && Check(alphabet, k) == 0 && suit == len) {
                    // cout << "(" << i - len + 1 << ", " << i << ")" << endl;
                    ans++;
                }
            } else {
                alphabet[word[i] - 'a']++;
                alphabet[word[i - len] - 'a']--;
                if (abs(word[i - len] - word[i - len + 1]) > 2) {
                    suit++;
                } else {
                    suit--;
                }
                if (Check(alphabet, k) == 0 && suit == len) {
                    // cout << "(" << i - len + 1 << ", " << i << ")" << endl;
                    ans++;
                }
            }
        }
        j++;
    }
    return ans;
}


// LC2250
vector<int> countRectangles(vector<vector<int>>& rectangles, vector<vector<int>>& points)
{
    sort(rectangles.begin(), rectangles.end());

    int i, j;
    int n = rectangles.size();
    // 大于等于h的rectangles[x][1]的集合heightsIdx[h]
    vector<vector<int>> heightsIdx(101); // h <= 100
    for (j = 0; j <= 100; j++) {
        for (i = 0; i < n; i++) {
            if (j <= rectangles[i][1]) {
                heightsIdx[j].emplace_back(i);
            }
        }
    }

    vector<int> ans;
    int left, right, mid;
    for (auto p : points) {
        left = 0;
        right = n - 1;
        // 第一个大于等于p[0]的rectangles[x][0], 所求left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (rectangles[mid][0] < p[0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        // cout << left << endl;
        // 再在heightsIdx[p[1]]找符合[left, n - 1]的所有下标
        auto target = left;
        left = 0;
        right = heightsIdx[p[1]].size() - 1;
        // 所求left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (heightsIdx[p[1]][mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        ans.emplace_back(heightsIdx[p[1]].size() - left);
    }
    return ans;
}


// LC1616
bool checkPalindromeFormation(string a, string b)
{
    int i, j;
    int n = a.size();
    string t, s;

    if (IsPalindrome(a) || IsPalindrome(b)) {
        return true;
    }

    i = 0;
    j = n - 1;
    if (a[i] == b[j]) {
        while (i < j) {
            if (a[i] == b[j]) {
                i++;
                j--;
            } else {
                break;
            }
        }
        t = a.substr(0, i) + b.substr(i);
        s = a.substr(0, j + 1) + b.substr(j + 1);
        // cout << t << " " << s << endl;
        if (IsPalindrome(t) || IsPalindrome(s)) {
            return true;
        }
    }

    i = 0;
    j = n - 1;
    if (a[j] == b[i]) {
        while (i < j) {
            if (a[j] == b[i]) {
                i++;
                j--;
            } else {
                break;
            }
        }
        t = b.substr(0, i) + a.substr(i);
        s = b.substr(0, j + 1) + a.substr(j + 1);
        // cout << t << " " << s << endl;
        if (IsPalindrome(t) || IsPalindrome(s)) {
            return true;
        }
    }
    return false;
}


// LC3244
vector<int> shortestDistanceAfterQueries(int n, vector<vector<int>>& queries)
{
    int i;
    int size;
    unordered_map<int, set<int, greater<>>> edges;
    set<pair<int, int>> specialRoads;

    size = queries.size();
    for (i = 0; i < size; i++) {
        edges[queries[i][0]].emplace(queries[i][1]);
    }

    // 从后往前找
    vector<int> ans(size);
    int nextPos;
    int pos = 0;
    int cnt = 0;
    while (pos != n - 1) {
        if (edges.count(pos) != 0) {
            nextPos = *edges[pos].begin();
            specialRoads.insert({pos, nextPos});
            pos = nextPos;
        } else {
            pos++;
        }
        cnt++;
    }
    ans[size - 1] = cnt;

    for (i = size - 1; i >= 1; i--) {
        if (edges[queries[i][0]].size() == 1) {
            edges.erase(queries[i][0]);
        } else {
            edges[queries[i][0]].erase(queries[i][1]);
        }
        if (specialRoads.count({queries[i][0], queries[i][1]}) == 0) {
            ans[i - 1] = cnt;
            specialRoads.erase({queries[i][0], queries[i][1]});
        } else {
            specialRoads.erase({queries[i][0], queries[i][1]});
            // 这条捷径不能走了cnt减一
            cnt--;
            pos = queries[i][0];
            while (pos != queries[i][1]) {
                if (edges.count(pos) != 0) {
                    nextPos = *edges[pos].begin();
                    specialRoads.insert({pos, nextPos});
                    // cout << pos << " " << nextPos << endl;
                    pos = nextPos;
                } else {
                    pos++;
                }
                cnt++;
            }
            ans[i - 1] = cnt;
        }
    }
    return ans;
}


// LC3247
int subsequenceCount(vector<int>& nums)
{
    // dp[i][0] - 第i位结尾和为偶数的子序列数量
    int i;
    int n = nums.size();
    int mod = 1e9 + 7;
    // 和为偶/奇的子序列个数
    long long sum0, sum1;
    vector<vector<long long>> dp(n, vector<long long>(2, 0));

    if (nums[0] % 2 == 0) {
        dp[0][0] = 1;
        sum0 = 1;
        sum1 = 0;
    } else {
        dp[0][1] = 1;
        sum0 = 0;
        sum1 = 1;
    }
    for (i = 1; i < n; i++) {
        if (nums[i] % 2 == 0) {
            dp[i][1] = sum1;
            dp[i][0] = (sum0 + 1) % mod;
            sum1 = (dp[i][1] + sum1) % mod;
            sum0 = (sum0 + dp[i][0]) % mod;
        } else {
            dp[i][1] = (sum0 + 1) % mod;
            dp[i][0] = sum1;
            sum1 = (sum1 + dp[i][1]) % mod;
            sum0 = (dp[i][0] + sum0) % mod;
        }
    }
    return sum1;
}


// LC3249
int countGoodNodes(vector<vector<int>>& edges)
{
    vector<vector<int>> e(edges.size() + 1);
    // unordered_map<int, set<int>> e;
    for (auto ed : edges) {
        e[ed[0]].emplace_back(ed[1]);
        e[ed[1]].emplace_back(ed[0]);
    }
    unordered_map<int, int> cnt;
    int ans = 0;
    function<int (int, int)> f = [&f, &e, &cnt, &ans](int node, int parent) {
        if (e[node].size() == 1 && *e[node].begin() == parent) {
            cnt[node] = 0;
            ans++;
            return 0;
        }
        int tol = 0;
        int data = -1;
        bool flag = true;
        for (auto it : e[node]) {
            if (it != parent) {
                if (cnt.count(it) == 0) {
                    cnt[it] = f(it, node) + 1;
                }
                tol += cnt[it];
                if (data == -1) {
                    data = cnt[it];
                } else if (data != cnt[it]) {
                    flag = false;
                }
            }
        }
        if (flag) {
            ans++;
        }
        cnt[node] = tol;
        return tol;
    };
    f(0, -1);
    // for (auto it : cnt) printf("%d %d\n", it.first, it.second);
    return ans;
}


// LC3251
int countOfPairs(vector<int>& nums)
{
    int i, j, k;
    int n = nums.size();
    int mod = 1e9 + 7;
    int maxVal;
    size_t len = *max_element(nums.begin(), nums.end()) + 1;
    // dp[i][j] - arr1在i处取j的pair数
    vector<vector<long long>> dp(n, vector<long long>(len, 0));
    // prefix[i][j] - arr1在i处取0 - j的pair数前缀和
    vector<vector<long long>> prefix(n, vector<long long>(len, 0));

    dp[0][0] = 1;
    prefix[0][0] = 1;
    for (i = 1; i <= nums[0]; i++) {
        dp[0][i] = 1;
        prefix[0][i] = prefix[0][i - 1] + 1;
    }
    for (i = 1; i < n; i++) {
        for (j = 0; j <= nums[i]; j++) {
            // i位arr2的值
            k = nums[i] - j;
            // k要小于等于nums[i - 1] - j' (0 <= j' <= j) 且 nums[i - 1] - j' >= 0 => j' <= nums[i - 1]
            // k <= nums[i - 1] - j' => j' <= nums[i - 1] - k
            // j'能取的最大值maxVal
            maxVal = min(nums[i - 1], nums[i - 1] - k);
            if (maxVal < 0) {
                continue;
            } else if (maxVal > j) {
                maxVal = j;
            }
            dp[i][j] = (dp[i][j] + prefix[i - 1][maxVal]) % mod;
            if (j == 0) {
                prefix[i][j] = dp[i][j];
            } else {
                prefix[i][j] = (prefix[i][j - 1] + dp[i][j]) % mod;
            }
        }
    }

    long long ans = 0;
    for (k = 0; k < len; k++) {
        ans = (ans + dp[n - 1][k]) % mod;
    }
    return ans;
}


// LC1842
string nextPalindrome(string num)
{
    int n = num.size();

    string s = num.substr(0, n / 2);
    string t = s;

    // 偷懒用stl
    next_permutation(t.begin(), t.end());
    if (t <= s) {
        return "";
    } 
    string ans = t;
    if (n % 2 == 1) {
        ans += num[n / 2];
    }
    reverse(t.begin(), t.end());
    ans += t;
    return ans;
}


// LC3253
int minimumCost(string target, vector<string>& words, vector<int>& costs) 
{
    int i, j;
    int n = target.size();
    int m = costs.size();
    vector<int> dp(n + 1, INT_MAX);

    dp[0] = 0;
    for (i = 1; i <= n; i++) {
        for (j = 0; j < m; j++) {
            if (i >= words[j].size() && words[j] == target.substr(i - words[j].size(), words[j].size()) &&
                dp[i - words[j].size()] != INT_MAX) {
                dp[i] = min({dp[i], dp[i - words[j].size()] + costs[j]});
            }
        }
    }
    return dp[n] == INT_MAX ? -1 : dp[n];
}


// LC3209
long long countSubarrays_LC3209(vector<int>& nums, int k)
{
    int i, j, idx;
    int t;
    int n = nums.size();
    vector<int> base(30);
    vector<vector<int>> prefix(n, vector<int>(30, 0));

    idx = 0;
    t = k;
    while (t) {
        base[idx] = t % 2;
        t /= 2;
        idx++;
    }
    vector<int> v;
    for (i = 0; i < n; i++) {
        idx = 0;
        t = nums[i];
        while (idx < 30) {
            if (i == 0) {
                prefix[i][idx] = t % 2;
            } else {
                prefix[i][idx] = prefix[i - 1][idx] + t % 2;
            }
            t /= 2;
            idx++;
        }
    }

    long long ans = 0;
    int left, right, mid;
    int a, b;
    for (i = 0; i < n; i++) {
        if (nums[i] < k) {
            continue;
        }
        // 找两个区间 a 和 b [l, a]的按位与和[l, b]的按位与都为k则a到b之间的子数组都符合要求
        // 右边界
        left = i;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (nums[mid] < k) {
                right = mid - 1;
                continue;
            }
            if (i == 0) {
                for (j = 0; j < 30; j++) {
                    if (mid - i + 1 != prefix[mid][j] && base[j] == 1) {
                        right = mid - 1;
                        break;
                    }
                }
            } else {
                for (j = 0; j < 30; j++) {
                    if (mid - i + 1 != prefix[mid][j] - prefix[i - 1][j] && base[j] == 1) {
                        right = mid - 1;
                        break;
                    }
                }
            }
            if (j == 30) {
                left = mid + 1;
            }
        }
        b = right;
        if (right < i) { // 没有一个符合
            continue;
        }
        // 左边界
        if (nums[i] == k) {
            a = i;
            ans += static_cast<long long>(b - a + 1);
            continue;
        }
        left = i;
        right = b;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (i == 0) {
                for (j = 0; j < 30; j++) {
                    if ((mid - i + 1 != prefix[mid][j] && base[j] == 1) || 
                        (mid - i + 1 == prefix[mid][j] && base[j] == 0)) {
                        left = mid + 1;
                        break;
                    }
                }
            } else {
                for (j = 0; j < 30; j++) {
                    if ((mid - i + 1 != prefix[mid][j] - prefix[i - 1][j] && base[j] == 1) || 
                        (mid - i + 1 == prefix[mid][j] - prefix[i - 1][j] && base[j] == 0)) {
                        left = mid + 1;
                        break;
                    }
                }
            }
            if (j == 30) {
                right = mid - 1;
            }
        }
        a = left;
        // cout << a << " " << b << endl;
        ans += static_cast<long long>(b - a + 1);
    }
    return ans;
}


// LC3256
long long maximumValueSum(vector<vector<int>>& board)
{
    int i, j, k;
    int m = board.size();
    int n = board[0].size();
    long long ans, cur;
    vector<vector<pair<long long, int>>> max3OfRow(m, vector<pair<long long, int>>(3, {-1, -1}));
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (max3OfRow[i][0].second == -1) {
                max3OfRow[i][0] = {board[i][j], j};
            } else if (max3OfRow[i][1].second == -1) {
                if (max3OfRow[i][0].first < board[i][j]) {
                    max3OfRow[i][1] = max3OfRow[i][0];
                    max3OfRow[i][0] = {board[i][j], j};
                } else {
                    max3OfRow[i][1] = {board[i][j], j};
                }
            } else if (max3OfRow[i][2].second == -1) {
                if (max3OfRow[i][0].first < board[i][j]) {
                    max3OfRow[i][2] = max3OfRow[i][1];
                    max3OfRow[i][1] = max3OfRow[i][0];
                    max3OfRow[i][0] = {board[i][j], j};
                } else if (max3OfRow[i][1].first < board[i][j]) {
                    max3OfRow[i][2] = max3OfRow[i][1];
                    max3OfRow[i][1] = {board[i][j], j};
                } else {
                    max3OfRow[i][2] = {board[i][j], j};
                }
            } else {
                if (max3OfRow[i][0].first < board[i][j]) {
                    max3OfRow[i][2] = max3OfRow[i][1];
                    max3OfRow[i][1] = max3OfRow[i][0];
                    max3OfRow[i][0] = {board[i][j], j};
                } else if (max3OfRow[i][1].first < board[i][j]) {
                    max3OfRow[i][2] = max3OfRow[i][1];
                    max3OfRow[i][1] = {board[i][j], j};
                } else if (max3OfRow[i][2].first < board[i][j]) {
                    max3OfRow[i][2] = {board[i][j], j};
                }
            }
        }
    }
    ans = -3000000001ll;
    int c1, c2;
    long long t1, t2, t;
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            if (i == j) {
                continue;
            }
            if (max3OfRow[i][0].second != max3OfRow[j][0].second) {
                cur = max3OfRow[i][0].first + max3OfRow[j][0].first;
                c1 = max3OfRow[i][0].second;
                c2 = max3OfRow[j][0].second;
            } else {
                t1 = max3OfRow[i][0].first + max3OfRow[j][1].first;
                t2 = max3OfRow[i][1].first + max3OfRow[j][0].first;
                if (t1 > t2) {
                    cur = t1;
                    c1 = max3OfRow[i][0].second;
                    c2 = max3OfRow[j][1].second;
                } else {
                    cur = t2;
                    c1 = max3OfRow[i][1].second;
                    c2 = max3OfRow[j][0].second;
                }
            }
            t = cur;
            for (k = 0; k < m; k++) {
                cur = t;
                if (k == i || k == j) {
                    continue;
                }
                if (max3OfRow[k][0].second == c1 || max3OfRow[k][0].second == c2) {
                    if (max3OfRow[k][1].second == c1 || max3OfRow[k][1].second == c2) {
                        cur += max3OfRow[k][2].first;
                    } else {
                        cur += max3OfRow[k][1].first;
                    }
                } else {
                    cur += max3OfRow[k][0].first;
                }
                ans = max(ans, cur);
            }
            
        }
    }
    return ans;
}


// LC3260
bool CanDivide(string& num, int digit)
{
    int remainder = 0;
    string reversedNum = num;

    reverse(reversedNum.begin(), reversedNum.end());

    for (auto c : reversedNum) {
        int currentDigit = (c - '0') + remainder * 10;
        remainder = currentDigit % digit;
    }
    return remainder == 0;
}
string largestPalindrome(int n, int k)
{
    int i;
    string ans;

    if (k == 1 || k == 3 || k == 9) {
        ans.assign(n, '9');
    }
    if (k == 2) {
        ans.assign(n, '9');
        ans[0] = '8';
        ans[n - 1] = '8';
    }
    if (k == 4) {
        if (n <= 4) {
            ans.assign(n, '8');
        } else {
            ans.assign(n, '9');
            ans[0] = '8';
            ans[1] = '8';
            ans[n - 1] = '8';
            ans[n - 2] = '8';
        }
    }
    if (k == 5) {
        ans.assign(n, '9');
        ans[0] = '5';
        ans[1] = '5';
    }
    if (k == 6) {
        if (n <= 2) {
            ans.assign(n, '6');
        } else {
            ans.assign(n, '9');
            ans[0] = '8';
            ans[n - 1] = '8';
            if (n % 2 == 1) {
                ans[n / 2] = '8';
            } else {
                ans[n / 2 - 1] = '7';
                ans[n / 2] = '7';
            }
        }
    }
    if (k == 7) {
        if (n <= 2) {
            ans.assign(n, '7');
        } else {
            ans.assign(n, '9');
            for (i = 9; i >= 1; i--) {
                if (n % 2 == 1) {
                    ans[n / 2] = i + '0';
                } else {
                    ans[n / 2 - 1] = i + '0';
                    ans[n / 2] = i + '0';
                }
                if (CanDivide(ans, 7)) {
                    break;
                }
            }
        }
    }
    if (k == 8) {
        if (n <= 6) {
            ans.assign(n, '8');
        } else {
            ans.assign(n, '9');
            ans[0] = '8';
            ans[1] = '8';
            ans[2] = '8';
            ans[n - 1] = '8';
            ans[n - 2] = '8';
            ans[n - 3] = '8';
        }
    }
    return ans;
}


// LC3007
long long CountBitSum(long long num, int x)
{
    int base;
    long long t;
    long long ans;

    ans = 0;
    base = 1;

    while (1) {
        t = 1ll << x * base;
        if (num * 2 < t) {
            break;
        }
        ans += (num + 1) / t * (t >> 1) + ((num + 1) % t > (t >> 1) ? (num + 1) % t - (t >> 1) : 0);
        base++;
    }
    return ans;
}
long long findMaximumNumber(long long k, int x)
{
    long long left, right, mid;

    left = 0;
    right = 1e16;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        if (CountBitSum(mid, x) <= k) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right;
}


// LC1335
int minDifficulty(vector<int>& jobDifficulty, int d)
{
    int i, j, k;
    int curMax;
    int n = jobDifficulty.size();
    vector<vector<int>> dp(n, vector<int>(d + 1, 0x3f3f3f3f));

    if (d > n) {
        return -1;
    }
    dp[0][1] = jobDifficulty[0];
    for (i = 1; i < n; i++) {
        for (k = 1; k <= d; k++) {
            if (k > i + 1) {
                break;
            }
            curMax = jobDifficulty[i];
            for (j = i; j >= 1; j--) {
                if (k - 1 > j + 1) {
                    continue;
                }
                curMax = max(jobDifficulty[j], curMax);
                // dp[i][k] = min(dp[j][k], curMax);
                if (k > 1) {
                    dp[i][k] = min(dp[i][k], dp[j - 1][k - 1] + curMax);
                } else {
                    dp[i][k] = max(dp[j - 1][k], curMax);
                }
            }
        }
    }
    return dp[n - 1][d];
}


// LC2741
int specialPerm(vector<int>& nums)
{
    int mod = 1e9 + 7;
    int i, j, k, p;
    int n = nums.size();

    int size = pow(2, n);
    // dp[i][j][k] - 第i位前一个数的下标为j且当前nums的使用情况为k(二进制表示)的排列数
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(n, vector<long long>(size, 0)));

    for (i = 0; i < n; i++) {
        dp[0][i][1 << i] = 1;
    }
    for (p = 1; p < n; p++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < size; k++) {
                if (dp[p - 1][j][k] == 0) {
                    continue;
                }
                for (i = 0; i < n; i++) {
                    if (i == j || (nums[j] % nums[i] != 0 && nums[i] % nums[j] != 0) || (k & 1 << i) == 1) {
                        continue;
                    }
                    dp[p][i][k | 1 << i] += dp[p - 1][j][k];
                }
            }
        }
    }
    long long ans = 0;
    for (i = 0; i < n; i++) {
        ans += dp[n - 1][i][size - 1];
    }
    return ans % mod;
}
int specialPerm_1(vector<int>& nums) // 更低的时间复杂度
{
    int mod = 1e9 + 7;
    int i, j, k;
    int n = nums.size();

    int size = pow(2, n);
    // dp[j][k] - 前一个数的下标为j且当前nums的使用情况为k(二进制表示)的排列数
    vector<vector<long long>> dp(n, vector<long long>(size, 0));

    for (i = 0; i < n; i++) {
        dp[i][1 << i] = 1;
    }

    for (k = 0; k < size; k++) {
        for (j = 0; j < n; j++) {
            if ((k & 1 << j) == 0) {
                continue;
            }
            for (i = 0; i < n; i++) {
                if (i == j || (nums[j] % nums[i] != 0 && nums[i] % nums[j] != 0) || (k & 1 << i) == 0) {
                    continue;
                }
                dp[j][k] += dp[i][k ^ (1 << j)];
            }
        }
    }
    long long ans = 0;
    for (i = 0; i < n; i++) {
        ans += dp[i][size - 1];
    }
    return ans % mod;
}


// LC37
void solveSudoku(vector<vector<char>>& board)
{
    vector<vector<char>> ans;
    auto Check = [](vector<vector<char>>& board, int row, int col, char val) {
        int i, j;
        int n = board.size();
        for (i = 0; i < n; i++) {
            if (i != row && board[i][col] == val) {
                return false;
            }
        }
        for (j = 0; j < n; j++) {
            if (j != col && board[row][j] == val) {
                return false;
            }
        }
        int r = (row / 3) * 3;
        int c = (col / 3) * 3;
        for (i = r; i < r + 3; i++) {
            for (j = c; j < c + 3; j++) {
                if ((i != row || j != col) && board[i][j] == val) {
                    return false;
                }
            }
        }
        return true;
    };
    function<void (int, int, bool&)> FillSudoku = [&FillSudoku, &board, &Check, &ans](int row, int col, bool& find) {
        char i;
        int n = board.size();

        if (find) {
            return;
        }
        if (row == n) {
            ans = board;
            find = true;
            return;
        }
        if (board[row][col] != '.') {
            if (col + 1 < n) {
                FillSudoku(row, col + 1, find);
            } else {
                FillSudoku(row + 1, 0, find);
            }
        } else {
            for (i = '1'; i <= '9'; i++) {
                if (Check(board, row, col, i)) {
                    board[row][col] = i;
                    
                    if (col + 1 < n) {
                        FillSudoku(row, col + 1, find);
                    } else {
                        FillSudoku(row + 1, 0, find);
                    }
                    board[row][col] = '.';
                }
            }
        }
    };
    bool find = false;
    FillSudoku(0, 0, find);
    board = ans;
}


// LC3267
int countPairs(vector<int>& nums)
{
    int i, j, k, p;
    int n, len;
    string t;
    unordered_map<int, int> cnt;
    unordered_set<int> data;
    unordered_set<int> usedNums;

    sort(nums.rbegin(), nums.rend());
    for (auto num : nums) {
        cnt[num]++;
    }
    queue<string> q;
    int ans = 0;
    for (auto num : nums) {
        if (usedNums.count(num)) {
            continue;
        }
        while (q.size()) {
            q.pop();
        }
        data.clear();
        q.push(to_string(num));
        k = 0;
        while (q.size()) {
            n = q.size();
            for (p = 0; p < n; p++) {
                auto front = q.front();
                q.pop();
                len = front.size();
                for (i = 0; i < len - 1; i++) {
                    for (j = i + 1; j < len; j++) {
                        t = front;
                        swap(t[i], t[j]);
                        data.emplace(atoi(t.c_str()));
                        q.push(t);
                    }
                }
            }
            k++;
            if (k == 2) {
                break;
            }
        }
        // for (auto d : data) cout << d << " "; cout << endl;
        // 自身两两成一对
        if (cnt[num] > 1) {
            ans += (cnt[num] - 1) * cnt[num] / 2;
        }
        k = cnt[num];
        data.erase(num);
        usedNums.emplace(num);
        for (auto it : data) {
            if (cnt.count(it) && usedNums.count(it) == 0) {
                ans += k * cnt[it];
            }
        }
    }
    return ans;
}


// LC3272
long long countGoodIntegers(int n, int k)
{
    int i;
    long long ans = 0;
    vector<long long> factor(11, 1);
    vector<long long> base(10);
    vector<int> bitsCnt(10);

    base[0] = 1;
    for (i = 2; i <= 10; i++) {
        factor[i] = i * factor[i - 1];
        base[i - 1] = base[i - 2] * 10;
    }
    set<vector<int>> dataSets;
    function<void (int, long long)> CreatePalindrome = 
        [&CreatePalindrome, &ans, &k, &n, &factor, &dataSets, &bitsCnt, &base](int idx, long long cur) {
        long long tt;
        int i;
        if ((n % 2 == 0 && idx >= n / 2) || (n % 2 == 1 && idx > n / 2)) {
            // cout << cur << endl;
            if (cur % k == 0) {
                if (dataSets.count(bitsCnt)) {
                    return;
                }
                dataSets.insert(bitsCnt);
                if (bitsCnt[0] > 0) {
                    tt = factor[n - bitsCnt[0]];
                    for (i = 1; i <= 9; i++) {
                        if (bitsCnt[i] != 0) {
                            tt /= factor[bitsCnt[i]];
                        }
                    }
                    tt *= factor[n - 1] / factor[bitsCnt[0]] / factor[n - 1 - bitsCnt[0]];
                } else {
                    tt = factor[n];
                    for (i = 1; i <= 9; i++) {
                        if (bitsCnt[i] != 0) {
                            tt /= factor[bitsCnt[i]];
                        }
                    }
                }
                ans += tt;
            }
            return;
        }
        for (i = 0; i <= 9; i++) {
            if (idx == 0 && i == 0) {
                continue;
            }
            if (idx != n - 1 - idx) {
                bitsCnt[i] += 2;
                CreatePalindrome(idx + 1, cur + base[idx] * i + base[n - 1 - idx] * i);
                bitsCnt[i] -= 2;
            } else {
                bitsCnt[i]++;
                CreatePalindrome(idx + 1, cur + base[idx] * i);
                bitsCnt[i]--;
            }
        }
    };

    CreatePalindrome(0, 0);
    return ans;
}


// LC2555
int maximizeWin(vector<int>& prizePositions, int k)
{
    map<int, int> prefix;
    int i;
    int n = prizePositions.size();
    map<int, int> pos;

    for (i = 0; i < n; i++) {
        if (pos.count(prizePositions[i]) == 0) {
            pos[prizePositions[i]] = i;
        }
    }

    auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        }
        return a.first > b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);

    int left, right, mid;
    map<int, int> cnt;
    for (auto it : pos) {
        left = it.second;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (prizePositions[mid] > it.first + k) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        cnt[it.first] = right - it.second + 1;
        pq.push({it.first, right - it.second + 1});
    }

    // for (auto it : cnt) cout << it.first << " " << it.second << endl;
    int ans = 0;
    int cur;
    for (auto it : cnt) {
        cur = it.second;
        while (pq.size()) {
            auto t = pq.top();
            if (it.first + k >= t.first) {
                pq.pop();
                continue;
            }
            cur += t.second;
            break;
        }
        ans = max(ans, cur);
    }
    return ans;
}


// LC2925
long long maximumScoreAfterOperations(vector<vector<int>>& edges, vector<int>& values)
{
    int i;
    int n = values.size();
    unordered_map<int, unordered_set<int>> edge;

    for (auto e : edges) {
        edge[e[0]].emplace(e[1]);
        edge[e[1]].emplace(e[0]);
    }

    vector<long long> treeSum(n, -1);
    function<long long (int, int)> dfs = [&dfs, &treeSum, &edge, &values](int node, int parent) {
        if (treeSum[node] != -1) {
            return treeSum[node];
        }
        long long sum = values[node];
        for (auto it : edge[node]) {
            if (it != parent) {
                sum += dfs(it, node);
            }
        }
        treeSum[node] = sum;
        return sum;
    };

    dfs(0, -1);
    // for (auto it : treeSum) cout << it << endl;

    vector<long long> maxTreeSum(n, -1);
    function<long long (int, int)> maxPoint = [&maxPoint, &treeSum, &edge, &values, &maxTreeSum](int node, int parent) {
        if (edge[node].size() == 1 && *edge[node].begin() == parent) { // 叶子节点
            maxTreeSum[node] = 0;
            return maxTreeSum[node];
        }
        if (maxTreeSum[node] != -1) {
            return maxTreeSum[node];
        }
        long long sum1 = treeSum[node] - values[node];
        long long sum2 = values[node];
        for (auto it : edge[node]) {
            if (it != parent) {
                sum2 += maxPoint(it, node);
            }
        }
        maxTreeSum[node] = max(sum1, sum2);
        return maxTreeSum[node];
    };
    maxPoint(0, -1);

    return maxTreeSum[0];
}


// LC2792
int countGreatEnoughNodes(TreeNode* root, int k)
{
    int cnt;
    // vector<int> t = dfs(node), t - 以node为树根节点，最小的k个子节点的值集合
    cnt = 0;
    function<vector<int> (TreeNode *)> dfs = [&k, &cnt, &dfs](TreeNode *node) {
        if (node == nullptr) {
            return vector<int>();
        }
        vector<int> merge;
        vector<int> left = dfs(node->left);
        vector<int> right = dfs(node->right);

        // sort(left.begin(), left.end());
        // sort(right.begin(), right.end());

        int i, j, p;
        int m = left.size();
        int n = right.size();

        i = j = 0;
        while (i < m && j < n) {
            if (left[i] <= right[j]) {
                merge.emplace_back(left[i]);
                i++;
            } else {
                merge.emplace_back(right[j]);
                j++;
            }
            if (merge.size() == k) {
                break;
            }
        }
        if (merge.size() < k) {
            if (i < m) {
                for (p = i; p < m; p++) {
                    merge.emplace_back(left[p]);
                    if (merge.size() == k) {
                        break;
                    }
                }
            }
            if (j < n) {
                for (p = j; p < n; p++) {
                    merge.emplace_back(right[p]);
                    if (merge.size() == k) {
                        break;
                    }
                }
            }
        }
        if (merge.size() == k && node->val > merge[k - 1]) {
            cnt++;
        }
        merge.emplace_back(node->val);
        sort(merge.begin(), merge.end());
        return merge;
    };
    dfs(root);
    return cnt;
}


// LC3288
vector<int> longestIncreasingSeq_2D(vector<vector<int>>& points)
{
    int i;
    int n = points.size();
    int m;
    int left, right, mid;
    vector<int> low;
    vector<int> ans(n, 0);
    low.emplace_back(points[0][1]);
    ans[0] = 1;
    for (i = 1; i < n; i++) {
        m = low.size();
        if (points[i][1] > low[m - 1]) {
            low.emplace_back(points[i][1]);
            ans[i] = low.size();
            continue;
        }
        left = 0;
        right = m - 1;
        while (left <= right) { // 所求为left
            mid = ((right - left) >> 1) + left;
            if (low[mid] < points[i][1]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        low[left] = points[i][1];
        ans[i] = left + 1;
    }
    return ans;
}
int maxPathLength(vector<vector<int>>& coordinates, int k)
{
    vector<vector<int>> co;
    int i;
    int n = coordinates.size();
    int maxVal;

    if (n == 1) {
        return 1;
    }

    // 小于coordinates[k]的最长上升子序列
    for (i = 0; i < n; i++) {
        if (coordinates[i][0] < coordinates[k][0] && coordinates[i][1] < coordinates[k][1]) {
            co.emplace_back(coordinates[i]);
        }
    }
    sort(co.begin(), co.end(), [](vector<int>& a, vector<int>& b) {
        if (a[0] != b[0]) {
            return a[0] < b[0];
        }
        return a[1] > b[1];
    });

    int ans;
    vector<int> a;

    if (co.empty()) {
        ans = 1;
    } else {
        a = longestIncreasingSeq_2D(co);
        maxVal = 0;
        for (auto c : a) {
            maxVal = max(maxVal, c);
        }
        ans = maxVal + 1;
    }

    // 大于coordinates[k]的最长上升子序列
    co.clear();
    for (i = 0; i < n; i++) {
        if (coordinates[i][0] > coordinates[k][0] && coordinates[i][1] > coordinates[k][1]) {
            co.emplace_back(coordinates[i]);
        }
    }
    sort(co.begin(), co.end(), [](vector<int>& a, vector<int>& b) {
        if (a[0] != b[0]) {
            return a[0] < b[0];
        }
        return a[1] > b[1];
    });

    if (!co.empty()) {
        a = longestIncreasingSeq_2D(co);
        maxVal = 0;
        for (auto c : a) {
            maxVal = max(maxVal, c);
        }
        ans += maxVal;
    }
    return ans;
}


// LC3291
int minValidStrings(vector<string>& words, string target)
{
    // 注意此处CreateWordTrie函数中的root->IsEnd = true应该放在循环内, 为了不破坏CreateWordTrie的普适性这里特此说明
    Trie<char> *root = new Trie<char>('/');
    for (auto w : words) {
        Trie<char>::CreateWordTrie(root, w);
    }

    int i, j, k, kk;
    int n = target.size();
    vector<int> dp(n + 1, 0x3f3f3f3f);

    dp[0] = 0;
    for (i = 0; i < n; i++) {
        auto node = root;
        for (j = i; j < n; j++) {
            kk = node->children.size();
            for (k = 0; k < kk; k++) {
                if (node->children[k]->val == target[j]) {
                    node = node->children[k];
                    break;
                }
            }
            if (k == kk) {
                break;
            }
            if (node->IsEnd) {
                dp[j + 1] = min(dp[j + 1], dp[i] + 1);
            }
        }
    }
    delete(root);
    return dp[n] == 0x3f3f3f3f ? -1 : dp[n];
}
// 另一思路, 超时
int minValidStrings_1(vector<string>& words, string target)
{
    // 字符串哈希
    int i, j, k, p;
    int n = target.size();
    vector<int> dp(n + 1, 0x3f3f3f3f);

    unsigned long long base = 1337;
    unsigned long long t;
    int m = words.size();
    vector<unordered_set<unsigned long long>> wordsHash(5001);
    for (i = 0; i < m; i++) {
        t = 0;
        p = words[i].size();
        for (j = 0; j < p; j++) {
            t = t * base + (words[i][j] - 'a' + 1);
            wordsHash[j + 1].emplace(t);
        }
    }

    vector<vector<unsigned long long>> targetHash(n);
    for (i = 0; i < n; i++) {
        t = 0;
        for (j = i; j < n; j++) {
            t = t * base + (target[j] - 'a' + 1);
            targetHash[i].emplace_back(t);
        }
    }

    dp[0] = 0;
    for (i = 1; i <= n; i++) {
        if (wordsHash[i].count(targetHash[0][i - 1])) {
            dp[i] = 1;
            continue;
        }
        for (j = 1; j <= i; j++) {
            auto hash = targetHash[j - 1][i - j];
            if (wordsHash[i - j + 1].count(hash)) {
                dp[i] = min(dp[i], dp[j - 1] + 1);
                break;
            }
        }
    }
    return dp[n] == 0x3f3f3f3f ? -1 : dp[n];
}


// LC3229
long long minimumOperations(vector<int>& nums, vector<int>& target)
{
    int i;
    int n = nums.size();
    int diff, canDecrease, canAdd;
    long long ans;

    canDecrease = canAdd = 0;
    ans = 0;
    // 贪心
    for (i = 0; i < n; i++) {
        diff = nums[i] - target[i];
        if (diff > 0) {
            canAdd = 0;
            if (canDecrease == 0) {
                canDecrease = diff;
                ans += diff;
            } else if (canDecrease >= diff) {
                canDecrease = diff;
            } else {
                ans += diff - canDecrease;
                canDecrease = diff;
            }
        } else if (diff < 0) {
            canDecrease = 0;
            if (canAdd == 0) {
                canAdd = -diff;
                ans += -diff;
            } else if (canAdd >= -diff) {
                canAdd = -diff;
            } else {
                ans += -diff - canAdd;
                canAdd = -diff;
            }
        } else {
            canDecrease = canAdd = 0;
        }
    }
    return ans;
}


// LC3284
int getSum(vector<int>& nums)
{
    int i;
    int n = nums.size();
    long long ans, sum;
    vector<long long> prefixSubArrSum(n, 0);
    vector<long long> prefix(n, 0);
    vector<int> visited(n, 1);

    prefixSubArrSum[0] = nums[0];
    prefix[0] = nums[0];
    sum = nums[0];
    for (i = 1; i < n; i++) {
        prefixSubArrSum[i] = prefixSubArrSum[i - 1] + static_cast<long long>(i + 1) * nums[i];
        prefix[i] = prefix[i - 1] + nums[i];
        sum += nums[i];
    }

    ans = 0;
    // 递增区间
    for (i = 1; i < n; i++) {
        if (nums[i] == nums[i - 1] + 1) {
            visited[i] = visited[i - 1] + 1;
        }
    }
    for (i = 0; i < n; i++) {
        if (visited[i] == 1) {
            continue;
        }
        if (i + 1 == visited[i]) {
            ans += prefixSubArrSum[i] - nums[i];
        } else {
            ans += prefixSubArrSum[i] - prefixSubArrSum[i - visited[i]] - 
                (prefix[i] - prefix[i - visited[i]]) * (i - visited[i] + 1) - nums[i]; 
        }
    }

    // 递减区间
    visited.assign(n, 1);
    for (i = 1; i < n; i++) {
        if (nums[i] == nums[i - 1] - 1) {
            visited[i] = visited[i - 1] + 1;
        }
    }
    for (i = 0; i < n; i++) {
        if (visited[i] == 1) {
            continue;
        }
        if (i + 1 == visited[i]) {
            ans += prefixSubArrSum[i] - nums[i];
        } else {
            ans += prefixSubArrSum[i] - prefixSubArrSum[i - visited[i]] - 
                (prefix[i] - prefix[i - visited[i]]) * (i - visited[i] + 1) - nums[i]; 
        }
    }
    int mod = 1e9 + 7;
    return (ans + sum) % mod;
}


// LC84 LC1856
int largestRectangleArea(vector<int>& heights)
{
    int i;
    int n = heights.size();
    vector<int> l(n), r(n);

    // l[i] - 左侧第一个比heights[i]小的下标
    // r[i] - 右侧第一个比heights[i]小的下标
    // 单调递增栈

    stack<int> st;
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        auto idx = st.top();
        while (heights[idx] > heights[i]) {
            r[idx] = i;
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        r[st.top()] = n;
        st.pop();
    }

    // 同理从右到左求l[i];
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        auto idx = st.top();
        while (heights[idx] > heights[i]) {
            l[idx] = i;
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        l[st.top()] = -1;
        st.pop();
    }

    // for (auto it : r) cout << it << " "; cout << endl;
    // for (auto it : l) cout << it << " "; cout << endl;

    int ans = 0;
    for (i = 0; i < n; i++) {
        ans = max(ans, heights[i] * (r[i] - l[i] - 1));
    }
    return ans;
}


// LC85
int maximalRectangle(vector<vector<char>>& matrix)
{
    // 将matrix的每一行转化为类似LC84的高度数组, 第i行是第0 到 i - 1行的前缀和
    int i, j;
    int m, n;

    m = matrix.size();
    n = matrix[0].size();
    vector<vector<int>> heights(m, vector<int>(n, 0));

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (i == 0) {
                heights[i][j] = matrix[i][j] - '0';
            } else {
                if (matrix[i][j] == '0') {
                    heights[i][j] = 0;
                } else {
                    heights[i][j] = heights[i - 1][j] + 1;
                }
            }
        }
    }

    int ans = 0;
    for (auto h : heights) {
        ans = max(ans, largestRectangleArea(h));
    }
    return ans;
}


// LC3310
vector<int> remainingMethods(int n, int k, vector<vector<int>>& invocations)
{
    int i;
    unordered_map<int, vector<int>> edges;
    vector<vector<int>> edges1(n);
    // 注意当数据量增大时开多个vector也会极大增加时间开销
    for (auto in : invocations) {
        edges[in[0]].emplace_back(in[1]);
    }
    vector<bool> visited(n, false);
    unordered_set<int> nodes;
    function<void (int)> dfs = [&visited, &dfs, &edges, &nodes](int cur) {
        visited[cur] = true;
        nodes.emplace(cur);
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                if (visited[it] == false) {
                    dfs(it);
                }
            }
        }
    };
    dfs(k);

    bool f = false;
    function<void (int, bool&)> dfs1 = [&visited, &dfs1, &edges, &nodes](int cur, bool& f) {
        if (f) {
            return;
        }
        visited[cur] = true;
        if (edges.count(cur)) {
            for (auto it : edges[cur]) {
                if (visited[it] == true) {
                    if (nodes.count(it)) {
                        f = true;
                        return;
                    }
                } else {
                    dfs1(it, f);
                }
            }
        }
    };
    vector<int> ori(n);
    for (i = 0; i < n; i++) {
        ori[i] = i;
    }
    vector<int> ans;
    for (i = 0; i < n; i++) {
        if (visited[i] == false) {
            ans.emplace_back(i);
        }
    }
    for (i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }
        f = false;
        dfs1(i, f);
        if (f) {
            return ori;
        }
    }

    return ans;
}


// LC416
bool canPartition(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int sum = accumulate(nums.begin(), nums.end(), 0);

    if (sum % 2 == 1) {
        return false;
    }

    int target = sum / 2;
    // dp[i][j] - 前i个数能否组成j
    vector<vector<bool>> dp(n + 1, vector<bool>(target + 1, false));

    for (i = 0; i <= n; i++) {
        dp[i][0] = true;
    }
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= target; j++) {
            if (dp[i - 1][j]) {
                dp[i][j] = true;
            } else if (j - nums[i - 1] >= 0) {
                dp[i][j] = dp[i - 1][j - nums[i - 1]];
            }
        }
    }
    return dp[n][target];
}


// LC3320
int countWinningSequences(string s)
{
    int mod = 1e9 + 7;
    int i, j;
    int n = s.size();

    if (n == 1) {
        return 1;
    }
    // dp[i][j][k] - 第i轮召唤j(F, W, E)最后的分为k-n (数组下标不能为负)的方案数
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(3, vector<long long>(n * 2 + 1, -1)));

    if (s[0] == 'F') {
        dp[0][0][n] = 1;
        dp[0][1][n + 1] = 1;
        dp[0][2][n - 1] = 1;
    } else if (s[0] == 'W') {
        dp[0][0][n - 1] = 1;
        dp[0][1][n] = 1;
        dp[0][2][n + 1] = 1;
    } else {
        dp[0][0][n + 1] = 1;
        dp[0][1][n - 1] = 1;
        dp[0][2][n] = 1;
    }
    for (i = 1; i < n; i++) {
        for (j = 0; j < n * 2 + 1; j++) {
            if (s[i] == 'F') {
                if (dp[i - 1][0][j] != -1) {
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][1][j + 1] == -1) {
                            dp[i][1][j + 1] = 0;
                        }
                        dp[i][1][j + 1] = (dp[i][1][j + 1] + dp[i - 1][0][j]) % mod;
                    }
                    if (j - 1 >= 0) {
                        if (dp[i][2][j - 1] == -1) {
                            dp[i][2][j - 1] = 0;
                        }
                        dp[i][2][j - 1] = (dp[i][2][j - 1] + dp[i - 1][0][j]) % mod;
                    }
                }
                if (dp[i - 1][1][j] != -1) {
                    if (j - 1 >= 0) {
                        if (dp[i][2][j - 1] == -1) {
                            dp[i][2][j - 1] = 0;
                        }
                        dp[i][2][j - 1] = (dp[i][2][j - 1] + dp[i - 1][1][j]) % mod;
                    }
                    if (dp[i][0][j] == -1) {
                        dp[i][0][j] = 0;
                    }
                    dp[i][0][j] = (dp[i][0][j] + dp[i - 1][1][j]) % mod;
                }
                if (dp[i - 1][2][j] != -1) {
                    if (dp[i][0][j] == -1) {
                        dp[i][0][j] = 0;
                    }
                    dp[i][0][j] = (dp[i][0][j] + dp[i - 1][2][j]) % mod;
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][1][j + 1] == -1) {
                            dp[i][1][j + 1] = 0;
                        }
                        dp[i][1][j + 1] = (dp[i][1][j + 1] + dp[i - 1][2][j]) % mod;
                    }
                }
            } else if (s[i] == 'W') {
                if (dp[i - 1][0][j] != -1) {
                    if (dp[i][1][j] == -1) {
                        dp[i][1][j] = 0;
                    }
                    dp[i][1][j] = (dp[i][1][j] + dp[i - 1][0][j]) % mod;
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][2][j + 1] == -1) {
                            dp[i][2][j + 1] = 0;
                        }
                        dp[i][2][j + 1] = (dp[i][2][j + 1] + dp[i - 1][0][j]) % mod;
                    }
                }
                if (dp[i - 1][1][j] != -1) {
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][2][j + 1] == -1) {
                            dp[i][2][j + 1] = 0;
                        }
                        dp[i][2][j + 1] = (dp[i][2][j + 1] + dp[i - 1][1][j]) % mod;
                    }
                    if (j - 1 >= 0) {
                        if (dp[i][0][j - 1] == -1) {
                            dp[i][0][j - 1] = 0;
                        }
                        dp[i][0][j - 1] = (dp[i][0][j - 1] + dp[i - 1][1][j]) % mod;
                    }
                }
                if (dp[i - 1][2][j] != -1) {
                    if (j - 1 >= 0) {
                        if (dp[i][0][j - 1] == -1) {
                            dp[i][0][j - 1] = 0;
                        }
                        dp[i][0][j - 1] = (dp[i][0][j - 1] + dp[i - 1][2][j]) % mod;
                    }
                    if (dp[i][1][j] == -1) {
                        dp[i][1][j] = 0;
                    }
                    dp[i][1][j] = (dp[i][1][j] + dp[i - 1][2][j]) % mod;
                }
            } else {
                if (dp[i - 1][0][j] != -1) {
                    if (j - 1 >= 0) {
                        if (dp[i][1][j - 1] == -1) {
                            dp[i][1][j - 1] = 0;
                        }
                        dp[i][1][j - 1] = (dp[i][1][j - 1] + dp[i - 1][0][j]) % mod;
                    }
                    if (dp[i][2][j] == -1) {
                        dp[i][2][j] = 0;
                    }
                    dp[i][2][j] = (dp[i][2][j] + dp[i - 1][0][j]) % mod;
                }
                if (dp[i - 1][1][j] != -1) {
                    if (dp[i][2][j] == -1) {
                        dp[i][2][j] = 0;
                    }
                    dp[i][2][j] = (dp[i][2][j] + dp[i - 1][1][j]) % mod;
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][0][j + 1] == -1) {
                            dp[i][0][j + 1] = 0;
                        }
                        dp[i][0][j + 1] = (dp[i][0][j + 1] + dp[i - 1][1][j]) % mod;
                    }
                }
                if (dp[i - 1][2][j] != -1) {
                    if (j + 1 < n * 2 + 1) {
                        if (dp[i][0][j + 1] == -1) {
                            dp[i][0][j + 1] = 0;
                        }
                        dp[i][0][j + 1] = (dp[i][0][j + 1] + dp[i - 1][2][j]) % mod;
                    }
                    if (j - 1 >= 0) {
                        if (dp[i][1][j - 1] == -1) {
                            dp[i][1][j - 1] = 0;
                        }
                        dp[i][1][j - 1] = (dp[i][1][j - 1] + dp[i - 1][2][j]) % mod;
                    }
                }
            }
        }
    }
    long long ans = 0;
    for (i = n + 1; i <= n * 2; i++) {
        for (j = 0; j < 3; j++) {
            if (dp[n - 1][j][i] != -1) {
                ans = (ans + dp[n - 1][j][i]) % mod;
            }
        }
    }
    return ans;
}


// LC214
string shortestPalindrome(string s)
{
    if (IsPalindrome(s)) {
        return s;
    }

    int i, idx;
    int n;
    string t = s;
    string str;
    reverse(t.begin(), t.end());

    vector<int> next;

    str = s + "#" + t;
    GenerateNextArr(str, next);

    int k = next.back();
    n = str.size();
    string midStr = str.substr(k, n - 2 * k);

    cout << midStr << endl;

    n = midStr.size();
    for (i = 0; i < n; i++) {
        if (midStr[i] == '#') {
            idx = i + 1;
            break;
        }
    }
    return midStr.substr(idx) + s;
}


// LC3316
int maxRemovals(string source, string pattern, vector<int>& targetIndices)
{
    int i, j;
    int n = source.size();
    int m = pattern.size();

    unordered_set<int> idx(targetIndices.begin(), targetIndices.end());

    // dp[i][j] - source前i位匹配 pattern前j位最大可删除数
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1));

    dp[0][0] = 0;
    // dp[i][0] - 只要把前i位在targetIndices满足的都删了, 剩下的一定匹配, 此时pattern可看做空串, 一定是删除后source的子序列
    for (i = 1; i <= n; i++) {
        if (idx.count(i - 1)) {
            dp[i][0] = dp[i - 1][0] + 1;
        } else {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            if (j > i) {
                break;
            }
            // 不选择匹配source[i - 1]
            dp[i][j] = dp[i - 1][j];
            if (idx.count(i - 1) && dp[i - 1][j] != -1) {
                dp[i][j] = dp[i - 1][j] + 1;
            }
            // 选择匹配source[i - 1]
            if (source[i - 1] == pattern[j - 1]) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1]);
            }
        }
    }
    return dp[n][m];
}


// LC425
vector<vector<string>> wordSquares(vector<string>& words)
{
    int i;
    int n = words.size();
    auto Check = [](vector<string>& brick) {
        int i, j;
        int n = brick.size();

        if (n == 1) {
            return true;
        }

        string t;
        for (j = 0; j < n; j++) {
            t.clear();
            for (i = 0; i < n; i++) {
                t += brick[i][j];
            }
            if (brick[j].substr(0, n) != t) {
                return false;
            }
        }
        return true;
    };
    unordered_map<char, vector<string>> dict;

    for (auto w : words) {
        dict[w[0]].emplace_back(w);
    }
    vector<string> record;
    vector<vector<string>> ans;
    function<void (string&, int)> dfs = [&dfs, &Check, &dict, &record, &ans](string& cur, int idx) {
        if (record.size() == cur.size()) {
            ans.emplace_back(record);
            return;
        }
        if (dict.count(cur[idx])) {
            for (auto w : dict[cur[idx]]) {
                record.emplace_back(w);
                if (!Check(record)) {
                    record.pop_back();
                    continue;
                }
                dfs(cur, idx + 1);
                record.pop_back();
            }
        }
    };
    for (i = 0; i < n; i++) {
        record.emplace_back(words[i]);
        dfs(words[i], 1);
        record.pop_back();
    }
    return ans;
}


// LC3323
int minConnectedGroups(vector<vector<int>>& intervals, int k)
{
    int i;
    int n = intervals.size();
    int start, end;
    vector<vector<int>> ranges;

    sort(intervals.begin(), intervals.end());
    start = intervals[0][0];
    end = intervals[0][1];
    for (i = 1; i < n; i++) {
        if (intervals[i][0] > end) {
            ranges.push_back({start, end});
            start = intervals[i][0];
            end = intervals[i][1];
        } else {
            end = max(end, intervals[i][1]);
        }
    }
    ranges.push_back({start, end});

    int left, right, mid;
    int target, ans;

    n = ranges.size();
    ans = INT_MAX;
    for (i = 0; i < n; i++) {
        left = i;
        right = n - 1;
        target = ranges[i][1] + k;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (ranges[mid][0] <= target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        ans = min(ans, i + 1 + n - 1 - right);
    }
    return ans;
}


// LC3326
int minOperations_LC3326(vector<int>& nums)
{
    auto PrimeChange = [](int n) {
        if (n == 1 || n == 2) {
            return -1;
        }
        int i;
        int m = sqrt(n);

        for (i = 2; i <= m; i++) {
            if (n % i == 0) {
                return i;
            }
        }
        return -1;
    };

    int i;
    int n = nums.size();
    int ans;
    int INF = 0x3f3f3f3f;

    vector<int> afterChange(n);
    for (i = 0; i < n; i++) {
        auto t = PrimeChange(nums[i]);
        if (t != -1) {
            afterChange[i] = t;
        } else {
            afterChange[i] = nums[i];
        }
    }
    // dp[i][0 - 1] - 第i位是否变换的最少操作次数
    vector<vector<int>> dp(n, vector<int>(2, INF));
    if (afterChange[0] == nums[0]) {
        dp[0][0] = 0;
    } else {
        dp[0][0] = 0;
        dp[0][1] = 1;
    }
    for (i = 1; i < n; i++) {
        if (nums[i] >= nums[i - 1]) {
            dp[i][0] = min(dp[i - 1][0], dp[i - 1][1]);
            if (afterChange[i] != nums[i]) {
                if (afterChange[i] >= nums[i - 1]) {
                    dp[i][1] = dp[i - 1][0] == INF ? INF : dp[i - 1][0] + 1;
                    if (afterChange[i - 1] != nums[i - 1] && afterChange[i] >= afterChange[i - 1]) {
                        dp[i][1] = min(dp[i][1], dp[i - 1][1] == INF ? INF : dp[i - 1][1] + 1);
                    }
                } else {
                    if (afterChange[i - 1] != nums[i - 1] && afterChange[i - 1] <= afterChange[i]) {
                        dp[i][1] = dp[i - 1][1] == INF ? INF : dp[i - 1][1] + 1;
                    }
                }
            }
        } else {
            if (nums[i] >= afterChange[i - 1]) {
                dp[i][0] = min(dp[i][0], dp[i - 1][1]);
            }
            if (afterChange[i] != nums[i] && afterChange[i - 1] != nums[i - 1]) {
                if (afterChange[i] >= afterChange[i - 1]) {
                    dp[i][1] = dp[i - 1][1] == INF ? INF : dp[i - 1][1] + 1;
                }
            }
        }
    }
    ans = min(dp[n - 1][0], dp[n - 1][1]);
    return ans == INF ? -1 : ans;
}


// 面试题 08.13. 堆箱子
int pileBox(vector<vector<int>>& box)
{
    int i, j;
    int n = box.size();
    int ans;

    sort(box.rbegin(), box.rend());

    // dp[i] - 以box[i]作为最上面箱子的最大高度
    vector<int> dp(n, 0);

    dp[0] = box[0][2];
    ans = dp[0];
    for (i = 1; i < n; i++) {
        dp[i] = box[i][2];
        for (j = i - 1; j >= 0; j--) {
            if (box[i][0] < box[j][0] && box[i][1] < box[j][1] && box[i][2] < box[j][2]) {
                dp[i] = max(dp[i], box[i][2] + dp[j]);
            }
        }
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC3329
long long numberOfSubstrings(string s, int k)
{
    int i, j;
    int n = s.size();
    int left, right, mid;
    int curMaxCnt;
    long long ans;
    bool find = false;
    vector<int> t(26);
    vector<vector<int>> prefix(n, vector<int>(26, 0));

    ans = 0;
    curMaxCnt = 0;
    for (i = 0; i < n; i++) {
        if (i > 0) {
            prefix[i] = prefix[i - 1];
        }
        prefix[i][s[i] - 'a']++;
        curMaxCnt = max(curMaxCnt, prefix[i][s[i] - 'a']);
        if (curMaxCnt < k) {
            continue;
        }
        left = 0;
        right = i;
        // 所求为right
        while (left <= right) {
            find = false;
            mid = (right - left) / 2 + left;
            if (mid == 0) {
                left = mid + 1;
                find = true;
            } else {
                for (j = 0; j < 26; j++) {
                    t[j] = prefix[i][j] - prefix[mid - 1][j];
                    if (t[j] >= k) {
                        left = mid + 1;
                        find = true;
                        break;
                    }
                }
            }
            if (find == false) {
                right = mid - 1;
            }
        }
        ans += right + 1;
    }
    return ans;
}


// LC685
vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges)
{
    int i;
    int n = edges.size();
    int cnt;
    // 每个节点的入度
    vector<int> degree(n + 1, 0);
    vector<int> visited(n + 1, 0);
    vector<vector<int>> edge(n + 1);
    for (auto e : edges) {
        edge[e[0]].emplace_back(e[1]);
        degree[e[1]]++;
    }
    int curRoot = -1;
    for (i = 1; i <= n; i++) {
        if (degree[i] == 0) {
            curRoot = i;
            break;
        }
    }
    function<void (int, vector<int>&, bool&, int&)> dfs = [&dfs, &edge, &visited]
        (int cur, vector<int>& deleteEdge, bool& findLoop, int& cnt) {
        if (findLoop) {
            return;
        }
        visited[cur] = 1;
        for (auto it : edge[cur]) {
            if (deleteEdge[0] != cur || deleteEdge[1] != it) {
                if (visited[it] == 1) {
                    findLoop = true;
                    return;
                }
                if (visited[it] == 0) {
                    cnt++;
                    dfs(it, deleteEdge, findLoop, cnt);
                }
            }
        }
        visited[cur] = 2;
    };
    vector<int> ans;
    bool findLoop = false;
    for (i = n - 1; i >= 0; i--) {
        degree[edges[i][1]]--;
        if (curRoot != -1 && degree[edges[i][1]] == 0) {
            // 出现了两个根节点
            degree[edges[i][1]]++;
            continue;
        }
        cnt = 1;
        findLoop = false;
        fill(visited.begin(), visited.end(), 0);
        if (curRoot != -1) {
            dfs(curRoot, edges[i], findLoop, cnt);
        } else {
            if (degree[edges[i][1]] == 0) {
                dfs(edges[i][1], edges[i], findLoop, cnt);
            }
        }
        if (findLoop == false && cnt == n) {
            ans = edges[i];
            break;
        }
        degree[edges[i][1]]++;
    }
    return ans;
}


// LC1449
string largestNumber(vector<int>& cost, int target)
{
    int i, j, k;
    int n = cost.size();

    // dp[target][x] - 用数字x构成target时能得到的最大字符串长度
    vector<vector<int>> dp(target + 1, vector<int>(10, -1));
    for (i = 0; i < 10; i++) {
        dp[0][i] = 0;
    }
    for (i = 1; i <= target; i++) {
        for (j = 0; j < n; j++) {
            for (k = 1; k <= n; k++) {
                if (i - cost[j] >= 0 && dp[i - cost[j]][k] != -1) {
                    dp[i][j + 1] = max(dp[i][j + 1], dp[i - cost[j]][k] + 1);
                }
            }
        }
    }
    for (i = 0; i < 10; i++) {
        if (dp[target][i] != -1) {
            break;
        }
    }
    if (i == 10) {
        return "0";
    }
    string ans;
    int curLen, t;
    while (target) {
        curLen = 0;
        for (i = 1; i <= 9; i++) {
            if (dp[target][i] != -1) {
                if (dp[target][i] >= curLen) {
                    curLen = dp[target][i];
                    t = i;
                }
            }
        }
        target -= cost[t - 1];
        ans += t + '0';
    }
    for (i = 19; i >= 1; i--) {
        if ((n & 1 << i) == (1 << i)) {
            printf ("%d ", 1 << i);
        }
    }
    return ans;
}


// LC3339
int countOfArrays(int n, int m, int k)
{
    int i, j;
    int mod = 1e9 + 7;
    // dp[i][0 - 1][k] - 第i位是 偶数/奇数 组成k个偶数数组的情况
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(2, vector<long long>(k + 1, 0)));
    int oddNum = (m + 1) / 2;
    int evenNum = m / 2;

    dp[0][0][0] = evenNum;
    dp[0][1][0] = oddNum;
    for (i = 1; i < n; i++) {
        for (j = 0; j <= k; j++) {
            dp[i][0][j] = evenNum * (dp[i - 1][1][j] + (j > 0 ? dp[i - 1][0][j - 1] : 0ll)) % mod;
            dp[i][1][j] = oddNum * (dp[i - 1][0][j] + dp[i - 1][1][j]) % mod;
        }
    }
    return (dp[n - 1][0][k] + dp[n - 1][1][k]) % mod;
}


// LC10
bool isMatch(string s, string p)
{
    int m = s.size();
    int n = p.size();
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));

    dp[0][0] = true;
    for (int i = 0; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p[j - 1] == '*') {
                dp[i][j] = dp[i][j - 2]; // '*'可以代表前面的字符出现0次
                if (i > 0 && (s[i - 1] == p[j - 2] || p[j - 2] == '.')) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j]; // 匹配1次或多次
                }
            } else if (i > 0 && (s[i - 1] == p[j - 1] || p[j - 1] == '.')) {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    return dp[m][n];
}


// LC44
bool isMatch_LC44(string s, string p)
{
    int i, j;
    int m = s.size();
    int n = p.size();
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));

    dp[0][0] = true;

    string t;
    for (j = 1; j <= n; j++) {
        t.append(1, '*');
        if (p.substr(0, j) == t) {
            dp[0][j] = true;
        } else {
            break;
        }
    }
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (s[i - 1] == p[j - 1] || p[j - 1] == '?') {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p[j - 1] == '*') {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1]; // 是否匹配'*'
            }
        }
    }
    return dp[m][n];
}


// LC638
int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs)
{
    int i;
    int n = price.size();
    vector<int> v(n + 1);

    for (i = 0; i < n; i++) {
        fill(v.begin(), v.end(), 0);
        v[i] = 1;
        v[n] = price[i];
        special.emplace_back(v);
    }

    auto add = [&needs](vector<int>& b) {
        int i;
        int n = needs.size();
        for (i = 0; i < n; i++) {
            needs[i] += b[i];
        }
    };

    auto minus = [&needs](vector<int>& b) {
        int i;
        int n = needs.size();
        bool f = true;
        for (i = 0; i < n; i++) {
            needs[i] -= b[i];
            if (needs[i] < 0) {
                f = false;
            }
        }
        return f;
    };

    unordered_map<vector<int>, int, VectorHash<int>> cost;
    vector<int> z(n, 0);
    cost[z] = 0;
    function<int (vector<int>&)> dfs = [&dfs, &special, &cost, &add, &minus](vector<int>& needs) {
        int i;
        int n = needs.size();

        if (cost.count(needs)) {
            return cost[needs];
        }
        int m = special.size();
        int ans = 0x3f3f3f3f;
        int a, b;
        for (i = 0; i < m; i++) {
            if (minus(special[i])) {
                a = special[i][n];
                b = dfs(needs);
                ans = min(ans, a + b);
            }
            add(special[i]);
        }
        cost[needs] = ans;
        return ans;
    };

    return dfs(needs);
}


int minTimeToReach(vector<vector<int>>& moveTime)
{
    int i;
    int n = moveTime.size();
    int m = moveTime[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int cost;
    // dist[i][j] - 从(0, 0)到(i, j)最少耗时
    vector<vector<int>> dist(n, vector<int>(m, INT_MAX / 2));
    auto cmp = [](pair<pair<int, int>, int>& a, pair<pair<int, int>, int>& b) {
        return a.second > b.second;
    };

    // 堆优化dijkstra
    priority_queue<pair<pair<int, int>, int>, vector<pair<pair<int, int>, int>>, decltype(cmp)> q(cmp);
    q.push({{0, 0}, 0});
    while (q.size()) {

        auto p = q.top();
        q.pop();
        if (dist[p.first.first][p.first.second] < p.second) {
            continue;
        }
        dist[p.first.first][p.first.second] = p.second;

        for (i = 0; i < 4; i++) {
            auto row = p.first.first + directions[i][0];
            auto col = p.first.second + directions[i][1];
            if (row < 0 || row >= n || col < 0 || col >= m) {
                continue;
            }
            if ((row + col) % 2 == 0) {
                cost = 2;
            } else {
                cost = 1;
            }
            if (p.second <= moveTime[row][col]) {
                if (dist[row][col] <= moveTime[row][col] + cost) {
                    continue;
                }
                dist[row][col] = moveTime[row][col] + cost;
                // printf ("1 %d %d, dist[%d][%d]=%d\n", p.second, cost, row, col, dist[row][col]);
                q.push({{row, col}, dist[row][col]});
                continue;
            }
            if (p.second + cost < dist[row][col]) {
                dist[row][col] = p.second + cost;
                // printf ("2 %d %d, dist[%d][%d]=%d\n", p.second, cost, row, col, dist[row][col]);
                q.push({{row, col}, dist[row][col]});
            }
        }
    }
    return dist[n - 1][m - 1];
}


// LCP74
int fieldOfGreatestBlessing(vector<vector<int>>& forceField)
{
    int i, j;
    // 对于每个正方形的两条纵向边进行扫描
    int n = forceField.size();
    int m;
    int ans, cnt;
    vector<pair<double, char>> line1, line2;
    // 比较函数, 区间的起点's'在前
    auto cmp = [](const pair<double, char>& a, const pair<double, char>& b) {
        if (a.first == b.first) {
            return a.second > b.second;
        }
        return a.first < b.first;
    };
    ans = 0;
    for (i = 0; i < n; i++) {
        line1.clear();
        line2.clear();
        for (j = 0; j < n; j++) {
            if (forceField[i][0] - forceField[i][2] / 2.0 <= forceField[j][0] + forceField[j][2] / 2.0 &&
                forceField[i][0] - forceField[i][2] / 2.0 >= forceField[j][0] - forceField[j][2] / 2.0) {
                line1.push_back({forceField[j][1] - forceField[j][2] / 2.0, 's'});
                line1.push_back({forceField[j][1] + forceField[j][2] / 2.0, 'e'});
            }
            if (forceField[i][0] + forceField[i][2] / 2.0 <= forceField[j][0] + forceField[j][2] / 2.0 &&
                forceField[i][0] + forceField[i][2] / 2.0 >= forceField[j][0] - forceField[j][2] / 2.0) {
                line2.push_back({forceField[j][1] - forceField[j][2] / 2.0, 's'});
                line2.push_back({forceField[j][1] + forceField[j][2] / 2.0, 'e'});
            }
        }
        sort(line1.begin(), line1.end(), cmp);
        sort(line2.begin(), line2.end(), cmp);
        cnt = 0;
        m = line1.size();
        for (j = 0; j < m; j++) {
            if (line1[j].second == 's') {
                cnt++;
                ans = max(ans, cnt);
            } else {
                cnt--;
            }
        }
        cnt = 0;
        m = line2.size();
        for (j = 0; j < m; j++) {
            if (line2[j].second == 's') {
                cnt++;
                ans = max(ans, cnt);
            } else {
                cnt--;
            }
        }
    }
    return ans;
}


// LCP56
int conveyorBelt(vector<string>& matrix, vector<int>& start, vector<int>& end)
{
    int i;
    int m = matrix.size();
    int n = matrix[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    // 到达(i, j)的最小修改次数
    vector<vector<int>> dist(m, vector<int>(n, 0x3f3f3f3f));
    auto cmp = [](const pair<vector<int>, int>& a, const pair<vector<int>, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<vector<int>, int>, vector<pair<vector<int>, int>>, decltype(cmp)> pq(cmp);

    // 堆优化dijkstra
    pq.push({start, 0});
    while (pq.size()) {
        auto p = pq.top();
        pq.pop();

        if (dist[p.first[0]][p.first[1]] < p.second) {
            continue;
        }
        dist[p.first[0]][p.first[1]] = p.second;
        for (i = 0; i < 4; i++) {
            auto nr = p.first[0] + directions[i][0];
            auto nc = p.first[1] + directions[i][1];
            if (nr < 0 || nr >= m || nc < 0 || nc >= n) {
                continue;
            }
            if ((matrix[p.first[0]][p.first[1]] == '>' && i != 0) ||
                (matrix[p.first[0]][p.first[1]] == 'v' && i != 1) ||
                (matrix[p.first[0]][p.first[1]] == '<' && i != 2) ||
                (matrix[p.first[0]][p.first[1]] == '^' && i != 3)) {
                if (p.second + 1 < dist[nr][nc]) {
                    dist[nr][nc] = p.second + 1;
                    pq.push({{nr, nc}, dist[nr][nc]});
                }
            } else {
                if (p.second < dist[nr][nc]) {
                    dist[nr][nc] = p.second;
                    pq.push({{nr, nc}, p.second});
                }
            }
        }
    }
    return dist[end[0]][end[1]];
}


// LC1723
int minimumTimeRequired(vector<int>& jobs, int k)
{
    int i, j, z;
    int t, idx;
    int n = jobs.size();
    // dp[k][p] - k个工人完成p工作的最小的最大工作时间, p - 二进制位
    // bits - 二进制位工作量和
    vector<vector<int>> dp(k + 1, vector<int>(1 << n, 0x3f3f3f3f));
    vector<int> bits;

    bits.emplace_back(0);
    for (i = 1; i < (1 << n); i++) {
        t = i;
        idx = 0;
        dp[1][i] = 0;
        while (t) {
            if (t % 2 == 1) {
                dp[1][i] += jobs[idx];
            }
            t >>= 1;
            idx++;
        }
        bits.emplace_back(dp[1][i]);
    }

    for (i = 2; i <= k; i++) {
        for (j = 1; j < (1 << n); j++) {
            // 由奇偶性双倍递增z, 效率更高
            if (j % 2 == 0) {
                z = 2;
            } else {
                z = 1;
            }
            for (; z < j; z += 2) {
                // 判断z是否是j的二进制子集 - 例如 j = 10111, z = 00101, 则z是j的子集
                if ((j | z) > j) {
                    continue;
                }
                // cout << z << " " << j << endl;
                dp[i][j] = min(dp[i][j], max(dp[i - 1][z], bits[j] - bits[z]));
            }
        }
    }
    return dp[k][(1 << n) - 1];
}


// LC3346 LC3347
int maxFrequency(vector<int>& nums, int k, int numOperations)
{
    int n = nums.size();

    unordered_map<int, int> num;
    unordered_set<int> p;
    for (auto n : nums) {
        num[n]++;
        p.emplace(n);
        p.emplace(n - k);
        p.emplace(n + k);
    }
    sort(nums.begin(), nums.end());

    int ans = 0;
    int cnt;
    int left, right, mid;
    int lrange, rrange;
    for (auto it : p) {
        left = 0;
        right = n - 1;
        // 最后一个小于等于 it + k
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (nums[mid] <= it + k) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        rrange = right;

        left = 0;
        right = n - 1;
        // 第一个大于等于 it - k
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (nums[mid] >= it - k) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        lrange = left;

        cnt = min(rrange - lrange + 1 - num[it], numOperations) + num[it];
        ans = max(ans, cnt);
    }
    return ans;
}


// LC3350
int maxIncreasingSubarrays(vector<int>& nums)
{
    int i;
    int n = nums.size();
    vector<int> f(n); // 以nums[i] 结尾最长递增子数组大小
    vector<int> g(n); // 以nums[i] 结尾从右到左最长递减子数组大小

    f[0] = 1;
    for (i = 1; i < n; i++) {
        if (nums[i] > nums[i - 1]) {
            f[i] = f[i - 1] + 1;
        } else {
            f[i] = 1;
        }
    }
    g[n - 1] = 1;
    for (i = n - 2; i >= 0; i--) {
        if (nums[i] < nums[i + 1]) {
            g[i] = g[i + 1] + 1;
        } else {
            g[i] = 1;
        }
    }
    int ans = 1;
    for (i = 0; i < n - 1; i++) {
        ans = max(ans, min(f[i], g[i + 1]));
    }
    return ans;
}


// LC540
int singleNonDuplicate(vector<int>& nums)
{
    int n = nums.size();
    int left, right, mid;

    left = 0;
    right = n - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        if (mid > 0 && mid < n - 1 && nums[mid] != nums[mid - 1] && nums[mid] != nums[mid + 1]) {
            return nums[mid];
        }
        if (mid % 2 == 0) { // odd
            if (mid > 0 && nums[mid] != nums[mid - 1]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        } else {
            if (nums[mid] != nums[mid - 1]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
    return nums[mid];
}


// LC1153
bool canConvert(string str1, string str2)
{
    if (str1 == str2) {
        return true;
    }

    vector<vector<int>> edges(26);
    int i;
    int n = str1.size();
    unordered_set<char> uniqueChar;
    // 无论形成几个环, 只要str2用的唯一字符数小于26, 总能找到一个字符转换这个环
    for (i = 0; i < n; i++) {
        if (edges[str1[i] - 'a'].empty()) {
            edges[str1[i] - 'a'].emplace_back(str2[i] - 'a');
        } else {
            // 不能指向两个节点
            if (edges[str1[i] - 'a'][0] != str2[i] - 'a') {
                return false;
            }
        }
        uniqueChar.emplace(str2[i]);
    }

    return uniqueChar.size() < 26;
}


// LC3355 & LC3356 强相关
bool isZeroArray(vector<int>& nums, vector<vector<int>>& queries)
{
    int i;
    int n = nums.size();
    vector<int> v(n + 1, 0);
    vector<int> diff(n + 1); // v[i] - v[i - 1];

    for (auto q : queries) {
        diff[q[0]]++;
        diff[q[1] + 1]--;
    }
    v[0] = diff[0];
    for (i = 1; i < n; i++) {
        v[i] = diff[i] + v[i - 1];
    }
    // for (auto num : v) cout << num << " ";
    for (i = 0; i < n; i++) {
        if (v[i] < nums[i]) {
            return false;
        }
    }
    return true;
}
int minZeroArray(vector<int>& nums, vector<vector<int>>& queries)
{
    int i;
    int n = nums.size();
    vector<int> v(n + 1, 0);
    vector<int> diff(n + 1); // v[i] - v[i - 1];
    bool ok = true;
    int left, right, mid;

    // 检查原数组是否是全0
    for (i = 0; i < n; i++) {
        if (nums[i] != 0) {
            ok = false;
            break;
        }
    }
    if (ok) {
        return 0;
    }

    left = 0;
    right = queries.size() - 1;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        ok = true;
        v.assign(n + 1, 0);
        diff.assign(n + 1, 0);
        for (i = 0; i <= mid; i++) {
            diff[queries[i][0]] += queries[i][2];
            diff[queries[i][1] + 1] -= queries[i][2];
        }
        v[0] = diff[0];
        if (v[0] < nums[0]) {
            left = mid + 1;
            continue;
        }
        for (i = 1; i < n; i++) {
            v[i] = diff[i] + v[i - 1];
            if (v[i] < nums[i]) {
                left = mid + 1;
                ok = false;
                break;
            }
        }
        if (ok) {
            right = mid - 1;
        }
    }
    if (left == queries.size()) {
        return -1;
    }
    return left + 1;
}


// LC3224
int minChanges(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    int absVal, maxVal;
    int ans;

    // abs(nums[i] - nums[n - 1 - i]) x的范围[0, k], 用差分数组统计得到每个x的改变次数
    vector<int> x(k + 1, 0);
    vector<int> diff(k + 2, 0); // diff[i] = x[i] - x[i - 1];
    for (i = 0; i < n / 2; i++) {
        absVal = abs(nums[i] - nums[n - 1 - i]);
        // [0, absVal - 1] 只需改变一个数
        diff[0]++;
        diff[absVal]--;

        // [absVal + 1, max(k - min(p, q), p, q)] 只需改变一个数
        maxVal = max({k - min(nums[i], nums[n - 1 - i]), nums[i], nums[n - 1 - i]});
        diff[absVal + 1]++;
        diff[maxVal + 1]--;

        // 剩下的情况需要改变两个数
        diff[maxVal + 1] += 2;
        diff[k + 1] -= 2;
    }
    x[0] = diff[0];
    ans = x[0];
    for (i = 1; i <= k; i++) {
        x[i] = diff[i] + x[i - 1];
        ans = min(ans, x[i]);
    }
    return ans;
}


// LC1674 - 与LC3224高度相似
int minMoves(vector<int>& nums, int limit)
{
    // 无论怎么改变两数之和只能在[2, 2 * limit]区间
    int i;
    int n = nums.size();
    int sum;

    // x[n] - 互补数组两数之和为n时操作数
    vector<int> x(limit * 2 + 1, 0);
    vector<int> diff(limit * 2 + 2, 0); // diff[i] = x[i] - x[i - 1];

    for (i = 0; i < n / 2; i++) {
        sum = nums[i] + nums[n - 1 - i];

        // [sum + 1, max(nums[i], nums[n - 1 - i]) + limit] 需要改变一次
        diff[sum + 1]++;
        diff[max(nums[i], nums[n - 1 - i]) + limit + 1]--;

        // [min(nums[i], nums[n - 1 - i]) + 1, sum - 1] 需要改变一次
        diff[min(nums[i], nums[n - 1 - i]) + 1]++;
        diff[sum]--;

        // 其它情况要改变两次
        diff[2] += 2;
        diff[min(nums[i], nums[n - 1 - i]) + 1] -= 2;

        diff[limit * 2 + 1] -= 2;
        diff[max(nums[i], nums[n - 1 - i]) + limit + 1] += 2;
    }
    int ans = 0x3f3f3f3f;
    for (i = 2; i <= limit * 2; i++) {
        x[i] = diff[i] + x[i - 1];
        ans = min(ans, x[i]);
    }
    return ans;
}


// LC446
int numberOfArithmeticSlices(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    int ans;
    // cnt[i][d] - 以nums[i]结尾, 公差为d的子序列个数
    vector<unordered_map<long long, int>> cnt(n);
    // cnt2[i][d] - 以nums[i]结尾, 公差为d的长度为2的子序列个数
    vector<unordered_map<long long, int>> cnt2(n);
    // dp[i] - 以第i位等差子序列个数
    vector<int> dp(n, 0);

    if (n < 3) {
        return 0;
    }

    ans = 0;
    cnt2[1][nums[1] * 1ll - nums[0]] = 1; // 防止溢出
    for (i = 2; i < n; i++) {
        for (j = 0; j <= i - 1; j++) {
            auto d = nums[i] * 1ll - nums[j]; // 防止溢出
            cnt2[i][d]++;
            if (cnt[j].count(d)) {
                dp[i] += cnt[j][d];
                cnt[i][d] += cnt[j][d];
            }
            if (cnt2[j].count(d)) {
                dp[i] += cnt2[j][d];
                cnt[i][d] += cnt2[j][d];
            }
        }
        ans += dp[i];
    }
    return ans;
}


// LC312
int maxCoins(vector<int>& nums)
{
    // 在nums两端添加1, 方便计算
    vector<int> nums1;

    nums1.emplace_back(1);
    nums1.insert(nums1.end(), nums.begin(), nums.end());
    nums1.emplace_back(1);

    int n = nums1.size();
    // dp[i][j] - nums1[i, j] 之间取得最大分数
    vector<vector<int>> dp(n, vector<int>(n, -1));
    // 把戳气球想象成在两个1之间从零添加气球
    function<int (int left, int right)> dfs = [&dfs, &dp, &nums1](int left, int right) {
        if (left + 1 >= right) {
            return 0;
        }
        if (dp[left][right] != -1) {
            return dp[left][right];
        }

        int i;
        int n = nums1.size();
        int a, b;
        int sum, ans;

        ans = 0;
        for (i = left + 1; i < right; i++) {
            a = dfs(left, i);
            b = dfs(i, right);
            sum = nums1[left] * nums1[i] * nums1[right] + a + b;
            ans = max(ans, sum);
        }
        dp[left][right] = ans;
        return ans;
    };

    auto ans = dfs(0, n - 1);
    return ans;
}


// LC847
int shortestPathLength(vector<vector<int>>& graph)
{
    int i;
    int n = graph.size();
    int ans;
    // 由于n很小, 可以用二进制bitmask表示访问状态
    // 多源bfs
    queue<tuple<int, int, int>> q; // {curNode, curBitMask, dist}
    vector<vector<int>> visited(n, vector<int>(1 << n, 0));
    for (i = 0; i < n; i++) {
        q.emplace(i, 1 << i, 0);
        visited[i][1 << i] = 1; // 此处预处理极大增加效率
    }

    int end = (1 << n) - 1;
    while (q.size()) {
        auto [curNode, curBitMask, dist] = q.front();
        q.pop();

        if (curBitMask == end) {
            ans = dist;
            // cout << ans << endl;
            break;
        }
        // visited[curNode][curBitMask] = 1;
        for (auto it : graph[curNode]) {
            auto bitMask = curBitMask | (1 << it);
            if (visited[it][bitMask]) {
                continue;
            }
            visited[it][bitMask] = 1;
            q.emplace(it, bitMask, dist + 1);
        }
    }
    return ans;
}


// LC3366
int minArraySum(vector<int>& nums, int k, int op1, int op2)
{
    int i, j, p;
    int n = nums.size();
    int ans, cnt;
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(op1 + 1, vector<int>(op2 + 1, 0x3f3f3f3f)));

    // 特殊情况处理
    sort(nums.rbegin(), nums.rend());
    ans = 0;
    if (op1 == 0 && op2 == 0) {
        return accumulate(nums.begin(), nums.end(), 0);
    }
    if (op1 == 0) {
        cnt = 0;
        for (i = 0; i < n; i++) {
            if (cnt < op2 && nums[i] >= k) {
                ans += nums[i] - k;
                cnt++;
            } else {
                ans += nums[i];
            }
        }
        return ans;
    }
    if (op2 == 0) {
        cnt = 0;
        for (i = 0; i < n; i++) {
            if (cnt < op1) {
                ans += (nums[i] + 1) / 2;
                cnt++;
            } else {
                ans += nums[i];
            }
        }
        return ans;
    }
    // 注意op1和op2可以处理同一个下标各一次
    // 边界情况处理
    dp[0][0][0] = nums[0];
    dp[0][1][0] = (nums[0] + 1) / 2;
    if (nums[0] >= k) {
        dp[0][0][1] = nums[0] - k;
    }
    dp[0][1][1] = dp[0][1][0] - k >= 0 ? dp[0][1][0] - k : dp[0][1][1];
    dp[0][1][1] = min(dp[0][1][1], dp[0][0][1] == 0x3f3f3f3f ? 0x3f3f3f3f : (dp[0][0][1] + 1) / 2);
    if (n == 1) {
        return min({dp[0][0][0], dp[0][0][1], dp[0][1][0], dp[0][1][1]});
    }
    ans = 0x3f3f3f3f;
    for (i = 1; i < n; i++) {
        for (j = 0; j <= op1; j++) {
            for (p = 0; p <= op2; p++) {
                // 什么都不做
                if (dp[i - 1][j][p] != 0x3f3f3f3f) {
                    dp[i][j][p] = dp[i - 1][j][p] + nums[i];
                }
                // printf("1 dp[%d][%d][%d] = %d\n",i, j, p, dp[i][j][p]);
                // op1
                if (j > 0 && dp[i - 1][j - 1][p] != 0x3f3f3f3f) {
                    dp[i][j][p] = min(dp[i][j][p], dp[i - 1][j - 1][p] + (nums[i] + 1) / 2);
                }
                // printf("2 dp[%d][%d][%d] = %d\n",i, j, p, dp[i][j][p]);
                // op2
                if (p > 0 && dp[i - 1][j][p - 1] != 0x3f3f3f3f && nums[i] >= k) {
                    dp[i][j][p] = min(dp[i][j][p], dp[i - 1][j][p - 1] + (nums[i] - k));
                }
                // printf("3 dp[%d][%d][%d] = %d\n",i, j, p, dp[i][j][p]);
                // op1 && op2
                if (p > 0 && j > 0 && dp[i - 1][j - 1][p - 1] != 0x3f3f3f3f && nums[i] >= k) {
                    auto t = nums[i];
                    t -= k;
                    t = (t + 1) / 2;
                    dp[i][j][p] = min(dp[i][j][p], dp[i - 1][j - 1][p - 1] + t);
                    auto r = nums[i];
                    r = (r + 1) / 2;
                    if (r >= k) {
                        r -= k;
                        dp[i][j][p] = min(dp[i][j][p], dp[i - 1][j - 1][p - 1] + r);
                    }
                }
                // printf("4 dp[%d][%d][%d] = %d\n",i, j, p, dp[i][j][p]);
                if (i == n - 1) {
                    ans = min(ans, dp[i][j][p]);
                }
            }
        }
    }
    return ans;
}


// LC743
int networkDelayTime(vector<vector<int>>& times, int n, int k)
{
    // 标准dijkstra模版
    int i;
    vector<int> dist(n + 1, 0x3f3f3f3f);
    vector<vector<pair<int, int>>> edges(n + 1);

    for (auto t : times) {
        edges[t[0]].emplace_back(make_pair(t[1], t[2]));
    }

    auto cmp = [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);
    pq.push({k, 0});
    while (pq.size()) {
        auto p = pq.top();
        pq.pop();

        if (dist[p.first] < p.second) {
            continue;
        }
        dist[p.first] = p.second;
        for (auto it : edges[p.first]) {
            if (p.second + it.second < dist[it.first]) {
                dist[it.first] = p.second + it.second;
                pq.push({it.first, p.second + it.second});
            }
        }
    }
    int ans = 0;
    for (i = 1; i <= n; i++) {
        if (dist[i] == 0x3f3f3f3f) {
            return -1;
        }
        ans = max(ans, dist[i]);
    }
    return ans;
}


// LC3373
vector<int> BFS(vector<vector<int>>& edges, int start, vector<int>& len)
{
    queue<int> q;
    int dist = 0;
    int n = edges.size();
    vector<int> visited(n, 0);
    vector<int> ans(2, 0);
    q.push(start);
    while (q.size()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            auto node = q.front();
            q.pop();
            visited[node] = 1;
            len[node] = dist;
            ans[dist % 2]++;
            for (auto e : edges[node]) {
                if (visited[e] == 0) {
                    q.push(e);
                }
            }
        }
        dist++;
    }
    return ans;
}
vector<int> maxTargetNodes(vector<vector<int>>& edges1, vector<vector<int>>& edges2)
{
    int i, j;
    int n = edges1.size() + 1;
    int m = edges2.size() + 1;

    vector<vector<int>> e1(n), e2(m);
    for (auto edge : edges1) {
        e1[edge[0]].emplace_back(edge[1]);
        e1[edge[1]].emplace_back(edge[0]);
    }
    for (auto edge : edges2) {
        e2[edge[0]].emplace_back(edge[1]);
        e2[edge[1]].emplace_back(edge[0]);
    }
    // len[j] - start到j的距离
    vector<int> len1(n, 0);
    vector<int> len2(m, 0);
    auto p1 = BFS(e1, 0, len1);
    auto p2 = BFS(e2, 0, len2);
    vector<int> ans(n);
    for (i = 0; i < n; i++) {
        if (len1[i] % 2 == 0) {
            ans[i] = p1[0] + max(p2[0], p2[1]);
        } else {
            ans[i] = p1[1] + max(p2[0], p2[1]);
        }
    }
    return ans;
}


// LC51
vector<vector<string>> solveNQueens(int n)
{
    vector<string> board(n, string(n, '.'));
    vector<vector<string>> ans;
    function<void (int)> SetQueen = [&SetQueen, &board, &ans](int row) {
        int i, j;
        int n = board.size();
        if (row == n) {
            ans.emplace_back(board);
            return;
        }

        int col;
        for (col = 0; col < n; col++) {
            // 校验
            bool condition = true;
            // 纵向
            for (i = row - 1; i >= 0; i--) {
                if (board[i][col] == 'Q') {
                    condition = false;
                    break;
                }
            }
            if (!condition) {
                continue;
            }

            // 左斜向上
            i = row - 1;
            j = col - 1;
            while (i >= 0 && j >= 0) {
                if (board[i][j] == 'Q') {
                    condition = false;
                    break;
                }
                i--;
                j--;
            }
            if (!condition) {
                continue;
            }

            // 右斜向上
            i = row - 1;
            j = col + 1;
            while (i >= 0 && j < n) {
                if (board[i][j] == 'Q') {
                    condition = false;
                    break;
                }
                i--;
                j++;
            }
            if (condition) {
                board[row][col] = 'Q';
                SetQueen(row + 1);
                board[row][col] = '.';
            }
        }
    };

    SetQueen(0);

    return ans;
}


// LC1564
int maxBoxesInWarehouse(vector<int>& boxes, vector<int>& warehouse)
{
    int i;
    int n = warehouse.size();
    int m = boxes.size();
    vector<int> h(n); // h[i] - i处能放的最高行李

    h[0] = warehouse[0];
    for (i = 1; i < n; i++) {
        if (warehouse[i] < h[i - 1]) {
            h[i] = warehouse[i];
        } else {
            h[i] = h[i - 1];
        }
    }

    int left, right, mid;
    int pos = n - 1;
    int ans;
    sort (boxes.begin(), boxes.end());
    for (i = 0; i < m; i++) {
        // h具有单调性, 求每个boxes[i]的最后一个满足 boxes[i] >= h[pos] 位置
        left = 0;
        right = pos;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (h[mid] >= boxes[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            ans = i;
            break;
        }
        pos = right - 1;
    }
    return ans;
}
int maxBoxesInWarehouse_betterWay(vector<int>& boxes, vector<int>& warehouse)
{
    int i, j;
    int n = warehouse.size();
    int m = boxes.size();
    vector<int> h(n); // h[i] - i处能放的最高行李

    h[0] = warehouse[0];
    for (i = 1; i < n; i++) {
        if (warehouse[i] < h[i - 1]) {
            h[i] = warehouse[i];
        } else {
            h[i] = h[i - 1];
        }
    }

    sort (boxes.begin(), boxes.end());
    j = 0;
    for (i = n - 1; i >= 0; i--) {
        if (h[i] >= boxes[j]) {
            j++;
            if (j == m) {
                break;
            }
        }
    }
    return j;
}


// LC3381
long long maxSubarraySum(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    long long curSum = 0;

    for (i = 0; i < k; i++) {
        curSum += nums[i];
    }

    vector<vector<long long>> coll(k);
    coll[0].emplace_back(curSum);
    for (i = k; i < n; i++) {
        curSum -= nums[i - k];
        curSum += nums[i];
        coll[(i + 1) % k].emplace_back(curSum);
    }
    long long ans = LLONG_MIN;
    vector<long long> dp(n);
    for (auto v : coll) {
        if (v.empty()) {
            continue;
        }
        dp[0] = v[0];
        ans = max(ans, dp[0]);
        for (i = 1; i < v.size(); i++) {
            dp[i] = dp[i - 1] > 0 ? dp[i - 1] + v[i] : v[i];
            ans = max(ans, dp[i]);
        }
        // printf ("ans = %d\n", ans);
    }
    return ans;
}


// LC940
int distinctSubseqII(string s)
{
    int i, j;
    int n = s.size();
    int mod = 1e9 + 7;
    long long ans;
    // dp[i] - s[i] 结尾的不同子序列数
    vector<long long> dp(n, 0);
    vector<int> alphabet(26, 0); // 已出现的字符
    vector<long long> cnt(26, 0);

    for (i = 0; i < n; i++) {
        alphabet.assign(26, 0);
        dp[i] = 1;
        for (j = i - 1; j >= 0; j--) {
            if (alphabet[s[j] - 'a'] == 0) {
                dp[i] = (dp[i] + dp[j]) % mod;
                alphabet[s[j] - 'a'] = 1;
            }
        }
        cnt[s[i] - 'a'] = dp[i];
    }
    ans = 0;
    for (auto c : cnt) {
        ans = (ans + c) % mod;
    }
    return ans;
}


// LC3378
int countComponents(vector<int>& nums, int threshold)
{
    int i;
    UnionFind uf = UnionFind(threshold + 1);
    for (auto num : nums) {
        for (i = num * 2; i <= threshold; i += num) {
            uf.unionSets(num, i);
        }
    }

    unordered_set<int> areas;
    int cnt = 0;
    for (auto num : nums) {
        if (num <= threshold) {
            areas.emplace(uf.findSet(num));
        } else {
            cnt++;
        }
    }
    return areas.size() + cnt;
}


// LC3388
int beautifulSplits(vector<int>& nums)
{
    int i, j;
    int n = nums.size();

    if (n < 3) {
        return 0;
    }

    // 数组哈希
    unsigned long long base = 1337;
    vector<vector<unsigned long long>> numsHash(n);

    unsigned long long t;
    for (i = 0; i < n; i++) {
        t = 0;
        for (j = i; j < n; j++) {
            // nums[j] - [0, 50]
            t = t * base + (nums[j] != 0 ? nums[j] * 1ull: 51ull);
            numsHash[i].emplace_back(t);
        }
    }
    int ans = 0;
    // i, j分别表示nums1和nums2的最后一位的下标
    for (i = 0; i < n - 2; i++) {
        for (j = i + 1; j < n - 1; j++) {
            // nums1是nums2前缀
            if (j - i >= i + 1 && numsHash[0][i] == numsHash[i + 1][i]) {
                ans++;
                continue;
            }
            // nums2是nums3前缀
            if (j - i <= n - 1 - j && numsHash[i + 1][j - i - 1] == numsHash[j + 1][j - i - 1]) {
                ans++;
            }
        }
    }
    return ans;
}


// LC1293
int shortestPath(vector<vector<int>>& grid, int k)
{
    int i;
    int m = grid.size();
    int n = grid[0].size();
    int routeLen;
    vector<vector<int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    vector<vector<int>> needBlock(m, vector<int>(n, 0x3f3f3f3f)); // 到达(i, j) 需要最少消除障碍数
    queue<tuple<int, int, int>> q; // 坐标 - x - y - 到达(x, y) 需要的消除障碍数

    // 横竖全打通
    if (k >= m + n - 3) {
        return m + n - 2;
    }

    q.push({0, 0, 0});
    routeLen = 0;
    while (q.size()) {
        int size = q.size();
        for (int p = 0; p < size; p++) {
            auto [x, y, blockCnt] = q.front();
            q.pop();
            if (blockCnt > k || blockCnt >= needBlock[x][y]) {
                continue;
            }
            needBlock[x][y] = blockCnt;
            if (x == m - 1 && y == n - 1) {
                return routeLen;
            }
            for (i = 0; i < 4; i++) {
                auto nx = x + directions[i][0];
                auto ny = y+ directions[i][1];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n) {
                    continue;
                }
                if (grid[nx][ny] == 1) {
                    q.push({nx, ny, blockCnt + 1});
                } else {
                    q.push({nx, ny, blockCnt});
                }
            }
        }
        routeLen++;
    }
    return -1;
}


// LC3394
bool checkValidCuts(int n, vector<vector<int>>& rectangles)
{
    vector<vector<int>> x; // 纵向切割
    vector<vector<int>> y; // 横向切割

    for (auto p : rectangles) {
        x.push_back({p[0], p[2]});
        y.push_back({p[1], p[3]});
    }

    sort (x.begin(), x.end());
    sort (y.begin(), y.end());

    // 分别合并区间, 只要有一个合并完有3个独立的区间, 就可以分割
    auto f = [](vector<vector<int>>& ranges) {
        vector<vector<int>> ans;
        int l, r;
        int i;
        int n = ranges.size();
        l = ranges[0][0];
        r = ranges[0][1];
        for (i = 1; i < n; i++) {
            if (ranges[i][0] < r) {
                r = max(r, ranges[i][1]);
            } else {
                ans.push_back({l, r});
                l = ranges[i][0];
                r = ranges[i][1];
            }
        }
        ans.push_back({l, r});
        return ans;
    };

    return f(x).size() >= 3 || f(y).size() >= 3;
}


// LC3397
int maxDistinctElements(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    unordered_map<int, int> cnt;

    sort(nums.begin(), nums.end());
    nums[0] -= k;
    for (i = 1; i < n; i++) {
        if (nums[i] - nums[i - 1] > k) {
            nums[i] -= k;
        } else {
            if (nums[i] <= nums[i - 1]) {
                if (nums[i - 1] + 1 - nums[i] <= k) {
                    nums[i] = nums[i - 1] + 1;
                } else {
                    nums[i] = nums[i - 1];
                }
            } else {
                nums[i] = nums[i - 1] + 1;
            }
        }
    }
    for (auto n : nums) {
        cnt[n]++;
    }
    return cnt.size();
}


// LC3344
int maxSizedArray(long long s)
{
    // s <= 10^15
    int i, j;
    int left, right, mid;
    long long sum;
    long long multiply;

    left = 0;
    right = 2000;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        sum = 0;
        for (i = 0; i <= mid; i++) {
            for (j = 0; j <= mid; j++) {
                sum += (i | j);
            }
        }
        multiply = mid * 1ll * (mid + 1) / 2;
        if (multiply * sum <= s) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return right + 1;
}


// LC483
string smallestGoodBase(string n)
{
    // n ~ [3, 10^18] < 2^60
    int i, j;
    long long num = atol(n.c_str());
    long long left, right, mid;
    long long cur;
    long long ans = num;
    bool tooBig = false;
    for (i = 2; i <= 60; i++) {
        left = 2;
        right = num - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            cur = 0;
            tooBig = false;
            for (j = 0; j < i; j++) {
                // 溢出
                if (cur != 0 && LLONG_MAX / cur < mid) {
                    right = mid - 1;
                    tooBig = true;
                    break;
                }
                cur = cur * mid + 1;
                if (cur > num) {
                    tooBig = true;
                    right = mid - 1;
                    break;
                }
            }
            if (!tooBig) {
                if (cur < num) {
                    left = mid + 1;
                } else if (cur == num) {
                    ans = min(ans, mid);
                    break;
                }
            }
        }
    }
    return to_string(ans);
}


// LC1705
int eatenApples(vector<int>& apples, vector<int>& days)
{
    int i;
    int n = apples.size();
    int ans = 0;
    map<int, int> nums; // 第i天过期苹果数
    for (i = 0; i < n; i++) {
        if (apples[i] != 0) {
            nums[i + days[i]] += apples[i];
        }
        while (!nums.empty()) {
            auto it = nums.begin();
            if (i >= it->first) {
                nums.erase(it);
                continue;
            }
            ans++;
            if (it->second == 1) {
                nums.erase(it);
            } else {
                nums[it->first]--;
            }
            break;
        }
    }
    // 还有未变质的苹果
    int day = n;
    while (!nums.empty()) {
        auto it = nums.begin();
        if (day >= it->first) {
            nums.erase(it);
            continue;
        }
        ans++;
        day++;
        if (it->second == 1) {
            nums.erase(it);
        } else {
            nums[it->first]--;
        }
    }
    return ans;
}


// LC1847
vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries)
{
    // 房间按面积优先排列
    sort(rooms.begin(), rooms.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    int i, j;
    int prev;
    int n = rooms.size();
    multiset<int> preference;
    bool tooBig = false;
    for (auto v : rooms) {
        preference.emplace(v[0]);
    }

    int m = queries.size();
    vector<vector<int>> qq;
    for (i = 0; i < m; i++) {
        qq.push_back({queries[i][0], queries[i][1], i});
    }
    // 提问也按面积优先排列
    sort(qq.begin(), qq.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    });

    vector<int> ans(m);
    int left, right, mid;
    int diff1, diff2;
    prev = 0;
    for (i = 0; i < m; i++) {
        if (tooBig) {
            ans[qq[i][2]] = -1;
            continue;
        }
        left = 0;
        right = n - 1;
        // 第一个大于等于qq[i][1], left
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (rooms[mid][1] >= qq[i][1]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (left == n) {
            tooBig = true;
            ans[qq[i][2]] = -1;
            continue;
        }
        // 删除面积小于rooms[left]的成员
        for (j = prev; j < left; j++) {
            preference.erase(preference.find(rooms[j][0]));
        }
        prev = left; // 防止重复操作

        diff1 = diff2 = INT_MAX;
        int val1, val2;
        auto it = preference.lower_bound(qq[i][0]);
        if (it != preference.end()) {
            val1 = *it;
            diff1 = abs(qq[i][0] - val1);
            if (it != preference.begin()) {
                val2 = *--it;
                diff2 = abs(qq[i][0] - val2);
            }
            if (diff1 < diff2) {
                ans[qq[i][2]] = val1;
            } else {
                ans[qq[i][2]] = val2;
            }
        } else {
            ans[qq[i][2]] = *--it;
        }
    }
    return ans;
}


// LCP19
int minimumOperations(string leaves)
{
    int i;
    int n = leaves.size();
    // dp[i][0 - 2] - 第i位属于leaves第几部分的最小变换, 同时
    // 要保证leaves[0] = 'r', leaves[n - 1] = 'r', 所求dp[n - 1][2]
    vector<vector<int>> dp(n, vector<int>(3, -1));

    if (leaves[0] == 'y') {
        dp[0][0] = 1;
    } else {
        dp[0][0] = 0;
    }
    for (i = 1; i < n; i++) {
        if (leaves[i] == 'r') {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = dp[i - 1][0] + 1;
            if (dp[i - 1][1] != -1) {
                dp[i][1] = min(dp[i][1], dp[i - 1][1] + 1);
                dp[i][2] = dp[i - 1][1];
                if (dp[i - 1][2] != -1) {
                    dp[i][2] = min(dp[i][2], dp[i - 1][2]);
                }
            }
        } else {
            dp[i][0] = dp[i - 1][0] + 1;
            dp[i][1] = dp[i - 1][0];
            if (dp[i - 1][1] != -1) {
                dp[i][1] = min(dp[i][1], dp[i - 1][1]);
                dp[i][2] = dp[i - 1][1] + 1;
                if (dp[i - 1][2] != -1) {
                    dp[i][2] = min(dp[i][2], dp[i - 1][2] + 1);
                }
            }
        }
    }
    return dp[n - 1][2];
}


// LC3406
string answerString(string word, int numFriends)
{
    if (numFriends == 1) {
        return word;
    }

    // 寻找word的最大后缀
    // 取后缀的前word.size() - numFriends + 1即可
    // 最大后缀 LC1163
    string maxSuffix = lastSubstring(word);
    int n = maxSuffix.size();
    int len = word.size() - numFriends + 1;
    string ans = maxSuffix.substr(0, min(len, n));
    return ans;
}


// LC731 LC732
class MyCalendarTwo_MyCalendarThree {
public:
    vector<vector<int>> ranges;
    vector<vector<int>> rangesTwo;
    int curMax;
    map<int, int> r;
    MyCalendarTwo_MyCalendarThree()
    {
        curMax = 0;
    }

    int book_K(int startTime, int endTime)
    {
        int k = 0;
        r[startTime]++;
        r[endTime]--;
        for (auto it : r) {
            k += it.second;
            curMax = max(curMax, k);
        }
        return curMax;
    }

    // 较为麻烦的实现方式, 更高效方法参考book_K
    bool book_Two(int startTime, int endTime)
    {
        if (ranges.empty()) {
            ranges.push_back({startTime, endTime});
        } else {
            int i;
            int n = rangesTwo.size();
            // 3重预定判断
            for (i = 0; i < n; i++)
            {
                if ((startTime >= rangesTwo[i][0] && startTime < rangesTwo[i][1]) ||
                    (endTime > rangesTwo[i][0] && endTime <= rangesTwo[i][1]) ||
                    (startTime < rangesTwo[i][0] && endTime > rangesTwo[i][1])) {
                    return false;
                }
            }
            n = ranges.size();
            for (i = 0; i < n; i++)
            {
                if (startTime >= ranges[i][0] && startTime < ranges[i][1]) {
                    rangesTwo.push_back({startTime, min(endTime, ranges[i][1])});
                } else if (endTime > ranges[i][0] && endTime <= ranges[i][1]) {
                    rangesTwo.push_back({max(startTime, ranges[i][0]), endTime});
                } else if (startTime < ranges[i][0] && endTime > ranges[i][1]) {
                    rangesTwo.push_back({ranges[i][0], ranges[i][1]});
                }
            }
            ranges.push_back({startTime, endTime});
            sort(ranges.begin(), ranges.end());
            sort(rangesTwo.begin(), rangesTwo.end());
        }
        return true;
    }
};


// LC3413
long long maximumCoins(vector<vector<int>>& coins, int k)
{
    sort(coins.begin(), coins.end());
    int i;
    int n = coins.size();
    vector<long long> prefix(n);

    prefix[0] = coins[0][2] * 1ll * (coins[0][1] - coins[0][0] + 1);
    for (i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] + coins[i][2] * 1ll * (coins[i][1] - coins[i][0] + 1);
    }
    if (k >= coins[n - 1][1] - coins[0][0] + 1) {
        return prefix[n - 1];
    }

    int left, right, mid;
    int pos, diff;
    long long ans, sum;
    ans = 0;
    // 最佳策略 1 从每个区间开始取  2 从每个区间最后从后往前取
    for (i = 0; i < n; i++) {
        left = i;
        right = n - 1;
        pos = coins[i][0] + k - 1;
        // 第一个区间右边界大于等于pos的区间
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (coins[mid][1] < pos) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (left == n) {
            sum = prefix[n - 1] - prefix[i - 1];
            ans = max(ans, sum);
            break;
        }
        if (i == left) {
            sum = k * 1ll * coins[left][2];
        } else {
            diff = pos - coins[left][0] + 1;
            if (i == 0) {
                sum = prefix[left - 1] + (diff > 0 ? diff : 0) * 1ll * coins[left][2];
            } else {
                sum = prefix[left - 1] - prefix[i - 1] + 
                    (diff > 0 ? diff : 0) * 1ll * coins[left][2];
            }
        }
        // printf ("left = %d sum = %lld\n", left, sum);
        ans = max(ans, sum);
    }
    // 从后往前算
    for (i = n - 1; i >= 0; i--) {
        left = 0;
        right = i;
        pos = coins[i][1] - k + 1;
        if (pos < 0) {
            ans = max(ans, prefix[i]);
            break;
        }
        // 最初一个左边界在pos右边的区间
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (coins[mid][0] < pos) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right < 0) {
            ans = max(ans, prefix[i]);
            break;
        } else if (left == n) {
            sum = k * 1ll * coins[left - 1][2];
        } else {
            sum = prefix[i] - prefix[left - 1];
            if (coins[left - 1][1] >= pos) {
                sum += (coins[left - 1][1] - pos + 1) * 1ll * coins[left - 1][2];
            }
        }
        ans = max(ans, sum);
    }
    return ans;
}


// LC972
bool isRationalEqual(string s, string t)
{
    // 分离字符串
    auto f = [](string &s) {
        if (s.find('.') == string::npos) {
            return vector<string>{s, "", ""};
        }
        string a = s.substr(0, s.find('.'));
        if (s.find('(') == string::npos) {
            string b = s.substr(s.find('.') + 1);
            return vector<string>{a, b, ""};
        }
        string b = s.substr(s.find('.') + 1, s.find('(') - s.find('.') - 1);
        string c = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
        // cout << a << " " << b << " " << c;
        return vector<string>{a, b, c};
    };
    auto vs = f(s);
    auto vt = f(t);
    
    // 重新组合
    auto f1 = [](vector<string>& vs) {
        int i;
        int n;
        double a, b, val;
        double dividend, divisor;
        if (!vs[1].empty()) {
            a = atof(vs[1].c_str());
            if (!vs[2].empty()) {
                b = atof(vs[2].c_str());
                dividend = a * pow(10, vs[2].size()) + b - a;
                divisor = 0;
                n = vs[2].size();
                for (i = 0; i < n; i++) {
                    divisor = divisor * 10 + 9;
                }
                divisor *= pow(10, vs[1].size());
                val = dividend / divisor;
            } else {
                n = vs[1].size();
                val = a / pow(10, n);
            }
        } else {
            if (!vs[2].empty()) {
                b = atof(vs[2].c_str());
                dividend = b;
                divisor = 0;
                n = vs[2].size();
                for (i = 0; i < n; i++) {
                    divisor = divisor * 10 + 9;
                }
                val = dividend / divisor;
            } else {
                val = 0.0;
            }
        }
        return val + atof(vs[0].c_str());
    };
    double dvs, dvt;
    dvs = f1(vs);
    dvt = f1(vt);
    // cout << dvs << " " << dvt << endl;
    return dvs == dvt;
}


// LC3418
int maximumAmount(vector<vector<int>>& coins)
{
    int inf = -0x3f3f3f3f;
    int i, j, k;
    int m = coins.size();
    int n = coins[0].size();
    vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(3, inf)));

    if (coins[0][0] < 0) {
        dp[0][0][1] = 0;
    }
    dp[0][0][0] = coins[0][0];
    for (i = 1; i < m; i++) {
        for (k = 0; k <= 2; k++) {
            // 不"感化"
            if (dp[i - 1][0][k] != inf) {
                dp[i][0][k] = dp[i - 1][0][k] + coins[i][0];
            }
            // "感化"
            if (coins[i][0] < 0 && k > 0 && dp[i - 1][0][k - 1] != inf) {
                dp[i][0][k] = max(dp[i][0][k], dp[i - 1][0][k - 1]);
            }
        }
    }
    for (j = 1; j < n; j++) {
        for (k = 0; k <= 2; k++) {
            if (dp[0][j - 1][k] != inf) {
                dp[0][j][k] = dp[0][j - 1][k] + coins[0][j];
            }
            if (coins[0][j] < 0 && k > 0 && dp[0][j - 1][k - 1] != inf) {
                dp[0][j][k] = max(dp[0][j][k], dp[0][j - 1][k - 1]);
            }
        }
    }
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            for (k = 0; k <= 2; k++) {
                if (dp[i - 1][j][k] != inf) {
                    dp[i][j][k] = dp[i - 1][j][k] + coins[i][j];
                }
                if (coins[i][j] < 0 && k > 0 && dp[i - 1][j][k - 1] != inf) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k - 1]);
                }
            }
            for (k = 0; k <= 2; k++) {
                if (dp[i][j - 1][k] != inf) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j - 1][k] + coins[i][j]);
                }
                if (coins[i][j] < 0 && k > 0 && dp[i][j - 1][k - 1] != inf) {
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j - 1][k - 1]);
                }
            }
        }
    }
    /*
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k <= 2; k++) {
                printf ("%d ", dp[i][j][k]);
            }
            cout << endl;
        }
    }
    */
    return *max_element(dp[m - 1][n - 1].begin(), dp[m - 1][n - 1].end());
}


// LC3419
int minMaxWeight(int n, vector<vector<int>>& edges, int threshold)
{
    int i;
    int maxVal, minVal;

    maxVal = 0;
    minVal = 0x3f3f3f3f;
    vector<vector<pair<int, int>>> edgesWithWeight(n);
    // 存反向边
    for (auto e : edges) {
        edgesWithWeight[e[1]].push_back({e[0], e[2]});
        maxVal = max(maxVal, e[2]);
        minVal = min(minVal, e[2]);
    }

    vector<int> visited(n, 0);
    int left, right, mid;

    left = minVal;
    right = maxVal;
    // 带权的连通性判断
    function<void (int, int)> dfs_w = [&dfs_w, &edgesWithWeight, &visited](int cur, int w) {
        visited[cur] = 1;
        for (auto it : edgesWithWeight[cur]) {
            if (visited[it.first] != 1 && it.second <= w) {
                dfs_w(it.first, w);
            }
        }
    };
    // 先判断一次连通性
    visited.assign(n, 0);
    dfs_w(0, maxVal);
    for (auto it : visited) {
        if (it == 0) {
            return -1;
        }
    }
    while (left <= right) {
        mid = (right - left) / 2 + left;
        visited.assign(n, 0);
        // 从节点0开始判断权重为mid连通性
        dfs_w(0, mid);
        for (i = 0; i < n; i++) {
            if (visited[i] == 0) {
                left = mid + 1;
                break;
            }
        }
        if (i == n) {
            right = mid - 1;
        }
    }
    return left > maxVal ? -1 : left;
}


// LC3428
int minMaxSums(vector<int>& nums, int k)
{
    int mod = 1e9 + 7;
    int i, j;
    int n = nums.size();
    int len, cnt;
    long long ans = 0;
    vector<vector<long long>> prefix(n, vector<long long>(k)); // 从n个数取0 - k - 1个数的前缀和

    prefix[0][0] = 1;
    for (i = 1; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (i < j) {
                break;
            }
            if (j == 0) {
                prefix[i][j] = 1;
            } else if (j == 1) {
                prefix[i][j] = prefix[i][j - 1] + i;
            } else if (j == i) {
                prefix[i][j] = prefix[i][j - 1] + 1;
            } else {
                // 组合数的性质
                prefix[i][j] = (prefix[i][j - 1] + prefix[i - 1][j] - prefix[i - 1][j - 2] + mod) % mod;
            }
        }
    }
    sort(nums.begin(), nums.end());
    for (i = 0; i < n; i++) {
        // nums[i] 作为最小数 从i + 1 取 k - 1个数
        len = n - 1 - i;
        cnt = prefix[len][min(len, k - 1)];
        ans = (ans + nums[i] * 1ll * cnt) % mod;

        // nums[i] 作为最大数 从i - 1 取 k - 1个数
        len = i;
        cnt = prefix[len][min(len, k - 1)];
        ans = (ans + nums[i] * 1ll * cnt) % mod;
    }
    return ans;
}


// LC3429
long long minCost_LC3429(int n, vector<vector<int>>& cost)
{
    int i, j, k;
    int jj, kk;
    // dp[i][j][k] - 第i个房子和第n - 1 - i分别涂成j、k两种颜色的最低成本
    vector<vector<vector<long long>>> dp(n / 2, vector<vector<long long>>(3, vector<long long>(3, LLONG_MAX / 2)));

    for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
            if (j == k) {
                continue;
            }
            dp[0][j][k] = cost[0][j] + cost[n - 1][k];
        }
    }
    for (i = 1; i < n / 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                if (k == j) {
                    continue;
                }
                for (jj = 0; jj < 3; jj++) {
                    for (kk = 0; kk < 3; kk++) {
                        if (jj == kk || jj == j || kk == k) {
                            continue;
                        }
                        dp[i][j][k] = min(dp[i][j][k], 
                            dp[i - 1][jj][kk] + cost[i][j] + cost[n - 1 - i][k]);
                    }
                }
            }
        }
    }
    long long ans = LLONG_MAX / 2;
    for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
            ans = min(ans, dp[n / 2 - 1][j][k]);
        }
    }
    /*for (i = 0; i < n / 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                cout << dp[i][j][k] << " ";
            }
        }
        cout << endl;
    }*/
    return ans;
}


// LC2218
int maxValueOfCoins(vector<vector<int>>& piles, int k)
{
    int i, j, p;
    int n = piles.size();
    int m;

    vector<vector<int>> prefix(n); // 每个栈的前缀和
    for (i = 0; i < n; i++) {
        prefix[i].emplace_back(piles[i][0]);
        m = piles[i].size();
        for (j = 1; j < m; j++) {
            prefix[i].emplace_back(prefix[i].back() + piles[i][j]);
        }
    }
    // 问题转化为 : 每个栈有体积为 1 - m (piles[i].size()) 的物品 价值分别为prefix[i][0 - m] 求凑成体积k的最大选择
    vector<vector<int>> dp(n, vector<int>(k + 1, -1)); // 前n个pile取k个最大面值和
    for (i = 0; i < n; i++) {
        dp[i][0] = 0;
    }
    m = prefix[0].size();
    for (i = 0; i < min(m, k); i++) {
        dp[0][i + 1] = prefix[0][i];
    }
    for (i = 1; i <= k; i++) {
        for (j = 1; j < n; j++) {
            m = prefix[j].size();
            dp[j][i] = dp[j - 1][i]; // 不选第j个栈
            for (p = 0; p < m; p++) {
                if (i - (p + 1) >= 0 && dp[j - 1][i - (p + 1)] != -1) {
                    dp[j][i] = max(dp[j][i], dp[j - 1][i - (p + 1)] + prefix[j][p]);
                }
            }
        }
    }
    int ans = 0;
    for (i = 0; i < n; i++) {
        ans = max(ans, dp[i][k]);
    }
    return ans;
}


// LC2378
long long maxScore_LC2378(vector<vector<int>>& edges)
{
    // 类似"打家劫舍III", 只不过此处是边的权重
    int i;
    int n = edges.size();
    vector<vector<pair<int, long long>>> e(n);
    for (i = 1; i < n; i++) {
        e[edges[i][0]].push_back({i, edges[i][1]});
        e[i].push_back({edges[i][0], edges[i][1]});
    }

    vector<vector<long long>> vals(n, vector<long long>(2, -1)); // vals[i][0 - 1] 节点i的父节点是否被选择所得到的最大权之和
    function<long long (int, int, int)> dfs = [&dfs, &e, &vals](int cur, int parent, int taken) {
        int i, j;
        long long val1, val2;
        long long ret = 0;
        if (vals[cur][taken] != -1) {
            return vals[cur][taken];
        }
        if (e[cur].size() == 1 && e[cur][0].first == parent) {
            vals[cur][taken] = 0;
            return 0ll;
        }

        vector<pair<int, long long>> nodes;
        if (taken == 0) {
            for (auto it : e[cur]) {
                if (it.first != parent) {
                    nodes.push_back(it);
                }
            }
            if (nodes.size() == 1) {
                ret = max({ret, dfs(nodes[0].first, cur, 1) + nodes[0].second, 
                    dfs(nodes[0].first, cur, 0)});
            } else {
                // 取其中一条边
                for (i = 0; i < nodes.size(); i++) {
                    val1 = dfs(nodes[i].first, cur, 1) + nodes[i].second;
                    for (j = 0; j < nodes.size(); j++) {
                        if (i == j) {
                            continue;
                        }
                        val2 = dfs(nodes[j].first, cur, 0);
                        if (val2 > 0) {
                            val1 += val2;
                        }

                    }
                    ret = max(ret, val1);
                }
                // 都不取
                val1 = 0;
                for (i = 0; i < nodes.size(); i++) {
                    val2 = dfs(nodes[i].first, cur, 0);
                    if (val2 > 0) {
                        val1 += val2;
                    }
                }
                ret = max(ret, val1);
            }
        } else {
            val1 = 0;
            for (auto it : e[cur]) {
                if (it.first != parent) {
                    val2 = dfs(it.first, cur, 0);
                    if (val2 > 0) {
                        val1 += val2;
                    }
                }
            }
            ret = val1;
        }
        vals[cur][taken] = ret;
        return ret;
    };

    return max(dfs(0, -1, 0), dfs(0, -1, 1));
}


// LC879
int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit)
{
    int i, j, k;
    int mod = 1e9 + 7;
    int m = profit.size();
    // dp[i][j][k] - 前i个项目雇佣k个人且盈利j的方案数
    vector<vector<vector<long long>>> dp(m + 1, vector<vector<long long>>(minProfit + 1, vector<long long>(n + 1, -1)));

    long long ans = 0;

    dp[0][0][0] = 1;
    for (j = 0; j <= minProfit; j++) {
        for (k = 0; k <= n; k++) {
            for (i = 1; i <= m; i++) {
                if (dp[i - 1][j][k] != -1) {
                    // 不选profit[i - 1]
                    if (dp[i][j][k] == -1) {
                        dp[i][j][k] = dp[i - 1][j][k];
                    } else {
                        dp[i][j][k] = (dp[i][j][k] + dp[i - 1][j][k]) % mod;
                    }
                    // 选择profit[i - 1]
                    if (group[i - 1] + k <= n) {
                        auto jj = min(j + profit[i - 1], minProfit);
                        // auto jjj = max(j + profit[i - 1], minProfit);
                        if (dp[i][jj][group[i - 1] + k] == -1) {
                            dp[i][jj][group[i - 1] + k] = dp[i - 1][j][k];
                        } else {
                            dp[i][jj][group[i - 1] + k] = (dp[i][jj][group[i - 1] + k] + dp[i - 1][j][k]) % mod;
                        }
                    }
                }
            }
        }
    }

    for (i = 0; i <= n; i++) {
        // cout << dp[m][minProfit][i] << " ";
        if (dp[m][minProfit][i] != -1) {
            ans = (ans + dp[m][minProfit][i]) % mod;
        }
    }
    return ans;
}


// LC2412
long long minimumMoney(vector<vector<int>>& transactions)
{
    int n, i;
    int tt; // 不赔钱交易需要的最大初始金额
    long long left, right, mid;
    long long cur, ans;
    vector<vector<int>> v;

    tt = 0;
    for (auto t : transactions) {
        if (t[0] > t[1]) {
            v.emplace_back(t);
        } else {
            tt = max(tt, t[0]);
        }
    }
    if (v.empty()) {
        return tt;
    }

    sort(v.begin(), v.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] == b[1]) {
            return a[0] > b[0];
        }
        return a[1] < b[1];
    });

    left = 0;
    right = 1e15;
    n = v.size();
    while (left <= right) {
        mid = (right - left) / 2 + left;
        cur = mid;
        for (i = 0; i < n; i++) {
            if (cur < v[i][0]) {
                left = mid + 1;
                break;
            } else {
                cur = cur - v[i][0] + v[i][1];
            }
        }
        if (i == n) {
            right = mid - 1;
        }
    }
    ans = left;
    cur = v[n - 1][1];
    if (cur < tt) {
        ans += tt - cur;
    }
    return ans;
}


// LC40
vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());

    int i, sum;
    int n = candidates.size();
    vector<int> record;
    vector<vector<int>> ans;
    vector<int> prefix(n);

    i = 0;
    sum = 0;
    for (auto it : candidates) {
        sum += it;
        prefix[i] = sum;
        i++;
    }
    if (sum < target) {
        return {};
    } else if (sum == target) {
        return {candidates};
    }

    unordered_set<string> us;
    function<void (int, int)> dfs = [&dfs, &record, &target, &candidates, &prefix, 
        &us, &ans](int idx, int cur) {
        int i, j;
        int n = candidates.size();
        int m;
        string t;
        if (cur == target) {
            // 去重 形如 [1,2,5] [1,7] [1,2,5]
            t.clear();
            for (auto it : record) {
                t += to_string(it) + "_";
            } 
            if (us.count(t) == 0) {
                ans.emplace_back(record);
                us.emplace(t);
            }
            return;
        } else if (cur > target) {
            return;
        }
        if (idx != 0 && cur + prefix[n - 1] - prefix[idx - 1] < target) {
            return;
        }
        
        for (i = idx; i < n; i++) {
            // 关键去重
            if (!ans.empty() && !record.empty() && ans.back().size() > record.size()) {
                m = record.size();
                for (j = 0; j < m; j++) {
                    if (ans.back()[j] != record[j]) {
                        break;
                    }
                }
                if (j == m && ans.back()[m] == candidates[i]) {
                    continue;
                }
            }
            record.push_back(candidates[i]);
            dfs(i + 1, cur + candidates[i]);
            record.pop_back();
        }
    };

    dfs(0, 0);
    return ans;
}


// LC1192
vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections)
{
    // Tarjan
    int i;
    int idx;
    vector<vector<int>> bridges;
    vector<int> visited(n, 0);
    vector<int> dfs_num(n);
    vector<int> low(n);

    vector<vector<int>> edges(n);
    for (auto c : connections) {
        edges[c[0]].emplace_back(c[1]);
        edges[c[1]].emplace_back(c[0]);
    }

    idx = 0;
    function<void (int, int)> dfs = [&dfs, &idx, &visited, &dfs_num, &low, &edges, &bridges]
    (int cur, int parent) {
        visited[cur] = 1;
        dfs_num[cur] = low[cur] = idx;
        idx++;

        for (auto it : edges[cur]) {
            if (it == parent) {
                continue;
            }
            if (visited[it] == 0) {
                dfs(it, cur);
                low[cur] = min(low[cur], low[it]);
                if (dfs_num[cur] < low[it]) {
                    bridges.push_back({cur, it});
                }
            } else {
                low[cur] = min(low[cur], dfs_num[it]);
            }
        }
    };

    for (i = 0; i < n; i++) {
        if (visited[i] == 0) {
            dfs(i, - 1);
        }
    }
    return bridges;
}


// LC45
int jump(vector<int>& nums)
{
    // dijkstra
    int i;
    int idx;
    int n = nums.size();
    vector<int> dist(n, INT_MAX);
    queue<pair<int, int>> q;

    q.push({0, 0});
    while (q.size()) {
        auto [idx, step] = q.front();
        q.pop();
        if (dist[idx] < step) {
            continue;
        }
        dist[idx] = step;
        for (i = 1; i <= nums[idx]; i++) {
            if (idx + i < n && step + 1 < dist[idx + i]) {
                dist[idx + i] = step + 1;
                q.push({idx + i, step + 1});
            }
        }
    }
    return dist[n - 1];
}


// LC1514
double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start_node, int end_node)
{
    // dijkstra 与上题类似
    int i;
    int m = edges.size();
    vector<double> dist(n, 0.0);
    queue<pair<int, double>> q;
    vector<vector<pair<int, double>>> e(n);
    for (i = 0; i < m; i++) {
        e[edges[i][0]].push_back({edges[i][1], succProb[i]});
        e[edges[i][1]].push_back({edges[i][0], succProb[i]});
    }

    q.push({start_node, 1.0});
    while (q.size()) {
        auto [node, val] = q.front();
        q.pop();
        if (dist[node] > val) {
            continue;
        }
        dist[node] = val;
        for (auto next : e[node]) {
            if (val * next.second > dist[next.first]) {
                dist[next.first] = val * next.second;
                q.push({next.first, dist[next.first]});
            }
        }
    }
    return dist[end_node];
}


// LC1546
int maxNonOverlapping(vector<int>& nums, int target)
{
    int i;
    int n = nums.size();
    int ans = 0;
    vector<int> dp(n, 0); // dp[i] - 以nums[i]结尾的最大子数组数
    vector<int> prefix(n, 0);
    unordered_map<int, int> sumIdx; // 子数组和的最后下标

    sumIdx[nums[0]] = 0;
    prefix[0] = nums[0];
    if (nums[0] == target) {
        dp[0] = 1;
        ans = 1;
    }
    for (i = 1; i < n; i++) {
        dp[i] = dp[i - 1];
        prefix[i] = nums[i] + prefix[i - 1];
        if (prefix[i] == target) {
            dp[i] = max(dp[i], 1);
        }
        if (nums[i] == target) {
            dp[i] = max(dp[i], dp[i - 1] + 1);
        }
        if (sumIdx.count(prefix[i] - target)) {
            dp[i] = max(dp[i], dp[sumIdx[prefix[i] - target]] + 1);
        }
        sumIdx[prefix[i]] = i;
        ans = max(ans, dp[i]);
    }
    return ans;
}


// LC3440
int maxFreeTime(int eventTime, vector<int>& startTime, vector<int>& endTime)
{
    int i;
    int n = startTime.size();
    int diff;
    vector<int> freeTime;
    map<int, int> dist; // dist[距离] = 个数

    dist[startTime[0]]++;
    freeTime.emplace_back(startTime[0]);
    for (i = 1; i < n; i++) {
        diff = startTime[i] - endTime[i - 1];
        freeTime.emplace_back(diff);
        dist[diff]++;
    }
    diff = eventTime - endTime[n - 1];
    dist[diff]++;
    freeTime.emplace_back(diff);

    int ans = 0;
    int cur, len;
    for (i = 0; i < n; i++) {
        if (i != 0 && i != n - 1) {
            cur = startTime[i + 1] - endTime[i - 1];
            if (cur <= ans) {
                continue;
            }
        }
        cur = freeTime[i] + freeTime[i + 1];
        // printf ("cur = %d\n", cur);
        len = endTime[i] - startTime[i];
        if (dist[freeTime[i]] == 1) {
            dist.erase(freeTime[i]);
        } else {
            dist[freeTime[i]]--;
        }
        if (dist[freeTime[i + 1]] == 1) {
            dist.erase(freeTime[i + 1]);
        } else {
            dist[freeTime[i + 1]]--;
        }
        // 还有能匹配len的空间
        if (dist.lower_bound(len) != dist.end()) {
            if (i == 0) {
                cur = startTime[i + 1];
            } else if (i == n - 1) {
                cur = eventTime - endTime[n - 2];
            } else {
                cur = startTime[i + 1] - endTime[i - 1];
            }
        }
        ans = max(ans, cur);
        dist[freeTime[i]]++;
        dist[freeTime[i + 1]]++;
    }
    return ans;
}


// LC3443
int maxDistance(string s, int k)
{
    int i;
    int n = s.size();
    int t, a, b, cur;
    vector<int> v;
    vector<vector<int>> dir(n, vector<int>(4, 0));
    int ans = 0;
    for (i = 0; i < n; i++) {
        if (i != 0) {
            dir[i] = dir[i - 1];
        }
        if (s[i] == 'E') {
            dir[i][0]++;
        } else if (s[i] == 'W') {
            dir[i][1]++;
        } else if (s[i] == 'N') {
            dir[i][2]++;
        } else {
            dir[i][3]++;
        }
        v = dir[i];
        t = k;
        cur = 0;
        a = min(v[0], v[1]);
        b = max(v[0], v[1]);
        if (a < t) {
            cur += b + a;
            t -= a;
        } else {
            cur += b + t - (a - t);
            t = 0;
        }
        if (t != 0) {
            a = min(v[2], v[3]);
            b = max(v[2], v[3]);
            if (a < t) {
                cur += b + a;
                t -= a;
            } else {
                cur += b + t - (a - t);
                t = 0;
            }
        } else {
            cur += abs(v[2] - v[3]);
        }
        ans = max(ans, cur);
    }
    return ans;
}


// 面试题 16.26. 计算器
int calculate(string s)
{
    auto Trim = [](string &s, char separate) {
        string ans;
        for (auto ch : s) {
            if (ch == separate) {
                continue;
            }
            ans += ch;
        }
        return ans;
    };
    auto Calc = [](int a, int b, char sign) -> optional<int> {
        switch (sign) {
            case '+' : return a + b;
            case '-' : return a - b;
            case '*' : return a * b;
            case '/' : {
                if (b == 0) {
                    return nullopt;
                }
                return a / b;
            }
            default : return nullopt;
        }
    };

    string t = Trim(s, ' ');
    stack<char> sign;
    stack<int> nums;
    int i, num;
    int n = t.size();
    int a, b;

    num = 0;
    for (i = 0; i < n; i++) {
        if (isdigit(t[i])) {
            num = num * 10 + (t[i] - '0');
        } else {
            nums.push(num);
            num = 0;
            while (!sign.empty()) {
                auto last_sign = sign.top();
                if (t[i] == '*' || t[i] == '/') {
                    if (last_sign == '+' || last_sign == '-') {
                        break;
                    }
                }
                a = nums.top();
                nums.pop();
                b = nums.top();
                nums.pop();
                auto c = Calc(b, a, last_sign);
                sign.pop();
                nums.push(*c);
            }
            sign.push(t[i]);
        }
    }
    nums.push(num);
    while (!sign.empty()) {
        auto last_sign = sign.top();
        a = nums.top();
        nums.pop();
        b = nums.top();
        nums.pop();
        auto c = Calc(b, a, last_sign);
        sign.pop();
        nums.push(*c);
    }
    return nums.top();
}


// LC1234
int balancedString(string s)
{
    int i, j;
    int n = s.size();
    int limit = n / 4;
    int cnt;
    vector<vector<int>> prefix(n, vector<int>(4, 0));
    for (i = 0; i < n; i++) {
        if (i != 0) {
            prefix[i] = prefix[i - 1];
        }
        if (s[i] == 'Q') {
            prefix[i][0]++;
        } else if (s[i] == 'W') {
            prefix[i][1]++;
        } else if (s[i] == 'E') {
            prefix[i][2]++;
        } else {
            prefix[i][3]++;
        }
    }
    if (prefix[n - 1][0] == limit &&
        prefix[n - 1][1] == limit &&
        prefix[n - 1][2] == limit &&
        prefix[n - 1][3] == limit) {
        return 0;
    }

    int left, right, mid;
    int ans = n;
    for (i = 0; i < n; i++) {
        if (i > 0) {
            for (j = 0; j < 4; j++) {
                if (prefix[i - 1][j] > limit) {
                    break;
                }
            }
            if (j != 4) {
                break;
            }
        }
        left = i;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            for (j = 0; j < 4; j++) {
                if (i == 0) {
                    cnt = prefix[n - 1][j] - prefix[mid][j];
                } else {
                    cnt = prefix[i - 1][j] + prefix[n - 1][j] - prefix[mid][j];
                }
                if (cnt > limit) {
                    left = mid + 1;
                    break;
                }
            }
            if (j == 4) {
                right = mid - 1;
            }
        }
        ans = min(ans, left - i + 1);
    }
    return ans;
}


// LC3447
vector<int> assignElements(vector<int>& groups, vector<int>& elements)
{
    int i, j;
    int n = groups.size();
    int m = elements.size();
    vector<int> ans(n, -1);
    unordered_map<int, int> p;

    for (i = 0; i < m; i++) {
        if (p.count(elements[i]) == 0) {
            p[elements[i]] = i;
        }
    }

    for (i = 0; i < n; i++) {
        if (p.count(groups[i])) {
            ans[i] = p[groups[i]];
        }
        for (j = 1; j <= sqrt(groups[i]); j++) {
            if (groups[i] % j == 0 && p.count(j)) { 
                if (ans[i] == -1) {
                    ans[i] = m;
                }
                ans[i] = min(ans[i], p[j]);
            }
            if (groups[i] % j == 0 && p.count(groups[i] / j)) {
                if (ans[i] == -1) {
                    ans[i] = m;
                }
                ans[i] = min(ans[i], p[groups[i] / j]);
            }
        }
    }
    return ans;
}


// LC3448
long long countSubstrings(string s)
{
    // 思路无误, 时间复杂度无误, 然而oj卡常, 使用传统malloc + free可过
    int i, j, k;
    int n = s.size();
    int d;
    long long ans;
    // dp[i][1 - 9][0 - 9] - 第i位结束能被 1 - 9 整除的余数为0 - 9的字符串总数
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(10, vector<long long>(10, 0)));

    ans = 0;
    for (i = 0; i < n; i++) {
        d = s[i] - '0';
        for (j = 1; j <= 9; j++) {
            dp[i][j][d % j] = 1;
            if (i == 0) {
                continue;
            }
            for (k = 0; k <= 9; k++) {
                if (k >= j) {
                    break;
                }
                if (j == 2 && k == 1) {
                    auto tt =12;
                }
                dp[i][j][(k * 10 + d) % j] += dp[i - 1][j][k];
            }
        }
        ans += dp[i][d][0];
    }
    return ans;
}


// LC3453 浮点数二分
double separateSquares(vector<vector<int>>& squares)
{
    double left, right, mid;
    right = 0.0;
    // 测试数据出现卡常, 要么传统下标, 要么引用
    for (auto& it : squares) {
        right = max(right, it[1] * 1.0 + it[2]);
    }

    double top, bottom, square, part;

    left = 0.0;
    while (fabs(right - left) > 1e-5) {
        mid = (right - left) / 2 + left;
        top = bottom = 0.0;
        for (auto& it : squares) {
            square = it[2] * 1.0 * it[2];
            if (it[1] - mid > 1e-5 || fabs(it[1] - mid) < 1e-5) {
                top += square;
            } else if (it[1] + it[2] - mid < 1e-5 || fabs(it[1] + it[2] - mid) < 1e-5) {
                bottom += square;
            } else {
                part = square * (mid - it[1]) / it[2];
                bottom += part;
                top += square - part;
            }
        }

        if (top > bottom) {
            left = mid;
        } else {
            right = mid;
        }
        if (fabs(right - left) < 1e-5) {
            break;
        }
    }
    return right;
}


// LC3458
bool maxSubstringLength(string s, int k)
{
    if (k == 0) {
        return true;
    }

    int i, j;
    int n = s.size();

    vector<vector<int>> ranges(26, vector<int>(2, - 1));
    for (i = 0; i < n; i++) {
        if (ranges[s[i] - 'a'][0] == -1) {
            ranges[s[i] - 'a'][0] = i;
            ranges[s[i] - 'a'][1] = i;
        } else {
            ranges[s[i] - 'a'][1] = i;
        }
    }

    // 查找每个字符的特殊子字符串边界, 注意扩边
    set<vector<int>> r; // 方便去重
    int left, right;
    int flag;
    for (i = 0; i < 26; i++) {
        if (ranges[i][0] == -1) {
            continue;
        }
        left = ranges[i][0];
        right = ranges[i][1];
        flag = 1;
        while (flag) {
            for (j = left; j <= right; j++) {
                if (ranges[s[j] - 'a'][0] < left) {
                    left = ranges[s[j] - 'a'][0];
                    right = max(right, ranges[s[j] - 'a'][1]);
                    break;
                }
                if (ranges[s[j] - 'a'][1] > right) {
                    right = ranges[s[j] - 'a'][1];
                    break;
                }
            }
            if (j == right + 1) {
                flag = 0;
            }
        }
        // printf ("%c %d %d\n", 'a' + i, left, right);
        if (left == 0 && right == n - 1) {
            continue;
        }
        r.insert({left, right});
    }
    vector<vector<int>> vt;
    for (auto it : r) {
        vt.emplace_back(it);
    }

    // 当前即判断vt中有无k个不重叠区间
    n = vt.size();
    if (vt.empty() || n < k) {
        return false;
    }
    if (k == 1) {
        return true;
    }
    // dp[i] - 以vt[i]结束能得到的最大不重叠区间数
    vector<int> dp(n, 0);
    dp[0] = 1;
    for (i = 1; i < n; i++) {
        dp[i] = 1;
        for (j = i - 1; j >= 0; j--) {
            if (vt[i][0] >= vt[j][1]) {
                dp[i] = max(dp[i], dp[j] + 1); 
            }
        }
        if (dp[i] >= k) {
            return true;
        }
    }
    return false;
}


// LC1036
bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target)
{
    int n = blocked.size();
    // 长度为n的block最多能围城的面积(三角形) sum = 1 + 2 + 3 + ... + n - 1
    // 即起点和终点能达到的点大于sum则无法被包围
    int sum;
    int len = 1e6;
    unordered_set<long long> visited;
    unordered_set<long long> block;

    for (auto b : blocked) {
        block.emplace(b[0]* 1ll * len + b[1]);
    }

    sum = n * (n - 1) / 2;
    auto BFS = [&len, &block, &visited](int row, int col, int limit, vector<int>& target) {
        queue<pair<int, int>> q;
        q.push({row, col});
        visited.emplace(row * 1ll * len + col);
        while (!q.empty()) {
            auto [row, col] = q.front();
            q.pop();
            if (target[0] == row && target[1] == col) {
                return true;
            }

            int i;
            int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
            for (i = 0; i < 4; i++) {
                auto nr = row + directions[i][0];
                auto nc = col + directions[i][1];
                if (nr < 0 || nr >= len || nc < 0 || nc >= len || visited.count(nr * 1ll * len + nc) || 
                    block.count(nr * 1ll * len + nc)) {
                    continue;
                }
                q.push({nr, nc});
                visited.emplace(nr * 1ll * len + nc);
                if (visited.size() > limit) {
                    return true;
                }
            }
        }
        return false;
    };

    visited.clear();
    bool f1 = BFS(source[0], source[1], sum, target);

    visited.clear();
    bool f2 = BFS(target[0], target[1], sum, source);

    return f1 && f2;
}


// LC474
int findMaxForm(vector<string>& strs, int m, int n)
{
    int i, j, k;
    int len = strs.size();
    int cnt0, cnt1;
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (i = 0; i < len; i++) {
        cnt0 = cnt1 = 0;
        for (auto ch : strs[i]) {
            if (ch == '0') {
                cnt0++;
            } else {
                cnt1++;
            }
        }
        for (j = m; j >= cnt0; j--) {
            for (k = n ; k >= cnt1; k--) {
                dp[j][k] = max(dp[j][k], dp[j - cnt0][k - cnt1] + 1);
            }
        }
    }
    return dp[m][n];
}


// LC3469
int minCost_LC3469(vector<int>& nums)
{
    int i, j;
    int n = nums.size();
    if (n == 1) {
        return nums[0];
    } else if (n == 2) {
        return max(nums[0], nums[1]);
    }

    // dp[i][j] - 以nums[i]结尾, 前面删除余下nums[j]的最小cost
    vector<vector<int>> dp(n, vector<int>(n, 0x3f3f3f3f));
    dp[2][0] = max(nums[1], nums[2]);
    dp[2][1] = max(nums[0], nums[2]);
    dp[2][2] = max(nums[0], nums[1]);
    for (i = 4; i < n; i += 2) {
        for (j = 0; j <= i - 2; j++) {
            dp[i][j] = dp[i - 2][j] + max(nums[i], nums[i - 1]);
            dp[i][i - 1] = min(dp[i][i - 1], dp[i - 2][j] + max(nums[i], nums[j]));
            dp[i][i] = min(dp[i][i], dp[i - 2][j] + max(nums[i - 1], nums[j]));
        }
    }
    int ans = 0x3f3f3f3f;
    if (n % 2 == 0) {
        for (j = 0; j <= n - 2; j++) {
            ans = min(ans, dp[n - 2][j] + max(nums[j], nums[n - 1]));
        }
    } else {
        for (j = 0; j < n; j++) {
            ans = min(ans, dp[n - 1][j] + nums[j]);
        }
    }
    return ans;
}


// LC3478
vector<long long> findMaxSum(vector<int>& nums1, vector<int>& nums2, int k)
{
    int i;
    int n = nums1.size();
    vector<vector<int>> p;
    vector<long long> ans(n);
    for (i = 0; i < n; i++) {
        p.push_back({nums1[i], nums2[i], i});
    }
    sort(p.begin(), p.end(), [](vector<int>& a, vector<int>& b) {
        if (a[0] == b[0]) {
            return a[1] > b[1];
        }
        return a[0] < b[0];
    });

    priority_queue<int, vector<int>, greater<>> pq;
    long long cur = 0;
    bool equal;
    for (i = 0; i < n; i++) {
        if (i == 0) {
            ans[p[i][2]] = 0;
            pq.push(p[i][1]);
            cur += p[i][1];
        } else {
            equal = false;
            if (p[i][0] == p[i - 1][0]) {
                ans[p[i][2]] = ans[p[i - 1][2]];
                equal = true;
            }
            // 虽然有相等的情况, 当前最大和还是需要更新
            if (pq.size() < k) {
                if (!equal) {
                    ans[p[i][2]] = cur;
                    // printf ("i = %d, cur = %lld\n",i,cur);
                }
                cur += p[i][1];
                pq.push(p[i][1]);
            } else {
                if (!equal) {
                    ans[p[i][2]] = cur;
                }
                auto t = pq.top();
                if (t < p[i][1]) {
                    pq.pop();
                    pq.push(p[i][1]);
                    cur = cur - t + p[i][1];
                }
            }
        }
    }
    return ans;
}


// LC3481
string applySubstitutions(vector<vector<string>>& replacements, string text)
{
    unordered_map<char, string> dict;
    for (auto& r : replacements) {
        dict[r[0][0]] = r[1];
    }

    function<string (string &)> dfs = [&dfs, &dict](string& cur) {
        int i;
        int n = cur.size();
        string ans;
        for (i = 0; i < n; i++) {
            if (cur[i] == '%') {
                ans += dfs(dict[cur[i + 1]]);
                i += 2;
            } else {
                ans += cur[i];
            }
        }
        return ans;
    };

    return dfs(text);
}


// LC891
int sumSubseqWidths(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int mod = 1e9 + 7;

    sort (nums.begin(), nums.end());

    long long small, big;
    small = big = 0;
    for (i = 0; i < n; i++) {
        // 以nums[i] 作为最小数的子序列个数
        small = (small + nums[i] * 1ll * FastPow(2, n - 1 - i)) % mod;

        // 以nums[i] 作为最大数的子序列个数
        big = (big + nums[i] * 1ll * FastPow(2, i)) % mod;
    }
    return (big + mod - small) % mod;
}


// LC3306
long long countOfSubstrings(string word, int k)
{
    int i, j;
    int n = word.size();
    vector<vector<int>> prefix(n, vector<int>(6, 0));
    unordered_map<int, set<int>> vowelIdx;
    for (i = 0; i < n; i++) {
        if (i > 0) {
            prefix[i] = prefix[i - 1];
        }
        if (word[i] == 'a') {
            prefix[i][0]++;
            vowelIdx[0].emplace(i);
        } else if (word[i] == 'e') {
            prefix[i][1]++;
            vowelIdx[1].emplace(i);
        } else if (word[i] == 'i') {
            prefix[i][2]++;
            vowelIdx[2].emplace(i);
        } else if (word[i] == 'o') {
            prefix[i][3]++;
            vowelIdx[3].emplace(i);
        } else if (word[i] == 'u') {
            prefix[i][4]++;
            vowelIdx[4].emplace(i);
        } else {
            prefix[i][5]++;
        }
    }
    // 边界
    if (prefix[n - 1][5] < k) {
        return 0;
    }
    for (i = 0; i < 5; i++) {
        if (prefix[n - 1][i] == 0) {
            return 0;
        }
    }
    int leftRange, rightRange;
    int left, right, mid;
    int idx;
    long long ans = 0;
    for (i = 0; i < n; i++) {
        if (i > 0) {
            for (j = 0; j < 5; j++) {
                if (prefix[n - 1][j] - prefix[i - 1][j] == 0) {
                    return ans;
                }
            }
        }
        // 二分分别找左右界(k个辅音)
        left = i;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (i == 0) {
                if (prefix[mid][5] < k) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (prefix[mid][5] - prefix[i - 1][5] < k) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        if (left == n) {
            break;
        }
        leftRange = left;

        left = i;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (i == 0) {
                if (prefix[mid][5] < k + 1) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (prefix[mid][5] - prefix[i - 1][5] < k + 1) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        if (left == n) {
            rightRange = n - 1;
        } else {
            rightRange = left - 1;
        }
        // [leftRange, rightRange] 满足的个数
        int start = leftRange;
        if (i == 0) {
            for (j = 0; j < 5; j++) {
                if (prefix[leftRange][j] == 0) {
                    auto it = vowelIdx[j].upper_bound(leftRange);
                    if (it == vowelIdx[j].end() || *it > rightRange) {
                        break;
                    }
                    start = max(start, *it);
                }
            }
            if (j == 5) {
                ans += rightRange - start + 1;
            }
        } else {
            for (j = 0; j < 5; j++) {
                if (prefix[leftRange][j] - prefix[i - 1][j] == 0) {
                    auto it = vowelIdx[j].upper_bound(leftRange);
                    if (it == vowelIdx[j].end() || *it > rightRange) {
                        break;
                    }
                    start = max(start, *it);
                }
            }
            if (j == 5) {
                ans += rightRange - start + 1;
            }
        }
    }
    return ans;
}


// LC3489
int minZeroArray_LC3489(vector<int>& nums, vector<vector<int>>& queries)
{
    int i, j, k;
    int n = nums.size();
    int m = queries.size();
    int cnt, ans;
    vector<int> w;
    vector<bool> dp;
    // 针对每一个nums[i]找最小的queries, 所求所有queries最大值
    ans = 0;
    for (i = 0; i < n; i++) {
        if (nums[i] == 0) {
            continue;
        }
        w.clear();
        for (k = 0; k < m; k++) {
            if (queries[k][0] > i || queries[k][1] < i) {
                w.emplace_back(0); // 贡献为0
            } else {
                w.emplace_back(queries[k][2]);
            }
        }
        cnt = 0x3f3f3f3f;
        dp.assign(nums[i] + 1, false);
        dp[0] = true;
        for (j = 0; j < m; j++) {
            for (k = nums[i]; k >= w[j]; k--) {
                if (dp[k - w[j]]) {
                    dp[k] = true;
                    if (k == nums[i]) {
                        cnt = min(cnt, j + 1);
                    }
                }
            }
        }
        if (dp[nums[i]] == false) {
            return -1;
        }
        // cout << cnt << endl;
        ans = max(ans, cnt);
    }
    return ans;
}


// LC3494
long long minTime(vector<int>& skill, vector<int>& mana)
{
    int i, j;
    int n = skill.size();
    int m = mana.size();
    vector<long long> w(n);
    // 第一瓶药水完成时间, 也即每个巫师最初空闲时间
    for (i = 0; i < n; i++) {
        if (i == 0) {
            w[i] = skill[i] * 1ll * mana[0];
        } else {
            w[i] = w[i - 1] + skill[i] * 1ll * mana[0];
        }
    }

    long long cur;
    for (i = 1; i < m; i++) {
        cur = w[0];
        for (j = 1; j < n; j++) {
            cur = max(cur + skill[j - 1] * 1ll * mana[i], w[j]);
        }
        // 最后一个巫师的空闲时间
        cur += skill[n - 1] * 1ll * mana[i];
        // 从cur反推w[0, n - 2]
        w[n - 1] = cur;
        for (j = n - 2; j >= 0; j--) {
            w[j] = w[j + 1] - skill[j + 1] * 1ll * mana[i];
        }
    }
    return w[n - 1];
}


// LC3495
long long minOperations_LC3495(vector<vector<int>>& queries)
{
    int i;
    long long ans = 0;
    long long sum;
    long long l, r;
    vector<long long> log4cnt(16);

    for (i = 1; i <= 15; i++) {
        log4cnt[i] = pow(4, i) - pow(4, i - 1);
    }
    for (auto& q : queries) {
        l = log(q[0]) / log(4) + 1;
        r = log(q[1]) / log(4) + 1;

        // cout << l << " " << r << endl;

        if (l == r) {
            sum = (q[1] - q[0] + 1) * l;
        } else {
            sum = l * (pow(4, l) - q[0]) + r * (q[1] - pow(4, r - 1) + 1);
            for (i = l + 1; i <= r - 1; i++) {
                sum += i * log4cnt[i];
            }
        }
        // cout << sum << endl;
        ans += (sum + 1) / 2;
    }
    return ans;
}


// LC656
vector<int> cheapestJump(vector<int>& coins, int maxJump)
{
    int i, j, k;
    int n = coins.size();
    // 考虑到最小字典序, 路径不能用string存储
    vector<pair<int, vector<int>>> dp(n, {0x3f3f3f3f, {}});

    dp[0] = {coins[0], {1}};
    for (i = 1; i < n; i++) {
        if (coins[i] == -1) {
            continue;
        }
        for (j = i - 1; j >= (i - maxJump < 0 ? 0 : i - maxJump); j--) {
            if (coins[j] == -1) {
                continue;
            }
            if (coins[i] + dp[j].first < dp[i].first) {
                auto idx = dp[j].second;
                idx.emplace_back(i + 1);
                dp[i] = {coins[i] + dp[j].first, idx};
            } else if (coins[i] + dp[j].first == dp[i].first) {
                auto idx = dp[j].second;
                idx.emplace_back(i + 1);
                auto len = min(dp[i].second.size(), idx.size());
                for (k = 0; k < len; k++) {
                    if (dp[i].second[k] > idx[k]) {
                        dp[i] = {dp[i].first, idx};
                        break;
                    }
                }
            }
        }
    }
    return dp[n - 1].second;
}


// LC1692
int waysToDistribute(int n, int k)
{
    int i, j;
    int mod = 1e9 + 7;
    // dp[n][k] = k * dp[n - 1][k] + dp[n - 1][k - 1];
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, 0));
    dp[0][0] = 1;
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= min(i, k); j++) {
            if (j == i) {
                dp[i][j] = 1;
            } else {
                dp[i][j] = (j * dp[i - 1][j] % mod + dp[i - 1][j - 1]) % mod;
            }
        }
    }
    return dp[n][k];
}


// LC3499
int maxActiveSectionsAfterTrade(string s)
{
    s = "1" + s + "1";
    int i;
    int n = s.size();
    int start;
    int find = 0;
    vector<vector<int>> range;
    for (i = 1; i < n - 1; i++) {
        if (s[i] == '0') {
            if (find == 0) {
                find = 1;
                start = i;
            }
        } else {
            if (find == 1) {
                find = 0;
                range.push_back({start, i - 1});
            }
        }
    }
    if (find) {
        range.push_back({start, n - 2});
    }
    // for (auto r : range) cout << r[0] << " " << r[1] << endl;
    int cnt = 0;
    int m = range.size();
    for (i = 0; i < n; i++) {
        if (s[i] == '1') {
            cnt++;
        }
    }
    if (m < 2) {
        return cnt - 2;
    }
    int ans = 0;
    for (i = 1; i < m; i++) {
        ans = max(ans, range[i - 1][1] - range[i - 1][0] + 1 + 
            range[i][1] - range[i][0] + 1 + cnt);
    }
    return ans - 2;
}


// LC317
int shortestDistance(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    // 相当于对每个已存在的建筑物到所有点的最短距离
    // dijkstra
    vector<vector<vector<int>>> tol_dist;
    auto dij = [&grid, &tol_dist](int row, int col) {
        int i;
        int m = grid.size();
        int n = grid[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        vector<vector<int>> dist(m, vector<int>(n, 0x3f3f3f3f));
        queue<pair<int, int>> q;

        q.push({row * n + col, 0});
        while (q.size()) {
            auto [pos, d] = q.front();
            q.pop();
            auto r = pos / n;
            auto c = pos % n;
            if (dist[r][c] < d) {
                continue;
            }
            dist[r][c] = d;
            for (i = 0; i < 4; i++) {
                auto nr = r + directions[i][0];
                auto nc = c + directions[i][1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] != 0) {
                    continue;
                }
                if (d + 1 < dist[nr][nc]) {
                    dist[nr][nc] = d + 1;
                    q.push({nr * n + nc, d + 1});
                }
            }
        }
        tol_dist.emplace_back(dist);
    };
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == 1) {
                dij(i, j);
            }
        }
    }
    int ans = 0x3f3f3f3f;
    int cur;
    bool reach = true;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] != 0) {
                continue;
            }
            reach = true;
            cur = 0;
            for (auto& dist : tol_dist) {
                if (dist[i][j] == 0x3f3f3f3f) {
                    reach = false;
                    break;
                }
                cur += dist[i][j];
            }
            if (reach) {
                ans = min(ans, cur);
            }
        }
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC1199 LC3506
int minBuildTime(vector<int>& blocks, int split)
{
    int i;
    int n = blocks.size();
    int a, b;

    if (n == 1) {
        return blocks[0];
    }
    priority_queue<int, vector<int>, greater<>> pq;
    for (i = 0; i < n; i++) {
        pq.push(blocks[i]);
    }
    while (pq.size() > 1) {
        a = pq.top();
        pq.pop();
        b = pq.top();
        pq.pop();
        pq.push(split + max(a, b));
    }
    return pq.top();
}


// LC1259
int numberOfWays(int numPeople)
{
    int i, j;
    int mod = 1e9 + 7;
    vector<long long> dp(numPeople + 1);

    dp[2] = 1;
    for (i = 4; i <= numPeople; i += 2) {
        dp[i] = dp[i - 2] * 2; // 分为两部分
        for (j = 2; j < i - 2; j+= 2) { // 三部分
            dp[i] = (dp[i] + dp[j] * dp[i - 2 - j]) % mod;
        }
    }
    return dp[numPeople];
}


// LC2093
int minimumCost(int n, vector<vector<int>>& highways, int discounts)
{
    int i;
    int m = highways.size();
    vector<vector<pair<int, int>>> edges(n);
    // dists[i][j] - 从城市0到城市i使用j次折扣的最小花费, dijkstra
    vector<vector<int>> dists(n, vector<int>(discounts + 1, 0x3f3f3f3f));

    for (i = 0; i < m; i++) {
        edges[highways[i][0]].push_back({highways[i][1], highways[i][2]});
        edges[highways[i][1]].push_back({highways[i][0], highways[i][2]});
    }
    auto cmp = [](tuple<int, int, int>& a, tuple<int, int, int>& b) {
        return get<1>(a) > get<1>(b);
    };
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, decltype(cmp)> pq(cmp);

    pq.push({0, 0, 0});
    while (!pq.empty()) {
        auto [city, dist, discount] = pq.top();
        pq.pop();

        if (dists[city][discount] < dist) {
            continue;
        }
        dists[city][discount] = dist;
        for (auto& it : edges[city]) {
            if (it.second + dist < dists[it.first][discount]) {
                dists[it.first][discount] = it.second + dist;
                pq.push({it.first, dists[it.first][discount], discount});
            }
            if (discount + 1 <= discounts && it.second / 2 + dist < dists[it.first][discount + 1]) {
                dists[it.first][discount + 1] = it.second / 2 + dist;
                pq.push({it.first, dists[it.first][discount + 1], discount + 1});
            }
        }
    }
    int cost = *min_element(dists[n - 1].begin(), dists[n - 1].end());
    return cost == 0x3f3f3f3f ? -1 : cost;
}


// LC632
vector<int> smallestRange(vector<vector<int>>& nums)
{
    int i, j;
    int n;
    int k = nums.size();
    int rightEdge = 1e5;
    queue<int> q;
    vector<pair<int, int>> vp;
    for (i = 0; i < k; i++) {
        n = nums[i].size();
        for (j = 0; j < n; j++) {
            vp.push_back({nums[i][j], i});
        }
        rightEdge = min(rightEdge, nums[i][n - 1]);
    }

    sort(vp.begin(), vp.end());

    int right, prev;
    vector<int> ans = {100000, 300000};
    right = -1e5;
    for (i = 0; i < vp.size(); i++) {
        if (vp[i].first > rightEdge) {
            break;
        }
        if (i == 0) {
            for (j = 0; j < k; j++) {
                auto it = lower_bound(nums[j].begin(), nums[j].end(), vp[i].first);
                if (it == nums[j].end()) {
                    return ans;
                }
                right = max(right, *it);
            }
        } else {
            if (vp[i].first == vp[i - 1].first) {
                q.push(vp[i - 1].second); // 如果有多个组有相同值, 则滑窗时要一起考虑右边界
                continue;
            }
            q.push(vp[i - 1].second);
            while (q.size()) {
                prev = q.front();
                q.pop();
                auto it = lower_bound(nums[prev].begin(), nums[prev].end(), vp[i].first);
                if (it == nums[prev].end()) {
                    // cout << vp.size() << " " << i << endl;
                    return ans;
                }
                right = max(right, *it);
            }
        }
        if ((right - vp[i].first < ans[1] - ans[0]) || 
            (right - vp[i].first == ans[1] - ans[0] && vp[i].first < ans[0])) {
            ans = {vp[i].first, right};
        }
    }
    // cout << vp.size() << " " << i << endl;
    return ans;
}


// LC3508
class Router {
public:
    int memoryLimit;
    deque<vector<int>> dq;
    unordered_set<tuple<int, int, int>, MyHash<int, int, int>> memory;
    unordered_map<int, deque<int>> data;
    Router(int memoryLimit)
    {
        this->memoryLimit = memoryLimit;
    }
    bool addPacket(int source, int destination, int timestamp)
    {
        tuple<int, int, int> t = {source, destination, timestamp};
        vector<int> front;
        if (memory.count(t)) {
            return false;
        }
        if (dq.size() == memoryLimit) {
            front = dq.front();
            tuple<int, int, int> f = {front[0], front[1], front[2]};
            memory.erase(f);
            data[front[1]].pop_front();
            if (data[front[1]].empty()) {
                data.erase(front[1]);
            }
            dq.pop_front();
            dq.push_back({source, destination, timestamp});
            memory.emplace(t);
            data[destination].push_back(timestamp);
        } else {
            dq.push_back({source, destination, timestamp});
            memory.emplace(t);
            data[destination].push_back(timestamp);
        }
        return true;
    }

    vector<int> forwardPacket()
    {
        if (dq.empty()) {
            return {};
        }
        vector<int> ans = dq.front();
        data[ans[1]].pop_front();
        if (data[ans[1]].empty()) {
            data.erase(ans[1]);
        }
        dq.pop_front();
        tuple<int, int, int> a = {ans[0], ans[1], ans[2]};
        memory.erase(a);
        return ans;
    }

    int getCount(int destination, int startTime, int endTime)
    {
        if (data.count(destination) == 0) {
            return 0;
        }
        // deque<int> t = data[destination]; 效率极低, 要么改成引用方式 deque<int>& t
        auto it1 = lower_bound(data[destination].begin(), data[destination].end(), startTime);
        if (it1 == data[destination].end()) {
            return 0;
        }
        auto it2 = upper_bound(data[destination].begin(), data[destination].end(), endTime);
        // cout << it2 - it1 << endl;
        return it2 - it1;
    }
};


// LC1918
int kthSmallestSubarraySum(vector<int>& nums, int k)
{
    int i;
    int cnt, start, end;
    int n = nums.size();
    long long left, right, mid;
    long long cur;

    left = *min_element(nums.begin(), nums.end());
    right = accumulate(nums.begin(), nums.end(), 0);;
    while (left <= right) {
        // 子数组和小于等于mid的子数组个数
        mid = (right - left) / 2 + left;
        cnt = cur = 0;
        start = end = 0;
        while (start < n) {
            if (cur + nums[end] <= mid) {
                cur += nums[end];
                end++;
                if (end == n) { // 从start开始的子数组和都小于等于mid
                    end--;
                    cnt += (end - start + 1) * (end - start + 2) / 2;
                    break;
                }
            } else {
                end--;
                if (end >= start) {
                    cnt += end - start + 1;
                    cur -= nums[start];
                    start++;
                    end++;
                } else {
                    start++;
                    end = start;
                }
            }
        }
        if (cnt < k) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1548
vector<int> mostSimilar(int n, vector<vector<int>>& roads, vector<string>& names, vector<string>& targetPath)
{
    int i, j, k;
    int m = roads.size();
    vector<vector<int>> edges(n);
    for (i = 0; i < m; i++) {
        edges[roads[i][0]].emplace_back(roads[i][1]);
        edges[roads[i][1]].emplace_back(roads[i][0]);
    }
    int len = targetPath.size();
    // dp[i][j][k] - 第i个target为names[j]且上一个target为names[k]的最小编辑距离
    vector<vector<vector<int>>> dp(len, vector<vector<int>>(n, vector<int>(n, 0x3f3f3f3f)));

    auto f = [](string& a, string& b) {
        return a == b ? 0 : 1;
    };
    // g[i][j] - 第i个target为names[j]的最小编辑距离
    vector<vector<int>> g(len, vector<int>(n, 0x3f3f3f3f));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            dp[0][i][j] = f(targetPath[0], names[i]);
            g[0][i] = min(g[0][i], dp[0][i][j]);
        }
    }
    for (i = 1; i < len; i++) {
        for (j = 0; j < n; j++) {
            for (auto& it : edges[j]) {
                dp[i][j][it] = f(targetPath[i], names[j]) + g[i - 1][it];
                g[i][j] = min(g[i][j], dp[i][j][it]);
            }
        }
    }
    cout << *min_element(g[len - 1].begin(), g[len - 1].end());

    // 回溯找路径
    vector<int> ans;
    int cur = 0x3f3f3f3f;
    int a, b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (dp[len - 1][i][j] < cur) {
                cur = dp[len - 1][i][j];
                a = i;
                b = j;
            }
        }
    }
    ans.emplace_back(a);
    if (len == 1) {
        return ans;
    }
    ans.emplace_back(b);
    a = b;

    int t = len - 2;
    while (t >= 1) {
        cur = 0x3f3f3f3f;
        for (i = 0; i < n; i++) {
            if (dp[t][a][i] < cur) {
                cur = dp[t][a][i];
                b = i;
            }
        }
        ans.emplace_back(b);
        a = b;
        t--;
    }
    reverse(ans.begin(), ans.end());
    return ans;
}


// LC568
int maxVacationDays(vector<vector<int>>& flights, vector<vector<int>>& days)
{
    int i, j, k;
    int weeks = days[0].size();
    int n = flights.size(); // days.size()

    // dp[i][j] - 第i周在j市停留的最大休假天数
    vector<vector<int>> dp(weeks, vector<int>(n, -1));
    for (i = 0; i < n; i++) {
        if (flights[0][i] == 1 || i == 0) {
            dp[0][i] = days[i][0];
        }
    }
    for (i = 1; i < weeks; i++) {
        for (j = 0; j < n; j++) { // 当前是j, 从k来
            for (k = 0; k < n; k++) {
                if ((flights[k][j] == 1 || j == k) && dp[i - 1][k] != -1) { // 旅行或停留不移动
                    dp[i][j] = max(dp[i][j], dp[i - 1][k] + days[j][i]);
                }
            }
        }
    }
    return *max_element(dp[weeks - 1].begin(), dp[weeks - 1].end());
}


// LC2247
int maximumCost(int n, vector<vector<int>>& highways, int k)
{
    if (k >= n) {
        return -1;
    }

    int i, j;
    // dp[i][j] -当前城市是i, 形成的二进制状态路线j的最大旅行费用
    vector<vector<int>> dp(n, vector<int>(1 << n, -1));
    // data[i][j] - 经过j条公路到达第i个城市的二进制状态路线;
    vector<vector<unordered_set<int>>> data(n, vector<unordered_set<int>>(k + 1));
    for (i = 0; i < n; i++) {
        dp[i][1 << i] = 0;
        data[i][0].emplace(1 << i);
    }

    int m = highways.size();
    vector<vector<pair<int, int>>> edges(n);
    for (i = 0; i < m; i++) {
        edges[highways[i][0]].push_back({highways[i][1], highways[i][2]});
        edges[highways[i][1]].push_back({highways[i][0], highways[i][2]});
    }

    int ans = -1;
    for (i = 1; i <= k; i++) {
        for (j = 0; j < n; j++) { // 当前城市j, 上一个城市是it.first
            for (auto& it : edges[j]) {
                for (auto& way : data[it.first][i - 1]) {
                    if ((way & (1 << j)) == (1 << j)) { // 城市j已访问过
                        continue;
                    }
                    dp[j][way | (1 << j)] = max(dp[j][way | (1 << j)], dp[it.first][way] + it.second);
                    data[j][i].emplace(way | (1 << j));
                    if (i == k) {
                        ans = max(ans, dp[j][way | (1 << j)]);
                    }
                }
            }
        }
    }
    return ans;
}


// LC1473
int minCost(vector<int>& houses, vector<vector<int>>& cost, int m, int n, int target)
{
    int i, j, k, t;
    // dp[i][j][t] - 第i个房子涂成color j 且形成t个街区的最小花费
    vector<vector<vector<int>>> dp(m, vector<vector<int>>(n + 1, vector<int>(target + 1, 0x3f3f3f3f)));
    if (houses[0] != 0) {
        dp[0][houses[0]][1] = 0;
    } else {
        for (i = 1; i <= n; i++) {
            dp[0][i][1] = cost[0][i - 1];
        }
    }
    for (i = 1; i < m; i++) {
        if (houses[i] != 0) {
            for (j = 1; j <= n; j++) {
                for (t = 1; t <= target; t++) {
                    if (dp[i - 1][j][t] == 0x3f3f3f3f) {
                        continue;
                    }
                    if (j == houses[i]) {
                        dp[i][houses[i]][t] = min(dp[i][houses[i]][t], dp[i - 1][j][t]);
                    } else {
                        if (t + 1 <= target) {
                            dp[i][houses[i]][t + 1] = min(dp[i][houses[i]][t + 1], dp[i - 1][j][t]);
                        }
                    }
                }
            }
        } else {
            for (j = 1; j <= n; j++) {
                for (k = 1; k <= n; k++) {
                    for (t = 1; t <= target; t++) {
                        if (dp[i - 1][j][t] == 0x3f3f3f3f) {
                            continue;
                        }
                        if (j == k) {
                            dp[i][k][t] = min(dp[i][k][t], dp[i - 1][j][t] + cost[i][k - 1]);
                        } else {
                            if (t + 1 <= target) {
                                dp[i][k][t + 1] = min(dp[i][k][t + 1], dp[i - 1][j][t] + cost[i][k - 1]);
                            }
                        }
                    }
                }
            }
        }
    }

    int ans = 0x3f3f3f3f;
    for (i = 1; i <= n; i++) {
        ans = min(ans, dp[m - 1][i][target]);
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC1976
int countPaths(int n, vector<vector<int>>& roads)
{
    int i;
    int m = roads.size();
    int mod = 1e9 + 7;
    vector<vector<pair<int, int>>> edges(n);

    for (i = 0; i < m; i++) {
        edges[roads[i][0]].push_back({roads[i][1], roads[i][2]});
        edges[roads[i][1]].push_back({roads[i][0], roads[i][2]});
    }
    // dijkstra
    vector<long long> dist(n, LLONG_MAX);
    vector<long long> cnt(n, 0);
    // 此处用普通队列会漏记数
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> q;
    q.push({0, 0});
    cnt[0] = 1;
    while (!q.empty()) {
        auto [cost, cur] = q.top();
        q.pop();
        if (dist[cur] < cost) {
            continue;
        }
        dist[cur] = cost;
        for (auto& it : edges[cur]) {
            if (cost + it.second < dist[it.first]) {
                dist[it.first] = cost + it.second;
                cnt[it.first] = cnt[cur];
                q.push({dist[it.first], it.first});
            } else if (cost + it.second == dist[it.first]) {
                cnt[it.first] = (cnt[it.first] + cnt[cur]) % mod;
            }
        }
    }
    // for (auto d : dist) cout << d << " "; cout << endl;
    // for (auto c : cnt) cout << c << " "; cout << endl;
    return cnt[n - 1];
}


// LC2714  与LC2093类似
int shortestPathWithHops(int n, vector<vector<int>>& edges, int s, int d, int k)
{
    int i;
    int m = edges.size();
    vector<vector<pair<int, int>>> edgesWithWeight(n);
    // dists[i][j] - 从城市0到城市i使用j次折扣的最小花费, dijkstra
    vector<vector<int>> dists(n, vector<int>(k + 1, 0x3f3f3f3f));

    for (i = 0; i < m; i++) {
        edgesWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
        edgesWithWeight[edges[i][1]].push_back({edges[i][0], edges[i][2]});
    }
    auto cmp = [](tuple<int, int, int>& a, tuple<int, int, int>& b) {
        return get<1>(a) > get<1>(b);
    };
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, decltype(cmp)> pq(cmp);

    pq.push({s, 0, 0});
    while (!pq.empty()) {
        auto [city, dist, discount] = pq.top();
        pq.pop();

        if (dists[city][discount] < dist) {
            continue;
        }
        dists[city][discount] = dist;
        for (auto& it : edgesWithWeight[city]) {
            if (it.second + dist < dists[it.first][discount]) {
                dists[it.first][discount] = it.second + dist;
                pq.push({it.first, dists[it.first][discount], discount});
            }
            if (discount + 1 <= k && dist < dists[it.first][discount + 1]) {
                dists[it.first][discount + 1] = dist;
                pq.push({it.first, dists[it.first][discount + 1], discount + 1});
            }
        }
    }
    int cost = *min_element(dists[d].begin(), dists[d].end());
    return cost;
}


// LC2814
int minimumSeconds(vector<vector<string>>& land)
{
    int i, j;
    int n = land.size();
    int m = land[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    auto bfs = [&land](queue<pair<int, int>>& q, vector<vector<int>>& dist) {
        int i, k;
        int n = land.size();
        int m = land[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int cur = 0;
        while (q.size()) {
            int size = q.size();
            for (i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();
                for (k = 0; k < 4; k++) {
                    auto nx = x + directions[k][0];
                    auto ny = y + directions[k][1];
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m || land[nx][ny] == "X" || 
                        land[nx][ny] == "D" || land[nx][ny] == "*") {
                        continue;
                    }
                    if (dist[nx][ny] > cur + 1) {
                        dist[nx][ny] = cur + 1;
                        q.push({nx, ny});
                    }
                }
            }
            cur++;
        }
    };
    vector<vector<int>> distWater(n, vector<int>(m, 0x3f3f3f3f));
    vector<vector<int>> distPerson(n, vector<int>(m, 0x3f3f3f3f));
    queue<pair<int, int>> qw, qp;
    int x, y;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (land[i][j] == "*") {
                qw.push({i, j});
                distWater[i][j] = 0;
            } else if (land[i][j] == "S") {
                qp.push({i, j});
                distPerson[i][j] = 0;
            } else if (land[i][j] == "D") {
                x = i;
                y = j;
            }
        }
    }
    bfs(qw, distWater);
    bfs(qp, distPerson);

    int ans = 0x3f3f3f3f;
    for (i = 0; i < 4; i++) {
        auto nx = x + directions[i][0];
        auto ny = y + directions[i][1];
        if (nx < 0 || nx >= n || ny < 0 || ny >= m || land[nx][ny] == "*" || land[nx][ny] == "X") {
            continue;
        }
        if (distPerson[nx][ny] < distWater[nx][ny]) {
            ans = min(ans, distPerson[nx][ny] + 1);
        }
    }
    return ans == 0x3f3f3f3f ? -1 : ans;
}


// LC1950
vector<int> findMaximums(vector<int>& nums)
{
    // 对于每一个nums[i]的左右分别找到第一个小于nums[i]的下标l, r, 则 r - l + 1长度的子数组对于nums[i]都成立
    // 单调递增栈
    int i;
    int n = nums.size();
    stack<int> st;
    vector<int> l(n), r(n);
    // nums[i]的右边界
    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        auto idx = st.top();
        while (nums[idx] > nums[i]) {
            st.pop();
            r[idx] = i;
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        r[st.top()] = n;
        st.pop();
    }

    // nums[i]的左边界
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        auto idx = st.top();
        while (nums[idx] > nums[i]) {
            st.pop();
            l[idx] = i;
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        l[st.top()] = -1;
        st.pop();
    }
    for (i = 0; i < n; i++) printf ("[%d %d]\n", l[i], r[i]);

    // 对于每一个nums[i], 它可以成为长度[1, r[i] - l[i] - 1]区间的子数组最小值
    map<int, int, greater<>> subarrVal;
    for (i = 0; i < n; i++) {
        subarrVal[nums[i]] = max(subarrVal[nums[i]], r[i] - l[i] - 1);
    }
    vector<int> ans(n);
    int curId = 0;
    for (auto& it : subarrVal) {
        if (it.second <= curId) {
            continue;
        }
        for (i = curId; i < it.second; i++) {
            ans[i] = it.first;
        }
        curId = i;
    }
    return ans;
}


// LC3511
int makeArrayPositive(vector<int>& nums)
{
    int i;
    int n = nums.size();
    int cnt = 0;

    long long cur, minSum = 1e18;
    for (i = 2; i < n;) {
        cur = nums[i];
        cur += nums[i - 1] + nums[i - 2];
        minSum = min(minSum + nums[i], cur);
        if (minSum <= 0) {
            cnt++;
            minSum = 1e18;
            i += 3;
        } else {
            i++;
        }
    }
    return cnt;
}


// LC644
double findMaxAverage(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    double left, right, mid;
    double sum, prev, minVal;

    left = *min_element(nums.begin(), nums.end());
    right = *max_element(nums.begin(), nums.end());

    while (left <= right) {
        mid = (right - left) / 2 + left;
        sum = prev = 0.0;
        minVal = 1e5;
        for (i = 0; i < k; i++) {
            sum += nums[i] - mid;
        }
        if (sum >= 0) {
            left = mid + 1e-5;
        } else {
            for (i = k; i < n; i++) {
                sum += nums[i] - mid;
                prev += nums[i - k] - mid;
                minVal = min(minVal, prev);
                // sum比从下标0开始的最小子数组前缀和大
                if (sum >= 0 || sum >= minVal) {
                    left = mid + 1e-5;
                    break;
                }
            }
            if (i == n) {
                right = mid - 1e-5;
            }
        }
    }
    return right;
}


// LC499
string findShortestWay(vector<vector<int>>& maze, vector<int>& ball, vector<int>& hole)
{
    int i, j;
    int m = maze.size();
    int n = maze[0].size();
    int x = ball[0];
    int y = ball[1];
    int nx, ny, dist;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<vector<vector<int>>> visited(m, vector<vector<int>>(n, vector<int>(4, 0)));
    vector<string> dir = {"r", "d", "l", "u"};
    queue<tuple<int, int, int, string>> q;

    for (i = 0; i < 4; i++) {
        nx = x + directions[i][0];
        ny = y + directions[i][1];
        if (nx < 0 || nx >= m || ny < 0 || ny >= n || maze[nx][ny] == 1) {
            continue;
        }
        q.push({nx, ny, i, dir[i]});
    }

    vector<pair<int, string>> route;
    dist = 0;
    while (q.size()) {
        int size = q.size();
        for (i = 0; i < size; i++) {
            auto [row, col, d, r] = q.front();
            q.pop();
            if (row == hole[0] && col == hole[1]) {
                route.push_back({dist, r});
                continue;
            }
            visited[row][col][d] = 1;
            nx = row + directions[d][0];
            ny = col + directions[d][1];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || maze[nx][ny] == 1 || visited[nx][ny][d]) { // 换方向
                for (j = 0; j < 4; j++) {
                    if (j == d) {
                        continue;
                    }
                    nx = row + directions[j][0];
                    ny = col + directions[j][1];
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n || maze[nx][ny] == 1 || visited[nx][ny][j]) {
                        continue;
                    }
                    q.push({nx, ny, j, r + dir[j]});
                }
            } else {
                q.push({nx, ny, d, r});
            }
        }
        dist++;
    }
    if (route.empty()) {
        return "impossible";
    }
    sort(route.begin(), route.end());

    // for (auto p : route) cout << p.first << " " << p.second << endl;
    return route[0].second;
}


// LC2184
int buildWall(int height, int width, vector<int>& bricks)
{
    int i, j, k;
    int mod = 1e9 + 7;
    // 由于width <= 10, 可先用bitmask表示每一层可能的铺设方式
    vector<int> bitmask;
    function<void (int, int, int)> dfs = [&dfs, &bricks, &bitmask](int cur, int bit, int len) {
        if (cur == 0) {
            bitmask.emplace_back(bit);
        }
        int i;
        int n = bricks.size();
        for (i = 0; i < n; i++) {
            if (cur - bricks[i] >= 0) {
                dfs(cur - bricks[i], bit + (1 << (len + bricks[i])), len + bricks[i]);
            }
        }
    };
    dfs(width, 0, 0);
    int n = bitmask.size();
    if (height == 1) {
        return n;
    }
    // dp[height][i] - 第height层是bitmask[i]的铺设方式的方案数, 所求 sum(dp[height - 1][0 ~ n - 1])
    vector<vector<long long>> dp(height, vector<long long>(n, 0));
    for (i = 0; i < n; i++) {
        dp[0][i] = 1;
    }
    long long ans = 0;
    for (i = 1; i < height; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                if ((bitmask[j] & bitmask[k]) == (1 << width)) {
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % mod;
                }
            }
            if (i == height - 1) {
                ans = (ans + dp[i][j]) % mod;
            }
        }
    }
    return ans;
}


// LC2403
long long minimumTime(vector<int>& power)
{
    // power.size() <= 17 可用二进制bit表示杀死怪物状态, 比如 1001 -> 1011
    // 从两个怪物到第三个怪物
    int i, j;
    int n = power.size();
    vector<long long> dp(1 << n, LLONG_MAX);
    vector<unordered_set<int>> bitmask(n);
    for (i = 0; i < n; i++) {
        dp[1 << i] = power[i];
        bitmask[0].emplace(1 << i);
    }
    for (i = 1; i < n; i++) {
        for (auto& bit : bitmask[i - 1]) {
            for (j = 0; j < n; j++) {
                if ((bit & (1 << j)) == 0) {
                    dp[bit | (1 << j)] = min(dp[bit | (1 << j)], 
                        static_cast<long long>(ceil(power[j] * 1.0 / (i + 1))) + dp[bit]);
                    bitmask[i].emplace(bit | (1 << j));
                }
            }
        }
    }
    return dp[(1 << n) - 1];
}
long long minimumTime_betterWay(vector<int>& power)
{
    int i, j;
    int n = power.size();
    vector<long long> dp(1 << n, LLONG_MAX);
    dp[0] = 0;
    for (i = 0; i < (1 << n); i++) {
        if (dp[i] == LLONG_MAX) {
            continue;
        }
        int k = __builtin_popcount(i);
        for (j = 0; j < n; j++) {
            if ((i & (1 << j)) == 0) {
                dp[i | (1 << j)] = min(dp[i | (1 << j)], 
                    dp[i] + static_cast<long long>(ceil(power[j] * 1.0 / (k + 1))));
            }
        }
    }
    return dp[(1 << n) - 1];
}


// LC2709
bool canTraverseAllPairs(vector<int>& nums)
{
    int i, j;
    int t;
    int n = nums.size();

    if (n == 1) {
        return true;
    }
    for (i = 0; i < n; i++) {
        if (nums[i] == 1) {
            return false;
        }
    }
    UnionFind uf = UnionFind(100001);
    for (i = 0; i < n; i++) {
        t = nums[i];
        if (nums[i] == 2) {
            uf.unionSets(nums[i], 2);
        } else if (nums[i] == 3) {
            uf.unionSets(nums[i], 3);
        }
        // 将质因数与nums[i]合并
        for (j = 2; j <= sqrt(nums[i]); j++) {
            if (t % j == 0) {
                uf.unionSets(nums[i], j);
                while (t % j == 0) {
                    t /= j;
                }
            }
        }
        if (t != 1) {
            uf.unionSets(nums[i], t);
        }
    }
    int no = uf.findSet(nums[0]);
    for (i = 1; i < n; i++) {
        // cout << uf.findSet(nums[i])  << " ";
        if (no != uf.findSet(nums[i])) {
            return false;
        }
    }
    return true;
}


// LC1866
int rearrangeSticks(int n, int k)
{
    int i, j;
    int mod = 1e9 + 7;
    // dp[n][k] - n根棍子从左侧看k根
    vector<vector<long long>> dp(n + 1, vector<long long>(k + 1, 0));

    dp[1][1] = 1;
    for (i = 2; i <= n; i++) {
        for (j = 1; j <= k; j++) {
            if (j > i) {
                break;
            }
            // 考虑 dp[n][k] - 长度为2 ~ n的排列方式 - dp[n - 1][k - 1], 现在放置长度1的木棍 组成 dp[n][k]
            dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] * (i - 1)) % mod;
        }
    }
    return dp[n][k];
}


// LC1883
int minSkips(vector<int>& dist, int speed, int hoursBefore)
{
    int i, j;
    int n = dist.size();
    int ans = n;
    const double epsilon = 1e-7;
    // 注意精度
    vector<vector<double>> cost(n, vector<double>(n + 1, 0x3f3f3f3f));

    cost[0][0] = ceil(dist[0] * 1.0 / speed);
    cost[0][1] = dist[0] * 1.0 / speed;
    for (i = 1; i < n - 1; i++) {
        for (j = 0; j <= i; j++) {
            // 精度处理
            auto t = cost[i - 1][j] + dist[i] * 1.0 / speed;
            if (fabs(round(t) - t) < epsilon) {
                t = round(t);
            } else {
                t = ceil(t);
            }
            cost[i][j] = min(cost[i][j], t); // 不跳过
            cost[i][j + 1] = min(cost[i][j + 1], cost[i - 1][j] + dist[i] * 1.0 / speed); // 跳过
        }
    }
    if (n > 1) {
        for (i = 0; i <= n - 1; i++) {
            cost[n - 1][i] = cost[n - 2][i] + dist[n - 1] * 1.0 / speed;
        }
    }
    for (i = 0; i < n; i++) {
        // cout << cost[n - 1][i] << endl;
        if (cost[n - 1][i] <= hoursBefore) {
            ans = min(ans, i);
        }
    }
    return ans == n ? -1 : ans;
}


// LC1931
int colorTheGrid(int m, int n)
{
    int i, j, k;
    int t, idx;
    int mod = 1e9 + 7;
    vector<int> v(m);
    // 由于m <= 5, 则对每一列, 其组合数是有限的, 可以用bitmask压缩状态
    // 为了方便代码书写, 可以把grid行列转置
    vector<vector<int>> bits;
    for (i = 0; i < pow(3, m); i++) {
        t = i;
        idx = 0;
        while (idx < m) {
            v[idx] = t % 3;
            t /= 3;
            if (idx > 0 && v[idx] == v[idx - 1]) {
                break; // 约束条件
            }
            idx++;
        }
        if (idx == m) {
            bits.emplace_back(v);
        }
    }

    auto Check = [](vector<int>& a, vector<int>& b) {
        int i;
        int n = a.size();
        for (i = 0; i < n; i++) {
            if (a[i] == b[i]) {
                return false;
            }
        }
        return true;
    };

    int len = bits.size();
    // cout << len << endl;

    vector<vector<long long>> dp(n, vector<long long>(len, 0));
    for (i = 0; i < len; i++) {
        dp[0][i] = 1;
    }
    for (i = 1; i < n; i++) {
        for (j = 0; j < len; j++) {
            for (k = 0; k < len; k++) {
                if (j == k || Check(bits[j], bits[k]) == false) {
                    continue;
                }
                dp[i][j] = (dp[i][j] + dp[i - 1][k]) % mod;
            }
        }
    }
    long long ans = 0;
    for (i = 0; i < len; i++) {
        ans = (ans + dp[n - 1][i]) % mod;
    }
    return ans;
}


// LC2350
int shortestSequence(vector<int>& rolls, int k)
{
    // 假设 k = 3, 则能满足所有长度为2的子序列rolls必然包含子序列 [1 2 3] [3 2 1]
    // 同理长度为3, 则必包含 [1 2 3] [3 2 1] [2 1 3], 以此类推（中括表示里面的数可以以任意顺序排列）
    int i;
    int n = rolls.size();
    int cnt, freq;
    vector<int> visited(k + 1, 0);

    cnt = 0;
    freq = 1;
    for (i = 0; i < n; i++) {
        if (visited[rolls[i]] == freq) {
            continue;
        }
        visited[rolls[i]] = freq;
        cnt++;
        if (cnt == k) {
            freq++;
            cnt = 0;
        }
    }
    return freq;
}


// LC1601
int maximumRequests(int n, vector<vector<int>>& requests)
{
    int i, j;
    int m = requests.size();
    int t, cnt, ans;
    vector<int> bits(m);
    vector<int> building;

    ans = 0;
    for (i = 1; i < (1 << m); i++) {
        t = i;
        cnt = 0;
        while (cnt < m) {
            bits[cnt] = t % 2;
            t >>= 1;
            cnt++;
        }
        building.assign(n, 0);
        cnt = 0;
        for (j = 0; j < m; j++) {
            if (bits[j]) {
                cnt++;
                building[requests[j][0]]--;
                building[requests[j][1]]++;
            }
        }
        for (j = 0; j < n; j++) {
            if (building[j] != 0) {
                break;
            }
        }
        if (j == n) {
            ans = max(ans, cnt);
        }
    }
    return ans;
}


// LC1691
int maxHeight_LC1691(vector<vector<int>>& cuboids)
{
    int i, j;
    int n = cuboids.size();
    vector<vector<int>> cube;
    // 长方体可旋转则它的放置总计有6种情况
    for (i = 0; i < n; i++) {
        cube.push_back({cuboids[i][0], cuboids[i][1], cuboids[i][2], i});
        cube.push_back({cuboids[i][1], cuboids[i][0], cuboids[i][2], i});
        cube.push_back({cuboids[i][0], cuboids[i][2], cuboids[i][1], i});
        cube.push_back({cuboids[i][2], cuboids[i][0], cuboids[i][1], i});
        cube.push_back({cuboids[i][1], cuboids[i][2], cuboids[i][0], i});
        cube.push_back({cuboids[i][2], cuboids[i][1], cuboids[i][0], i});
    }

    sort (cube.rbegin(), cube.rend());
    // for (auto c : cube) printf("%d %d %d %d\n", c[0],c[1],c[2],c[3]);

    int ans = 0;
    int m = cube.size();
    vector<int> dp(m, 0);
    vector<int> visited(n, 0);
    for (i = 0; i < m; i++) {
        dp[i] = cube[i][2]; // 只考虑一个长方体
        ans = max(ans, dp[i]);
    }
    for (i = 1; i < m; i++) {
        visited[cube[i][3]] = 1;
        for (j = i - 1; j >= 0; j--) {
            if (visited[cube[j][3]] || cube[i][0] > cube[j][0] ||
                cube[i][1] > cube[j][1] ||
                cube[i][2] > cube[j][2]) {
                continue;
            }
            dp[i] = max(dp[i], dp[j] + cube[i][2]);
            ans = max(ans, dp[i]);
        }
        visited[cube[i][3]] = 0;
    }
    return ans;
}


// LC1665
int minimumEffort(vector<vector<int>>& tasks)
{
    sort(tasks.begin(), tasks.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] - a[0] == b[1] - b[0]) {
            return a[1] > b[1];
        }
        return a[1] - a[0] > b[1] - b[0];
    });

    int i;
    int n = tasks.size();
    int cur;
    int left, right, mid;
    left = tasks[0][1];
    right = 1e9 + 1;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        cur = mid;
        for (i = 0; i < n; i++) {
            if (cur < tasks[i][1]) {
                break;
            }
            cur -= tasks[i][0];
        }
        if (i != n) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC1559
bool containsCycle(vector<vector<char>>& grid)
{
    int i, j;
    int m = grid.size();
    int n = grid[0].size();
    bool loop = false;
    vector<vector<int>> visited(m, vector<int>(n, 0));

    function<void (int, int)> dfs = [&dfs, &loop, &visited, &grid](int cur, int from) {
        if (loop) {
            return;
        }

        int m = grid.size();
        int n = grid[0].size();
        int x, y;

        x = cur / n;
        y = cur % n;
        if (visited[x][y] == 1) {
            loop = true;
            return;
        }
        visited[x][y] = 1;

        int i;
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        for (i = 0; i < 4; i++) {
            auto nx = x + directions[i][0];
            auto ny = y + directions[i][1];
            auto npos = nx * n + ny;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] != grid[x][y] || npos == from) {
                continue;
            }
            dfs(npos, cur);
        }
        visited[x][y] = 2;
    };

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (visited[i][j] == 0) {
                loop = false;
                dfs(i * n + j, -1);
                if (loop) {
                    return true;
                }
            }
        }
    }
    return false;
}


// LC2444
long long countSubarrays(vector<int>& nums, int minK, int maxK)
{
    int i;
    int n = nums.size();
    int minIdx, maxIdx;
    int startIdx = 0;
    long long ans = 0;
    minIdx = maxIdx = -1;
    // 找到最接近的minK, maxK下标,则它两区间及之前到startIdx的子数组都满足条件
    // 如果当前nums[i]在两者之间且minIdx, maxIdx存在,则子数组个数与上面计算方式相同
    for (i = 0; i < n; i++) {
        if (nums[i] < minK || nums[i] > maxK) {
            minIdx = maxIdx = -1;
            startIdx = i + 1;
            // cout << i << endl;
            continue;
        }
        if (nums[i] == minK) {
            minIdx = i;
        }
        if (nums[i] == maxK) { // 避免minK == maxK情况
            maxIdx = i;
        }
        if (minIdx != -1 && maxIdx != -1) {
            ans += min(minIdx, maxIdx) - startIdx + 1;
        }
    }
    // cout << startIdx << endl;
    return ans;
}


// LC1349
int maxStudents(vector<vector<char>>& seats)
{
    // 回溯是条死路
    // 采用状态压缩dp, 时间复杂度为 O(m * n * 2 ^ (2n))
    int i, j, k;
    int n = seats[0].size();
    int cnt;
    vector<int> bits(n);
    // num对应的二进制状态是否能匹配seats[i]
    auto Check = [&bits, &cnt](vector<char>& seat, int num) {
        int n = seat.size();
        int idx = 0;
        while (idx < n) {
            bits[idx] = num % 2;
            num >>= 1;
            if (bits[idx] == 0 && seat[idx] == '#') {
                return false;
            }
            if (seat[idx] == '.' && bits[idx] == 1) {
                if (idx > 0 && seat[idx - 1] != '#' && bits[idx - 1] == 1) {
                    return false;
                }
                cnt++;
            }
            idx++;
        }
        return true;
    };
    int m = seats.size();
    int ans = 0;
    vector<vector<int>> dp(m, vector<int>(1 << n, -1));
    for (i = 0; i < (1 << n); i++) {
        cnt = 0;
        if (Check(seats[0], i)) {
            dp[0][i] = cnt;
            ans = max(ans, cnt);
        }
    }
    vector<int> bits1(n), bits2(n);
    // 上下两行以num1, num2方式放置是否不冲突
    auto Conflict = [&bits1, &bits2](vector<char>& seat1, vector<char>& seat2, int num1, int num2) {
        int i;
        int n = seat1.size();
        int idx = 0;
        while (idx < n) {
            bits1[idx] = num1 % 2;
            num1 >>= 1;
            idx++;
        }

        idx = 0;
        while (idx < n) {
            bits2[idx] = num2 % 2;
            num2 >>= 1;
            idx++;
        }

        for (i = 0; i < n; i++) {
            if (seat2[i] == '#') {
                continue;
            }
            if (bits2[i] == 1) {
                if (i > 0 && bits1[i - 1] == 1 && seat1[i - 1] == '.') {
                    return false;
                }
                if (i < n - 1 && bits1[i + 1] == 1 && seat1[i + 1] == '.') {
                    return false;
                }
            }
        }
        return true;
    };

    for (i = 1; i < m; i++) {
        for (j = 0; j < (1 << n); j++) {
            cnt = 0;
            if (Check(seats[i], j) == false) {
                continue;
            }
            for (k = 0; k < (1 << n); k++) {
                if (dp[i - 1][k] == -1) {
                    continue;
                }
                if (Conflict(seats[i - 1], seats[i], k, j)) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][k] + cnt);
                }
            }
            if (i == m - 1) {
                ans = max(ans, dp[i][j]);
            }
        }
    }
    return ans;
}


// LC3530
int maxProfit(int n, vector<vector<int>>& edges, vector<int>& score)
{
    int i, j, k;
    int m;
    vector<vector<int>> parent(n);
    for (auto& edge : edges) {
        parent[edge[1]].emplace_back(edge[0]);
    }
    vector<int> dp(1 << n, -1);
    dp[0] = 0;
    for (i = 0; i < (1 << n); i++) {
        if (dp[i] == -1) {
            continue;
        }
        for (j = 0; j < n; j++) {
            if ((i & (1 << j)) != 0) {
                continue;
            }
            m = parent[j].size();
            for (k = 0; k < m; k++) {
                if ((i & (1 << parent[j][k])) == 0) {
                    break;
                }
            }
            if (k == m) {
                unsigned int state = (i | (1 << j));
                dp[state] = max(dp[state], dp[i] + score[j] * popcount(state));
            }
        }
    }
    // for (auto d : dp) cout << d << " ";
    return dp[(1 << n) - 1];
}


// LC3533
vector<int> concatenatedDivisibility(vector<int>& nums, int k)
{
    int i, j, p;
    int n = nums.size();
    int remainder;
    unsigned int state;
    // dp[i][j] - 连接状态为i, 被k整除后余数为j是否存在
    vector<vector<bool>> dp(1 << n, vector<bool>(k, false));
    unordered_map<int, unordered_map<int, vector<int>>> data;

    // 每连接一个数n, 余数要乘以pow(1, n的位数)
    auto f = [](int n) {
        int ans = 1;
        while (n) {
            ans *= 10;
            n /= 10;
        }
        return ans;
    };

    dp[0][0] = true;
    data[0][0] = {};
    for (i = 0; i < (1 << n); i++) {
        for (p = 0; p < k; p++) {
            if (dp[i][p] == false) {
                continue;
            }
            for (j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) {
                    continue;
                }
                state = (i | (1 << j));
                remainder = (p * f(nums[j]) + nums[j]) % k;
                dp[state][remainder] = true;
                auto v = data[i][p];
                v.emplace_back(nums[j]);
                // printf ("s %d  r %d\n", state, remainder);
                // for (auto vv : v) cout << vv << " "; cout << endl;
                if (data[state].count(remainder)) {
                    data[state][remainder] = min(data[state][remainder], v);
                } else {
                    data[state][remainder] = v;
                }
            }
        }
    }
    return data[(1 << n) - 1][0];
}


// LC1627
vector<bool> areConnected(int n, int threshold, vector<vector<int>>& queries)
{
    int i;
    int m = queries.size();
    vector<bool> ans(m);

    if (threshold == 0) {
        ans.assign(m, true);
        return ans;
    }
    int p;
    UnionFind uf = UnionFind(n + 1);
    // 此处时间复杂度近似为O(nlog(n))
    for (p = 2; p <= n; p++) {
        for (i = p * 2; i <= n; i += p) {
            if (p > threshold) {
                uf.unionSets(p, i);
            }
        }
    }
    for (i = 0; i < m; i++) {
        ans[i] = (uf.findSet(queries[i][0]) == uf.findSet(queries[i][1]));
    }
    return ans;
}


// LC1595
int connectTwoGroups(vector<vector<int>>& cost)
{
    int i, j, k;
    int size1 = cost.size();
    int size2 = cost[0].size();

    // size1 >= size2, 以size2状态压缩效率更高
    // dp[i][j] - 第一组前i个点连接成为状态j的最小cost
    vector<vector<int>> dp(size1 + 1, vector<int>(1 << size2, 0x3f3f3f3f));

    dp[0][0] = 0;
    for (i = 1; i <= size1; i++) {
        for (j = 1; j < (1 << size2); j++) {
            for (k = 0; k < size2; k++) {
                if ((j & (1 << k)) != 0) {
                    // dp[i - 1][j ^ (1 << k)] + cost[i - 1][k] k只与i - 1相连
                    // dp[i - 1][j] + cost[i - 1][k] k与之前的点有相连
                    // dp[i][j ^ (1 << k)] + cost[i - 1][k] i - 1这个点与非k点都相连
                    dp[i][j] = min(dp[i][j], min({dp[i - 1][j ^ (1 << k)], dp[i - 1][j],
                        dp[i][j ^ (1 << k)]}) + cost[i - 1][k]);
                }
            }
        }
    }
    // for (auto d : dp[size1]) cout << d << " "; cout << endl;
    return dp[size1][(1 << size2) - 1];
}


// LC1320
int minimumDistance(string word)
{
    // 5 * 6键盘
    int i, j, k;
    int n = 6;
    int len = word.size();

    if (len <= 2) {
        return 0;
    }

    // dp[i][0 ~ 1][k] - 第0 - 1个手指在word[i], 另一个手指在k的最小cost
    vector<vector<vector<int>>> dp(len, vector<vector<int>>(2, vector<int>(26, 0x3f3f3f3f)));

    auto Dist = [](char a, char b, int n) {
        int x1, x2, y1, y2;
        x1 = (a - 'A') / n;
        y1 = (a - 'A') % n;
        x2 = (b - 'A') / n;
        y2 = (b - 'A') % n;
        return abs(x1 - x2) + abs(y1 - y2);
    };

    for (i = 0; i < 26; i++) {
        for (j = 0; j < 2; j++) {
            dp[0][j][i] = 0;
        }
    }
    for (k = 1; k < len; k++) {
        for (i = 0; i < 26; i++) {
            dp[k][0][i] = min({dp[k][0][i], dp[k - 1][0][i] + Dist(word[k - 1], word[k], n), 
                dp[k - 1][1][word[k] - 'A'] + Dist(word[k - 1], i + 'A', n)});
            dp[k][1][i] = min({dp[k][1][i], dp[k - 1][1][i] + Dist(word[k - 1], word[k], n), 
                dp[k - 1][0][word[k] - 'A'] + Dist(word[k - 1], i + 'A', n)});
        }
    }
    int ans = 0x3f3f3f3f;
    for (i = 0; i < 26; i++) {
        for (j = 0; j < 2; j++) {
            ans = min(ans, dp[len - 1][j][i]);
        }
    }
    return ans;
}


// LC3537
vector<vector<int>> specialGrid(int N)
{
    vector<vector<int>> grid(1 << N , vector<int>(1 << N));
    function<void (int, int, int, int)> dfs = [&dfs, &grid](int cur, int n, int x, int y) {
        if (n == 0) {
            // cout << x << " " << y << endl;
            grid[x][y] = cur;
            return;
        }
        int dist = (1 << (n - 1)) * (1 << (n - 1));
        dfs(cur, n - 1, x, y + (1 << (n - 1)));
        dfs(cur + dist, n - 1, x + (1 << (n - 1)), y + (1 << (n - 1)));
        dfs(cur + dist * 2, n - 1, x + (1 << (n - 1)), y);
        dfs(cur + dist * 3, n - 1, x, y);
    };

    dfs(0, N, 0, 0);
    return grid;
}


// LC472
vector<string> findAllConcatenatedWordsInADict(vector<string>& words)
{
    int i, j, k;
    int n = words.size();
    int m;
    unsigned long long base = 1337ull;
    unsigned long long hash;
    unordered_set<unsigned long long> hashcode; // words[i] 后缀hash
    vector<string> ans;
    for (i = 0; i < n; i++) {
        m = words[i].size();
        hash = 0;
        for (j = m - 1; j >= 0; j--) {
            hash = hash * base + (words[i][j] - 'a' + 1);
        }
        hashcode.emplace(hash);
    }
    vector<bool> dp;
    for (i = 0; i < n; i++) {
        m = words[i].size();
        dp.assign(m + 1, false);
        dp[0] = true;
        for (j = 1; j <= m; j++) {
            hash = 0;
            for (k = j; k >= 1; k--) {
                hash = hash * base + (words[i][k - 1] - 'a' + 1);
                if (hashcode.count(hash) && dp[k - 1]) {
                    if (!(j == m && k == 1)) { // 不是整个单词
                        dp[j] = true;
                        break;
                    }
                }
            }
        }
        if (dp[m]) {
            ans.emplace_back(words[i]);
        }
    }
    return ans;
}


// LC1547
int minCost_LC1547(int n, vector<int>& cuts)
{
    int i, j, k;
    int m = cuts.size();
    int cur = 0;
    vector<int> v, prefix;

    sort(cuts.begin(), cuts.end());
    for (i = 0; i < m; i++) {
        v.emplace_back(cuts[i] - cur);
        if (i == 0) {
            prefix.emplace_back(cuts[i] - cur);
        } else {
            prefix.emplace_back(prefix.back() + cuts[i] - cur);
        }
        cur = cuts[i];
    }
    v.emplace_back(n - cur);
    prefix.emplace_back(prefix.back() + n - cur);

    // 反过来分析, 将m + 1段合并为n的最小花费
    // dp[i][j] - 合并i到j的最小花费
    vector<vector<int>> dp(m + 1, vector<int>(m + 1, 0x3f3f3f3f));
    auto f = [&prefix](int i, int j) {
        if (i == 0) {
            return prefix[j];
        }
        return prefix[j] - prefix[i - 1];
    }; 
    for (j = 0; j <= m; j++) {
        for (i = j; i >= 0; i--) {
            if (i == j) {
                dp[i][j] = v[i];
            } else if (i + 1 == j) {
                dp[i][j] = v[i] + v[j];
            } else {
                for (k = i; k < j; k++) {
                    if (k == i) {
                        dp[i][j] = min(dp[i][j], f(i, j) + dp[k + 1][j]);
                    } else if (k + 1 == j) {
                        dp[i][j] = min(dp[i][j], f(i, j) + dp[i][k]);
                    } else {
                        dp[i][j] = min(dp[i][j], (dp[i][k] + dp[k + 1][j]) + f(i, j));
                    }
                    
                }
            }
        }
    }
    return dp[0][m];
}


// LC960
int minDeletionSize_LC960(vector<string>& strs)
{
    int i, j, k;
    int n = strs.size();
    int m = strs[0].size();
    vector<int> dp(m, 0);

    dp[0] = 1;
    for (i = 1; i < m; i++) {
        dp[i] = 1;
        for (j = i - 1; j >= 0; j--) {
            for (k = 0; k < n; k++) {
                if (strs[k][i] < strs[k][j]) {
                    break;
                }
            }
            if (k == n) {
                dp[i] = max(dp[i], dp[j] + 1);
                // printf ("dp[%d] = %d\n", i, dp[i]);
            }
        }
    }
    int ans = m;
    for (i = 0; i < m; i++) {
        ans = min(ans, m - dp[i]);
    }
    return ans;
}


// LC689
vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    auto cmp = [](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first == b.first) {
            return a.second > b.second;
        }
        return a.first < b.first;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pqFront(cmp), pqTail(cmp);
    vector<int> subsum(n - k + 1);
    int sum = 0;
    for (i = 0; i < k; i++) {
        sum += nums[i];
    }
    subsum[0] = sum;
    for (i = 1; i <= n - k; i++) {
        subsum[i] = subsum[i - 1] - nums[i - 1] + nums[i + k - 1];
    }

    for (i = n - k; i >= 1; i--) {
        pqTail.push({subsum[i], i});
    }
    int cur;
    int maxSum = 0;
    vector<int> ans(3);
    for (i = k; i <= n - k - k; i++) {
        pqFront.push({subsum[i - k], i - k});
        auto front = pqFront.top();
        auto tail = pqTail.top();
        while (tail.second <= i + k - 1) {
            pqTail.pop();
            tail = pqTail.top();
        }
        cur = front.first + subsum[i] + tail.first;
        // cout << i << " " << cur << endl;
        if (cur > maxSum) {
            maxSum = cur;
            ans[0] = front.second;
            ans[1] = i;
            ans[2] = tail.second;
        }
    }
    return ans;
}


// LC3543
int maxWeight(int n, vector<vector<int>>& edges, int k, int t)
{
    int i;
    int m = edges.size();
    vector<vector<pair<int, int>>> edgeWithWeight(n);

    if (k == 0) {
        return 0;
    }

    for (i = 0; i < m; i++) {
        edgeWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
    }
    vector<vector<int>> dist(n, vector<int>(k + 1, -1));
    auto f = [&edgeWithWeight, &k, &t, &dist](int node, int len) {
        int i;
        int n = edgeWithWeight.size();
        unordered_map<int, unordered_map<int, unordered_set<int>>> cnt;
        int ans = -1;
        queue<tuple<int, int, int>> pq;
        pq.push({0, 0, node});
        while (pq.size()) {
            auto [d, e, curNode] = pq.front();
            pq.pop();
            if (e == k && d < t) {
                dist[curNode][e] = max(dist[curNode][e], d);
                ans = max(ans, dist[curNode][e]);
                cnt[node][e].emplace(dist[curNode][e]);
                continue;
            } 
            for (auto& it : edgeWithWeight[curNode]) {
                if (e + 1 <= k && d + it.second < t && cnt[it.first][e + 1].count(d + it.second) == 0) {
                    pq.push({d + it.second, e + 1, it.first});
                    cnt[it.first][e + 1].emplace(d + it.second);
                }
            }
        }
        return ans;
    };

    int maxVal;
    int ans = -1;
    for (i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist[j].assign(k + 1, -1);
        }
        maxVal = f(i, 0);
        // printf("i = %d, maxVal = %d\n", i, maxVal);
        ans = max(ans, maxVal);
    }
    return ans;
}


// LC2156
string subStrHash(string s, int power, int modulo, int k, int hashValue)
{
    int i;
    int n = s.size();
    int idx = -1;
    long long val;
    string ans;

    val = 0;
    for (i = n - 1; i >= n - k; i--) {
        val = (val * power + (s[i] - 'a' + 1)) % modulo;
    }
    if (val == hashValue) {
        idx = n - k;
    }
    for (i = n - k - 1; i >= 0; i--) {
        val = ((val + modulo - (s[i + k] - 'a' + 1) * FastPow(power, k - 1, modulo) % modulo) * power
            + (s[i] - 'a' + 1)) % modulo;
        if (val == hashValue) {
            idx = i;
        }
    }
    if (idx == -1) {
        return ans;
    }

    ans = s.substr(idx, k);
    return ans;
}


// LC2681
int sumOfPower(vector<int>& nums)
{
    // 既然是算全子序列, 则按贡献度方式统计
    int i;
    int n = nums.size();
    int mod = 1e9 + 7;
    long long ans, cur;
    vector<long long> nums_ll(nums.begin(), nums.end());

    sort(nums_ll.begin(), nums_ll.end());
    ans = nums_ll[0] * nums_ll[0] % mod * nums_ll[0] % mod;
    cur = nums_ll[0];
    for (i = 1; i < n; i++) {
        ans = (ans + nums_ll[i] * nums_ll[i] % mod * (cur + nums_ll[i]) % mod) % mod;
        cur = (cur * 2 + nums_ll[i]) % mod;
    }
    return ans;
}


// LC1368
int minCost_LC1368(vector<vector<int>>& grid)
{
    int i;
    int m = grid.size();
    int n = grid[0].size();
    int directions[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    vector<vector<int>> dist(m, vector<int>(n, 0x3f3f3f3f));

    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
    pq.push({0, 0, 0});
    while (!pq.empty()) {
        auto [d, r, c] = pq.top();
        pq.pop();

        if (dist[r][c] < d) {
            continue;
        }
        dist[r][c] = d;
        for (i = 0; i < 4; i++) {
            auto nr = r + directions[i][0];
            auto nc = c + directions[i][1];
            if (nr < 0 || nr >= m || nc < 0 || nc >= n) {
                continue;
            }
            if (i == grid[r][c] - 1) {
                if (dist[nr][nc] > dist[r][c]) {
                    dist[nr][nc] = dist[r][c];
                    pq.push({dist[nr][nc], nr, nc});
                }
            } else {
                if (dist[nr][nc] > dist[r][c] + 1) {
                    dist[nr][nc] = dist[r][c] + 1;
                    pq.push({dist[nr][nc], nr, nc});
                }
            }
        }
    }
    return dist[m - 1][n - 1];
}


// LC2203
long long minimumWeight(int n, vector<vector<int>>& edges, int src1, int src2, int dest)
{
    // 从src1出发经过src2到达dest的最短路径和
    // 从src2出发经过src1到达dest的最短路径
    // 分别从src1, src2出发汇聚到某一点再到达dest的最短距离之和 这三种情况的最小值
    int i;
    int m = edges.size();
    vector<vector<pair<int, int>>> edgesWithWeight(n);

    for (i = 0; i < m; i++) {
        edgesWithWeight[edges[i][0]].push_back({edges[i][1], edges[i][2]});
    }

    // 反向边图, 方便计算dest到各点最短路径
    vector<vector<pair<int, int>>> edgesWithWeight_rev(n);
    for (i = 0; i < m; i++) {
        edgesWithWeight_rev[edges[i][1]].push_back({edges[i][0], edges[i][2]});
    }

    auto dij = [n](int src, vector<vector<pair<int, int>>>& edgesWithWeight) {
        vector<long long> dist(n, LLONG_MAX / 4);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;

        pq.push({0ll, src});
        while (!pq.empty()) {
            auto [d, node] = pq.top();
            pq.pop();

            if (dist[node] < d) {
                continue;
            }
            dist[node] = d;
            for (auto& e : edgesWithWeight[node]) {
                if (dist[e.first] > d + e.second) {
                    dist[e.first] = d + e.second;
                    pq.push({dist[e.first], e.first});
                }
            }
        }
        return dist;
    };

    long long ans;
    vector<long long> distSrc1 = dij(src1, edgesWithWeight);
    vector<long long> distSrc2 = dij(src2, edgesWithWeight);
    vector<long long> distDest = dij(dest, edgesWithWeight_rev);

    ans = LLONG_MAX;
    for (i = 0; i < n; i++) {
        ans = min(ans, distDest[i] + distSrc1[i] + distSrc2[i]);
    }
    return ans >= LLONG_MAX / 4 ? -1 : ans;
}


// LC3552
int minMoves(vector<string>& matrix)
{
    int i, j;
    int m = matrix.size();
    int n = matrix[0].size();

    if (matrix[m - 1][n - 1] == '#') {
        return -1;
    }

    int directions[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    vector<vector<int>> dist(m, vector<int>(n, 0x3f3f3f3f));
    unordered_map<char, vector<pair<int, int>>> gates;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (isalpha(matrix[i][j])) {
                gates[matrix[i][j]].push_back({i, j});
            }
        }
    }

    // [距离, r, c, 使用传送门状态]
    priority_queue<tuple<int, int, int, int>, vector<tuple<int, int, int, int>>, greater<>> pq;

    pq.push({0, 0, 0, 0});

    while (!pq.empty()) {
        auto [d, r, c, used] = pq.top();
        pq.pop();

        if (dist[r][c] < d) {
            continue;
        }
        dist[r][c] = d;
        if (r == m - 1 && c == n - 1) {
            break;
        }
        if (isalpha(matrix[r][c]) && (used & 1 << (matrix[r][c] - 'A')) == 0) {
            for (auto& p : gates[matrix[r][c]]) {
                if (dist[p.first][p.second] > d) {
                    dist[p.first][p.second] = d;
                    used |= 1 << (matrix[r][c] - 'A');
                    pq.push({d, p.first, p.second, used});
                }
            }
        }
        for (i = 0; i < 4; i++) {
            auto nr = r + directions[i][0];
            auto nc = c + directions[i][1];
            if (nr < 0 || nr >= m || nc < 0 || nc >= n || matrix[nr][nc] == '#') {
                continue;
            }
            if (dist[nr][nc] > d + 1) {
                dist[nr][nc] = d + 1;
                pq.push({d + 1, nr, nc, used});
            }
        }
    }
    return dist[m - 1][n - 1] == 0x3f3f3f3f ? -1 : dist[m - 1][n - 1];
}


// LC1590
int minSubarray(vector<int>& nums, int p)
{
    int i;
    int n = nums.size();
    int remainder;
    long long sum;

    sum = 0;
    for (i = 0; i < n; i++) {
        sum += nums[i];
    }
    remainder = sum % p;
    if (remainder == 0) {
        return 0;
    }

    int t = 0;
    int ans = n;
    unordered_map<int, int> r;
    // cout << remainder << endl;
    for (i = 0; i < n; i++) {
        if (nums[i] % p == remainder) {
            return 1;
        }
        t = (t + nums[i]) % p;
        if (t == 0) {
            ans = min(ans, n - 1 - i);
        } else if (t == remainder) {
            ans = min(ans, i + 1);
            if (r.count(t)) {
                ans = min(ans, i - r[t] - 1);
            }
        } else if (r.count(t + p - remainder)) {
            ans = min(ans, i - r[t + p - remainder]);
        } else if (r.count(t - remainder)) {
            ans = min(ans, i - r[t - remainder]);
        }
        // cout << t << endl;
        r[t] = i;
    }
    return ans == n ? -1 : ans;
}


// LC321
vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k)
{
    int i, j;
    int m = nums1.size();
    int n = nums2.size();

    // 从nums选取长度为k的最大字典序子序列
    auto FindMaxSubsquence = [](vector<int>& nums, int k) {
        int i;
        int n = nums.size();
        vector<int> ans;
        stack<int> st;
        for (i = 0; i < n; i++) {
            if (st.empty()) {
                st.push(nums[i]);
                continue;
            }
            auto t = st.top();
            while (t < nums[i] && st.size() - 1 + n - i >= k) {
                st.pop();
                if (st.empty()) {
                    break;
                }
                t = st.top();
            }
            if (st.size() < k) {
                st.push(nums[i]);
            }
        }
        while (!st.empty()) {
            ans.emplace_back(st.top());
            st.pop();
        }
        reverse(ans.begin(), ans.end());
        return ans;
    };

    // 比较两个数组大小
    auto Cmp = [](vector<int>& nums1, int idx1, vector<int>& nums2, int idx2) {
        int i, j;
        int m = nums1.size();
        int n = nums2.size();

        i = idx1;
        j = idx2;
        while (i < m && j < n) {
            if (nums1[i] != nums2[j]) {
                return nums1[i] > nums2[j];
            }
            i++;
            j++;
        }
        if (i != m) {
            return true;
        }
        return false;
    };

    // 合并两个数组使字典序最大
    auto Combine = [&Cmp](vector<int>& nums1, vector<int>& nums2) {
        int i, j;
        int m = nums1.size();
        int n = nums2.size();
        vector<int> ans;

        if (nums1.empty()) {
            return nums2;
        }
        if (nums2.empty()) {
            return nums1;
        }

        i = j = 0;
        while (i < m && j < n) {
            if (nums1[i] > nums2[j]) {
                ans.emplace_back(nums1[i]);
                i++;
            } else if (nums1[i] < nums2[j]) {
                ans.emplace_back(nums2[j]);
                j++;
            } else { // 出现相同数字, 字典序比较
                if (Cmp(nums1, i, nums2, j)) {
                    ans.emplace_back(nums1[i]);
                    i++;
                } else {
                    ans.emplace_back(nums2[j]);
                    j++;
                }
            }
        }
        while (i != m) {
            ans.emplace_back(nums1[i]);
            i++;
        }
        while (j != n) {
            ans.emplace_back(nums2[j]);
            j++;
        }
        return ans;
    };

    vector<vector<int>> subsequence1(m + 1);
    vector<vector<int>> subsequence2(n + 1);

    for (i = 1; i <= m; i++) {
        subsequence1[i] = FindMaxSubsquence(nums1, i);
    }
    for (i = 1; i <= n; i++) {
        subsequence2[i] = FindMaxSubsquence(nums2, i);
    }
    vector<int> ans(k, 0);
    for (i = 0; i <= m; i++) {
        if ((k - i > 0 && k - i <= n && !subsequence2[k - i].empty()) || (k - i == 0)) {
            ans = max(ans, Combine(subsequence1[i], subsequence2[k - i]));
        }
    }
    return ans;
}


// LC675
int cutOffTree(vector<vector<int>>& forest)
{
    int i, j;
    int m = forest.size();
    int n = forest[0].size();
    int d, ans;
    queue<pair<int, int>> q;
    vector<pair<int, int>> trees;

    if (forest[0][0] == 0) {
        return -1;
    }
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (forest[i][j] > 1) {
                trees.push_back({forest[i][j], i * n + j});
            }
        }
    }

    sort(trees.begin(), trees.end());

    auto dij = [&forest](int start, int end) {
        int i;
        int m = forest.size();
        int n = forest[0].size();
        int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        vector<vector<int>> dist(m, vector<int>(n, 0x3f3f3f3f));
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

        pq.push({0, start});
        while (!pq.empty()) {
            auto [d, pos] = pq.top();
            pq.pop();

            auto r = pos / n;
            auto c = pos % n;
            if (dist[r][c] < d) {
                continue;
            }

            dist[r][c] = d;
            for (i = 0; i < 4; i++) {
                auto nr = r + directions[i][0];
                auto nc = c + directions[i][1];
                if (nr < 0 || nr >= m || nc < 0 || nc >= n || forest[nr][nc] == 0) {
                    continue;
                }
                if (dist[nr][nc] > d + 1) {
                    dist[nr][nc] = d + 1;
                    pq.push({d + 1, nr * n + nc});
                }
            }
        }
        return dist[end / n][end % n];
    };

    n = trees.size();
    // 从(0, 0)到最矮的树
    d = dij(0, trees[0].second);
    if (d == 0x3f3f3f3f) {
        return -1;
    }
    ans = d;
    for (i = 1; i < n; i++) {
        d = dij(trees[i - 1].second, trees[i].second);
        if (d == 0x3f3f3f3f) {
            return -1;
        }
        ans += d;
    }
    return ans;
}


// LC1383
int maxPerformance(int n, vector<int>& speed, vector<int>& efficiency, int k)
{
    int i;
    int mod = 1e9 + 7;
    long long curSum, multiply;
    vector<pair<int, int>> vp;

    for (i = 0; i < n; i++) {
        vp.push_back({efficiency[i], speed[i]});
    }

    sort(vp.rbegin(), vp.rend());

    auto cmp = [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);

    multiply = curSum = 0;
    // 注意是最多k个
    // efficiency - speed
    for (i = 0; i < n; i++) {
        if (i < k) {
            curSum += vp[i].second;
            pq.push(vp[i]);
            multiply = max(multiply, curSum * vp[i].first);
            continue;
        }
        
        auto t = pq.top();
        if (t.second >= vp[i].second) {
            continue;
        }

        // 更大的curSum
        curSum = curSum - t.second + vp[i].second;
        pq.pop();
        pq.push(vp[i]);

        multiply = max(multiply, curSum * vp[i].first);
    }
    return multiply % mod;
}


// LC864
int shortestPathAllKeys(vector<string>& grid)
{
    int i, j;
    int k;
    int m = grid.size();
    int n = grid[0].size();
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    priority_queue<tuple<int, int, int, int>, vector<tuple<int, int, int, int>>, greater<>> pq;

    k = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] == '@') {
                pq.push({0, i, j, 0});
            } else if (islower(grid[i][j])) {
                k++;
            }
        }
    }
    vector<vector<vector<int>>> dist(m, vector<vector<int>>(n, vector<int>(1 << k, 0x3f3f3f3f)));

    while (!pq.empty()) {
        // state 获取钥匙状态
        auto [d, r, c, state] = pq.top();
        pq.pop();

        if (dist[r][c][state] < d) {
            continue;
        }
        dist[r][c][state] = d;
        if (state == (1 << k) - 1) {
            return dist[r][c][state];
        }
        for (i = 0; i < 4; i++) {
            auto nr = r + directions[i][0];
            auto nc = c + directions[i][1];
            if (nr < 0 || nr >= m || nc < 0 || nc >= n || grid[nr][nc] == '#') {
                continue;
            }
            if (isupper(grid[nr][nc])) {
                // 还没找到锁grid[nr][nc]的钥匙
                if ((1 << (grid[nr][nc] - 'A') & state) == 0) {
                    continue;
                }
                if (dist[nr][nc][state] > d + 1) {
                    dist[nr][nc][state] = d + 1;
                    pq.push({d + 1, nr, nc, state});
                }
            } else if (islower(grid[nr][nc])) {
                auto n_state = 1 << (grid[nr][nc] - 'a') | state;
                if (dist[nr][nc][n_state] > d + 1) {
                    dist[nr][nc][n_state] = d + 1;
                    pq.push({d + 1, nr, nc, n_state});
                }
            } else {
                if (dist[nr][nc][state] > d + 1) {
                    dist[nr][nc][state] = d + 1;
                    pq.push({d + 1, nr, nc, state});
                }
            }
        }
    }

    return -1;
}


// LC909
int snakesAndLadders(vector<vector<int>>& board)
{
    int i, j;
    int n = board.size();
    int cur = n * n;
    int limit;
    int inf = 0x3f3f3f3f;
    vector<vector<int>> grid(n, vector<int>(n));

    // 构建n * n折线棋盘
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            grid[i][j] = cur;
            cur--;
        }
    }
    // 根据n反转grid的行
    if (n % 2 == 0) {
        for (i = 1; i < n; i += 2) {
            reverse(grid[i].begin(), grid[i].end());
        }
    } else {
        for (i = 0; i < n; i += 2) {
            reverse(grid[i].begin(), grid[i].end());
        }
    }

    vector<vector<int>> edges(n * n + 1);
    vector<pair<int, int>> pos(n * n + 1);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            pos[grid[i][j]] = {i, j};
            if (board[i][j] != -1) {
                edges[grid[i][j]].emplace_back(board[i][j]);
                // edges[board[i][j]].emplace_back(grid[i][j]); 梯子和蛇都是单向
            }
        }
    }

    vector<vector<int>> dist(n, vector<int>(n, inf));
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    pq.push({0, 1});
    limit = n * n;
    while (!pq.empty()) {
        auto [d, p] = pq.top();
        pq.pop();

        auto x = pos[p].first;
        auto y = pos[p].second;

        if (dist[x][y] < d) {
            continue;
        }

        dist[x][y] = d;
        for (i = p + 1; i <= p + 6; i++) {
            if (i > limit) {
                break;
            }
            auto nx = pos[i].first;
            auto ny = pos[i].second;
            if (!edges[i].empty()) {
                for (auto& node : edges[i]) {
                    nx = pos[node].first;
                    ny = pos[node].second;
                    if (d + 1 < dist[nx][ny]) {
                        dist[nx][ny] = d + 1;
                        pq.push({d + 1, grid[nx][ny]});
                    }
                }
            } else {
                if (d + 1 < dist[nx][ny]) {
                    dist[nx][ny] = d + 1;
                    pq.push({d + 1, grid[nx][ny]});
                }
            }
        }
    }

    auto ex = pos[limit].first;
    auto ey = pos[limit].second;
    return dist[ex][ey] == inf ? -1 : dist[ex][ey];
}


// LC2467
int mostProfitablePath(vector<vector<int>>& edges, int bob, vector<int>& amount)
{
    int i, j;
    int m;
    int n = edges.size() + 1;
    int inf = 0x3f3f3f3f;
    vector<vector<int>> e(n);

    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    vector<int> route, record;
    function<void (int, int, int)> dfs = [&dfs, &e, &record, &route](int cur, int parent, int target) {
        if (cur == target) {
            record.emplace_back(cur);
            route = record;
            return;
        }
        record.emplace_back(cur);
        for (auto& node : e[cur]) {
            if (node != parent) {
                dfs(node, cur, target);
            }
        }
        record.pop_back();
    };

    dfs(0, -1, bob);
    reverse(route.begin(), route.end());

    vector<int> bobsRoute(n, inf);
    m = route.size();
    for (i = 0; i < m; i++) {
        bobsRoute[route[i]] = i;
    }

    int step = 0;
    int ans = -inf;
    queue<tuple<int, int, int>> q;
    q.push({0, -1, 0});
    while (!q.empty()) {
        int size = q.size();
        for (i = 0; i < size; i++) {
            auto [cur, parent, val] = q.front();
            // cout << cur << " " << parent << " " << val << " " << step << endl;
            q.pop();
            m = e[cur].size();
            if (bobsRoute[cur] == inf || step < bobsRoute[cur]) {
                if (m == 1 && e[cur][0] == parent) {
                    ans = max(ans, val + amount[cur]);
                    continue;
                }
                for (j = 0; j < m; j++) {
                    if (e[cur][j] == parent) {
                        continue;
                    }
                    q.push({e[cur][j], cur, val + amount[cur]});
                }
            } else if (bobsRoute[cur] == step) {
                if (m == 1 && e[cur][0] == parent) {
                    ans = max(ans, val + amount[cur] / 2);
                    continue;
                }
                for (j = 0; j < m; j++) {
                    if (e[cur][j] == parent) {
                        continue;
                    }
                    q.push({e[cur][j], cur, val + amount[cur] / 2});
                }
            } else {
                if (m == 1 && e[cur][0] == parent) {
                    ans = max(ans, val);
                    continue;
                }
                for (j = 0; j < m; j++) {
                    if (e[cur][j] == parent) {
                        continue;
                    }
                    q.push({e[cur][j], cur, val});
                }
            }
        }
        step++;
    }
    return ans;
}


// LC3568
int minMoves(vector<string>& classroom, int energy)
{
    int i, j;
    int inf = 0x3f3f3f3f;
    int m = classroom.size();
    int n = classroom[0].size();
    int pos, start, cnt;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    vector<int> litter(m * n);
    cnt = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (classroom[i][j] == 'L') {
                pos = i * n + j;
                litter[pos] = cnt;
                cnt++;
            } else if (classroom[i][j] == 'S') {
                start = i * n + j; 
            }
        }
    }
    if (cnt == 0) {
        return 0;
    }
    // [当前移动次数, 剩余能量, 位置, 垃圾是否清理(二进制压缩)]
    priority_queue<tuple<int, int, int, int>, vector<tuple<int, int, int, int>>, greater<>> pq;
    pq.emplace(0, energy, start, 0);
    vector<vector<vector<int>>> dist(m * n, vector<vector<int>>(energy + 1, vector<int>(1 << cnt, inf)));
    vector<vector<int>> data(m * n, vector<int>(1 << cnt, -1)); 
    while (!pq.empty()) {
        auto [step, e, p, val] = pq.top();
        pq.pop();
        // printf ("%d %d %d %d\n", step, e, p, val);

        if (dist[p][e][val] < step) {
            continue;
        }
        dist[p][e][val] = step;
        if (val == (1 << cnt) - 1) {
            return step;
        }
        // 剪枝
        if (data[p][val] == -1) {
            data[p][val] = e;
        } else if (data[p][val] > e) {
            continue;
        }
        for (i = 0; i < 4; i++) {
            auto x = p / n;
            auto y = p % n;
            auto nx = x + directions[i][0];
            auto ny = y + directions[i][1];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || classroom[nx][ny] == 'X') {
                continue;
            }
            auto npos = nx * n + ny;
            if (classroom[nx][ny] == 'R') {
                if (e > 0 && dist[npos][energy][val] > step + 1) {
                    dist[npos][energy][val] = step + 1;
                    pq.emplace(step + 1, energy, npos, val);
                }
            } else if (classroom[nx][ny] == 'L') {
                auto nval = (val | (1 << litter[npos]));
                if (e > 0 && dist[npos][e - 1][nval] > step + 1) {
                    dist[npos][e - 1][nval] = step + 1;
                    pq.emplace(step + 1, e - 1, npos, nval);
                }
            } else {
                if (e > 0 && dist[npos][e - 1][val] > step + 1) {
                    dist[npos][e - 1][val] = step + 1;
                    pq.emplace(step + 1, e - 1, npos, val);
                }
            }
        }
    }
    return -1;
}


// LCP 03. 机器人大冒险
bool robot(string command, vector<vector<int>>& obstacles, int x, int y)
{
    int i;
    int len = command.size();
    int u, r;

    u = r = 0;
    for (i = 0; i < len; i++) {
        if (command[i] == 'U') {
            u++;
        } else {
            r++;
        }
    }

    auto CanReach = [](int x, int y, int u, int r, string& command) {
        int i;
        int len = command.size();
        int m = x / r;
        int n = y / u;

        m = min(m, n);

        int ex = x - m * r;
        int ey = y - m * u;
        int sx, sy;

        sx = sy = 0;
        if (sx == ex && sy == ey) {
            return true;
        }
        for (i = 0; i < len; i++) {
            if (command[i] == 'U') {
                sy++;
            } else {
                sx++;
            }
            if (sx == ex && sy == ey) {
                return true;
            }
        }
        return false;
    };

    for (auto& o : obstacles) {
        // 障碍物在终点之后
        if (o[0] > x || o[1] > y) {
            continue;
        }
        if (CanReach(o[0], o[1], u, r, command)) {
            return false;
        }
    }
    return CanReach(x, y, u, r, command);
}


// LC3573
long long maximumProfit(vector<int>& prices, int k)
{
    int i, j;
    int n = prices.size();
    long long inf = LLONG_MIN / 2;
    long long ans = inf;

    // dp[n][k][0 ~ 2] - 已经进行了k次交易, 在第n天是否进行交易的最大利润 0 - 不持有股票
    // 1 - 进行普通交易 2 - 做空交易
    vector<vector<vector<long long>>> dp(n, vector<vector<long long>>(k + 1,
        vector<long long>(3, inf)));

    dp[0][0][0] = 0;
    dp[0][1][1] = -prices[0];
    dp[0][1][2] = prices[0];
    for (i = 1; i < n; i++) {
        for (j = 0; j <= k; j++) {
            if (j > 0) {
                dp[i][j][1] = dp[i - 1][j][1];
                dp[i][j][2] = dp[i - 1][j][2];
                if (dp[i - 1][j - 1][0] != inf) {
                    dp[i][j][1] = max(dp[i][j][1], dp[i - 1][j - 1][0] - prices[i]);
                    dp[i][j][2] = max(dp[i][j][2], dp[i - 1][j - 1][0] + prices[i]);
                }
            }

            dp[i][j][0] = dp[i - 1][j][0];
            if (dp[i - 1][j][1] != inf) {
                dp[i][j][0] = max(dp[i][j][0], dp[i - 1][j][1] + prices[i]);
            }
            if (dp[i - 1][j][2] != inf) {
                dp[i][j][0] = max(dp[i][j][0], dp[i - 1][j][2] - prices[i]);
            }
            ans = max(ans, dp[i][j][0]);
        }
    }
    return ans;
}


// LC2472
int maxPalindromes(string s, int k)
{
    // 类似LC5先找所有大于等于k的回文串
    int i, j;
    int n = s.size();
    int ans;
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    vector<vector<int>> ranges;

    for (j = 0; j < n; j++) {
        for (i = j; i >= 0; i--) {
            if (s[i] != s[j]) {
                continue;
            }
            if (i == j || i + 1 == j) {
                dp[i][j] = true;
            } else if (dp[i + 1][j - 1]) {
                dp[i][j] = true;
            }
            if (dp[i][j] && j - i + 1 >= k) {
                ranges.push_back({i, j});
            }
        }
    }

    if (ranges.empty()) {
        return 0;
    }

    // 按右边界排序
    sort(ranges.begin(), ranges.end(), [](const vector<int>& a, const vector<int>& b) {
        return a[1] < b[1];
    });

    int right;

    n = ranges.size();
    ans = 1;
    right = ranges[0][1];
    for (i = 1; i < n; i++) {
        if (ranges[i][0] > right) {
            ans++;
            right = ranges[i][1];
        }
    }
    return ans;
}


// LC2392
vector<vector<int>> buildMatrix(int k, vector<vector<int>>& rowConditions, vector<vector<int>>& colConditions)
{
    // 分别满足rowConditions和colConditions, 且两个condition都不能形成环
    auto CreateGraph = [](vector<vector<int>>& conditions, vector<vector<int>>& edges, vector<int>& indegree) {
        for (auto& c : conditions) {
            edges[c[0]].emplace_back(c[1]);
            indegree[c[1]]++;
        }
    };

    auto TopoOrder = [](vector<vector<int>>& edges, vector<int>& indegree, vector<int>& order) {
        queue<int> q;
        int i;
        int n = indegree.size();
        for (i = 1; i < n; i++) {
            if (indegree[i] == 0) {
                q.push(i);
            }
        }
        while (!q.empty()) {
            auto t = q.front();
            q.pop();

            order.emplace_back(t);
            for (auto& next : edges[t]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.emplace(next);
                }
            }
        }
    };

    vector<vector<int>> edgesR(k + 1);
    vector<int> indegreeR(k + 1, 0);
    vector<int> orderR;

    CreateGraph(rowConditions, edgesR, indegreeR);
    TopoOrder(edgesR, indegreeR, orderR);
    // 有环则orderR节点数量不等于k
    if (orderR.size() != k) {
        return {};
    }

    vector<vector<int>> edgesC(k + 1);
    vector<int> indegreeC(k + 1, 0);
    vector<int> orderC;

    CreateGraph(colConditions, edgesC, indegreeC);
    TopoOrder(edgesC, indegreeC, orderC);
    if (orderC.size() != k) {
        return {};
    }

    int i;
    vector<vector<int>> grid(k, vector<int>(k, 0));
    vector<pair<int, int>> pos(k + 1);
    for (i = 0; i < k; i++) {
        pos[orderR[i]].first = i;
        pos[orderC[i]].second = i;
    }
    for (i = 1; i <= k; i++) {
        grid[pos[i].first][pos[i].second] = i;
    }
    return grid;
}


// LC395
unordered_map<string, int> data_LC395;
int longestSubstring(string s, int k)
{
    if (data_LC395.count(s)) {
        return data_LC395[s];
    }

    int i;
    int n = s.size();
    int m;
    string t;
    vector<string> strs;
    unordered_map<char, vector<int>> idx;
    for (i = 0; i < n; i++) {
        idx[s[i]].emplace_back(i);
    }
    bool f = true;
    for (auto& it : idx) {
        if (it.second.size() < k) {
            f = false;
            // it.second的每一个区间内考虑情况
            auto& v = it.second;
            m = v.size();
            if (v[0] != 0 && v[0] >= k) {
                t = s.substr(0, v[0]);
                strs.emplace_back(t);
            }
            for (i = 1; i < m; i++) {
                if (v[i] - v[i - 1] - 1 < k) {
                    continue;
                }
                t = s.substr(v[i - 1] + 1, v[i] - v[i - 1] - 1);
                strs.emplace_back(t);
            }
            if (v[m - 1] != n - 1 && n - v[m - 1] - 1 >= k) {
                t = s.substr(v[m - 1] + 1);
                strs.emplace_back(t);
            }
        }
    }
    if (f) {
        return n;
    }
    // for (auto str :strs) cout << str << endl;
    int maxLen = 0;
    for (auto& str : strs) {
        maxLen = max(maxLen, longestSubstring(str, k));
    }
    data_LC395[s] = maxLen;
    return maxLen;
}


// LC3593
int minIncrease(int n, vector<vector<int>>& edges, vector<int>& cost)
{
    long long maxCost;
    vector<vector<int>> e(n);
    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    // maxVal[i] - 以i为根节点到其任意叶子节点路径最大cost
    vector<long long> maxVal(n, -1);
    function<long long (int, int)> dfs = [&dfs, &cost, &e, &maxVal](int cur, int parent) {
        if (maxVal[cur] != -1) {
            return maxVal[cur];
        }

        long long ans = 0;
        long long sum;

        ans = cost[cur];
        for (auto& next : e[cur]) {
            sum = cost[cur];
            if (next != parent) {
                sum += dfs(next, cur);
                ans = max(ans, sum);
            }
        }
        maxVal[cur] = ans;
        return ans;
    };

    dfs(0, -1);
    maxCost = 0;
    for (auto m : maxVal) {
        // cout << m << " ";
        maxCost = max(maxCost, m);
    }

    int cnt = 0;
    function <void (int, int, long long)> GetIncrease = [&GetIncrease, &maxVal, &e, &cost, &cnt](int cur, int parent, long long curVal) {
        if (maxVal[cur] == curVal) {
            for (auto& next : e[cur]) {
                if (next != parent) {
                    GetIncrease(next, cur, curVal - cost[cur]);
                }
            }
        } else {
            cnt++;
            for (auto& next : e[cur]) {
                if (next != parent) {
                    GetIncrease(next, cur, curVal - cost[cur] - (curVal - maxVal[cur]));
                }
            }
        }
    };

    GetIncrease(0, -1, maxCost);

    return cnt;
}


// LC239
vector<int> maxSlidingWindow(vector<int>& nums, int k)
{
    if (k == 1) {
        return nums;
    }

    // 单调递减队列
    deque<int> dq;

    int i;
    int n = nums.size();
    vector<int> ans;
    for (i = 0; i < n; i++) {
        if (dq.empty()) {
            dq.push_back(i);
            continue;
        }
        while (i - dq.front() >= k) {
            dq.pop_front();
            if (dq.empty()) {
                break;
            }
        }
        while (nums[dq.back()] < nums[i]) {
            dq.pop_back();
            if (dq.empty()) {
                break;
            }
        }
        dq.push_back(i);
        if (i >= k - 1) {
            ans.emplace_back(nums[dq.front()]);
        }
    }
    return ans;
}


// LC3559
vector<int> assignEdgeWeights(vector<vector<int>>& edges, vector<vector<int>>& queries)
{
    int mod = 1e9 + 7;
    int n = edges.size() + 1;
    int a, b;
    vector<vector<int>> e(n + 1);
    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    BinaryLiftingLCA bll(e, 1);
    vector<int> ans;
    vector<int> depth = bll.GetDepth();
    for (auto& q : queries) {
        if (q[0] == q[1]) {
            ans.emplace_back(0);
            continue;
        }
        int node = bll.lca(q[0], q[1]);
        ans.emplace_back(FastPow(2, depth[q[1]] + depth[q[0]] - depth[node] * 2 - 1, mod));
    }
    return ans;
}


// LC1353
int maxEvents(vector<vector<int>>& events)
{
    int i, j;
    int n = events.size();
    int lastDay;
    int ans;

    sort(events.begin(), events.end());
    lastDay = 0;
    for (auto& e : events) {
        lastDay = max(lastDay, e[1]);
    }

    for (auto e : events) cout << e[0] << " " << e[1] << endl;

    // 存储event end day
    priority_queue<int, vector<int>, greater<>> pq; 
    ans = 0;
    j = 0;
    for (i = 1; i <= lastDay; i++) {
        while (j < n && events[j][0] <= i) {
            pq.push(events[j][1]);
            j++;
        }
        // 小于i的会议没办法参加了
        while (!pq.empty() && pq.top() < i) {
            pq.pop();
        }
        if (!pq.empty()) {
            ans++;
            pq.pop();
        }
    }
    return ans;
}


// LC2448
long long minCost_LC2448(vector<int>& nums, vector<int>& cost)
{
    int i;
    int n = nums.size();
    vector<pair<long long, long long>> vp;
    for (i = 0; i < n; i++) {
        vp.push_back({nums[i], cost[i]});
    }

    sort(vp.begin(), vp.end());

    vector<long long> prefixnXcSum(n);
    vector<long long> prefixCostSum(n);

    prefixCostSum[0] = vp[0].second;
    prefixnXcSum[0] = vp[0].first * vp[0].second;
    for (i = 1; i < n; i++) {
        prefixnXcSum[i] = prefixnXcSum[i - 1] + vp[i].first * vp[i].second;
        prefixCostSum[i] = prefixCostSum[i - 1] + vp[i].second;
        // cout << prefixnXcSum[i] << " " << prefixCostSum[i] << endl;
    }

    long long ans = LLONG_MAX;
    long long cur;
    for (i = 0; i < n; i++) {
        cur = prefixCostSum[i] * vp[i].first - prefixnXcSum[i] + 
            (prefixnXcSum[n - 1] - prefixnXcSum[i]) - 
            (prefixCostSum[n - 1] - prefixCostSum[i]) * vp[i].first;
        ans = min(ans, cur);
        // cout << cur << endl;
    }

    return ans;
}


// LC3331
vector<int> findSubtreeSizes(vector<int>& parent, string s)
{
    int i;
    int n = parent.size();

    vector<vector<int>> edges(n);

    for (i = 0; i < n; i++) {
        if (parent[i] != -1) {
            edges[parent[i]].emplace_back(i);
        }
    }

    // 字符所在的最近节点编号
    vector<int> ancestor(26, -1);
    function<void (int)> ScanTree = [&ScanTree, &edges, &parent, &ancestor, &s](int cur) {
        int t = ancestor[s[cur] - 'a'];
        if (edges[cur].empty()) {
            if (t != -1 && t != parent[cur]) {
                parent[cur] = t;
            }
            return;
        }
        for (auto& next : edges[cur]) {
            if (t == -1) {
                ancestor[s[cur] - 'a'] = cur;
                ScanTree(next);
                ancestor[s[cur] - 'a'] = t;
            } else {
                if (t != parent[cur]) {
                    parent[cur] = t;
                }
                ancestor[s[cur] - 'a'] = cur;
                ScanTree(next);
                ancestor[s[cur] - 'a'] = t;
            }
        }
    };

    ScanTree(0);

    // 由更新后的parent重新建图
    for (i = 0; i < n; i++) {
        edges[i].clear();
    }
    for (i = 0; i < n; i++) {
        // cout << parent[i] << " ";
        if (parent[i] != -1) {
            edges[parent[i]].emplace_back(i);
        }
    }

    vector<int> nodeNums(n, 0);
    function<int (int)> dfs = [&dfs, &edges, &nodeNums](int cur) {
        if (nodeNums[cur] != 0) {
            return nodeNums[cur];
        }

        int ans = 1;
        for (auto& next : edges[cur]) {
            ans += dfs(next);
        }
        nodeNums[cur] = ans;
        return ans;
    };

    dfs(0);
    return nodeNums;
}


// LC2327
vector<bool> friendRequests(int n, vector<vector<int>>& restrictions, vector<vector<int>>& requests)
{
    int i;
    int m = restrictions.size();
    UnionFind uf = UnionFind(n);
    vector<bool> ans;
    for (auto& r : requests) {
        auto rk0 = uf.findSet(r[0]);
        auto rk1 = uf.findSet(r[1]);
        if (rk0 == rk1) {
            ans.emplace_back(true);
            continue;
        }
        // 从限制条件入手, 如果r0, r1合并则rk0 rk1相等, 可能导致限制条件合并
        for (i = 0; i < m; i++) {
            auto rk2 = uf.findSet(restrictions[i][0]);
            auto rk3 = uf.findSet(restrictions[i][1]);
            if ((rk0 == rk2 && rk1 == rk3) || (rk0 == rk3 && rk1 == rk2)) {
                ans.emplace_back(false);
                break;
            }
        }
        if (i == m) {
            uf.unionSets(r[0], r[1]);
            ans.emplace_back(true);
        }
    }
    return ans;
}


// LC3614
char processStr(string s, long long k)
{
    int i;
    int n = s.size();
    // 执行s[i]后目标字符串的长度
    vector<long long> opLen(n);
    if (isalpha(s[0])) {
        opLen[0] = 1;
    } else {
        opLen[0] = 0;
    }
    for (i = 1; i < n; i++) {
        if (isalpha(s[i])) {
            opLen[i] = opLen[i - 1] + 1;
        } else if (s[i] == '*') {
            if (opLen[i - 1] > 0) {
                opLen[i] = opLen[i - 1] - 1;
            }
        } else if (s[i] == '#') {
            opLen[i] = opLen[i - 1] * 2;
        } else {
            // '%'
            opLen[i] = opLen[i - 1];
        }
    }
    if (opLen[n - 1] <= k) {
        return '.';
    }

    // cout << opLen[n - 1] << endl;
    long long len = opLen[n - 1];
    for (i = n - 1; i >= 0; i--) {
        if (isalpha(s[i])) {
            if (opLen[i] == k + 1) {
                return s[i];
            }
            len--;
        } else if (s[i] == '*') {
            len++;
        } else if (s[i] == '#') {
            if (i > 0) {
                auto tLen = opLen[i - 1];
                if (k >= tLen) {
                    k -= tLen;
                }
                len = tLen;
            }
        } else if (s[i] == '%') {
            // idx ~ n - 1 - idx
            k = len - 1 - k;
        }
        // cout << "k = " << k << endl;
    }
    return s[k];
}


// LC3302
vector<int> validSequence(string word1, string word2)
{
    int i, j;
    int m = word1.size();
    int n = word2.size();

    if (m < n) {
        return {};
    }

    vector<int> ans(n);
    // 预处理, suffix[i] - 以word1的i位开始的后缀最多能覆盖word2后缀的位置
    vector<int> suffix(m, n);
    i = m - 1;
    j = n - 1;
    while (i >= 0 && j >= 0) {
        if (word1[i] == word2[j]) {
            suffix[i] = j;
            j--;
        } else {
            if (i < m - 1) {
                suffix[i] = suffix[i + 1];
            }
        }
        i--;
    }
    while (i >= 0) {
        if (i < m - 1) {
            suffix[i] = suffix[i + 1];
        }
        i--;
    }
    // for (auto s : suffix) cout << s << " ";
    bool used = false;
    i = j = 0;
    while (i < m && j < n) {
        if (word2[j] == word1[i]) {
            ans[j] = i;
            j++;
        } else {
            if (used == false) {
                if (i == m - 1) {
                    if (j != n - 1) {
                        break;
                    }
                    ans[j] = m - 1;
                    j++;
                    continue;
                }
                // i < m - 1
                if (j == n - 1) {
                    ans[j] = i;
                    j++;
                    continue;
                }
                if (suffix[i + 1] <= j + 1) {
                    used = true;
                    ans[j] = i;
                    j++;
                }
            }
        }
        i++;
    }
    if (j == n) {
        return ans;
    }
    return {};
}


// LC2876
vector<int> countVisitedNodes(vector<int>& edges)
{
    // 条件已限定n个节点, n条边且每个点的出度均为1, 则通过拓扑排序后剩下的点一定是一个环或多个环
    int i;
    int n = edges.size();
    vector<int> inDegree(n, 0);
    vector<int> isloopNodes(n, 1);
    for (i = 0; i < n; i++) {
        inDegree[edges[i]]++;
    }

    queue<int> q;
    for (i = 0; i < n; i++) {
        if (inDegree[i] == 0) {
            isloopNodes[i] = 0;
            q.push(i);
        }
    }
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        inDegree[edges[node]]--;
        if (inDegree[edges[node]] == 0) {
            q.push(edges[node]);
            isloopNodes[edges[node]] = 0;
        }
    }

    // 注意可能不止一个环(多个连通分量)
    int len;
    vector<int> visited(n, 0);
    vector<int> loopNodes;
    vector<int> loopLen(n, 0);
    auto FindLoop = [&edges, &visited, &len, &loopNodes](auto&& self, int cur) -> void {
        visited[cur] = 1;
        len++;
        loopNodes.emplace_back(cur);
        if (visited[edges[cur]] == 0) {
            self(self, edges[cur]);
        }
    };
    for (i = 0; i < n; i++) {
        if (isloopNodes[i] && visited[i] == 0) {
            len = 0;
            FindLoop(FindLoop, i);
            while (!loopNodes.empty()) {
                loopLen[loopNodes.back()] = len;
                loopNodes.pop_back();
            }
            // cout << "loopLen = " << len << endl;
        }
    }

    // 反向图
    vector<vector<int>> rev_edges(n);
    for (i = 0; i < n; i++) {
        rev_edges[edges[i]].emplace_back(i);
    }

    vector<int> ans(n);
    auto dfs = [&isloopNodes, &rev_edges, &ans](auto&& self, int cur, int len) -> void {
        ans[cur] = len;
        for (auto& next : rev_edges[cur]) {
            if (isloopNodes[next] == 0) {
                self(self, next, len + 1);
            }
        }
    };
    
    for (i = 0; i < n; i++) {
        if (isloopNodes[i]) {
            dfs(dfs, i, loopLen[i]);
        }
    }
    return ans;
}


// LC2050
int minimumTime(int n, vector<vector<int>>& relations, vector<int>& time)
{
    int i;
    // 第i门课程可以学习的时刻
    vector<int> learningTime(n + 1, 0);

    vector<int> inDegree(n + 1, 0);
    // 条件已保证relations是一个DAG, 但一门课程可能有多个出度
    vector<vector<int>> edges(n + 1);
    for (auto& r : relations) {
        inDegree[r[1]]++;
        edges[r[0]].emplace_back(r[1]);
    }
    queue<pair<int, int>> q;
    for (i = 1; i <= n; i++) {
        if (inDegree[i] == 0) {
            q.push({i, time[i - 1]});
        }
    }

    while (!q.empty()) {
        auto [course, needTime] = q.front();
        q.pop();
        learningTime[course] = max(learningTime[course], needTime);
        for (auto& next : edges[course]) {
            inDegree[next]--;
            learningTime[next] = max(learningTime[next], needTime + time[next - 1]);
            if (inDegree[next] == 0) {
                q.push({next, learningTime[next]});
            }
        }
    }
    return *max_element(learningTime.begin(), learningTime.end());
}


// LC2290
int minimumObstacles(vector<vector<int>>& grid)
{
    // dijkstra
    int i;
    int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
    // {obstaclesNum, pos}
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    pq.push({0, 0});
    while (!pq.empty()) {
        auto [obstaclesNum, pos] = pq.top();
        pq.pop();
        auto r = pos / n;
        auto c = pos % n;
        if (dist[r][c] < obstaclesNum) {
            continue;
        }
        dist[r][c] = obstaclesNum;

        for (i = 0; i < 4; i++) {
            auto nr = r + directions[i][0];
            auto nc = c + directions[i][1];
            if (nr >= m || nr < 0 || nc >= n || nc < 0) {
                continue;
            }
            if (obstaclesNum + grid[nr][nc] < dist[nr][nc]) {
                dist[nr][nc] = obstaclesNum + grid[nr][nc];
                pq.push({dist[nr][nc], nr * n + nc});
            }
        }
    }

    return dist[m - 1][n - 1];
}


// LC3615
int maxLen(int n, vector<vector<int>>& edges, string label)
{
    int i, j;
    vector<vector<int>> adj(n);
    for (auto& e : edges) {
        adj[e[0]].emplace_back(e[1]);
        adj[e[1]].emplace_back(e[0]);
    }

    // dp[i] - i型路径(二进制压缩)能否组成回文路径
    vector<bool> dp;

    int ans = 0;
    // 可以从中心一(两)点向两端扩展构造回文
    auto dfs = [&label, &dp, &adj, &ans](auto&& self, int start, int end, unsigned int route, int nums) -> void {
        ans = max(ans, nums);
        int i, j;
        int size1 = adj[start].size();
        int size2 = adj[end].size();
        for (i = 0; i < size1; i++) {
            for (j = 0; j < size2; j++) {
                auto n_start = adj[start][i];
                auto n_end = adj[end][j];
                auto n_route = route | (1 << n_start) | (1 << n_end);
                if (n_start == n_end || (route & 1 << n_start) != 0 ||
                    (route & 1 << n_end) != 0 || label[n_start] != label[n_end] || dp[n_route]) {
                    continue;
                }

                dp[n_route] = true;
                self(self, n_start, n_end, n_route, nums + 2);
            }
        }
    };

    // 中心一个点
    unsigned int route;
    for (i = 0; i < n; i++) {
        dp.assign(1 << n, false);
        route = 0;
        route |= 1 << i;
        dp[route] = true;
        dfs(dfs, i, i, route, 1);
    }

    // 两个点
    int size;
    for (i = 0; i < n; i++) {
        size = adj[i].size();
        dp.assign(1 << n, false);
        for (j = 0; j < size; j++) {
            if (label[i] == label[adj[i][j]]) {
                route = 0;
                route = route | 1 << i | 1 << adj[i][j];
                dp[route] = true;
                dfs(dfs, i, adj[i][j], route, 2);
            }
        }
    }
    return ans;
}


// LC3600
int maxStability(int n, vector<vector<int>>& edges, int k)
{
    int i, m;
    int cnt, t;
    int maxLen;
    UnionFind uf(n), uf_connect(n);
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>> pq; // [weight, from, to]
    cnt = 0;
    maxLen = 2e5;
    for (auto& edge: edges) {
        if (edge[3] == 1) {
            if (uf.findSet(edge[0]) != uf.findSet(edge[1])) {
                uf.unionSets(edge[0], edge[1]);
                maxLen = min(maxLen, edge[2]);
            } else { // 形成了环
                return -1;
            }
            cnt++;
        } else {
            pq.push({edge[2], edge[0], edge[1]});
        }
        uf_connect.unionSets(edge[0], edge[1]);
    }

    // must边太多可以形成环
    if (cnt >= n) {
        return -1;
    }
    // 有多个连通分量
    int area = uf_connect.findSet(0);
    for (i = 0; i < n; i++) {
        if (uf_connect.findSet(i) != area) {
            return -1;
        }
    }

    vector<tuple<int, int, int>> e;
    while (!pq.empty()) {
        e.emplace_back(pq.top());
        pq.pop();
    }

    int left, right, mid;

    left = 1;
    right = maxLen;
    m = e.size();
    while (left <= right) {
        t = k;
        mid = (right - left) / 2 + left;
        auto tempUf = uf;
        // 从大到小遍历可升级边
        for (i = 0; i < m; i++) {
            auto [weight, a, b] = e[i];
            if (tempUf.findSet(a) != tempUf.findSet(b)) {
                if (weight >= mid) {
                    tempUf.unionSets(a, b);
                } else {
                    if (t == 0 || weight * 2 < mid) {
                        break;
                    } else {
                        tempUf.unionSets(a, b);
                        t--;
                    }
                }
            }
        }
        if (i != m) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return right;
}


// LC2322
int minimumScore(vector<int>& nums, vector<vector<int>>& edges)
{
    int i, j;
    int n = nums.size();
    vector<vector<int>> e(n);

    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    vector<int> xorVal(n, -1);
    vector<vector<int>> adj; // 以0为根的树的所有边
    auto dfs = [&nums, &e, &xorVal, &adj](auto&& self, int cur, int parent) -> int {
        if (xorVal[cur] != -1) {
            return xorVal[cur];
        }
        if (e[cur].size() == 1 && e[cur][0] == parent) {
            xorVal[cur] = nums[cur];
            return xorVal[cur];
        }
        int ans = nums[cur];
        for (auto& next : e[cur]) {
            if (next != parent) {
                adj.push_back({cur, next});
                ans ^= self(self, next, cur);
            }
        }

        xorVal[cur] = ans;
        return ans;
    };

    dfs(dfs, 0, -1);
    // for (auto x : xorVal) cout << x << " "; cout << endl;
    int m = adj.size();
    int a, b, c;
    int curXorVal;
    int ancestor, ans;
    BinaryLiftingLCA bll(e, 0);
    ans = INT_MAX;
    for (i = 0; i < m - 1; i++) {
        for (j = i + 1; j < m; j++) {
            // cout << adj[i][1] << " " << adj[j][1] << endl;
            ancestor = bll.lca(adj[i][1], adj[j][1]);
            if (ancestor == adj[i][1]) { // 子树
                a = xorVal[0] ^ xorVal[adj[i][1]];
                b = xorVal[adj[i][1]] ^ xorVal[adj[j][1]];
                c = xorVal[adj[j][1]];
            } else {
                a = xorVal[adj[i][1]];
                b = xorVal[adj[j][1]];
                c = xorVal[0] ^ a ^ b;
            }
            ans = min(ans, max({a, b, c}) - min({a, b, c}));
        }
    }
    return ans;
}
// 不使用lca, 而是采用每个节点的进入和退出时间戳来判断父子关系
int minimumScore_betterWay(vector<int>& nums, vector<vector<int>>& edges)
{
    int i, j;
    int n = nums.size();
    vector<vector<int>> e(n);

    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    vector<int> xorVal(n, -1);
    vector<vector<int>> adj; // 以0为根的树的所有边
    vector<int> in(n); // 节点进入的时间
    vector<int> out(n); // 节点退出的时间
    auto dfs = [&nums, &e, &xorVal, &adj, &in, &out](auto&& self, int cur, int parent, int& time) -> int {
        in[cur] = time;
        time++;
        if (xorVal[cur] != -1) {
            return xorVal[cur];
        }
        if (e[cur].size() == 1 && e[cur][0] == parent) {
            xorVal[cur] = nums[cur];
            out[cur] = time;
            time++;
            return xorVal[cur];
        }
        int ans = nums[cur];
        for (auto& next : e[cur]) {
            if (next != parent) {
                adj.push_back({cur, next});
                ans ^= self(self, next, cur, time);
            }
        }

        xorVal[cur] = ans;
        out[cur] = time;
        time++;
        return ans;
    };

    int time = 0;

    dfs(dfs, 0, -1, time);
    // for (auto x : in) cout << x << " "; cout << endl;
    // for (auto x : out) cout << x << " "; cout << endl;
    int m = adj.size();
    int a, b, c;
    int ans;
    ans = INT_MAX;
    for (i = 0; i < m - 1; i++) {
        for (j = i + 1; j < m; j++) {
            if (in[adj[i][1]] < in[adj[j][1]] && out[adj[i][1]] > out[adj[j][1]]) { // 子树
                a = xorVal[0] ^ xorVal[adj[i][1]];
                b = xorVal[adj[i][1]] ^ xorVal[adj[j][1]];
                c = xorVal[adj[j][1]];
            } else {
                a = xorVal[adj[i][1]];
                b = xorVal[adj[j][1]];
                c = xorVal[0] ^ a ^ b;
            }
            ans = min(ans, max({a, b, c}) - min({a, b, c}));
        }
    }
    return ans;
}


// LC2421
int numberOfGoodPaths(vector<int>& vals, vector<vector<int>>& edges)
{
    // 从最小值点扩充, 查并集计算连通分量和各个分量含有的当前最小点个数
    int i;
    int n = vals.size();
    int ans;
    map<int, vector<int>> nodes;

    for (i = 0; i < n; i++) {
        nodes[vals[i]].emplace_back(i);
    }
    vector<vector<int>> e(n);
    for (auto& edge : edges) {
        e[edge[0]].emplace_back(edge[1]);
        e[edge[1]].emplace_back(edge[0]);
    }

    UnionFind uf(n);
    unordered_map<int, int> nodesSet;
    ans = 0;
    for (auto& [val, node] : nodes) {
        for (auto& id : node) {
            for (auto& next : e[id]) {
                if (vals[next] <= val && uf.findSet(next) != uf.findSet(id)) {
                    uf.unionSets(next, id);
                }
            }
        }
        nodesSet.clear();
        for (auto& id : node) {
            nodesSet[uf.findSet(id)]++;
        }
        for (auto& [sets, nums] : nodesSet) {
            ans += (nums + 1) * nums / 2;
        }
    }
    // for (i = 0; i < n; i++) cout << uf.findSet(i) << " "; cout << endl;
    return ans;
}


// LC1574
int findLengthOfShortestSubarray(vector<int>& arr)
{
    int i;
    int n = arr.size();
    int left, right, mid;
    bool f;
    // 前缀是否不递减
    vector<bool> isIncreasing(n);
    // 后缀是否不递增
    vector<bool> isDecreasing(n);

    f = true;
    isIncreasing[0] = f;
    for (i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            f = false;
        }
        isIncreasing[i] = f;
    }
    f = true;
    isDecreasing[n - 1] = f;
    for (i = n - 2; i >= 0; i--) {
        if (arr[i] > arr[i + 1]) {
            f = false;
        }
        isDecreasing[i] = f;
    }

    left = 0;
    right = n;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        for (i = 0; i < n; i++) {
            if (i == 0) {
                if (isDecreasing[mid]) {
                    right = mid - 1;
                    break;
                }
            } else {
                if (isIncreasing[i - 1] == false) {
                    left = mid + 1;
                    break;
                }
                if (isIncreasing[i]) {
                    if (i + mid >= n || (arr[i - 1] <= arr[i + mid] && isDecreasing[i + mid])) {
                        right = mid - 1;
                        break;
                    }
                } else {
                    if (i + mid >= n) {
                        right = mid - 1;
                    } else {
                        if (arr[i - 1] <= arr[i + mid] && isDecreasing[i + mid]) {
                            right = mid - 1;
                        } else {
                            left = mid + 1;
                        }
                    }
                    break;
                }
            }
        }
    }
    return left;
}


// LC467
int findSubstringInWraproundString(string s)
{
    int i;
    int n = s.size();
    int ans, curMaxLen;
    vector<int> cnt(26, 0);

    cnt[s[0] - 'a'] = 1;
    curMaxLen = 1;
    for (i = 1; i < n; i++) {
        if ('a' + (s[i - 1] - 'a' + 1) % 26 == s[i]) {
            curMaxLen++;
        } else {
            curMaxLen = 1;
        }
        cnt[s[i] - 'a'] = max(cnt[s[i] - 'a'], curMaxLen);
    }
    ans = 0;
    for (i = 0; i < 26; i++) {
        ans += cnt[i];
    }
    return ans;
}


// LC3635
int earliestFinishTime(vector<int>& landStartTime, vector<int>& landDuration, vector<int>& waterStartTime, vector<int>& waterDuration)
{
    int i;
    int m, n;
    int lastTime;

    m = landStartTime.size();
    n = waterStartTime.size();
    vector<pair<int, int>> land;
    vector<pair<int, int>> water;

    for (i = 0; i < m; i++) {
        land.push_back({landStartTime[i], landStartTime[i] + landDuration[i]});
    }
    for (i = 0; i < n; i++) {
        water.push_back({waterStartTime[i], waterStartTime[i] + waterDuration[i]});
    }

    // 最早结束时间后缀数组
    vector<int> suffixWater(n);
    vector<int> suffixLand(m);
    sort(water.begin(), water.end());
    sort(land.begin(), land.end());

    suffixLand[m - 1] = land[m - 1].second;
    for (i = m - 2; i >= 0; i--) {
        suffixLand[i] = min(suffixLand[i + 1], land[i].second);
    }
    suffixWater[n - 1] =  water[n - 1].second;
    for (i = n - 2; i >= 0; i--) {
        suffixWater[i] = min(suffixWater[i + 1], water[i].second);
    }
    vector<int> prefixWater(n);
    vector<int> prefixLand(m);
    // 最少持续时间前缀数组
    prefixLand[0] = land[0].second - land[0].first;
    for (i = 1; i < m; i++) {
        prefixLand[i] = min(prefixLand[i - 1], land[i].second - land[i].first);
    }

    prefixWater[0] = water[0].second - water[0].first;
    for (i = 1; i < n; i++) {
        prefixWater[i] = min(prefixWater[i - 1], water[i].second - water[i].first);
    }

    int left, right, mid;
    lastTime = INT_MAX;
    // 先land
    for (i = 0; i < m; i++) {
        // 无外乎两种情况: waterStartTime小于landStartTime[i] + landDuration[i], 加上最小持续时间;
        // waterStartTime大于landStartTime[i] + landDuration[i], 选当前最早结束时间
        left = 0;
        right = n - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (water[mid].first <= landStartTime[i] + landDuration[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right >= 0) {
            lastTime = min(lastTime, landStartTime[i] + landDuration[i] + prefixWater[right]);
        }
        if (right + 1 < n) {
            lastTime = min(lastTime, suffixWater[right + 1]);
        }
    }
    // cout << lastTime << endl;
    // 先water
    for (i = 0; i < n; i++) {
        left = 0;
        right = m - 1;
        while (left <= right) {
            mid = (right - left) / 2 + left;
            if (land[mid].first <= waterStartTime[i] + waterDuration[i]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (right >= 0) {
            lastTime = min(lastTime, waterStartTime[i] + waterDuration[i] + prefixLand[right]);
        }
        if (right + 1 < m) {
            lastTime = min(lastTime, suffixLand[right + 1]);
        }
    }
    return lastTime;
}


// LC3640
long long maxSumTrionic(vector<int>& nums)
{
    int i;
    int n = nums.size();
    long long INF = -1e14;
    vector<long long> prefix(n);
    for (i = 0; i < n; i++) {
        prefix[i] = nums[i];
        if (i > 0) {
            prefix[i] += prefix[i - 1];
        }
    }

    // 寻找单减区间
    vector<vector<int>> downRange;
    int start;
    bool f = true;
    for (i = 1; i < n; i++) {
        if (nums[i] < nums[i - 1] && f) {
            start = i;
            f = false;
        }
        if (nums[i] > nums[i - 1] && f == false) {
            downRange.push_back({start, i - 1});
            f = true;
        }
        if (nums[i] == nums[i - 1] && f == false) { // 必须严格单调
            f = true;
        }
    }

    vector<long long> dp(n); // dp[i] - i结尾最大递增子数组和, 且子数组长度大于1
    dp[0] = INF;
    for (i = 1; i < n; i++) {
        if (nums[i] > nums[i - 1]) {
            dp[i] = max(nums[i - 1] * 1ll, dp[i - 1]) + nums[i];
        } else {
            dp[i] = INF;
        }
    }
    vector<long long> dp1(n); // dp1[i] - i起始最大递增子数组和, 且子数组长度大于1
    dp1[n - 1] = INF;
    for (i = n - 2; i >= 0; i--) {
        if (nums[i] < nums[i + 1]) {
            dp1[i] = max(dp1[i + 1], nums[i + 1] * 1ll) + nums[i];
        } else {
            dp1[i] = INF;
        }
    }

    long long ans = INF;
    int m = downRange.size();
    for (i = 0; i < m; i++) {
        // cout << downRange[i][0] << " " << downRange[i][1] << endl;
        ans = max(ans, dp[downRange[i][0] - 1] + dp1[downRange[i][1]] + 
            prefix[downRange[i][1] - 1] - prefix[downRange[i][0] - 1]);
    }
    return ans;
}


// LC3398 LC3399
int minLength(string s, int numOps)
{
    int i;
    int n = s.size();
    int left, right, mid;
    int cntOps, cnt0, cnt1;
    string t;
    vector<int> cnt;

    // 特判最小长度可否为1
    int type1, type2;

    // type1 - "010101..."
    // types - "101010..."
    type1 = type2 = 0;
    for (i = 0; i < n; i++) {
        if (i % 2 == 0) {
            if (s[i] == '1') {
                type1++;
            } else {
                type2++;
            }
        } else {
            if (s[i] == '0') {
                type1++;
            } else {
                type2++;
            }
        }
    }
    // cout << type1 << " " << type2 << endl;
    if (type1 <= numOps || type2 <= numOps) {
        return 1;
    }

    // 二分最小相同子字符串长度
    left = 2;
    right = n;
    while (left <= right) {
        mid = (right - left) / 2 + left;
        cntOps = 0;
        cnt.assign(2, 0);
        t = s;
        if (t[0] == '0') {
            cnt[0]++;
        } else {
            cnt[1]++;
        }
        for (i = 1; i < n; i++) {
            if (t[i] == t[i - 1]) {
                cnt[t[i] - '0']++;
                if (cnt[t[i] - '0'] > mid) {
                    // 此处分别考虑 00011 00111 mid = 2的情况
                    if (i + 1 < n && t[i] == t[i + 1]) {
                        t[i] = '1' - (t[i] - '0');
                    } else if (i + 1 < n && t[i] != t[i + 1]) {
                        t[i - 1] = '1' - (t[i - 1] - '0');
                    }
                    cnt[t[i] - '0'] = 1;
                    cnt['1' - t[i]] = 0;
                    cntOps++;
                    if (cntOps > numOps) {
                        break;
                    }
                }
            } else {
                cnt[t[i] - '0'] = 1;
                cnt['1' - t[i]] = 0;
            }
        }
        if (i != n) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}


// LC316 LC1081
string removeDuplicateLetters(string s)
{
    int i, j;
    int n = s.size();
    int cnt, prev, idx;
    char ch;
    vector<vector<int>> alphabet(26);
    unordered_set<char> nums;
    for (i = 0; i < n; i++) {
        alphabet[s[i] - 'a'].emplace_back(i);
        nums.emplace(s[i]);
    }

    string ans;
    cnt = 0;
    prev = -1;
    while (cnt < nums.size()) {
        for (i = 0; i < 26; i++) {
            if (alphabet[i].empty()) {
                continue;
            }
            idx = 0;
            for (j = 0; j < 26; j++) {
                if (j == i || alphabet[j].empty()) {
                    continue;
                }
                // alphabet[i]第一个大于prev的idx
                auto it = upper_bound(alphabet[i].begin(), alphabet[i].end(), prev);
                if (it == alphabet[i].end()) {
                    break;
                }
                idx = it - alphabet[i].begin();
                if (alphabet[i][idx] > alphabet[j].back()) {
                    break;
                }
            }
            if (j == 26) {
                ans += 'a' + i;
                prev = alphabet[i][idx];
                alphabet[i].clear();
                cnt++;
                break;
            } else {
                // cout << i << " " << j << endl;
            }
        }
        // cout << ans << endl;
    }
    return ans;
}


// LC1210
int minimumMoves(vector<vector<int>>& grid)
{
    int i, j;
    int n = grid.size();
    // 蛇长度为2, 以pair<int, int> 表示其形态 {头, 尾}
    // 所求 dist[pair<(n - 1, n - 1), (n - 1, n - 2)>]
    map<pair<int, int>, int> dist;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (j - 1 >= 0) {
                dist[{i * n + j, i * n + j - 1}] = INT_MAX;
            }
            if (j + 1 < n) {
            //    dist[{i * n + j, i * n + j + 1}] = INT_MAX;
            }
            if (i - 1 >= 0) {
                dist[{i * n + j, (i - 1) * n + j}] = INT_MAX;
            }
            if (i + 1 < n) {
            //    dist[{i * n + j, (i + 1) * n + j}] = INT_MAX;
            }
        }
    }

    priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<>> pq;
    pq.push({0, {1, 0}});
    while (!pq.empty()) {
        auto [d, state] = pq.top();
        pq.pop();

        if (dist[state] < d) {
            continue;
        }
        dist[state] = d;
        auto [head, tail] = state;
        // printf ("dist[%d %d] = %d\n", head, tail, dist[state]);
        if (head - tail == 1) { // 平行的
            if (head % n + 1 < n && grid[head / n][head % n + 1] == 0 && d + 1 < dist[{head + 1, head}]) { // 向右进一格
                dist[{head + 1, head}] = d + 1;
                pq.push({d + 1, {head + 1, head}});
            }
            if (head / n + 1 < n && grid[head / n + 1][head % n] == 0 && grid[tail / n + 1][tail % n] == 0) {
                if (d + 1 < dist[{head + n, head + n - 1}]) { // 向下平移一格
                    dist[{head + n, tail + n}] = d + 1;
                    pq.push({d + 1, {head + n, tail + n}});
                }
                if (d + 1 < dist[{tail + n, tail}]) { // 顺时针旋转90度
                    dist[{tail + n, tail}]= d + 1;
                    pq.push({d + 1, {tail + n, tail}});
                }
            }
        } else {
            if (head / n + 1 < n && grid[head / n + 1][head % n] == 0 && d + 1 < dist[{head + n, head}]) { // 向下进一格
                dist[{head + n, head}] = d + 1;
                pq.push({d + 1, {head + n, head}});
            }
            if (head % n + 1 < n && grid[head / n][head % n + 1] == 0 && grid[tail / n][tail % n + 1] == 0) {
                if (d + 1 < dist[{head + 1, tail + 1}]) { // 向右平移一格
                    dist[{head + 1, tail + 1}] = d + 1;
                    pq.push({d + 1, {head + 1, tail + 1}});
                }
                if (d + 1 < dist[{tail + 1, tail}]) { // 逆时针旋转90度
                    dist[{tail + 1, tail}] = d + 1;
                    pq.push({d + 1, {tail + 1, tail}});
                }
            }
        }
    }

    // int ans = min(dist[{n * n - 1, n * n - 2}], dist[{n * n - 1, n * n - 1 - n}]); 只能横着到终点
    int ans = dist[{n * n - 1, n * n - 2}];
    return ans == INT_MAX ? -1 : ans;
}


// LC1371
int findTheLongestSubstring(string s)
{
    int i;
    int n = s.size();
    int ans;
    string hash = "00000"; // 元音状态
    unordered_map<string, int> prefix;

    ans = 0;
    prefix[hash] = -1;
    for (i = 0; i < n; i++) {
        if (s[i] == 'a') {
            hash[0] = '0' + (hash[0] - '0' + 1) % 2;
        } else if (s[i] == 'e') {
            hash[1] = '0' + (hash[1] - '0' + 1) % 2;
        } else if (s[i] == 'i') {
            hash[2] = '0' + (hash[2] - '0' + 1) % 2;
        } else if (s[i] == 'o') {
            hash[3] = '0' + (hash[3] - '0' + 1) % 2;
        } else if (s[i] == 'u') {
            hash[4] = '0' + (hash[4] - '0' + 1) % 2;
        }
        if (prefix.count(hash) == 0) {
            prefix[hash] = i;
        } else {
            ans = max(ans, i - prefix[hash]);
        }
    }
    return ans;
}
int findTheLongestSubstring_beterrWay(string s)
{
    int i;
    int n = s.size();
    int ans;
    int state = 0; // 元音状态, 用整数代替字符串, 效率更高
    vector<int> prefix(32, -2);

    ans = 0;
    prefix[state] = -1;
    for (i = 0; i < n; i++) {
        if (s[i] == 'a') { // 异或模拟不进位加法
            state ^= (1 << 0);
        } else if (s[i] == 'e') {
            state ^= (1 << 1);
        } else if (s[i] == 'i') {
            state ^= (1 << 2);
        } else if (s[i] == 'o') {
            state ^= (1 << 3);
        } else if (s[i] == 'u') {
            state ^= (1 << 4);
        }
        if (prefix[state] == -2) {
            prefix[state] = i;
        } else {
            ans = max(ans, i - prefix[state]);
        }
    }
    return ans;
}


// LC2246
int longestPath(vector<int>& parent, string s)
{
    int i;
    int n = parent.size();
    int ans;
    vector<int> data(n); // data[i] - i为首节点最长路径
    vector<vector<int>> edges(n);

    for (i = 0; i < n; i++) {
        if (parent[i] != -1) {
            edges[i].emplace_back(parent[i]);
            edges[parent[i]].emplace_back(i);
        }
    }

    data.assign(n, -1);
    ans = 0;
    auto dfs = [&s, &edges, &data, &ans](auto&& self, int cur, int parent) -> int {
        if (data[cur] != -1) {
            return data[cur];
        }
        if (edges[cur].size() == 1 && edges[cur][0] == parent) {
            data[cur] = 1;
            return 1;
        }

        int len;
        int a, b; // 统计第一和第二长的路径

        a = b = 1;
        for (auto next : edges[cur]) {
            if (next != parent) {
                if (s[next] != s[cur]) {
                    len = self(self, next, cur) + 1;
                    if (len > a) {
                        b = a;
                        a = len;
                    } else if (len > b) {
                        b = len;
                    }
                } else {
                    self(self, next, cur); // 不满足条件, 向子节点寻找
                }
            }
        }

        data[cur] = a;
        ans = max(ans, a + b - 1);
        return data[cur];
    };

    dfs(dfs, 0, -1);
    return ans;
}


// LC3645
long long maxTotal(vector<int>& value, vector<int>& limit)
{
    int i;
    int t;
    int n = value.size();
    long long ans = 0;
    auto cmp = [](const tuple<int, int, int>& a, const tuple<int, int, int>& b) {
        if (get<0>(a) == get<0>(b)) {
            return get<1>(a) < get<1>(b);
        }
        return get<0>(a) > get<0>(b);
    };
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, decltype(cmp)> pq(cmp);
    vector<pair<int, int>> vp;
    for (i = 0; i < n; i++) {
        pq.push({limit[i], value[i], i});
        vp.push_back({limit[i], i});
    }
    sort(vp.begin(), vp.end());

    int last_max_idx = 0;
    int cur = 0;
    vector<int> active(n, 0);
    while (!pq.empty()) {
        auto [l, v, idx] = pq.top();
        pq.pop();
        if (active[idx] == -1) { // 永久非活跃
            continue;
        }
        active[idx] = 1;
        ans += v;
        cur++;
        // cout << v << endl;
        t = cur;
        for (i = last_max_idx; i < n; i++) {
            if (vp[i].first <= cur) {
                if (active[vp[i].second] == 1) {
                    t--;
                }
                active[vp[i].second] = -1;
            } else {
                last_max_idx = i;
                break;
            }
        }
        cur = t;
    }
    return ans;
}


// LC2528
long long maxPower(vector<int>& stations, int r, int k)
{
    int i;
    int n = stations.size();
    int g;
    long long left, right, mid;
    long long cur;

    left = 0;
    right = 1e15;

    while (left <= right) {
        mid = (right - left) / 2 + left;
        auto t = stations;
        cur = 0;
        g = k;
        for (i = 0; i <= r; i++) {
            cur += t[i];
        }
        if (cur < mid) {
            if (mid - cur > g) {
                right = mid - 1;
                continue;
            } else {
                t[r] += mid - cur;
                g -= mid - cur;
                cur = mid;
            }
        }
        for (i = 1; i < n; i++) {
            if (i + r < n) {
                cur += t[i + r];
            }
            if (i - r - 1 >= 0) {
                cur -= t[i - r - 1];
            }
            if (cur < mid) {
                if (mid - cur > g) {
                    right = mid - 1;
                    break;
                } else {
                    t[i + r < n ? i + r : n - 1] += mid - cur;
                    g -= mid - cur;
                    cur = mid;
                }
            }
        }
        if (i == n) {
            left = mid + 1;
        }
    }
    return right;
}


// LC3654
long long minArraySum(vector<int>& nums, int k)
{
    int i;
    int n = nums.size();
    long long inf = 1e15;
    long long cur;

    if (k == 1) {
        return 0;
    }

    // dp[i] - 前i个元素最小剩余元素和
    vector<long long> dp(n + 1);
    // r[i] - 前缀和除以k余数对应的最小剩余元素和
    vector<long long> r(k + 1, inf);

    r[0] = dp[0]= 0;
    cur = 0;
    for (i = 1; i <= n; i++) {
        cur += nums[i - 1];
        if (r[cur % k] == inf) {
            dp[i] = dp[i - 1] + nums[i - 1];
            r[cur % k] = dp[i];
        } else {
            dp[i] = min(dp[i - 1] + nums[i - 1], r[cur % k]);
            r[cur % k] = min(r[cur % k], dp[i]);
        }
    }
    return dp[n];
}


// LC679
bool judgePoint24(vector<int>& cards)
{
    int i;
    int p;
    int t;
    vector<vector<int>> permutation = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3},
        {2, 3, 1}, {3, 1, 2}, {3, 2, 1}};

    vector<char> operators = {'+', '-', '*', '/'};
    vector<char> op(3, ' ');

    // 计算后缀表达式
    auto Calc = [](string& expression) {
        int i;
        int n = expression.size();
        double a, b;
        double inf = 1e9;
        stack<double> st;

        for (i = 0; i < n; i++) {
            if (isdigit(expression[i])) {
                st.push(expression[i] - '0');
            } else {
                a = st.top();
                st.pop();
                b = st.top();
                st.pop();
                if (expression[i] == '+') {
                    st.push(b + a);
                } else if (expression[i] == '-') {
                    st.push(b - a);
                } else if (expression[i] == '*') {
                    st.push(b * a);
                } else {
                    if (a == 0) {
                        return inf;
                    }
                    st.push(b / a);
                }
            }
        }
        return st.top();
    };

    string expression;
    vector<vector<int>> c = GetArrPermutation(cards);
    for (auto& cards : c) {
        for (p = 0; p < 6; p++) {
            for (i = 0; i < 64; i++) { // 以4进制表示三个符号
                t = i;
                op[0] = operators[t % 4];
                t /= 4;

                op[1] = operators[t % 4];
                t /= 4;

                op[2] = operators[t % 4];

                expression.clear();
                if (permutation[p][0] == 1) {
                    expression += cards[0] + '0';
                    expression += cards[1] + '0';
                    expression += op[0];
                } else if (permutation[p][1] == 1) {
                    expression += cards[1] + '0';
                    expression += cards[2] + '0';
                    expression += op[1];
                } else {
                    expression += cards[2] + '0';
                    expression += cards[3] + '0';
                    expression += op[2];
                }

                if (permutation[p][0] == 2) {
                    if (permutation[p][1] == 1) {
                        expression = to_string(cards[0]) + expression;
                        expression += op[0];
                    }
                    if (permutation[p][2] == 1) {
                        expression += cards[0] + '0';
                        expression += cards[1] + '0';
                        expression += op[0];
                    }
                } else if (permutation[p][1] == 2) {
                    if (permutation[p][0] == 1) {
                        expression += cards[2] + '0';
                        expression += op[1];
                    }
                    if (permutation[p][2] == 1) {
                        expression = to_string(cards[1]) + expression;
                        expression += op[1];
                    }
                } else {
                    if (permutation[p][1] == 1) {
                        expression += cards[3] + '0';
                        expression += op[2];
                    }
                    if (permutation[p][0] == 1) {
                        expression += cards[2] + '0';
                        expression += cards[3] + '0';
                        expression += op[2];
                    }
                }

                if (permutation[p][0] == 3) {
                    expression = to_string(cards[0]) + expression;
                    expression += op[0];
                } else if (permutation[p][1] == 3) {
                    expression += op[1];
                } else {
                    expression += cards[3] + '0';
                    expression += op[2];
                }
                auto res = Calc(expression);
                if (fabs(res - 24) < 1e-5) {
                    cout << "expression = " << expression << endl;
                    return true;
                }
            }
        }
    }
    return false;
}


// LC1277
int countSquares(vector<vector<int>>& matrix)
{
    int i, j;
    int m = matrix.size();
    int n = matrix[0].size();
    int ans = 0;
    // dp[i][j] - 以matrix[i][j]为正方形右下角顶点的最大边长
    vector<vector<int>> dp(m, vector<int>(n, 0));

    for (i = 0; i < m; i++) {
        if (matrix[i][0]) {
            dp[i][0] = 1;
            ans++;
        }
    }
    for (j = 1; j < n; j++) {
        if (matrix[0][j]) {
            dp[0][j] = 1;
            ans++;
        }
    }
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            if (matrix[i][j]) {
                dp[i][j] = min({dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]}) + 1;
                // printf ("dp[%d][%d] = %d\n", i, j, dp[i][j]);
                ans += dp[i][j];
            }
        }
    }
    return ans;
}


// LC1316
int distinctEchoSubstrings(string text)
{
    // 字符串哈希
    int i, j;
    int n = text.size();
    int mod = 1e9 + 7;
    long long base = 1337;
    vector<vector<long long>> hash(n, vector<long long>(n));
    for (i = 0; i < n; i++) {
        hash[i][i] = text[i] - 'a' + 1;
        for (j = i + 1; j < n; j++) {
            hash[i][j] = (hash[i][j - 1] * base + text[j] - 'a' + 1) % mod;
        }
    }
    set<int> chosenHash;
    int ans = 0;
    int len;
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j += 2) {
            len = j - i + 1;
            if (hash[i][i + len / 2 - 1] == hash[i + len / 2][j] && 
                chosenHash.count(hash[i][j]) == 0) {
                ans++;
                chosenHash.emplace(hash[i][j]);
            }
        }
    }
    return ans;
}


// LC1520
vector<string> maxNumOfSubstrings(string s)
{
    int i, j;
    int n = s.size();
    vector<vector<int>> idx(26);
    for (i = 0; i < n; i++) {
        idx[s[i] - 'a'].emplace_back(i);
    }
    bool flag;
    vector<vector<int>> ranges;
    for (i = 0; i < 26; i++) {
        if (idx[i].empty()) {
            continue;
        }
        flag = true;
        int end = idx[i].back();
        for (j = idx[i][0]; j <= end; j++) {
            if (s[j] == 'a' + i) {
                continue;
            }
            // 不包含
            if (idx[s[j] - 'a'][0] < idx[i][0]) {
                flag = false;
                break;
            }
            // 延长end
            if (idx[s[j] - 'a'].back() > end) {
                end = idx[s[j] - 'a'].back();
            }
        }
        if (flag) {
            ranges.push_back({idx[i][0], end});
        }
    }

    // for (auto r : ranges) cout << r[0] << " " << r[1] << endl;
    int size = ranges.size();
    sort(ranges.begin(), ranges.end());
    // dp[i] - 以ranges[i]结尾的最多区间数
    vector<int> dp(size, 0);
    // prev[i] - 以ranges[i]结尾实现最多区间数时上一个range的idx
    vector<int> prev(size);
    // sum[i] - 以ranges[i]结尾实现最多区间数时最短区间字符串长度之和
    vector<int> sum(size);
    dp[0] = 1;
    prev[0] = -1;
    sum[0] = ranges[0][1] - ranges[0][0] + 1;

    int maxRanges = 0;
    int minLen = INT_MAX;
    int endIdx = 0;
    for (i = 1; i < size; i++) {
        dp[i] = 1;
        prev[i] = -1;
        sum[i] = ranges[i][1] - ranges[i][0] + 1;
        for (j = i - 1; j >= 0; j--) {
            if (ranges[j][1] < ranges[i][0]) {
                if (dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    prev[i] = j;
                    sum[i] = sum[j] + ranges[i][1] - ranges[i][0] + 1;
                } else if (dp[i] == dp[j] + 1) {
                    if (sum[j] + ranges[i][1] - ranges[i][0] + 1 < sum[i]) {
                        prev[i] = j;
                        sum[i] = sum[j] + ranges[i][1] - ranges[i][0] + 1;
                    }
                }
            }
        }
        // 顺便找最大分隔区间数量且总长度最短的最后一个区间idx
        if (dp[i] > maxRanges) {
            maxRanges = dp[i];
            minLen = sum[i];
            endIdx = i;
        } else if (dp[i] == maxRanges && minLen > sum[i]) {
            minLen = sum[i];
            endIdx = i;
        }
    }
    // cout << endIdx << endl;

    // 从后向前找子字符串
    i = endIdx;
    vector<string> ans;
    ans.emplace_back(s.substr(ranges[i][0], ranges[i][1] - ranges[i][0] + 1));
    while (prev[i] != -1) {
        i = prev[i];
        ans.emplace_back(s.substr(ranges[i][0], ranges[i][1] - ranges[i][0] + 1));
    }
    return ans;
}


// LC1579
int maxNumEdgesToRemove(int n, vector<vector<int>>& edges)
{
    // 查并集, 优先连通type3的路径
    int i;
    vector<vector<int>> e3;
    vector<vector<int>> e1;
    vector<vector<int>> e2;

    for (auto& edge : edges) {
        if (edge[0] == 3) {
            e3.push_back({edge[1], edge[2]});
        } else if (edge[0] == 1) {
            e1.push_back({edge[1], edge[2]});
        } else {
            e2.push_back({edge[1], edge[2]});
        }
    }

    int ans = 0;
    UnionFind uf(n + 1);
    for (auto& edge : e3) {
        auto u = edge[0];
        auto v = edge[1];
        if (uf.findSet(u) != uf.findSet(v)) {
            uf.unionSets(u, v);
        } else {
            ans++;
        }
    }
    auto t = uf;
    for (auto& edge : e1) {
        auto u = edge[0];
        auto v = edge[1];
        if (t.findSet(u) != t.findSet(v)) {
            t.unionSets(u, v);
        } else {
            ans++;
        }
    }
    auto f = uf;
    for (auto& edge : e2) {
        auto u = edge[0];
        auto v = edge[1];
        if (f.findSet(u) != f.findSet(v)) {
            f.unionSets(u, v);
        } else {
            ans++;
        }
    }
    int area = t.findSet(1);
    for (i = 1; i <= n; i++) {
        if (area != t.findSet(i)) {
            return -1;
        }
    }
    area = f.findSet(1);
    for (i = 1; i <= n; i++) {
        if (area != f.findSet(i)) {
            return -1;
        }
    }
    return ans;
}


// LC1354
bool isPossible(vector<int>& target)
{
    int i;
    int n = target.size();

    if (n == 1) {
        if (target[0] != 1) {
            return false;
        }
        return true;
    }

    priority_queue<int, vector<int>> pq;
    long long sum = 0;
    for (i = 0; i < n; i++) {
        sum += target[i];
        pq.push(target[i]);
    }
    while (1) {
        auto t1 = pq.top();
        if (t1 == 1) {
            break;
        }
        pq.pop();
        auto t2 = pq.top();
        if (t1 == t2) {
            return false;
        }

        // t1 - num * d <= t2;
        auto d = sum - t1;
        auto num = ceil((t1 - t2) * 1.0 / d);
        auto t3 = t1 - num * d;
        // printf ("%d %d %d\n", d, num, t3);
        if (t3 < 1) {
            return false;
        }
        sum = d + t3;
        pq.push(t3);
    }
    return true;
}


// LC3665
int uniquePaths(vector<vector<int>>& grid)
{
    int i, j;
    int m = grid.size();
    int n  = grid[0].size();
    vector<vector<vector<long long>>> dp(m, vector<vector<long long>>(n, vector<long long>(2, 0)));

    // dp[i][j][0 ~ 1] - 0 - 从左边进入  1 - 从上面进入
    dp[0][0][0] = 1;
    dp[0][0][1] = 1;
    for (i = 1; i < m; i++) {
        if (grid[i][0] == 0) {
            dp[i][0][1] = 1;
        } else {
            dp[i][0][1] = 1;
            break;
        }
    }
    for (j = 1; j < n; j++) {
        if (grid[0][j] == 0) {
            dp[0][j][0] = 1;
        } else {
            dp[0][j][0] = 1;
            break;
        }
    }
    int mod = 1e9 + 7;
    for (i = 1; i < m; i++) {
        for (j = 1; j < n; j++) {
            if ((grid[i][j] == 1 && i + 1 < m) || grid[i][j] == 0) {
                if (grid[i][j - 1] == 1) {
                    dp[i][j][0] = (dp[i][j][0] + dp[i][j - 1][1]) % mod;
                } else {
                    dp[i][j][0] = (dp[i][j][0] + dp[i][j - 1][0] + dp[i][j - 1][1]) % mod;
                }
            }
            if ((grid[i][j] == 1 && j + 1 < n) || grid[i][j] == 0) {
                if (grid[i - 1][j] == 1) {
                    dp[i][j][1] = (dp[i][j][1] + dp[i - 1][j][0]) % mod;
                } else {
                    dp[i][j][1] = (dp[i][j][1] + dp[i - 1][j][0] + dp[i - 1][j][1]) % mod;
                }
            }
        }
    }
    /* for (auto d : dp) {
        for (auto dd : d) cout << dd[0] + dd[1] << " "; cout << endl;
    } */
    return (dp[m - 1][n - 1][0] + dp[m - 1][n - 1][1]) % mod;
}


// LC3676 (原题long long 的返回值有点搞)
long long bowlSubarrays(vector<int>& nums)
{
    int i;
    int idx;
    int n = nums.size();
    long long ans = 0;
    stack<int> st;

    for (i = 0; i < n; i++) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        idx = st.top();
        while (nums[idx] < nums[i]) {
            // cout << idx << " " << i << endl;
            if (idx + 1 < i) {
                ans++;
            }
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    while (!st.empty()) {
        st.pop();
    }
    for (i = n - 1; i >= 0; i--) {
        if (st.empty()) {
            st.push(i);
            continue;
        }
        idx = st.top();
        while (nums[idx] < nums[i]) {
            // cout << idx << " " << i << endl;
            if (i + 1 < idx) {
                ans++;
            }
            st.pop();
            if (st.empty()) {
                break;
            }
            idx = st.top();
        }
        st.push(i);
    }
    return ans;
}


// LC2607
long long makeSubKSumEqual(vector<int>& arr, int k)
{
    int i, j;
    int n = arr.size();
    vector<vector<int>> groups(k);
    vector<int> visited(n, 0);
    for (i = 0; i < k; i++) {
        if (visited[i]) {
            continue;
        }
        j = i;
        while (visited[j] == 0) {
            visited[j] = 1;
            groups[i].emplace_back(arr[j]);
            j = (j + k) % n;
        }
    }

    // 每个groups[i] 取中位数
    int mid, len;
    long long ans = 0;
    for (i = 0; i < k; i++) {
        if (groups[i].empty()) {
            continue;
        }
        sort(groups[i].begin(), groups[i].end());
        len = groups[i].size();
        if (len % 2 == 1) {
            mid = groups[i][len / 2];
        } else {
            mid = (groups[i][len / 2 - 1] + groups[i][len / 2]) / 2;
        }
        for (j = 0; j < len; j++) {
            ans += abs(mid - groups[i][j]);
        }
    }
    return ans;
}